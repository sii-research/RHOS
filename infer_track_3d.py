#!/usr/bin/env python3
"""Track first-frame query pixels through a video with a D4RT checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from src.core import build_logger, load_checkpoint, load_yaml_config, seed_everything
from src.eval.tasks import _encode_model_memory, _model_clip_frames, _run_model_for_queries, _umeyama_sim3
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer 3D point tracks from a video using OpenD4RT.")
    parser.add_argument("--config", required=True, help="Model config yaml.")
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint path.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output-dir", default="tmp/infer_track_3d")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--query-chunk-size", type=int, default=4096)
    parser.add_argument("--points", default="", help="Semicolon-separated pixel coordinates: 'x1,y1;x2,y2'.")
    parser.add_argument("--grid-cols", type=int, default=8, help="Used when --points is empty.")
    parser.add_argument("--grid-rows", type=int, default=8, help="Used when --points is empty.")
    parser.add_argument("--margin-ratio", type=float, default=0.08)
    parser.add_argument("--max-points", type=int, default=64)
    parser.add_argument("--trail-length", type=int, default=12)
    parser.add_argument("--fps", type=float, default=0.0, help="Override output FPS. <=0 uses input video FPS.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on decoded video frames.")
    parser.add_argument(
        "--umeyama-slide-window",
        "--umeyama_slide_window",
        action="store_true",
        dest="umeyama_slide_window",
        help="Use paper-style overlapped sliding windows with Umeyama Sim(3) stitching for long sequences.",
    )
    parser.add_argument(
        "--umeyama-slide-window-dense",
        "--umeyama_slide_window_dense",
        action="store_true",
        dest="umeyama_slide_window_dense",
        help="Use dense high-confidence overlap point clouds to estimate chunk Sim(3) stitching.",
    )
    return parser.parse_args()


def _resolve_device(raw: str) -> torch.device:
    key = raw.strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but it is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    return {}


def _load_video_rgb(path: Path, max_frames: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 10.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if max_frames > 0 and len(frames) >= int(max_frames):
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0), fps


def _resize_video(video_rgb: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    h, w = image_hw
    resized = [
        cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if frame.shape[0] >= h else cv2.INTER_LINEAR)
        for frame in video_rgb
    ]
    return np.stack(resized, axis=0)


def _parse_query_points(spec: str, width: int, height: int, max_points: int) -> np.ndarray:
    coords: list[tuple[float, float]] = []
    for item in spec.split(";"):
        raw = item.strip()
        if not raw:
            continue
        parts = [v.strip() for v in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid point '{raw}'. Expected 'x,y'.")
        x = float(parts[0])
        y = float(parts[1])
        coords.append((x, y))
    if not coords:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.asarray(coords, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(max(width - 1, 0)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(max(height - 1, 0)))
    if pts.shape[0] > max_points:
        pts = pts[:max_points]
    return pts


def _grid_query_points(
    width: int,
    height: int,
    cols: int,
    rows: int,
    margin_ratio: float,
    max_points: int,
) -> np.ndarray:
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    margin_x = float(max(width - 1, 0)) * float(np.clip(margin_ratio, 0.0, 0.45))
    margin_y = float(max(height - 1, 0)) * float(np.clip(margin_ratio, 0.0, 0.45))
    xs = np.linspace(margin_x, float(max(width - 1, 0)) - margin_x, num=cols, dtype=np.float32)
    ys = np.linspace(margin_y, float(max(height - 1, 0)) - margin_y, num=rows, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    if grid.shape[0] > max_points:
        pick = np.linspace(0, grid.shape[0] - 1, num=max_points, dtype=np.int64)
        grid = grid[pick]
    return grid.astype(np.float32)


def _make_normalized_uv_grid(grid_size: int) -> np.ndarray:
    size = max(2, int(grid_size))
    coords = (np.arange(size, dtype=np.float32) + 0.5) / float(size)
    uv = np.stack(np.meshgrid(coords, coords, indexing="xy"), axis=-1).reshape(-1, 2)
    return uv.astype(np.float32)


def _make_anchor_clip_indices(num_frames: int, clip_frames: int, target_idx: int) -> np.ndarray:
    num_frames = int(num_frames)
    clip_frames = max(1, int(clip_frames))
    target_idx = int(np.clip(int(target_idx), 0, max(0, num_frames - 1)))
    if num_frames <= clip_frames:
        return np.arange(num_frames, dtype=np.int64)
    if clip_frames == 1:
        return np.asarray([target_idx], dtype=np.int64)
    tail_len = clip_frames - 1
    seg_end = target_idx + 1
    seg_start = max(1, seg_end - tail_len)
    seg_end = min(num_frames, seg_start + tail_len)
    seg_start = max(1, seg_end - tail_len)
    tail = np.arange(seg_start, seg_end, dtype=np.int64)
    if target_idx not in tail:
        tail[-1] = target_idx
        tail = np.unique(tail)
        while tail.shape[0] < tail_len:
            cand = max(1, int(tail[0]) - 1)
            if cand in tail:
                break
            tail = np.concatenate([np.asarray([cand], dtype=np.int64), tail], axis=0)
        tail = np.sort(tail)[-tail_len:]
    return np.concatenate([np.asarray([0], dtype=np.int64), tail], axis=0)


def _make_sliding_window_clip_ranges(
    num_frames: int,
    clip_frames: int,
    overlap_frames: int | None = None,
) -> list[tuple[int, int]]:
    num_frames = int(num_frames)
    clip_frames = max(1, int(clip_frames))
    if num_frames <= clip_frames:
        return [(0, num_frames)]
    overlap = clip_frames // 2 if overlap_frames is None else int(overlap_frames)
    overlap = max(1, min(overlap, clip_frames - 1))
    stride = max(1, clip_frames - overlap)
    starts = list(range(0, max(num_frames - clip_frames, 0) + 1, stride))
    last_start = max(0, num_frames - clip_frames)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [(int(start), int(min(num_frames, start + clip_frames))) for start in starts]


def _apply_sim3_to_xyz(
    xyz: np.ndarray,
    scale: float,
    rot: np.ndarray,
    trans: np.ndarray,
) -> np.ndarray:
    src = np.asarray(xyz, dtype=np.float64)
    out = np.full_like(src, np.nan, dtype=np.float64)
    flat_src = src.reshape(-1, 3)
    flat_out = out.reshape(-1, 3)
    valid = np.isfinite(flat_src).all(axis=1)
    if np.any(valid):
        flat_out[valid] = (float(scale) * (np.asarray(rot, dtype=np.float64) @ flat_src[valid].T)).T + np.asarray(trans, dtype=np.float64)
    return out.astype(np.float32)


def _estimate_overlap_sim3(
    *,
    prev_xyz_qt3: np.ndarray,
    curr_xyz_qt3: np.ndarray,
    prev_vis_qt: np.ndarray,
    curr_vis_qt: np.ndarray,
    prev_conf_qt: np.ndarray,
    curr_conf_qt: np.ndarray,
    keep_ratio: float = 0.85,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    prev_xyz = np.asarray(prev_xyz_qt3, dtype=np.float64)
    curr_xyz = np.asarray(curr_xyz_qt3, dtype=np.float64)
    prev_vis = np.asarray(prev_vis_qt, dtype=bool)
    curr_vis = np.asarray(curr_vis_qt, dtype=bool)
    prev_conf = np.asarray(prev_conf_qt, dtype=np.float64)
    curr_conf = np.asarray(curr_conf_qt, dtype=np.float64)

    valid = (
        np.isfinite(prev_xyz).all(axis=-1)
        & np.isfinite(curr_xyz).all(axis=-1)
        & prev_vis
        & curr_vis
        & np.isfinite(prev_conf)
        & np.isfinite(curr_conf)
    )
    if int(np.count_nonzero(valid)) < 3:
        return None
    src = curr_xyz[valid]
    dst = prev_xyz[valid]
    scores = np.minimum(prev_conf[valid], curr_conf[valid])
    if scores.size >= 4:
        order = np.argsort(scores)[::-1]
        keep = max(3, int(np.ceil(float(scores.size) * float(np.clip(keep_ratio, 0.0, 1.0)))))
        pick = order[:keep]
        src = src[pick]
        dst = dst[pick]
    sim3 = _umeyama_sim3(src, dst)
    if sim3 is None:
        return None
    scale, rot, trans = sim3
    return float(scale), np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64)


def _estimate_overlap_sim3_with_diagnostics(
    *,
    prev_xyz_qt3: np.ndarray,
    curr_xyz_qt3: np.ndarray,
    prev_vis_qt: np.ndarray,
    curr_vis_qt: np.ndarray,
    prev_conf_qt: np.ndarray,
    curr_conf_qt: np.ndarray,
    keep_ratio: float = 0.85,
) -> tuple[tuple[float, np.ndarray, np.ndarray] | None, dict[str, Any]]:
    prev_xyz = np.asarray(prev_xyz_qt3, dtype=np.float64)
    curr_xyz = np.asarray(curr_xyz_qt3, dtype=np.float64)
    prev_vis = np.asarray(prev_vis_qt, dtype=bool)
    curr_vis = np.asarray(curr_vis_qt, dtype=bool)
    prev_conf = np.asarray(prev_conf_qt, dtype=np.float64)
    curr_conf = np.asarray(curr_conf_qt, dtype=np.float64)
    diag: dict[str, Any] = {
        "candidate_count": 0,
        "selected_count": 0,
        "keep_ratio": float(keep_ratio),
        "sim3_success": False,
    }

    valid = (
        np.isfinite(prev_xyz).all(axis=-1)
        & np.isfinite(curr_xyz).all(axis=-1)
        & prev_vis
        & curr_vis
        & np.isfinite(prev_conf)
        & np.isfinite(curr_conf)
    )
    candidate_count = int(np.count_nonzero(valid))
    diag["candidate_count"] = candidate_count
    if candidate_count < 3:
        diag["reason"] = "not_enough_overlap_points"
        return None, diag

    src = curr_xyz[valid]
    dst = prev_xyz[valid]
    scores = np.minimum(prev_conf[valid], curr_conf[valid])
    selected_count = int(src.shape[0])
    if scores.size >= 4:
        order = np.argsort(scores)[::-1]
        selected_count = max(3, int(np.ceil(float(scores.size) * float(np.clip(keep_ratio, 0.0, 1.0)))))
        pick = order[:selected_count]
        src = src[pick]
        dst = dst[pick]
        scores = scores[pick]
    diag["selected_count"] = int(selected_count)
    if scores.size > 0:
        diag["selected_conf_min"] = float(np.nanmin(scores))
        diag["selected_conf_mean"] = float(np.nanmean(scores))

    sim3 = _umeyama_sim3(src, dst)
    if sim3 is None:
        diag["reason"] = "umeyama_failed"
        return None, diag
    scale, rot, trans = sim3
    diag.update(
        {
            "sim3_success": True,
            "scale": float(scale),
            "rotation": np.asarray(rot, dtype=np.float64).tolist(),
            "translation": np.asarray(trans, dtype=np.float64).tolist(),
        }
    )
    return (float(scale), np.asarray(rot, dtype=np.float64), np.asarray(trans, dtype=np.float64)), diag


def _merge_chunk_predictions(
    *,
    global_frame_ids: np.ndarray,
    chunk_xyz_local_qt3: np.ndarray,
    chunk_xyz_ref_qt3: np.ndarray,
    chunk_uv_qt2: np.ndarray,
    chunk_vis_qt: np.ndarray,
    chunk_conf_qt: np.ndarray,
    out_xyz_local_qt3: np.ndarray,
    out_xyz_ref_qt3: np.ndarray,
    out_uv_qt2: np.ndarray,
    out_vis_qt: np.ndarray,
    out_conf_qt: np.ndarray,
) -> None:
    frame_ids = np.asarray(global_frame_ids, dtype=np.int64).reshape(-1)
    for local_idx, global_idx in enumerate(frame_ids.tolist()):
        current_conf = np.asarray(chunk_conf_qt[:, local_idx], dtype=np.float32)
        existing_conf = np.asarray(out_conf_qt[:, global_idx], dtype=np.float32)
        current_ok = np.isfinite(current_conf)
        existing_ok = np.isfinite(existing_conf)
        better = (~existing_ok & current_ok) | (current_ok & existing_ok & (current_conf >= existing_conf))
        better |= (~existing_ok & ~current_ok & np.asarray(chunk_vis_qt[:, local_idx], dtype=bool))
        if not np.any(better):
            continue
        out_xyz_local_qt3[better, global_idx] = chunk_xyz_local_qt3[better, local_idx]
        out_xyz_ref_qt3[better, global_idx] = chunk_xyz_ref_qt3[better, local_idx]
        out_uv_qt2[better, global_idx] = chunk_uv_qt2[better, local_idx]
        out_vis_qt[better, global_idx] = chunk_vis_qt[better, local_idx]
        out_conf_qt[better, global_idx] = chunk_conf_qt[better, local_idx]


def _fill_overlap_query_sources(
    *,
    source_uv_chunk: np.ndarray,
    source_idx_local: np.ndarray,
    carry_query_ids: np.ndarray,
    overlap_frames: np.ndarray,
    tracks_uv: np.ndarray,
    tracks_visibility: np.ndarray,
    window_start: int,
    allow_invisible_fallback: bool,
) -> tuple[int, int]:
    carry_visible_count = 0
    carry_fallback_count = 0
    overlap_frames = np.asarray(overlap_frames, dtype=np.int64)
    for qi in np.asarray(carry_query_ids, dtype=np.int64).tolist():
        finite_overlap = np.isfinite(tracks_uv[qi, overlap_frames]).all(axis=-1)
        visible_frames = overlap_frames[tracks_visibility[qi, overlap_frames] & finite_overlap]
        if visible_frames.size > 0:
            src_global = int(visible_frames[-1])
            carry_visible_count += 1
        elif bool(allow_invisible_fallback):
            finite_frames = overlap_frames[finite_overlap]
            if finite_frames.size <= 0:
                continue
            src_global = int(finite_frames[-1])
            carry_fallback_count += 1
        else:
            continue
        source_uv_chunk[qi] = tracks_uv[qi, src_global]
        source_idx_local[qi] = src_global - int(window_start)
    return int(carry_visible_count), int(carry_fallback_count)


def _build_query_for_targets(
    query_uv_norm: np.ndarray,
    t_src: np.ndarray,
    t_tgt: np.ndarray,
    t_cam: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "u": torch.from_numpy(query_uv_norm[:, 0]).to(device=device, dtype=torch.float32),
        "v": torch.from_numpy(query_uv_norm[:, 1]).to(device=device, dtype=torch.float32),
        "t_src": torch.from_numpy(t_src).to(device=device, dtype=torch.long),
        "t_tgt": torch.from_numpy(t_tgt).to(device=device, dtype=torch.long),
        "t_cam": torch.from_numpy(t_cam).to(device=device, dtype=torch.long),
    }


def _run_queries_once(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    memory: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    t_src_idx: int,
    t_tgt_idx: int,
    t_cam_idx: int,
    query_chunk_size: int,
) -> dict[str, np.ndarray]:
    num_queries = int(query_uv_norm.shape[0])
    query = _build_query_for_targets(
        query_uv_norm=query_uv_norm,
        t_src=np.full((num_queries,), int(t_src_idx), dtype=np.int64),
        t_tgt=np.full((num_queries,), int(t_tgt_idx), dtype=np.int64),
        t_cam=np.full((num_queries,), int(t_cam_idx), dtype=np.int64),
        device=video_clip.device,
    )
    pred = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    return {key: value.numpy() for key, value in pred.items()}


def _run_full_clip_queries(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    query_chunk_size: int,
    num_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    num_queries = int(query_uv_norm.shape[0])
    repeated_uv = np.repeat(query_uv_norm, num_frames, axis=0)
    t_src = np.repeat(np.zeros((num_queries,), dtype=np.int64), num_frames, axis=0)
    t_tgt = np.tile(np.arange(num_frames, dtype=np.int64), num_queries)
    memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_ratio)
    query_local = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=t_tgt.copy(),
        device=video_clip.device,
    )
    query_ref = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=np.zeros_like(t_tgt),
        device=video_clip.device,
    )
    pred_local = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_local,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    pred_ref = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_ref,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    def _reshape(pred: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in pred.items():
            arr = value.numpy()
            if arr.ndim == 1:
                out[key] = arr.reshape(num_queries, num_frames)
            else:
                out[key] = arr.reshape(num_queries, num_frames, *arr.shape[1:])
        return out
    return _reshape(pred_local), _reshape(pred_ref)


def _run_point_cloud_queries_for_target_indices(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    memory: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    local_target_indices: np.ndarray,
    query_chunk_size: int,
) -> dict[str, np.ndarray]:
    target_ids = np.asarray(local_target_indices, dtype=np.int64).reshape(-1)
    num_queries = int(query_uv_norm.shape[0])
    num_targets = int(target_ids.shape[0])
    if num_targets <= 0:
        return {}
    repeated_uv = np.tile(np.asarray(query_uv_norm, dtype=np.float32), (num_targets, 1))
    t_src = np.repeat(target_ids, num_queries)
    t_tgt = np.repeat(target_ids, num_queries)
    t_cam = np.zeros_like(t_tgt)
    query = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=t_cam,
        device=video_clip.device,
    )
    pred = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    out: dict[str, np.ndarray] = {}
    for key, value in pred.items():
        arr = value.numpy()
        if arr.ndim == 1:
            out[key] = arr.reshape(num_targets, num_queries)
        else:
            out[key] = arr.reshape(num_targets, num_queries, *arr.shape[1:])
    return out


def _run_clip_queries_for_target_indices(
    *,
    model: torch.nn.Module,
    video_clip: torch.Tensor,
    aspect_ratio: torch.Tensor | None,
    memory: torch.Tensor | None,
    query_uv_norm: np.ndarray,
    query_src_indices: np.ndarray | None,
    local_target_indices: np.ndarray,
    query_chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    target_ids = np.asarray(local_target_indices, dtype=np.int64).reshape(-1)
    num_queries = int(query_uv_norm.shape[0])
    num_targets = int(target_ids.shape[0])
    if num_targets <= 0:
        return {}, {}

    repeated_uv = np.repeat(query_uv_norm, num_targets, axis=0)
    if query_src_indices is None:
        query_src = np.zeros((num_queries,), dtype=np.int64)
    else:
        query_src = np.asarray(query_src_indices, dtype=np.int64).reshape(-1)
        if query_src.shape[0] != num_queries:
            raise ValueError(
                f"query_src_indices must have shape [{num_queries}], got {query_src.shape}"
            )
    t_src = np.repeat(query_src, num_targets)
    t_tgt = np.tile(target_ids, num_queries)
    query_local = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=t_tgt.copy(),
        device=video_clip.device,
    )
    query_ref = _build_query_for_targets(
        query_uv_norm=repeated_uv,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=np.zeros_like(t_tgt),
        device=video_clip.device,
    )
    pred_local = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_local,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )
    pred_ref = _run_model_for_queries(
        model=model,
        video_b=video_clip,
        aspect_b=aspect_ratio,
        query=query_ref,
        chunk_size=max(1, int(query_chunk_size)),
        memory_b=memory,
    )

    def _reshape(pred: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in pred.items():
            arr = value.numpy()
            if arr.ndim == 1:
                out[key] = arr.reshape(num_queries, num_targets)
            else:
                out[key] = arr.reshape(num_queries, num_targets, *arr.shape[1:])
        return out

    return _reshape(pred_local), _reshape(pred_ref)


def _infer_tracks(
    *,
    model: torch.nn.Module,
    video_model_rgb: np.ndarray,
    query_uv_norm: np.ndarray,
    query_chunk_size: int,
    query_src_indices_global: np.ndarray | None = None,
    umeyama_slide_window: bool = False,
    umeyama_slide_window_dense: bool = False,
    dense_grid_size: int = 32,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    num_frames = int(video_model_rgb.shape[0])
    num_queries = int(query_uv_norm.shape[0])
    clip_frames = _model_clip_frames(model)
    aspect_value = np.asarray([[float(video_model_rgb.shape[2]) / float(max(1, video_model_rgb.shape[1]))]], dtype=np.float32)
    aspect_tensor = torch.from_numpy(aspect_value).to(device=device, dtype=torch.float32)

    tracks_xyz_local = np.full((num_queries, num_frames, 3), np.nan, dtype=np.float32)
    tracks_xyz_ref0 = np.full_like(tracks_xyz_local, np.nan)
    tracks_uv = np.full((num_queries, num_frames, 2), np.nan, dtype=np.float32)
    tracks_visibility = np.zeros((num_queries, num_frames), dtype=bool)
    tracks_confidence = np.full((num_queries, num_frames), np.nan, dtype=np.float32)
    dense_mode = bool(umeyama_slide_window_dense)
    slide_window_enabled = bool(umeyama_slide_window) or dense_mode
    stitch_diagnostics: dict[str, Any] = {
        "mode": "umeyama_slide_window_dense" if dense_mode else ("umeyama_slide_window" if slide_window_enabled else "anchor_clip"),
        "clip_frames": int(clip_frames),
        "dense_grid_size": int(dense_grid_size) if dense_mode else 0,
        "keep_ratio": 0.85,
        "chunks": [],
    }
    if query_src_indices_global is None:
        query_src_indices_global = np.zeros((num_queries,), dtype=np.int64)
    else:
        query_src_indices_global = np.asarray(query_src_indices_global, dtype=np.int64).reshape(num_queries)

    video_tensor = torch.from_numpy(video_model_rgb).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0

    with torch.no_grad():
        if num_frames <= clip_frames:
            pred_local, pred_ref = _run_full_clip_queries(
                model=model,
                video_clip=video_tensor,
                aspect_ratio=aspect_tensor,
                query_uv_norm=query_uv_norm,
                query_chunk_size=query_chunk_size,
                num_frames=num_frames,
            )
            tracks_xyz_local[:] = pred_local["xyz_3d"].astype(np.float32)
            tracks_xyz_ref0[:] = pred_ref["xyz_3d"].astype(np.float32)
            tracks_uv[:] = pred_local["uv_2d"].astype(np.float32)
            tracks_visibility[:] = 1.0 / (1.0 + np.exp(-pred_local["visibility"])) > 0.5
            tracks_confidence[:] = pred_local["confidence"].astype(np.float32)
        elif slide_window_enabled:
            window_ranges = _make_sliding_window_clip_ranges(num_frames=num_frames, clip_frames=clip_frames)
            dense_uv = _make_normalized_uv_grid(dense_grid_size) if dense_mode else None
            prev_dense_tail: dict[str, np.ndarray] | None = None
            for window_idx, (start, end) in enumerate(window_ranges):
                clip_indices = np.arange(start, end, dtype=np.int64)
                local_len = int(clip_indices.shape[0])
                video_clip = video_tensor[:, clip_indices]
                memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_tensor)
                chunk_diag: dict[str, Any] = {
                    "window_idx": int(window_idx),
                    "start": int(start),
                    "end": int(end),
                    "sim3_success": bool(window_idx == 0),
                    "candidate_count": 0,
                    "selected_count": 0,
                }

                source_uv_chunk = np.full((num_queries, 2), np.nan, dtype=np.float32)
                source_idx_local = np.full((num_queries,), -1, dtype=np.int64)
                use_original = (query_src_indices_global >= start) & (query_src_indices_global < end)
                carry_mask = np.zeros((num_queries,), dtype=bool)
                if np.any(use_original):
                    source_uv_chunk[use_original] = query_uv_norm[use_original]
                    source_idx_local[use_original] = query_src_indices_global[use_original] - int(start)
                if start > 0:
                    carry_mask = (~use_original) & (query_src_indices_global < start)
                    overlap_hi = min(end, start + max(1, clip_frames // 2))
                    if overlap_hi > start:
                        overlap_frames = np.arange(start, overlap_hi, dtype=np.int64)
                        carry_visible_count, carry_fallback_count = _fill_overlap_query_sources(
                            source_uv_chunk=source_uv_chunk,
                            source_idx_local=source_idx_local,
                            carry_query_ids=np.flatnonzero(carry_mask),
                            overlap_frames=overlap_frames,
                            tracks_uv=tracks_uv,
                            tracks_visibility=tracks_visibility,
                            window_start=int(start),
                            allow_invisible_fallback=bool(dense_mode),
                        )
                        chunk_diag["carry_visible_count"] = int(carry_visible_count)
                        if dense_mode:
                            # Dense stitching can keep a query alive even when the
                            # overlap visibility bit flickers low, as long as the
                            # overlap UV remains finite.
                            chunk_diag["carry_invisible_fallback_count"] = int(carry_fallback_count)
                valid_queries = np.isfinite(source_uv_chunk).all(axis=-1) & (source_idx_local >= 0)
                chunk_diag["carry_unresolved_count"] = int(np.count_nonzero(carry_mask) - np.count_nonzero(valid_queries & carry_mask))
                if not np.any(valid_queries):
                    continue

                valid_idx = np.flatnonzero(valid_queries)
                pred_local, pred_ref = _run_clip_queries_for_target_indices(
                    model=model,
                    video_clip=video_clip,
                    aspect_ratio=aspect_tensor,
                    memory=memory,
                    query_uv_norm=source_uv_chunk[valid_idx],
                    query_src_indices=source_idx_local[valid_idx],
                    local_target_indices=np.arange(local_len, dtype=np.int64),
                    query_chunk_size=query_chunk_size,
                )

                chunk_xyz_local = np.full((num_queries, local_len, 3), np.nan, dtype=np.float32)
                chunk_xyz_ref = np.full_like(chunk_xyz_local, np.nan)
                chunk_uv = np.full((num_queries, local_len, 2), np.nan, dtype=np.float32)
                chunk_vis = np.zeros((num_queries, local_len), dtype=bool)
                chunk_conf = np.full((num_queries, local_len), np.nan, dtype=np.float32)
                chunk_xyz_local[valid_idx] = pred_local["xyz_3d"].astype(np.float32)
                chunk_xyz_ref[valid_idx] = pred_ref["xyz_3d"].astype(np.float32)
                chunk_uv[valid_idx] = pred_local["uv_2d"].astype(np.float32)
                chunk_vis[valid_idx] = 1.0 / (1.0 + np.exp(-pred_local["visibility"])) > 0.5
                chunk_conf[valid_idx] = pred_local["confidence"].astype(np.float32)

                if window_idx > 0:
                    prev_start, prev_end = window_ranges[window_idx - 1]
                    overlap_start = int(max(start, prev_start))
                    overlap_end = int(min(end, prev_end))
                    chunk_diag["overlap_start"] = int(overlap_start)
                    chunk_diag["overlap_end"] = int(overlap_end)
                    if overlap_end > overlap_start:
                        overlap_global = np.arange(overlap_start, overlap_end, dtype=np.int64)
                        overlap_local = overlap_global - int(start)
                        if dense_mode and dense_uv is not None and prev_dense_tail is not None:
                            pred_dense = _run_point_cloud_queries_for_target_indices(
                                model=model,
                                video_clip=video_clip,
                                aspect_ratio=aspect_tensor,
                                memory=memory,
                                query_uv_norm=dense_uv,
                                local_target_indices=overlap_local,
                                query_chunk_size=query_chunk_size,
                            )
                            curr_dense_xyz = pred_dense["xyz_3d"].astype(np.float32)
                            curr_dense_conf = pred_dense.get(
                                "confidence",
                                np.ones(curr_dense_xyz.shape[:2], dtype=np.float32),
                            ).astype(np.float32)
                            curr_dense_vis = np.isfinite(curr_dense_xyz).all(axis=-1)
                            prev_frames = np.asarray(prev_dense_tail["frames"], dtype=np.int64)
                            prev_lookup = {int(frame): idx for idx, frame in enumerate(prev_frames.tolist())}
                            prev_indices = np.asarray([prev_lookup.get(int(frame), -1) for frame in overlap_global.tolist()], dtype=np.int64)
                            valid_prev = prev_indices >= 0
                            if np.all(valid_prev):
                                sim3, dense_diag = _estimate_overlap_sim3_with_diagnostics(
                                    prev_xyz_qt3=np.transpose(prev_dense_tail["xyz"][prev_indices], (1, 0, 2)),
                                    curr_xyz_qt3=np.transpose(curr_dense_xyz, (1, 0, 2)),
                                    prev_vis_qt=np.transpose(prev_dense_tail["vis"][prev_indices], (1, 0)),
                                    curr_vis_qt=np.transpose(curr_dense_vis, (1, 0)),
                                    prev_conf_qt=np.transpose(prev_dense_tail["conf"][prev_indices], (1, 0)),
                                    curr_conf_qt=np.transpose(curr_dense_conf, (1, 0)),
                                )
                                chunk_diag.update(dense_diag)
                            else:
                                sim3 = None
                                chunk_diag["reason"] = "previous_dense_overlap_missing"
                        else:
                            sim3, sparse_diag = _estimate_overlap_sim3_with_diagnostics(
                                prev_xyz_qt3=tracks_xyz_ref0[:, overlap_global],
                                curr_xyz_qt3=chunk_xyz_ref[:, overlap_local],
                                prev_vis_qt=tracks_visibility[:, overlap_global],
                                curr_vis_qt=chunk_vis[:, overlap_local],
                                prev_conf_qt=tracks_confidence[:, overlap_global],
                                curr_conf_qt=chunk_conf[:, overlap_local],
                            )
                            chunk_diag.update(sparse_diag)
                        if sim3 is not None:
                            scale, rot, trans = sim3
                            chunk_xyz_ref = _apply_sim3_to_xyz(chunk_xyz_ref, scale, rot, trans)
                            chunk_diag["sim3_success"] = True
                        else:
                            scale = 1.0
                            rot = np.eye(3, dtype=np.float64)
                            trans = np.zeros((3,), dtype=np.float64)
                    else:
                        scale = 1.0
                        rot = np.eye(3, dtype=np.float64)
                        trans = np.zeros((3,), dtype=np.float64)
                        chunk_diag["reason"] = "no_overlap"
                else:
                    scale = 1.0
                    rot = np.eye(3, dtype=np.float64)
                    trans = np.zeros((3,), dtype=np.float64)

                _merge_chunk_predictions(
                    global_frame_ids=clip_indices,
                    chunk_xyz_local_qt3=chunk_xyz_local,
                    chunk_xyz_ref_qt3=chunk_xyz_ref,
                    chunk_uv_qt2=chunk_uv,
                    chunk_vis_qt=chunk_vis,
                    chunk_conf_qt=chunk_conf,
                    out_xyz_local_qt3=tracks_xyz_local,
                    out_xyz_ref_qt3=tracks_xyz_ref0,
                    out_uv_qt2=tracks_uv,
                    out_vis_qt=tracks_visibility,
                    out_conf_qt=tracks_confidence,
                )
                if dense_mode and dense_uv is not None and window_idx + 1 < len(window_ranges):
                    next_start, _ = window_ranges[window_idx + 1]
                    tail_start = int(max(start, next_start))
                    tail_end = int(end)
                    if tail_end > tail_start:
                        tail_global = np.arange(tail_start, tail_end, dtype=np.int64)
                        tail_local = tail_global - int(start)
                        pred_tail = _run_point_cloud_queries_for_target_indices(
                            model=model,
                            video_clip=video_clip,
                            aspect_ratio=aspect_tensor,
                            memory=memory,
                            query_uv_norm=dense_uv,
                            local_target_indices=tail_local,
                            query_chunk_size=query_chunk_size,
                        )
                        tail_xyz = pred_tail["xyz_3d"].astype(np.float32)
                        tail_conf = pred_tail.get(
                            "confidence",
                            np.ones(tail_xyz.shape[:2], dtype=np.float32),
                        ).astype(np.float32)
                        tail_vis = np.isfinite(tail_xyz).all(axis=-1)
                        tail_xyz_global = _apply_sim3_to_xyz(tail_xyz, scale, rot, trans)
                        prev_dense_tail = {
                            "frames": tail_global,
                            "xyz": tail_xyz_global,
                            "vis": tail_vis,
                            "conf": tail_conf,
                        }
                stitch_diagnostics["chunks"].append(chunk_diag)
        else:
            clip_groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
            for frame_idx in range(num_frames):
                clip_indices = _make_anchor_clip_indices(num_frames=num_frames, clip_frames=clip_frames, target_idx=frame_idx)
                local_tgt_idx = int(np.flatnonzero(clip_indices == frame_idx)[0])
                clip_groups.setdefault(tuple(int(v) for v in clip_indices.tolist()), []).append((frame_idx, local_tgt_idx))

            for clip_key, assignments in clip_groups.items():
                clip_indices = np.asarray(clip_key, dtype=np.int64)
                video_clip = video_tensor[:, clip_indices]
                memory = _encode_model_memory(model=model, video_b=video_clip, aspect_b=aspect_tensor)

                global_frame_ids = np.asarray([item[0] for item in assignments], dtype=np.int64)
                local_target_ids = np.asarray([item[1] for item in assignments], dtype=np.int64)
                local_src_ids = np.full((num_queries,), -1, dtype=np.int64)
                for qi in range(num_queries):
                    hits = np.flatnonzero(clip_indices == int(query_src_indices_global[qi]))
                    if hits.size > 0:
                        local_src_ids[qi] = int(hits[0])
                valid_idx = np.flatnonzero(local_src_ids >= 0)
                if valid_idx.size <= 0:
                    continue
                pred_local, pred_ref = _run_clip_queries_for_target_indices(
                    model=model,
                    video_clip=video_clip,
                    aspect_ratio=aspect_tensor,
                    memory=memory,
                    query_uv_norm=query_uv_norm[valid_idx],
                    query_src_indices=local_src_ids[valid_idx],
                    local_target_indices=local_target_ids,
                    query_chunk_size=query_chunk_size,
                )

                tracks_xyz_local[valid_idx[:, None], global_frame_ids[None, :]] = pred_local["xyz_3d"].astype(np.float32)
                tracks_xyz_ref0[valid_idx[:, None], global_frame_ids[None, :]] = pred_ref["xyz_3d"].astype(np.float32)
                tracks_uv[valid_idx[:, None], global_frame_ids[None, :]] = pred_local["uv_2d"].astype(np.float32)
                tracks_visibility[valid_idx[:, None], global_frame_ids[None, :]] = 1.0 / (1.0 + np.exp(-pred_local["visibility"])) > 0.5
                tracks_confidence[valid_idx[:, None], global_frame_ids[None, :]] = pred_local["confidence"].astype(np.float32)

    return {
        "tracks_xyz_local": tracks_xyz_local,
        "tracks_xyz_ref0": tracks_xyz_ref0,
        "tracks_uv_norm": tracks_uv,
        "tracks_visibility": tracks_visibility,
        "tracks_confidence": tracks_confidence,
        "clip_frames": np.asarray(clip_frames, dtype=np.int32),
        "stitch_diagnostics": stitch_diagnostics,
    }


def _palette_bgr(num_colors: int) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int]] = []
    for idx in range(max(1, int(num_colors))):
        hue = 179.0 * float(idx) / float(max(1, num_colors))
        hsv = np.uint8([[[int(round(hue)), 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


def _draw_overlay_tracks(
    frame_rgb: np.ndarray,
    tracks_uv_px: np.ndarray,
    visibility: np.ndarray,
    query_points_px: np.ndarray,
    frame_idx: int,
    trail_length: int,
    colors_bgr: list[tuple[int, int, int]],
) -> np.ndarray:
    canvas = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    num_queries = int(query_points_px.shape[0])
    start = max(0, int(frame_idx) - max(1, int(trail_length)) + 1)
    for qid in range(num_queries):
        color = colors_bgr[qid % len(colors_bgr)]
        for tidx in range(start + 1, frame_idx + 1):
            if not bool(visibility[qid, tidx - 1]) or not bool(visibility[qid, tidx]):
                continue
            p0 = tracks_uv_px[qid, tidx - 1]
            p1 = tracks_uv_px[qid, tidx]
            if not np.isfinite(p0).all() or not np.isfinite(p1).all():
                continue
            cv2.line(
                canvas,
                (int(round(float(p0[0]))), int(round(float(p0[1])))),
                (int(round(float(p1[0]))), int(round(float(p1[1])))),
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        if bool(visibility[qid, frame_idx]) and np.isfinite(tracks_uv_px[qid, frame_idx]).all():
            pt = tracks_uv_px[qid, frame_idx]
            center = (int(round(float(pt[0]))), int(round(float(pt[1]))))
            cv2.circle(canvas, center, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, center, 3, color, -1, lineType=cv2.LINE_AA)
    if frame_idx == 0:
        for qid in range(num_queries):
            pt = query_points_px[qid]
            center = (int(round(float(pt[0]))), int(round(float(pt[1]))))
            cv2.drawMarker(canvas, center, colors_bgr[qid % len(colors_bgr)], markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    return canvas


def _rotation_matrix(elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = math.radians(float(elev_deg))
    azim = math.radians(float(azim_deg))
    cz, sz = math.cos(azim), math.sin(azim)
    cx, sx = math.cos(elev), math.sin(elev)
    rz = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return rx @ rz


def _project_view(points_xyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    return pts @ rotation.T


def _view_bounds(points_xyz: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    valid = np.isfinite(pts).all(axis=-1)
    if not np.any(valid):
        return np.zeros((2,), dtype=np.float32), 1.0
    proj = _project_view(pts[valid], rotation)
    xy = proj[:, :2]
    center = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    radius = float(np.max(xy.max(axis=0) - xy.min(axis=0)) * 0.6)
    return center.astype(np.float32), max(radius, 1e-3)


def _canvas_xy(points_xyz: np.ndarray, center: np.ndarray, radius: float, panel_size: int, rotation: np.ndarray) -> np.ndarray:
    proj = _project_view(points_xyz, rotation)
    xy = proj[:, :2]
    xy = (xy - center[None, :]) / float(radius)
    xy_px = np.empty_like(xy, dtype=np.float32)
    xy_px[:, 0] = (xy[:, 0] * 0.42 + 0.5) * float(panel_size - 1)
    xy_px[:, 1] = (0.5 - xy[:, 1] * 0.42) * float(panel_size - 1)
    return xy_px


def _draw_view_panel(
    tracks_xyz: np.ndarray,
    visibility: np.ndarray,
    frame_idx: int,
    trail_length: int,
    colors_bgr: list[tuple[int, int, int]],
    rotation: np.ndarray,
    center: np.ndarray,
    radius: float,
    panel_size: int,
    title: str,
) -> np.ndarray:
    panel = np.full((panel_size, panel_size, 3), 248, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_size - 1, panel_size - 1), (220, 220, 220), 1)
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    num_queries = int(tracks_xyz.shape[0])
    start = max(0, int(frame_idx) - max(1, int(trail_length)) + 1)
    for qid in range(num_queries):
        color = colors_bgr[qid % len(colors_bgr)]
        for tidx in range(start + 1, frame_idx + 1):
            if not bool(visibility[qid, tidx - 1]) or not bool(visibility[qid, tidx]):
                continue
            seg_xyz = tracks_xyz[qid, [tidx - 1, tidx]]
            if not np.isfinite(seg_xyz).all():
                continue
            seg_xy = _canvas_xy(seg_xyz, center=center, radius=radius, panel_size=panel_size, rotation=rotation)
            cv2.line(
                panel,
                (int(round(float(seg_xy[0, 0]))), int(round(float(seg_xy[0, 1])))),
                (int(round(float(seg_xy[1, 0]))), int(round(float(seg_xy[1, 1])))),
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        if bool(visibility[qid, frame_idx]) and np.isfinite(tracks_xyz[qid, frame_idx]).all():
            point_xy = _canvas_xy(
                tracks_xyz[qid, frame_idx : frame_idx + 1],
                center=center,
                radius=radius,
                panel_size=panel_size,
                rotation=rotation,
            )[0]
            center_px = (int(round(float(point_xy[0]))), int(round(float(point_xy[1]))))
            cv2.circle(panel, center_px, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(panel, center_px, 3, color, -1, lineType=cv2.LINE_AA)
    return panel


def _write_track_video(
    *,
    output_path: Path,
    video_rgb: np.ndarray,
    tracks_uv_px: np.ndarray,
    tracks_xyz_ref0: np.ndarray,
    visibility: np.ndarray,
    query_points_px: np.ndarray,
    fps: float,
    trail_length: int,
) -> None:
    num_queries = int(query_points_px.shape[0])
    colors_bgr = _palette_bgr(num_queries)
    panel_size = int(max(video_rgb.shape[1], video_rgb.shape[2]))
    views = [
        ("Input Video", None, None, None),
        ("3D View A", 24.0, -58.0, _rotation_matrix(24.0, -58.0)),
        ("3D View B", 16.0, 24.0, _rotation_matrix(16.0, 24.0)),
        ("3D View C", 75.0, -92.0, _rotation_matrix(75.0, -92.0)),
    ]

    visible_xyz = tracks_xyz_ref0[np.isfinite(tracks_xyz_ref0).all(axis=-1) & visibility]
    bounds: dict[str, tuple[np.ndarray, float]] = {}
    for title, elev, azim, rotation in views:
        if rotation is None:
            continue
        bounds[title] = _view_bounds(visible_xyz, rotation)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(fps, 1.0)),
        (panel_size * 2, panel_size * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

    try:
        for frame_idx in range(int(video_rgb.shape[0])):
            overlay = _draw_overlay_tracks(
                frame_rgb=video_rgb[frame_idx],
                tracks_uv_px=tracks_uv_px,
                visibility=visibility,
                query_points_px=query_points_px,
                frame_idx=frame_idx,
                trail_length=trail_length,
                colors_bgr=colors_bgr,
            )
            overlay = cv2.resize(overlay, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR)
            panels = [overlay]
            for title, elev, azim, rotation in views[1:]:
                center, radius = bounds[title]
                panel = _draw_view_panel(
                    tracks_xyz=tracks_xyz_ref0,
                    visibility=visibility,
                    frame_idx=frame_idx,
                    trail_length=trail_length,
                    colors_bgr=colors_bgr,
                    rotation=rotation,
                    center=center,
                    radius=radius,
                    panel_size=panel_size,
                    title=title,
                )
                panels.append(panel)
            top = np.concatenate([panels[0], panels[1]], axis=1)
            bottom = np.concatenate([panels[2], panels[3]], axis=1)
            canvas = np.concatenate([top, bottom], axis=0)
            writer.write(canvas)
    finally:
        writer.release()


def _save_first_frame_preview(output_path: Path, first_frame_rgb: np.ndarray, query_points_px: np.ndarray) -> None:
    colors_bgr = _palette_bgr(int(query_points_px.shape[0]))
    canvas = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)
    for qid, point in enumerate(query_points_px):
        center = (int(round(float(point[0]))), int(round(float(point[1]))))
        cv2.drawMarker(canvas, center, colors_bgr[qid % len(colors_bgr)], markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        cv2.putText(
            canvas,
            str(qid),
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(qid),
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors_bgr[qid % len(colors_bgr)],
            1,
            cv2.LINE_AA,
        )
    imageio.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger("infer_track_3d", output_dir)

    config = load_yaml_config(args.config)
    seed_everything(int(config.get_path("experiment.seed", 42)), deterministic=True)

    video_path = Path(args.video)
    ckpt_path = Path(args.ckpt_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    video_rgb, video_fps = _load_video_rgb(video_path, max_frames=int(args.max_frames))
    original_t, original_h, original_w = video_rgb.shape[:3]
    image_size = config.get_path("model.input.image_size", [original_h, original_w])
    model_h = int(image_size[0])
    model_w = int(image_size[1])
    video_model_rgb = _resize_video(video_rgb, image_hw=(model_h, model_w))

    query_points_px = _parse_query_points(args.points, width=original_w, height=original_h, max_points=int(args.max_points))
    if query_points_px.size == 0:
        query_points_px = _grid_query_points(
            width=original_w,
            height=original_h,
            cols=int(args.grid_cols),
            rows=int(args.grid_rows),
            margin_ratio=float(args.margin_ratio),
            max_points=int(args.max_points),
        )
    if query_points_px.size == 0:
        raise RuntimeError("No query points were generated.")
    query_uv_norm = query_points_px.copy()
    query_uv_norm[:, 0] /= float(max(original_w - 1, 1))
    query_uv_norm[:, 1] /= float(max(original_h - 1, 1))

    model = build_model(config["model"]).eval()
    payload = load_checkpoint(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"No model weights found in checkpoint: {ckpt_path}")
    load_result = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded checkpoint %s", ckpt_path)
    logger.info("Missing keys: %d  Unexpected keys: %d", len(load_result.missing_keys), len(load_result.unexpected_keys))

    device = _resolve_device(args.device)
    model.to(device).eval()

    logger.info(
        "Running track inference: frames=%d, video_hw=%dx%d, model_hw=%dx%d, queries=%d, clip_frames=%d",
        original_t,
        original_h,
        original_w,
        model_h,
        model_w,
        query_points_px.shape[0],
        _model_clip_frames(model),
    )
    results = _infer_tracks(
        model=model,
        video_model_rgb=video_model_rgb,
        query_uv_norm=query_uv_norm.astype(np.float32),
        query_chunk_size=int(args.query_chunk_size),
        umeyama_slide_window=bool(args.umeyama_slide_window),
        umeyama_slide_window_dense=bool(args.umeyama_slide_window_dense),
    )

    tracks_uv_norm = results["tracks_uv_norm"].astype(np.float32)
    tracks_uv_px = tracks_uv_norm.copy()
    tracks_uv_px[..., 0] *= float(max(original_w - 1, 1))
    tracks_uv_px[..., 1] *= float(max(original_h - 1, 1))

    npz_path = output_dir / "tracks_3d_predictions.npz"
    preview_path = output_dir / "first_frame_queries.png"
    video_out_path = output_dir / "track_views.mp4"
    meta_path = output_dir / "metadata.json"

    np.savez_compressed(
        npz_path,
        query_points_px=query_points_px.astype(np.float32),
        query_uv_norm=query_uv_norm.astype(np.float32),
        tracks_uv_norm=tracks_uv_norm.astype(np.float32),
        tracks_uv_px=tracks_uv_px.astype(np.float32),
        tracks_visibility=results["tracks_visibility"].astype(np.bool_),
        tracks_confidence=results["tracks_confidence"].astype(np.float32),
        tracks_xyz_local=results["tracks_xyz_local"].astype(np.float32),
        tracks_xyz_ref0=results["tracks_xyz_ref0"].astype(np.float32),
        video_shape_original=np.asarray(video_rgb.shape, dtype=np.int32),
        video_shape_model=np.asarray(video_model_rgb.shape, dtype=np.int32),
        model_clip_frames=np.asarray(results["clip_frames"], dtype=np.int32),
    )
    _save_first_frame_preview(preview_path, video_rgb[0], query_points_px)
    _write_track_video(
        output_path=video_out_path,
        video_rgb=video_rgb,
        tracks_uv_px=tracks_uv_px,
        tracks_xyz_ref0=results["tracks_xyz_ref0"].astype(np.float32),
        visibility=results["tracks_visibility"].astype(bool),
        query_points_px=query_points_px.astype(np.float32),
        fps=float(args.fps if args.fps > 0.0 else video_fps),
        trail_length=int(args.trail_length),
    )

    metadata = {
        "video": str(video_path),
        "checkpoint": str(ckpt_path),
        "config": str(args.config),
        "num_frames": int(video_rgb.shape[0]),
        "video_size_original": [int(original_h), int(original_w)],
        "video_size_model": [int(model_h), int(model_w)],
        "num_queries": int(query_points_px.shape[0]),
        "clip_frames": int(results["clip_frames"]),
        "umeyama_slide_window": bool(args.umeyama_slide_window),
        "umeyama_slide_window_dense": bool(args.umeyama_slide_window_dense),
        "stitch_diagnostics": results.get("stitch_diagnostics", {}),
        "video_fps_input": float(video_fps),
        "video_fps_output": float(args.fps if args.fps > 0.0 else video_fps),
        "notes": {
            "tracks_xyz_local": "Per-frame local camera coordinates with t_src=0, t_tgt=t, t_cam=t.",
            "tracks_xyz_ref0": "Shared reference coordinates with t_src=0, t_tgt=t, t_cam=0. Used for 3D multi-view visualization.",
            "tracks_uv_px": "Predicted target-frame pixel coordinates in the original video resolution.",
        },
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved results to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
