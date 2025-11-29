"""
Usage:
python eval.py --checkpoint_dir data/outputs4/pusht/flow_L1_10/run_4/checkpoints
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import yaml
import re
import numpy as np
import random
from termcolor import colored
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf


def find_best_checkpoint(checkpoint_dir):
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    best_ckpt = None
    best_score = -float('inf')

    pattern = re.compile(r'epoch=\d+-test_mean_score=([\d.]+)\.ckpt')
    for f in ckpt_files:
        match = pattern.match(f)
        if match:
            score = float(match.group(1))
            if score > best_score:
                best_score = score
                best_ckpt = f

    if best_ckpt is None:
        raise FileNotFoundError(
            colored(f"No valid checkpoint found in {checkpoint_dir} matching pattern 'epoch=...-test_mean_score=....ckpt'", "red")
        )
    
    return os.path.join(checkpoint_dir, best_ckpt)


def get_base_log_path(base_dir, policy, nfe):
    """Return the base path without version suffix: eval_log_p1_n2.yaml"""
    base_dir = pathlib.Path(base_dir)
    stem = f"eval_log_p{policy}_n{nfe}"
    return base_dir / (stem + ".yaml")

def get_log_path(base_dir, policy, nfe):
    """Return the base path with version suffix: eval_log_p1_n2.yaml"""
    base_dir = pathlib.Path(base_dir)
    stem = f"eval_log_p{policy}_n{nfe}"
    
    # Try base name first
    candidate = base_dir / (stem + ".yaml")
    if not candidate.exists():
        return candidate
    # Then try with version suffix: _v2, _v3, ...
    version = 2
    while True:
        versioned_stem = f"{stem}_v{version}"
        candidate = base_dir / (versioned_stem + ".yaml")
        if not candidate.exists():
            return candidate
        version += 1


def to_python_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            # print("This is Tensor")
            return value.item()
        else:
            return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or (value.ndim == 1 and value.size == 1):
            # print("This is ndarray")
            return value.item()
        else:
            return None
    if np.isscalar(value):
        try:
            # print("This is isscalar")
            return float(value) if isinstance(value, (np.floating, float)) else int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float, bool)) or value is None:
        print("This is float")
        return value
    return None


@click.command()
@click.option('-c', '--checkpoint_dir', required=True, help="Directory containing .ckpt files")
@click.option('-n', '--nfe', default=2, type=float)
@click.option('-p', '--policy', default=1, type=int)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint_dir, nfe, policy, device):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(colored(f"checkpoint_dir must be a directory: {checkpoint_dir}", "red"))

    output_dir = checkpoint_dir.parent / "eval_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Check if evaluation already exists ---
    base_log_path = get_base_log_path(output_dir, policy, nfe)
    # if base_log_path.exists():
    #     print(colored(f"[SKIP] Evaluation already exists at {base_log_path}. Skipping.", "yellow"))
    #     return  # Early exit

    # --- Proceed with evaluation ---
    checkpoint = find_best_checkpoint(str(checkpoint_dir))
    print(colored(f"[INFO] Using checkpoint: {checkpoint}", "cyan"))

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(cfg, False)
    
    log_policy = "flow_L1sample"
    if policy == 1:
        # 使用原始 FM 的 inference
        cfg.policy._target_ = "diffusion_policy.policy.flow_test_NFE_origin.FlowUnetL1SampleHybridImagePolicy"
        log_policy = "flow_test_NFE_origin"
        # 转成 int
        cfg.policy.num_inference_steps = int(nfe)
    else:
        # 使用论文中的两步 inference
        cfg.policy._target_ = "diffusion_policy.policy.flow_test_NFE_ours.FlowUnetL1SampleHybridImagePolicy"
        log_policy = "flow_test_NFE_ours"
        # 这里是当做 first_t 用
        cfg.policy.num_inference_steps = nfe
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy_model = workspace.model
    if cfg.training.use_ema:
        policy_model = workspace.ema_model

    device = torch.device(device)
    policy_model.to(device)
    policy_model.eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(output_dir))
    runner_log = env_runner.run(policy_model)
    

    test_mean_score = float(runner_log.get("test/mean_score", None))
    train_mean_score = float(runner_log.get("train/mean_score", None))

    cli_args = {
        "checkpoint": str(checkpoint),
        "nfe": nfe,
        "policy": log_policy,
        "test_mean_score": test_mean_score,
        "train_mean_score": train_mean_score,
        "device": str(device)
    }

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    metrics = {}
    for key, value in runner_log.items():
        scalar_val = to_python_scalar(value)
        if scalar_val is not None:
            metrics[key] = scalar_val

    final_log = {
        "cli_args": cli_args,
        "config": cfg_dict,
        "metrics": metrics
    }

    # Save to base name
    out_path = base_log_path
    with open(out_path, 'w') as f:
        yaml.dump(final_log, f, default_flow_style=False, indent=2, sort_keys=False)

    print(colored(f"[INFO] Evaluation log saved to {out_path}", "green"))


if __name__ == '__main__':
    main()