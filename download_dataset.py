"""
Basic Enhanced Script: Download and extract training datasets for Diffusion Policy.
✓ Download only if missing + extract only if not present  
✓ Automatic retry (up to 3 attempts)  
✓ Resume interrupted downloads (HTTP Range support)  
✓ Standard library only — no external dependencies
"""

from __future__ import annotations

import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
import shutil


CHUNK_SIZE = 1024 * 1024  # 1 MiB per chunk


def repo_root() -> Path:
    """Infer repository root directory."""
    return Path(__file__).resolve().parents[1]


def human_mb(num_bytes: int) -> str:
    """Convert bytes to human-readable MiB string."""
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def download_if_missing(url: str, dst: Path, max_retries: int = 3) -> None:
    """Download file only if it does not exist; supports resume + auto-retry."""
    if dst.exists():
        print(f"✓ {dst.name} already exists — skipping download.")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        # Check if partial download exists
        resume_size = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"Range": f"bytes={resume_size}-"} if resume_size > 0 else {}

        try:
            req = urllib.request.Request(url.strip(), headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                status = response.getcode()
                content_range = response.headers.get("Content-Range", "")
                content_length = response.headers.get("Content-Length")

                # Determine download mode
                if resume_size > 0 and status == 206:
                    # Partial content — resume
                    mode = "ab"
                    total = int(content_range.split("/")[-1]) if "/" in content_range else None
                elif resume_size == 0 and status == 200:
                    # Full download
                    mode = "wb"
                    total = int(content_length) if content_length else None
                else:
                    # Server doesn't support Range — start over
                    if tmp_path.exists():
                        tmp_path.unlink()
                    resume_size = 0
                    mode = "wb"
                    total = int(content_length) if content_length else None

                action = f" (resuming from {human_mb(resume_size)})" if resume_size > 0 else ""
                print(f"↓ Downloading: {dst.name}{action}")

                downloaded = resume_size
                with open(tmp_path, mode) as fout:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        fout.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded / total * 100
                            sys.stdout.write(
                                f"\r    Attempt {attempt}/{max_retries}: {percent:5.1f}% "
                                f"({human_mb(downloaded)}/{human_mb(total)})"
                            )
                        else:
                            sys.stdout.write(f"\r    Attempt {attempt}/{max_retries}: {human_mb(downloaded)} downloaded")
                        sys.stdout.flush()

                # Success — break retry loop
                break

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            print(f"\n⚠ Attempt {attempt}/{max_retries} failed: {type(e).__name__}: {e}")
            if attempt == max_retries:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(f"Download failed after {max_retries} attempts: {url}") from e
            time.sleep(1.5 ** attempt)  # Exponential backoff: ~1.5s, ~2.25s, ~3.4s

    # Rename temp file to final destination
    tmp_path.replace(dst)
    sys.stdout.write("\n")
    print(f"✓ {dst.name} download complete.")


def extract_if_missing(zip_path: Path, target_dir: Path, marker_name: str) -> None:
    """Extract zip only if marker directory does not exist."""
    marker_path = target_dir / marker_name
    if marker_path.exists():
        print(f"✓ {marker_name}/ already exists — skipping extraction.")
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP archive not found: {zip_path}")

    print(f"📦 Extracting: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=target_dir)
        print(f"✓ {marker_name}/ extraction complete.")
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Corrupted ZIP file: {zip_path}. Please delete and re-download.") from e


def main() -> None:
    data_dir = (repo_root() / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {"name": "pusht", "url": "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip", "zip": "pusht.zip"},
        {"name": "robomimic_image", "url": "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip", "zip": "robomimic_image.zip"},
    ]

    for ds in datasets:
        zip_path = data_dir / ds["zip"]

        # 1. Download (with resume + retry)
        download_if_missing(ds["url"], zip_path)

        # 2. Extract (if marker dir missing)
        extract_if_missing(zip_path, data_dir, ds["name"])

    print(f"\n🎉 All datasets ready at: {data_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user. Temporary files cleaned up automatically.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)