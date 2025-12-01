#!/usr/bin/env python3
"""Single-GPU task launcher with file-based lock coordination.

Supports incremental task execution on a single GPU (specified via --gpu-id).
Uses `.locked`/`.done` marker files to coordinate task state.

Usage:
    python launch_worker_single_gpu.py --gpu-id 2
"""

import json
import os
import sys
import subprocess
import time
import random
from pathlib import Path
from multiprocessing import Process, Queue
import argparse
import re
import shutil
from typing import Optional, Union


# --- Interactive task selection ---
while True:
    try:
        EXP_id = int(input("Select EXP_id: "))
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")

TASK_FILE = f"TASKS/EXP{EXP_id}.jsonl"

if not os.path.exists(TASK_FILE):
    print(f"Error: Task file '{TASK_FILE}' does not exist.")
    sys.exit(1)

LOCK_SUFFIX = ".locked"
DONE_SUFFIX = ".done"

TASK_FILE_PATH = Path(TASK_FILE)
LOCKS_DIR = Path("locks") / TASK_FILE_PATH.stem
LOCKS_DIR.mkdir(parents=True, exist_ok=True)


def try_claim_task(line_idx: int) -> Optional[str]:
    """Atomically claim a task by creating a lock file."""
    lock_path = LOCKS_DIR / f"{line_idx}{LOCK_SUFFIX}"
    done_path = LOCKS_DIR / f"{line_idx}{DONE_SUFFIX}"

    if done_path.exists():
        return None

    try:
        with lock_path.open("x") as f:
            f.write(str(os.getpid()))
        return str(lock_path)
    except FileExistsError:
        return None


def mark_done(lock_path: Optional[str]) -> None:
    """Rename `.locked` to `.done` to mark successful completion."""
    if not lock_path:
        return
    lock_p = Path(lock_path)
    done_p = lock_p.with_name(lock_p.name.replace(LOCK_SUFFIX, DONE_SUFFIX))
    try:
        lock_p.rename(done_p)
    except Exception:
        pass


def has_successful_ckpt(ckpt_dir: Union[Path, str], threshold: float = 0.7) -> bool:
    """Check if a valid checkpoint exists with sufficient score.
    Pattern: `epoch=<int>-test_mean_score=<float>.ckpt`, epoch > 100, score > threshold.
    """
    pattern = r"epoch=(\d+)-test_mean_score=(\d\.\d+)\.ckpt$"
    ckpt_dir = Path(ckpt_dir)

    for ckpt_path in ckpt_dir.glob("*.ckpt"):
        match = re.search(pattern, ckpt_path.name)
        if match:
            epoch = int(match.group(1))
            score = float(match.group(2))
            if score > threshold and epoch > 100:
                return True
    return False


def worker_process(gpu_id: int, log_queue: Queue) -> None:
    """Worker: claim *one* available task, run it on `cuda:{gpu_id}`."""
    def log(msg: str):
        log_queue.put(f"[GPU{gpu_id}] {msg}")

    log(f"🚀 Worker started, using GPU cuda:{gpu_id}")

    # Load all tasks
    tasks = []
    try:
        with open(TASK_FILE, "r") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log(f"⚠️ Skipping invalid JSON at line {line_idx}: {e}")
                    tasks.append(None)
    except Exception as e:
        log(f"💥 Failed to read task file '{TASK_FILE}': {e}")
        return

    if not tasks:
        log("📭 Task file is empty.")
        return

    # Shuffle to reduce contention (helpful in multi-machine setups)
    # But we only claim *first available* task in order — still deterministic per round.
    valid_indices = [i for i, t in enumerate(tasks) if t is not None]

    claimed = False
    for i in valid_indices:
        task = tasks[i]
        lock_file = try_claim_task(i)
        if not lock_file:
            continue

        claimed = True
        task_id = task.get("task_id", "N/A")
        log(f"✅ Claimed task {i}: {task_id}")

        success = False
        try:
            cmd_str = task["cmd"].replace("cuda:0", f"cuda:{gpu_id}")
            cmd = cmd_str.split()

            output_dir = Path(task["output_dir"])
            ckpt_dir = output_dir / "checkpoints"

            # Skip / resume logic
            if output_dir.exists():
                if has_successful_ckpt(ckpt_dir):
                    log(f"⏭️ Skipping task {i} ({task_id}): valid checkpoint exists.")
                    success = True
                else:
                    log(f"🧹 Clearing output directory for retraining: {task_id}")
                    try:
                        shutil.rmtree(output_dir)
                    except OSError as e:
                        log(f"⚠️ Failed to remove output dir: {e}")

            # Run task if needed
            if not success:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    log(f"🎉 Task succeeded: {task_id}")
                    success = True
                else:
                    log(f"❌ Task failed (exit {result.returncode}): {task_id}")
                    if result.stderr:
                        log(f"STDERR: {result.stderr}")

        except Exception as e:
            log(f"💥 Execution error: {e}")

        # Finalize
        if success:
            mark_done(lock_file)
        else:
            # Release lock for retry
            try:
                Path(lock_file).unlink(missing_ok=True)
                log(f"🔓 Released lock for retry: {task_id}")
            except Exception as e:
                log(f"⚠️ Failed to remove lock: {e}")

        # Exit after processing *one* task
        break

    if not claimed:
        log("📭 No unclaimed/pending tasks found this round.")
    else:
        log("👋 Worker exiting after completing one task.")


def logger_process(log_queue: Queue) -> None:
    """Dedicated logger for clean, ordered output."""
    while True:
        msg = log_queue.get()
        if msg == "STOP":
            break
        print(msg, flush=True)


def count_task_states() -> tuple[int, int, int, int]:
    if not Path(TASK_FILE).exists():
        return 0, 0, 0, 0

    with open(TASK_FILE) as f:
        total = sum(1 for line in f if line.strip())

    done = len(list(LOCKS_DIR.glob(f"*{DONE_SUFFIX}")))
    locked = len(list(LOCKS_DIR.glob(f"*{LOCK_SUFFIX}")))
    pending = max(0, total - done - locked)
    return total, done, locked, pending


def main():
    parser = argparse.ArgumentParser(description="Launch single-GPU task worker.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID to use")
    args = parser.parse_args()
    gpu_id = args.gpu_id

    print(f"🔍 Monitoring task file: {TASK_FILE}")
    print(f"📁 Lock directory: {LOCKS_DIR}")
    print(f"🎯 Using GPU: cuda:{gpu_id}\n")

    while True:
        total, done, locked, pending = count_task_states()

        print(
            f"\n📊 Task status | "
            f"Total: {total} | "
            f"✅ Done: {done} | "
            f"🟡 Locked: {locked} | "
            f"⏳ Pending: {pending}"
        )

        if pending == 0:
            print("🎉 All tasks completed or claimed. Exiting.")
            break

        print(f"🔄 Launching worker on GPU {gpu_id} to claim one pending task...\n")

        log_queue = Queue()
        logger = Process(target=logger_process, args=(log_queue,))
        logger.start()

        worker = Process(target=worker_process, args=(gpu_id, log_queue))
        worker.start()
        worker.join()

        log_queue.put("STOP")
        logger.join()

        # Small sleep to avoid overwhelming I/O on tight loops
        time.sleep(random.uniform(0.5, 1.5))


if __name__ == "__main__":
    main()