"""
Multi-GPU task launcher with file-based lock coordination.

Supports incremental task execution on multi GPUs (specified via --gpu-nums).

Usage:
    python task_worker.py --gpu-nums 2
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
    """Atomically claim a task by creating a lock file.

    Args:
        line_idx: Zero-based index of the task in the JSONL file.

    Returns:
        Path to the created lock file if claimed successfully; None if already claimed or done.
    """
    lock_path = LOCKS_DIR / f"{line_idx}{LOCK_SUFFIX}"
    done_path = LOCKS_DIR / f"{line_idx}{DONE_SUFFIX}"

    # Skip if already completed
    if done_path.exists():
        return None

    try:
        with lock_path.open("x") as f:  # 'x' ensures atomic creation
            f.write(str(os.getpid()))
        return str(lock_path)
    except FileExistsError:
        return None  # Already locked by another process/machine


def mark_done(lock_path: Optional[str]) -> None:
    """Rename `.locked` to `.done` to mark task completion (atomic on same FS)."""
    if not lock_path:
        return
    lock_p = Path(lock_path)
    done_p = lock_p.with_name(lock_p.name.replace(LOCK_SUFFIX, DONE_SUFFIX))
    try:
        lock_p.rename(done_p)
    except Exception:
        pass  # Ignore errors (e.g., file already removed)


def has_successful_ckpt(ckpt_dir: Union[Path, str], threshold: float = 0.7) -> bool:
    """Check if a valid checkpoint exists with sufficient score.

    Filename pattern: `epoch=<int>-test_mean_score=<float>.ckpt`
    Requires: epoch > 100 and score > threshold.
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
    """Worker process that claims and executes tasks on a specific GPU."""

    def log(msg: str) -> None:
        log_queue.put(f"[GPU{gpu_id}] {msg}")

    log(f"🚀 Worker started, using GPU cuda:{gpu_id}")

    # --- 1. Preload all tasks ---
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
        log("📭 Task file is empty. Exiting.")
        return

    valid_indices = [i for i, t in enumerate(tasks) if t is not None]
    if not valid_indices:
        log("📭 No valid tasks found. Exiting.")
        return

    claimed_count = 0
    for i in valid_indices:
        task = tasks[i]
        lock_file = try_claim_task(i)
        if not lock_file:
            continue  # Already claimed or done

        claimed_count += 1
        task_id = task.get("task_id", "N/A")
        log(f"✅ Claimed task {i}: {task_id}")

        success = False
        try:
            # Replace GPU device in command
            cmd_str = task["cmd"].replace("cuda:0", f"cuda:{gpu_id}")
            cmd = cmd_str.split()

            output_dir = Path(task["output_dir"])
            ckpt_dir = output_dir / "checkpoints"

            # Checkpoint-based resume/skip logic
            if output_dir.exists():
                if has_successful_ckpt(ckpt_dir):
                    log(f"⏭️ Skipping task {i} ({task_id}): valid checkpoint already exists.")
                    success = True  # Mark as done
                else:
                    # Clean output dir for retraining
                    log(f"🧹 Clearing output directory for retraining task {i} ({task_id})")
                    try:
                        shutil.rmtree(output_dir)
                    except OSError as e:
                        log(f"⚠️ Failed to remove output directory: {e}")

            # Execute command if not skipped
            if not success:
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    log(f"🎉 Task succeeded: {task_id}")
                    success = True
                else:
                    log(f"❌ Task failed (exit code {result.returncode}): {task_id}")
                    if result.stderr:
                        log(f"STDERR: {result.stderr}")
                    # Do NOT mark as done on failure

        except Exception as e:
            log(f"💥 Execution error: {e}")
            # Do NOT mark as done on exception

        # --- Finalize task state ---
        if success:
            mark_done(lock_file)
        else:
            # Release lock to allow retry
            try:
                lock_path = Path(lock_file)
                if lock_path.exists():
                    lock_path.unlink()
                    log(f"🔓 Released lock for retry: {task_id}")
            except Exception as e:
                log(f"⚠️ Failed to remove lock file: {e}")

        time.sleep(random.uniform(0.1, 0.5))  # Reduce contention

    log(f"👋 Worker exiting. Claimed {claimed_count} task(s).")


def logger_process(log_queue: Queue) -> None:
    """Dedicated process for ordered log output (avoids interleaving)."""
    while True:
        msg = log_queue.get()
        if msg == "STOP":
            break
        print(msg, flush=True)


def count_task_states() -> tuple[int, int, int, int]:
    """Count task states based on lock files.

    Returns:
        (total_tasks, done_count, locked_count, pending_count)
    """
    if not Path(TASK_FILE).exists():
        return 0, 0, 0, 0

    # Count non-empty lines as total tasks
    with open(TASK_FILE) as f:
        total = sum(1 for line in f if line.strip())

    done_count = len(list(LOCKS_DIR.glob(f"*{DONE_SUFFIX}")))
    locked_count = len(list(LOCKS_DIR.glob(f"*{LOCK_SUFFIX}")))
    pending_count = max(0, total - done_count - locked_count)

    return total, done_count, locked_count, pending_count


def main():
    parser = argparse.ArgumentParser(description="Launch concurrent training workers.")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to utilize (e.g., 4)")
    args = parser.parse_args()
    n_gpus = args.num_gpus

    print(f"🔍 Monitoring task file: {TASK_FILE}")
    print(f"📁 Lock directory: {LOCKS_DIR}")

    while True:
        total, done, locked, pending = count_task_states()

        print(
            f"\n📊 Task status | "
            f"Total: {total} | "
            f"✅ Done: {done} | "
            f"🟡 Locked: {locked} | "
            f"⏳ Pending: {pending}"
        )

        # Exit when no pending tasks remain (all claimed or completed)
        if pending == 0:
            print("🎉 All tasks have been claimed. Exiting.")
            break

        print(f"🔄 Launching {n_gpus} worker(s) (GPU 0 ~ {n_gpus - 1}) to claim {pending} pending task(s)...\n")

        log_queue = Queue()
        logger = Process(target=logger_process, args=(log_queue,))
        logger.start()

        workers = []
        for gpu_id in range(n_gpus):
            p = Process(target=worker_process, args=(gpu_id, log_queue))
            p.start()
            workers.append(p)

        # Wait for all workers in this round to finish
        for p in workers:
            p.join()

        # Terminate logger process
        log_queue.put("STOP")
        logger.join()

        # Small sleep to avoid overwhelming I/O on tight loops
        time.sleep(random.uniform(0.5, 1.5))


if __name__ == "__main__":
    main()