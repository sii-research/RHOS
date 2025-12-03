"""
Usage:
    python eval_worker.py --gpu-nums 2
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
import shutil

# ------------------------------------------------------------------------------
# Interactive task selection
while True:
    try:
        EVAL_id = int(input("Select EVAL_id: "))
        TASK_FILE = f"TASKS/EVAL{EVAL_id}.jsonl"
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")
# ------------------------------------------------------------------------------

# Or You can hardcode the task file here:
# TASK_FILE = "TASKS/EVAL1.jsonl"

# ------------------------------------------------------------------------------

if not os.path.exists(TASK_FILE):
    print(f"Error: Task file '{TASK_FILE}' does not exist.")
    sys.exit(1)

LOCK_SUFFIX = ".locked" 
DONE_SUFFIX = ".done"

TASK_FILE_PATH = Path(TASK_FILE)
LOCKS_DIR = Path("locks") / TASK_FILE_PATH.stem
LOCKS_DIR.mkdir(parents=True, exist_ok=True)


def try_claim_task(line_idx):
    """Atomically claim a task by creating a lock file."""
    lock_path = LOCKS_DIR / f"{line_idx}{LOCK_SUFFIX}"
    done_path = LOCKS_DIR / f"{line_idx}{DONE_SUFFIX}"

    if done_path.exists():
        return None

    try:
        with open(lock_path, 'x') as f:
            f.write(str(os.getpid()))
        return str(lock_path)
    except FileExistsError:
        return None


def mark_done(lock_path):
    """Mark a claimed task as completed by renaming its lock file."""
    if lock_path:
        lock_p = Path(lock_path)
        done_p = lock_p.with_name(lock_p.name.replace(LOCK_SUFFIX, DONE_SUFFIX))
        try:
            lock_p.rename(done_p)
        except Exception:
            pass


def worker_process(gpu_id, log_queue):
    def log(msg):
        log_queue.put(f"[GPU{gpu_id}] {msg}")

    log(f"🚀 Eval worker started, using GPU cuda:{gpu_id}")

    # --- Preload all eval tasks ---
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
        log(f"💥 Failed to read eval task file {TASK_FILE}: {e}")
        return

    if not tasks:
        log("📭 Eval task file is empty. Exiting.")
        return

    valid_indices = [i for i, t in enumerate(tasks) if t is not None]
    if not valid_indices:
        log("📭 No valid eval tasks found. Exiting.")
        return

    claimed_count = 0
    for i in valid_indices:
        task = tasks[i]
        lock_file = try_claim_task(i)
        if not lock_file:
            continue

        claimed_count += 1
        task_id = task.get("task_id", f"task_{i}")
        log(f"✅ Claimed eval task {i}: {task_id}")

        success = False
        try:
            # Build the eval command
            cmd_str = task["cmd"].replace("cuda:0", f"cuda:{gpu_id}")
            cmd = cmd_str.split()

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                log(f"🎉 Eval succeeded: {task_id}")
                success = True
            else:
                log(f"❌ Eval failed (exit code {result.returncode}): {task_id}")
                if result.stderr:
                    log(f"STDERR: {result.stderr[:500]}...")  # Truncate output to avoid flooding

        except Exception as e:
            log(f"💥 Eval raised an exception: {e}")

        # Mark completion or clean up the lock
        if success:
            mark_done(lock_file)
        else:
            try:
                lock_path = Path(lock_file)
                if lock_path.exists():
                    lock_path.unlink()
                    log(f"🔓 Removed lock; task can be retried: {task_id}")
            except Exception as e:
                log(f"⚠️ Failed to remove lock: {e}")

        time.sleep(random.uniform(0.1, 0.3))

    log(f"👋 Eval worker exiting after processing {claimed_count} task(s)")


def logger_process(log_queue):
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
    parser = argparse.ArgumentParser(description="Run multiple evaluation tasks concurrently.")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to utilize (e.g., 4)")
    args = parser.parse_args()
    n_gpus = args.num_gpus

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
        return

    print(f"🔄 Launching {n_gpus} worker(s) (GPU 0 ~ {n_gpus - 1}) to claim {pending} pending task(s)...\n")

    log_queue = Queue()
    logger = Process(target=logger_process, args=(log_queue,))
    logger.start()

    workers = []
    for gpu_id in range(n_gpus):
        p = Process(target=worker_process, args=(gpu_id, log_queue))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    log_queue.put("STOP")
    logger.join()

    done_files = list(LOCKS_DIR.glob(f"*{DONE_SUFFIX}"))
    print(f"\n✅ Eval summary: {len(done_files)} / {total} tasks completed")


if __name__ == "__main__":
    main()
