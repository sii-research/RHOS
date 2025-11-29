# eval_all.py
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

EVAL_TASK_FILE = "TASKS_eval.jsonl"
LOCK_SUFFIX = ".locked" 
DONE_SUFFIX = ".done"

TASK_FILE_PATH = Path(EVAL_TASK_FILE)
LOCKS_DIR = Path("locks") / TASK_FILE_PATH.stem
LOCKS_DIR.mkdir(parents=True, exist_ok=True)


def try_claim_task(line_idx):
    """原子领取任务"""
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
    """标记任务完成"""
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

    log(f"🚀 启动 Eval Worker，使用 GPU cuda:{gpu_id}")

    # --- 加载所有 eval 任务 ---
    tasks = []
    try:
        with open(EVAL_TASK_FILE, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log(f"⚠️ 跳过无效 JSON 行 {line_idx}: {e}")
                    tasks.append(None)
    except Exception as e:
        log(f"💥 无法读取评估任务文件 {EVAL_TASK_FILE}: {e}")
        return

    if not tasks:
        log("📭 评估任务文件为空，退出。")
        return

    valid_indices = [i for i, t in enumerate(tasks) if t is not None]
    if not valid_indices:
        log("📭 无有效评估任务，退出。")
        return

    claimed_count = 0
    for i in valid_indices:
        task = tasks[i]
        lock_file = try_claim_task(i)
        if not lock_file:
            continue

        claimed_count += 1
        task_id = task.get("task_id", f"task_{i}")
        log(f"✅ 领取评估任务 {i}: {task_id}")

        success = False
        try:
            # 构建 eval 命令
            cmd_str = task['cmd'].replace("cuda:0", f"cuda:{gpu_id}")
            cmd = cmd_str.split()

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                log(f"🎉 评估成功: {task_id}")
                success = True
            else:
                log(f"❌ 评估失败（退出码 {result.returncode}）: {task_id}")
                if result.stderr:
                    log(f"STDERR: {result.stderr[:500]}...")  # 截断避免刷屏

        except Exception as e:
            log(f"💥 评估异常: {e}")

        # 标记或清理锁
        if success:
            mark_done(lock_file)
        else:
            try:
                lock_path = Path(lock_file)
                if lock_path.exists():
                    lock_path.unlink()
                    log(f"🔓 删除锁，任务可重试: {task_id}")
            except Exception as e:
                log(f"⚠️ 删除锁失败: {e}")

        time.sleep(random.uniform(0.1, 0.3))

    log(f"👋 Eval Worker 退出，共处理 {claimed_count} 个任务")


def logger_process(log_queue):
    while True:
        msg = log_queue.get()
        if msg == "STOP":
            break
        print(msg, flush=True)


def count_total_tasks():
    if not Path(EVAL_TASK_FILE).exists():
        print(f"❌ 评估任务文件 {EVAL_TASK_FILE} 不存在")
        return 0
    with open(EVAL_TASK_FILE) as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(description="并发执行多个评估任务")
    parser.add_argument("num_gpus", type=int, help="使用的 GPU 数量（如 4）")
    args = parser.parse_args()

    n_gpus = args.num_gpus
    total_tasks = count_total_tasks()
    if total_tasks == 0:
        return

    print(f"📋 共 {total_tasks} 个评估任务，启动 {n_gpus} 个 worker (GPU 0 ~ {n_gpus-1})...\n")

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
    print(f"\n✅ 评估总结: {len(done_files)} / {total_tasks} 个任务已完成")


if __name__ == "__main__":
    main()