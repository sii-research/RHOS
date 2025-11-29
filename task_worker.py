# launch_workers.py
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

while True:
    try:
        task_num = int(input("请选择任务："))
        break  # 如果转换成功，跳出循环
    except ValueError:
        print("输入无效，请输入一个整数。")

TASK_FILE = f"TASKS{task_num}.jsonl"

if not os.path.exists(TASK_FILE):
    print(f"文件 {TASK_FILE} 不存在，请重新选择。")
    exit(1)
    
LOCK_SUFFIX = ".locked"
DONE_SUFFIX = ".done"

TASK_FILE_PATH = Path(TASK_FILE)
LOCKS_DIR = Path("locks") / TASK_FILE_PATH.stem
LOCKS_DIR.mkdir(parents=True, exist_ok=True)


def try_claim_task(line_idx):
    """尝试原子领取一个任务，锁文件存于 ./locks/<TASK_FILE_STEM>/"""
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
    """将 .locked 重命名为 .done（在同一目录）"""
    if lock_path:
        lock_p = Path(lock_path)
        done_p = lock_p.with_name(lock_p.name.replace(LOCK_SUFFIX, DONE_SUFFIX))
        try:
            lock_p.rename(done_p)  # 原子重命名
        except Exception:
            pass  # 忽略错误（如文件已被删除）
        
def has_successful_ckpt(ckpt_dir, threshold=0.7):
    # 注意：epoch= 后是整数（\d+），test_mean_score= 后是浮点数（\d\.\d+）
    pattern = r"epoch=(\d+)-test_mean_score=(\d\.\d+)\.ckpt$"
    ckpt_dir = Path(ckpt_dir)  # 确保是 Path 对象（若传入的是 str）
    
    for ckpt_path in ckpt_dir.glob("*.ckpt"):
        match = re.search(pattern, ckpt_path.name)
        if match:
            epoch = int(match.group(1))      # ✅ group(1): epoch 数字
            score = float(match.group(2))    # ✅ group(2): score
            if score > threshold and epoch > 100:
                return True
    return False

def worker_process(gpu_id, log_queue):
    def log(msg):
        log_queue.put(f"[GPU{gpu_id}] {msg}")

    log(f"🚀 启动 Worker，使用 GPU cuda:{gpu_id}")

    # --- 1. 预加载所有任务 ---
    tasks = []
    try:
        with open(TASK_FILE, 'r') as f:
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
        log(f"💥 无法读取任务文件 {TASK_FILE}: {e}")
        return

    if not tasks:
        log("📭 任务文件为空，退出。")
        return

    valid_indices = [i for i, t in enumerate(tasks) if t is not None]
    if not valid_indices:
        log("📭 无有效任务，退出。")
        return

    # --- 2. 全局随机打乱 ---
    # random.shuffle(valid_indices)
    # log(f"🔀 打乱任务顺序，共 {len(valid_indices)} 个有效任务")

    claimed_count = 0
    for i in valid_indices:
        task = tasks[i]
        lock_file = try_claim_task(i)
        if not lock_file:
            continue

        claimed_count += 1
        log(f"✅ 领取任务 {i}: {task.get('task_id', 'N/A')}")

        success = False
        try:
            cmd_str = task['cmd'].replace("cuda:0", f"cuda:{gpu_id}")
            cmd = cmd_str.split()

            output_dir = Path(task['output_dir'])
            ckpt_dir = output_dir / "checkpoints"
            
            if output_dir.exists():
                if has_successful_ckpt(ckpt_dir):
                    log(f"⏭️ 跳过任务 {i}: {task.get('task_id')}, 已存在 checkpoint ")
                    success = True  # 视为成功，标记 done
                else:
                    # 清空输出目录，重新训练
                    log(f"🧹 清空输出目录，重新训练任务 {i}: {task.get('task_id')}")
                    try:
                        shutil.rmtree(output_dir)
                    except OSError as e:
                        log(f"⚠️ 删除失败（可能被占用）: {e}")
                        
            if not success:
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    log(f"🎉 任务成功: {task.get('task_id')}")
                    success = True
                else:
                    log(f"❌ 任务失败（退出码 {result.returncode}）: {task.get('task_id')}")
                    if result.stderr:
                        log(f"STDERR: {result.stderr}")
                    # 不标记成功，后续删除锁

        except Exception as e:
            log(f"💥 执行异常: {e}")
            # 不标记成功

        # --- 关键：根据 success 决定是 mark_done 还是 删除锁 ---
        if success:
            mark_done(lock_file)
        else:
            # 删除 .locked 文件，允许重试
            try:
                lock_path = Path(lock_file)
                if lock_path.exists():
                    lock_path.unlink()
                    log(f"🔓 删除锁文件，任务可重试: {task.get('task_id')}")
            except Exception as e:
                log(f"⚠️ 删除锁文件失败: {e}")

        time.sleep(random.uniform(0.1, 0.5))

    log(f"👋 Worker 退出，共处理 {claimed_count} 个任务")

def logger_process(log_queue):
    """统一打印日志，避免多进程输出混乱"""
    while True:
        msg = log_queue.get()
        if msg == "STOP":
            break
        print(msg, flush=True)

def count_total_tasks():
    if not Path(TASK_FILE).exists():
        print(f"❌ 任务文件 {TASK_FILE} 不存在，请先运行 generate_tasks.py")
        return 0
    with open(TASK_FILE) as f:
        return sum(1 for _ in f)

def main():
    parser = argparse.ArgumentParser(description="并发启动多个训练 worker")
    parser.add_argument("num_gpus", type=int, help="要使用的 GPU 数量（如 4）")
    args = parser.parse_args()

    n_gpus = args.num_gpus
    total_tasks = count_total_tasks()
    if total_tasks == 0:
        return

    print(f"📋 共 {total_tasks} 个任务，启动 {n_gpus} 个 worker (GPU 0 ~ {n_gpus-1})...\n")

    log_queue = Queue()
    logger = Process(target=logger_process, args=(log_queue,))
    logger.start()

    workers = []
    for gpu_id in range(n_gpus):
        p = Process(target=worker_process, args=(gpu_id, log_queue))
        p.start()
        workers.append(p)

    # 等待所有 worker 结束
    for p in workers:
        p.join()

    # 停止日志进程
    log_queue.put("STOP")
    logger.join()

    # 统计完成情况
    done_files = list(LOCKS_DIR.glob(f"*{DONE_SUFFIX}"))  # ← 关键修复
    print(f"\n✅ 总结: {len(done_files)} / {total_tasks} 个任务已完成")

if __name__ == "__main__":
    main()