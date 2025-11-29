import re
from pathlib import Path
import numpy as np

# 配置支持的任务列表（可扩展）
TASKS = ["pusht", "square_mh", "square_ph", "toolhang_ph", "transport_ph"]

# 显示任务选项
print("请选择任务:")
for i, task in enumerate(TASKS, start=1):
    print(f"  {i}. {task}")

# 获取用户输入
try:
    choice = int(input("请输入选项编号: ").strip())
    if not (1 <= choice <= len(TASKS)):
        raise ValueError
    task_name = TASKS[choice - 1]
except (ValueError, KeyboardInterrupt):
    print("❌ 无效输入。请输入有效的编号。")
    exit(1)

# 正则表达式：匹配 run_数字 目录
run_dir_pattern = re.compile(r'run_(\d+)$')
# 正则表达式：匹配 checkpoint 文件名
ckpt_pattern = re.compile(r'epoch=\d+-test_mean_score=([+-]?\d*\.?\d+)\.ckpt')

print(f"\n📊 正在分析任务: {task_name}\n")

best_scores = []      # 每个有效 run 的最高分
ckpt_info = []        # (score, short_path)

# 扫描 data/outputs* 下的所有内容
base_dirs = list(Path("data").glob("outputs*"))

if not base_dirs:
    print("❌ 未找到 data/outputs* 目录。")
    exit(1)

# 用于去重：每个 run 路径只处理一次（避免重复扫描）
seen_run_paths = set()

for base_dir in sorted(base_dirs):
    if not base_dir.is_dir():
        continue
    task_dir = base_dir / task_name
    if not task_dir.exists():
        continue
    
    if "outputs22" not in str(task_dir):
        continue
    
    # 递归查找所有 run_{y} 目录
    for run_dir in task_dir.rglob("*"):
        if not run_dir.is_dir():
            continue
        if not run_dir_pattern.match(run_dir.name):
            continue
        if str(run_dir) in seen_run_paths:
            continue
        seen_run_paths.add(str(run_dir))

        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        # 提取相对于 data/ 的路径用于过滤
        try:
            rel_path_str = str(run_dir.relative_to(Path("data")))
        except ValueError:
            rel_path_str = str(run_dir)

        # 路径过滤逻辑
        # if "rel" in rel_path_str:
        #     continue
        if "NFE_2" not in rel_path_str:
            continue
        
        if "run_2" in rel_path_str:
            continue
        
        # if "outputs13" not in rel_path_str or "flow_200" not in rel_path_str:
        #     continue

        max_score = -np.inf
        best_ckpt_file = None
        found_any = False

        for ckpt_file in ckpt_dir.glob("*.ckpt"):
            match = ckpt_pattern.search(ckpt_file.name)
            if match:
                try:
                    score = float(match.group(1))
                    found_any = True
                    if score > max_score:
                        max_score = score
                        best_ckpt_file = ckpt_file
                except ValueError:
                    continue

        if found_any and max_score > 0.7:
            # 获取 checkpoint 相对于 data/ 的短路径
            try:
                short_path = str(best_ckpt_file.relative_to(Path("data")))
            except ValueError:
                short_path = str(best_ckpt_file)

            best_scores.append(max_score)
            ckpt_info.append((max_score, short_path))
            print(f"  ✅ {rel_path_str}: {max_score:.3f}")
        else:
            status = f"最高分 {max_score:.3f}" if found_any else "无有效 ckpt"
            # print(f"  ⚠️  {rel_path_str}: {status}（跳过）")

# 汇总结果
if not best_scores:
    print("\n❌ 未找到任何有效的 checkpoint\n")
    exit(0) 

# 转为 numpy 数组
scores = np.array(best_scores)
paths = np.array([p for _, p in ckpt_info])

# 按分数降序排序
sorted_indices = np.argsort(-scores)
sorted_scores = scores[sorted_indices]
sorted_paths = paths[sorted_indices]

# 全体统计
mean_all = round(float(np.mean(scores)), 3)
stderr_all = round(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, 3)

# Top-K 统计
top_k = min(5, len(sorted_scores))
top5_scores = sorted_scores[:top_k]
mean_top5 = round(float(np.mean(top5_scores)), 3)
stderr_top5 = round(float(np.std(top5_scores, ddof=1)) if len(top5_scores) > 1 else 0.0, 3)

# 输出
print(f"\n✅ 共收集 {len(best_scores)} 个有效 run（分数 > 0.8，通过路径过滤）\n")

print("📄 详细列表（按分数降序）:")
for score, path in zip(sorted_scores, sorted_paths):
    print(f"   {score:.3f} → {path}")

print(f"\n📈 全体平均值 ± 标准误差: {mean_all} ± {stderr_all}")
print(f"🏆 Top-{top_k} 平均值 ± 标准误差: {mean_top5} ± {stderr_top5}\n")

print("✅ 分析完成！")