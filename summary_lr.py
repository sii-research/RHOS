import re
from pathlib import Path
import numpy as np

# 配置支持的任务列表（可扩展）
TASKS = ["pusht", "square_mh", "square_ph", "toolhang_ph", "transport_mh", "transport_ph"]

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


# 配置 x 和 y 范围
x_values = [1,2,4,6,8,10]
# x_values = [100,11]
y_values = [0, 1, 2, 3, 4]
# y_values = [0, 1]

# 正则表达式匹配 checkpoint 文件名
ckpt_pattern = re.compile(r'epoch=\d+-test_mean_score=([+-]?\d*\.?\d+)\.ckpt')

print(f"\n📊 正在分析任务: {task_name}\n")

for x in x_values:
    best_scores = []
    print(f"🔍 处理 x = {x} ...")

    for y in y_values:
        ckpt_dir = Path(f"data/outputs9/{task_name}/flow_{x}/run_{y}/checkpoints")
        
        if not ckpt_dir.exists():
            # print(f"  ⚠️ 警告: 目录不存在 {ckpt_dir}")
            continue
        
        max_score = -np.inf
        found_any = False
        for ckpt_file in ckpt_dir.glob("*.ckpt"):
            match = ckpt_pattern.search(ckpt_file.name)
            if match:
                found_any = True
                try:
                    score = float(match.group(1))
                    if score > max_score:
                        max_score = score
                except ValueError:
                    continue
        if found_any :
        # if found_any and max_score>0.6:
            best_scores.append(max_score)
        # else:
        #     print(f"  ⚠️ 警告: 在 {ckpt_dir} 中未找到有效的 checkpoint 文件")

    # 汇总当前 x 的结果
    if not best_scores:
        print(f"  ❌ 无有效数据，跳过 x = {x}\n")
    else:
        scores = np.array(best_scores)
        # if len(scores)>2:
        #     scores = scores[:2]
        
        mean = np.mean(scores)
        if len(scores) > 1:
            stderr = np.std(scores, ddof=1)
        else:
            stderr = 0.0

        mean = round(float(mean), 3)
        stderr = round(float(stderr), 3)
        score_list = [round(s, 3) for s in scores]

        print(f"  ✅ x = {x}")
        print(f"     最高分列表: {score_list}")
        print(f"     平均值 ± 标准误差: {mean} ± {stderr}\n")

print("✅ 所有任务分析完成！")