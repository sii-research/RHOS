import re
from pathlib import Path
import numpy as np
import pandas as pd

# 配置支持的任务列表
# TASKS = ["pusht", "square_mh", "square_ph", "toolhang_ph", "transport_mh", "transport_ph"]
TASKS =  ["pusht","square_ph"]

# 配置 NFE 和 run 范围
NFE_values = [round(i * 0.1, 2) for i in range(1,10)] 
# NFE_values = [0.5] 
NFE_values += [1, 2, 5, 10]
y_values = [0, 1, 2, 3, 4]

# 正则表达式匹配 test_mean_score
score_pattern = re.compile(r'test_mean_score:\s*([+-]?\d*\.?\d+)')

# 存储结果：{ (task, nfe): (mean, stderr, count) }
results = {}

print("📊 正在分析所有任务...")

for task_name in TASKS:
    for nfe in NFE_values:
        scores = []
        for y in y_values:
            output_dir = Path(f"result/{task_name}/run_{y}/eval_logs")
            if nfe < 1:
                log = output_dir / f"eval_log_p0_n{nfe}.yaml"
            else:
                log = output_dir / f"eval_log_p1_n{nfe}.0.yaml"
            
            if not log.exists():
                continue
            
            try:
                with open(log, 'r', encoding='utf-8') as f:
                    content = f.read()
                    match = score_pattern.search(content)
                    if match:
                        score = float(match.group(1))
                        scores.append(score)
            except Exception as e:
                print(f"⚠️ 读取失败: {log} - {e}")
                continue

        if scores:
            scores = np.array(scores)
            mean = float(np.mean(scores))
            if len(scores) > 1:
                stderr = float(np.std(scores, ddof=1))
            else:
                stderr = 0.0
            count = len(scores)
        else:
            mean = np.nan
            stderr = np.nan
            count = 0

        results[(task_name, nfe)] = (round(mean, 3), round(stderr, 3), count)

# 构建 DataFrame
rows = []
for task in TASKS:
    for nfe in NFE_values:
        mean, stderr, count = results.get((task, nfe), (np.nan, np.nan, 0))
        if count == 0:
            display = "–"
        else:
            display = f"{mean} ± {stderr}"
        rows.append({"Task": task, "NFE": nfe, "Score (mean ± stderr)": display})

df = pd.DataFrame(rows)

# 透视表：任务为行，NFE 为列
pivot_df = df.pivot(index="NFE", columns="Task", values="Score (mean ± stderr)")

# 按 NFE 排序列（确保 0.1, 0.2, ..., 1, 2, 5... 顺序正确）
pivot_df = pivot_df.reindex(index=NFE_values)

# 打印表格
print("\n✅ 汇总结果表格：")
print(pivot_df.to_string(na_rep="–"))
pivot_df.to_csv("results.csv")
print("\n✅ 所有任务分析完成！")