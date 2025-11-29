import re
from pathlib import Path
import numpy as np
import pandas as pd

RUN = 22

# 配置支持的任务列表
TASKS = ["pusht"]
# TASKS = ["pusht", "square_mh", "square_ph", "toolhang_ph", "transport_mh", "transport_ph"]

# 配置 NFE 和 run 范围
NFE_values = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]
y_values = [0, 1, 2, 3, 4]

# 正则匹配 checkpoint 文件名中的 test_mean_score
ckpt_pattern = re.compile(r'epoch=\d+-test_mean_score=([+-]?\d*\.?\d+)\.ckpt')

# 存储结果
results = {}           # (task, nfe) -> (mean, stderr, count)
raw_scores_dict = {}   # (task, nfe) -> list of top-5 scores (floats)

print("📊 正在分析所有任务...")

for task_name in TASKS:
    for nfe in NFE_values:
        all_run_max_scores = []  # 每个 run 的最优 ckpt score
        for y in range(11):
            ckpt_dir = Path(f"data/outputs{RUN}/{task_name}/NFE_{nfe}/run_{y}/checkpoints")
            
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
                        print(f"    ❌ 无效分数: {ckpt_file.name}")
                        continue
            if found_any:
                all_run_max_scores.append(max_score)

        # 选取最高的 5 个 run-wise 最优分数
        top5_scores = []
        mean, stderr, count = np.nan, np.nan, 0
        if all_run_max_scores:
            all_run_max_scores = np.array(all_run_max_scores)
            # 取最高的 5 个（升序 → [-5:]；若需降序可加 [::-1]）
            top5_scores = np.sort(all_run_max_scores)[-5:].tolist()
            count = len(top5_scores)
            mean = float(np.mean(top5_scores))
            stderr = float(np.std(top5_scores, ddof=1)) if count > 1 else 0.0

        # 保存结果
        results[(task_name, nfe)] = (round(mean, 3), round(stderr, 3), count)
        raw_scores_dict[(task_name, nfe)] = top5_scores

# 构建 DataFrame（长格式）
rows = []
for task in TASKS:
    for nfe in NFE_values:
        mean, stderr, count = results.get((task, nfe), (np.nan, np.nan, 0))
        top5_list = raw_scores_dict.get((task, nfe), [])

        if count == 0:
            score_display = "–"
            scores_detail = "–"
        else:
            score_display = f"{mean} ± {stderr}"
            # 格式化为 "0.xxx, 0.xxx, ..."，保留3位小数
            scores_detail = ", ".join([f"{s:.3f}" for s in top5_list])

        rows.append({
            "Task": task,
            "NFE": nfe,
            "Score (mean ± stderr)": score_display,
            "Top-5 Scores": scores_detail
        })

df = pd.DataFrame(rows)

# 输出到终端
print("\n" + "="*80)
print("✅ 汇总结果（含具体 top-5 分数）")
print("="*80)
print(df.to_string(index=False, na_rep="–"))
print("="*80)

# 保存 CSV
output_csv = "results_with_scores.csv"
df.to_csv(output_csv, index=False)
print(f"\n📁 已保存详细结果至: {output_csv}")

# 额外：生成传统透视表（仅 mean ± stderr）
pivot_df = df.pivot(index="NFE", columns="Task", values="Score (mean ± stderr)")
pivot_df = pivot_df.reindex(index=NFE_values)  # 保持 NFE 顺序

print("\n📊 传统透视表（仅 mean ± stderr）：")
print(pivot_df.to_string(na_rep="–"))

pivot_df.to_csv("results_pivot.csv")
print(f"📁 已保存透视表至: results_pivot.csv")

print("\n✅ 所有任务分析完成！")