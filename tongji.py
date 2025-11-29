import re
from pathlib import Path
import numpy as np

# ===== 配置参数 =====
task_name = "pusht"
x_values = [1, 2, 10, 100, 11]
y_values = [0, 1, 2, 3, 4]  # run_0 到 run_4

# 可调节的阈值参数
EARLY_THRESHOLD = 300   # 用于判断“早期”（epoch <= EARLY_THRESHOLD）
MAX_EPOCHS = 1000       # 用于判断是否接近或超过常规训练长度（仅用于分析，不影响逻辑）

ckpt_pattern = re.compile(r'epoch=(\d+)-test_mean_score=([+-]?\d*\.?\d+)\.ckpt')

all_scores = []
all_epochs = []
late_improvements = []  # 存储 epoch > EARLY_THRESHOLD 的 run 的 (score_gap, x, y, best_epoch)

print("🔍 正在扫描所有 flow_x / run_y 目录...")

for x in x_values:
    for y in y_values:
        ckpt_dir = Path(f"data/outputs/{task_name}/flow_{x}/run_{y}/checkpoints")
        
        if not ckpt_dir.exists():
            print(f"⚠️ 跳过：目录不存在 {ckpt_dir}")
            continue
        
        best_score = -np.inf
        best_epoch = None
        best_score_under_threshold = -np.inf  # epoch <= EARLY_THRESHOLD 的最高分

        found_any = False

        for ckpt_file in ckpt_dir.glob("*.ckpt"):
            match = ckpt_pattern.search(ckpt_file.name)
            if match:
                found_any = True
                try:
                    epoch = int(match.group(1))
                    score = float(match.group(2))
                    
                    # 全局最佳
                    if score > best_score:
                        best_score = score
                        best_epoch = epoch
                    
                    # epoch <= EARLY_THRESHOLD 的最佳
                    if epoch <= EARLY_THRESHOLD and score > best_score_under_threshold:
                        best_score_under_threshold = score
                        
                except ValueError:
                    continue

        if not found_any:
            print(f"⚠️ flow_{x}/run_{y}: 未找到有效 checkpoint")
            continue

        all_scores.append(best_score)
        all_epochs.append(best_epoch)

        print(f"✅ flow_{x}/run_{y}: score={best_score:.3f} @ epoch={best_epoch}")

        # 如果最佳出现在 epoch > EARLY_THRESHOLD，且存在 epoch <= EARLY_THRESHOLD 的 checkpoint
        if best_epoch > EARLY_THRESHOLD:
            if best_score_under_threshold > -np.inf:
                gap = best_score - best_score_under_threshold
                late_improvements.append({
                    'gap': gap,
                    'x': x,
                    'y': y,
                    'best_epoch': best_epoch,
                    'best_score': best_score,
                    'best_under_500': best_score_under_threshold
                })
                print(f"    📈 epoch>{EARLY_THRESHOLD} 提升: +{gap:.3f} (≤{EARLY_THRESHOLD} 最高: {best_score_under_threshold:.3f})")
            else:
                print(f"    ⚠️ 最佳在 epoch>{EARLY_THRESHOLD}，但 epoch≤{EARLY_THRESHOLD} 无有效 checkpoint")

# ===== 总体统计 =====
if not all_scores:
    print("\n❌ 错误：未找到任何有效的 checkpoint 数据。")
else:
    scores = np.array(all_scores)
    epochs = np.array(all_epochs)

    mean_score = round(float(np.mean(scores)), 3)
    stderr_score = round(float(np.std(scores, ddof=1) / np.sqrt(len(scores))), 3) if len(scores) > 1 else 0.0
    mean_epoch = round(float(np.mean(epochs)), 1)
    max_epoch_seen = int(np.max(epochs))

    print("\n" + "="*70)
    print("📊 总体汇总结果（所有 x 和 run 合并）")
    print("="*70)
    print(f"总有效 runs 数量: {len(all_scores)}")
    print(f"test_mean_score: {mean_score} ± {stderr_score}")
    print(f"最佳 epoch（平均）: {mean_epoch}")
    print(f"最大 epoch 出现值: {max_epoch_seen}")

    # ===== 后期提升分析 =====
    if late_improvements:
        gaps = [item['gap'] for item in late_improvements]
        avg_gap = np.mean(gaps)
        max_gap = np.max(gaps)
        num_late = len(late_improvements)

        print(f"\n📈 后期提升分析 (最佳 epoch > {EARLY_THRESHOLD}):")
        print(f"  • 共 {num_late} 个 runs 在 epoch > {EARLY_THRESHOLD} 时达到最佳")
        print(f"  • 平均提升幅度: +{avg_gap:.3f}")
        print(f"  • 最大提升幅度: +{max_gap:.3f} (来自 flow_{late_improvements[np.argmax(gaps)]['x']}/run_{late_improvements[np.argmax(gaps)]['y']})")
        
        if avg_gap < 0.01:
            print("  💡 提示：后期提升较小 (<0.01)，可考虑提前终止训练")
        elif avg_gap > 0.05:
            print("  ⚠️ 提示：后期提升显著 (>0.05)，建议保留完整训练周期")
        else:
            print("  ℹ️ 提示：后期有中等提升，可根据资源权衡")
    else:
        print(f"\n✅ 所有 runs 的最佳 checkpoint 均出现在 epoch ≤ {EARLY_THRESHOLD}")