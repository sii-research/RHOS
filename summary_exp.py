import re
from pathlib import Path
import numpy as np

# --- Interactive task selection (now for EXP only) ---
while True:
    try:
        EXP_id = int(input("Select EXP_id: "))
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")

# Supported tasks
TASKS = ["pusht", "square_mh", "square_ph", "toolhang_ph", "transport_mh", "transport_ph"]
RUN_ids = [0, 1, 2, 3, 4]

# Regex pattern for checkpoint files
ckpt_pattern = re.compile(r"epoch=\d+-test_mean_score=([+-]?\d*\.?\d+)\.ckpt")

print(f"\n[INFO] Analyzing EXP{EXP_id}\n")

# Store results per task
results = {}

for task_name in TASKS:
    best_scores = []  # scores across runs for this task
    print(f"Task: {task_name}")

    for run_id in RUN_ids:
        ckpt_dir = Path(f"results/EXP{EXP_id}/{task_name}/run_{run_id}/checkpoints")
        
        if not ckpt_dir.exists():
            print(f"  ⚠️ Run {run_id}: Directory not found")
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
        
        if found_any:
            best_scores.append(max_score)
            print(f"  ✅ Run {run_id}: best score = {max_score:.3f}")
        else:
            print(f"  ⚠️ Run {run_id}: No valid checkpoint found")

    # Analyze this task
    if best_scores:
        scores = np.array(best_scores)
        mean = np.mean(scores)
        stderr = np.std(scores, ddof=1) if len(scores) > 1 else 0.0

        results[task_name] = {
            "scores": [round(s, 3) for s in best_scores],
            "mean": round(float(mean), 3),
            "stderr": round(float(stderr), 3),
            "n_runs": len(best_scores)
        }
    else:
        results[task_name] = None
        print(f"  ❌ No valid scores for {task_name}")

    print()  # blank line between tasks

# ===== Unified Summary Output =====
print("=" * 60)
print(f"📊 EXP{EXP_id} Summary (Mean ± StdErr over runs)")
print("=" * 60)

# Optional: compute overall mean across tasks (if needed)
all_means = []
for task_name in TASKS:
    res = results[task_name]
    if res is not None:
        print(f"{task_name:<15}: {res['mean']:.3f} ± {res['stderr']:.3f} (n={res['n_runs']})  {res['scores']}")
        all_means.append(res['mean'])
    else:
        print(f"{task_name:<15}: N/A")

# Optional global average (across task means, not across all runs)
if all_means:
    global_mean = np.mean(all_means)
    print("-" * 60)
    print(f"Overall (task means): {global_mean:.3f}")

print("\n✅ All tasks analysis completed!")