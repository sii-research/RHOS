import re
from pathlib import Path
import numpy as np

EXP_id = 1

# Configure supported tasks list
TASKS = ["pusht", "square_mh", "square_ph", "toolhang_ph", "transport_mh", "transport_ph"]

# Display task options
print("Select a task:")
for i, task in enumerate(TASKS, start=1):
    print(f"  {i}. {task}")

# Get user input
try:
    choice = int(input("Enter option number: ").strip())
    if not (1 <= choice <= len(TASKS)):
        raise ValueError
    task_name = TASKS[choice - 1]
except (ValueError, KeyboardInterrupt):
    print("Invalid input. Please enter a valid number.")
    exit(1)

# Regex pattern for checkpoint files
ckpt_pattern = re.compile(r"epoch=\d+-test_mean_score=([+-]?\d*\.?\d+)\.ckpt")

print(f"\n[INFO] Analyzing task: {task_name}\n")

RUN_ids = [0, 1, 2, 3, 4]

best_scores = []

for run_id in RUN_ids:
    ckpt_dir = Path(f"results/EXP{EXP_id}/{task_name}/run_{run_id}/checkpoints")
    
    if not ckpt_dir.exists():
        print(f"  ⚠️ Warning: Directory does not exist {ckpt_dir}")
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
        best_scores.append(max_score)
    else:
        print(f"  ⚠️ Warning: No valid checkpoint files found in {ckpt_dir}")

# Summarize results for the current task
if not best_scores:
    print(f"  ❌ No valid data.\n")
else:
    scores = np.array(best_scores)
    mean = np.mean(scores)
    if len(scores) > 1:
        stderr = np.std(scores, ddof=1)
    else:
        stderr = 0.0

    mean = round(float(mean), 3)
    stderr = round(float(stderr), 3)
    score_list = [round(s, 3) for s in best_scores]

    print(f" Scores: {score_list}")
    print(f" Mean ± StdErr: {mean} ± {stderr}\n")

print("✅ All tasks analysis completed!")