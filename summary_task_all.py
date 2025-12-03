import re
from pathlib import Path
import numpy as np

# --- Interactive task selection ---
while True:
    try:
        EXP_id = int(input("Select EXP_id (-1 represents select all): "))
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")

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

# Determine base directories based on EXP_id
if EXP_id == -1:
    base_dirs = list(Path("results").glob("EXP*"))
    if not base_dirs:
        print("[WARN] No results/EXP* directories found.")
        exit(1)
else:
    base_dirs = list(Path("results").glob(f"EXP{EXP_id}"))
    if not base_dirs:
        print(f"[WARN] No results/EXP{EXP_id} directory found.")
        exit(1)

best_scores = []  # Store best scores for each run
ckpt_info = []    # Store (score, relative_checkpoint_path) tuples
seen_run_paths = set()  # Track processed runs (though not used in current logic)
path_score_list = []

# Process each experiment directory
for base_dir in sorted(base_dirs):
    if not base_dir.is_dir():
        continue

    task_dir = base_dir / task_name
    if not task_dir.exists():
        continue

    # Recursively search for run directories
    for run_dir in task_dir.rglob("*"):
        if not run_dir.is_dir():
            continue

        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue

        # Get relative path for display
        try:
            rel_path_str = str(run_dir.relative_to(Path("results")))
        except ValueError:
            rel_path_str = str(run_dir)

        max_score = -np.inf
        best_ckpt_file = None
        found_any = False

        # Scan checkpoint files
        for ckpt_file in ckpt_dir.glob("*.ckpt"):
            match = ckpt_pattern.search(ckpt_file.name)
            if not match:
                continue
            try:
                score = float(match.group(1))
            except ValueError:
                continue
            found_any = True
            if score > max_score:
                max_score = score
                best_ckpt_file = ckpt_file

        # Record best checkpoint for this run
        if found_any and best_ckpt_file is not None:
            try:
                # Create shortened path relative to 'data' directory
                short_path = str(best_ckpt_file.relative_to(Path("data")))
            except ValueError:
                short_path = str(best_ckpt_file)
            best_scores.append(max_score)
            ckpt_info.append((max_score, short_path))
            print(f"{rel_path_str}: {max_score:.3f}")

# Summary statistics
if not best_scores:
    print("\n[WARN] No valid checkpoints found.\n")
    exit(0)

scores = np.array(best_scores)
paths = np.array([path for _, path in ckpt_info])

# Sort results by score (descending)
sorted_indices = np.argsort(-scores)
sorted_scores = scores[sorted_indices]
sorted_paths = paths[sorted_indices]

# Calculate overall statistics
mean_all = round(float(np.mean(scores)), 3)
stderr_all = round(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, 3)

# Calculate top-k statistics
top_k = min(5, len(sorted_scores))
top_scores = sorted_scores[:top_k]
mean_top = round(float(np.mean(top_scores)), 3)
stderr_top = round(float(np.std(top_scores, ddof=1)) if len(top_scores) > 1 else 0.0, 3)

# Display detailed results
print(f"\nDetailed results (sorted by score):")
for score, path in zip(sorted_scores, sorted_paths):
    print(f"   {score:.3f} -> {path}")

# Display summary statistics
print(f"\nOverall mean ± std: {mean_all} ± {stderr_all}")
print(f"Top-{top_k} mean ± std: {mean_top} ± {stderr_top}\n")

print(f"✅ {task_name} tasks analysis completed!")