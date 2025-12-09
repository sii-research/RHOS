import json
from pathlib import Path

EXP_id = 1

# TASKS = ["pusht", "square_mh", "square_ph", "toolhang_ph", "transport_mh", "transport_ph"]
TASKS = ["pusht", "square_mh", "square_ph", "toolhang_ph", "transport_ph"]

task_list = []
RUN_ids  = [0, 1, 2, 3, 4]

#------------------------------------------------------------------------------
# Configuration Options (adjust as needed):
#------------------------------------------------------------------------------

# Global training parameters
infer_strategy = "L1Flow"         # Options: "L1Flow", "FM"
num_inference_steps = 2           # Only used when infer_strategy="FM"
t_first = 0.5                     # Only used when infer_strategy="L1Flow"

#------------------------------------------------------------------------------


for run_id in RUN_ids:
    for task_name in TASKS:
        if infer_strategy == "L1Flow":
            strategy_name = "L1Flow"
        else:
            strategy_name = f"FM_NFE{num_inference_steps}"

        task_id = f"{task_name}{EXP_id}_{strategy_name}_{run_id}"
        ckpt_dir = f"results/EXP{EXP_id}/{task_name}/run_{run_id}"

        cmd = [
            "python", "eval.py",
            f"-c {ckpt_dir}",
            f"-i {infer_strategy}",
            f"-n {num_inference_steps}",
            f"-t {t_first}",
            f"-d cuda:0"
        ]
        
        task_list.append({
            "task_id": task_id,
            "run_id": run_id,
            "cmd": " ".join(cmd),
        })

# output 
output_path = Path(__file__).parent / f"EVAL{EXP_id}.jsonl"

with open(output_path, "w") as f:
    for t in task_list:
        f.write(json.dumps(t) + "\n")

print(f"✅ Generated {len(task_list)} tasks in L1FLOW/TASKS/EVAL{EXP_id}.jsonl")