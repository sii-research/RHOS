import json
from pathlib import Path

EXP_id = 2

TASKS = ["square_ph", "toolhang_ph", "transport_mh", "transport_ph"]
# TASKS = ["pusht", "square_mh"]

task_list = []
RUN_ids  = [0]

#------------------------------------------------------------------------------
# Configuration Options (adjust as needed):
#------------------------------------------------------------------------------

# Global training parameters
GLOBAL_PARAMS = {
    "infer_strategy": "L1Flow",         # Options: "L1Flow" (default), "FM"
    "num_inference_steps": 2,           # Only used when infer_strategy="FM"
    "t_first": 0.5,                     # Only used when infer_strategy="L1Flow"
    "loss_type": "l1",                  # Options: "l1" (default), "mse"
    "loss_space": "sample",             # Options: "sample" (default), "velocity"
    "timestep_sampler_type": "mixed",   # Options: "mixed" (default), "uniform", "beta"
    "test_episodes": 10,               # Number of test episodes per evaluation
}

# Task-specific hyperparameters (epochs and learning rates)
TASK_CONFIG = {
    "pusht":        {"epochs": 200, "lr": "1e-4"},
    "square_mh":    {"epochs": 1000, "lr": "2e-5"},
    "square_ph":    {"epochs": 200, "lr": "1e-4"},
    "toolhang_ph":  {"epochs": 500, "lr": "5e-5"},
    "transport_mh": {"epochs": 200, "lr": "1e-4"},
    "transport_ph": {"epochs": 200, "lr": "6e-5"},
}
#------------------------------------------------------------------------------


run_id = 0 
for task_name in TASKS:
    for loss_space in ["sample", "velocity"]:
        for loss_type in ["l1", "mse"]:
            for infer_strategy in ["L1Flow", "FM"]:
                for timestep_sampler_type in ["mixed", "uniform", "beta"]:
                    run_id += 1
                    if GLOBAL_PARAMS["infer_strategy"] == "L1Flow":
                        strategy_name = "L1Flow"
                    else:
                        strategy_name = f"FM_NFE{GLOBAL_PARAMS['num_inference_steps']}"

                    task_id = f"{task_name}{EXP_id}_{strategy_name}_{run_id}"
                    log_name = f"{task_name}{EXP_id}_{strategy_name}_{run_id}"
                    output_dir = f"results/EXP{EXP_id}/{task_name}/run_{run_id}"

                    lr = TASK_CONFIG[task_name]["lr"]
                    epoch = 10

                    cmd = [
                        "python", "train.py",
                        f"--config-dir=./yamls",
                        f"--config-name={task_name}_flow",
                        f"training.device=cuda:0",
                        f"hydra.run.dir={output_dir}",
                        f"logging.name={log_name}",
                        f"policy.infer_strategy={infer_strategy}",
                        f"policy.num_inference_steps={GLOBAL_PARAMS['num_inference_steps']}",
                        f"policy.t_first={GLOBAL_PARAMS['t_first']}",
                        f"policy.loss_type={loss_type}",
                        f"policy.loss_space={loss_space}",
                        f"policy.timestep_sampler_type={timestep_sampler_type}",
                        f"task.env_runner.n_test={GLOBAL_PARAMS['test_episodes']}",
                        f"training.num_epochs={epoch}",
                        f"optimizer.lr={lr}",
                        f"policy._target_=diffusion_policy.policy.L1Flow_unet_hybrid_image_policy.L1FlowUnetHybridImagePolicy"
                    ]
                    
                    task_list.append({
                        "task_id": task_id,
                        "run_id": run_id,
                        "cmd": " ".join(cmd),
                        "output_dir": output_dir
                    })

EXP_file = f"EXP{EXP_id}.jsonl"

with open(EXP_file, "w") as f:
    for t in task_list:
        f.write(json.dumps(t) + "\n")

print(f"✅ Generated {len(task_list)} tasks in TASKS/{EXP_file}")