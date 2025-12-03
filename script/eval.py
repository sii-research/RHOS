"""
Usage:
python eval.py -c results/EXP1/pusht/run_0/checkpoints 
python eval.py -c results/EXP1/pusht/run_0/checkpoints -i L1Flow -t 0.5 -d cuda:0
"""

import sys
import os
import pathlib
import click
import hydra
import torch
import dill
import yaml
import re
import numpy as np
import random
from termcolor import colored
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf


def find_best_checkpoint(checkpoint_dir):
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    best_ckpt = None
    best_score = -float('inf')

    pattern = re.compile(r'epoch=\d+-test_mean_score=([\d.]+)\.ckpt')
    for f in ckpt_files:
        match = pattern.match(f)
        if match:
            score = float(match.group(1))
            if score > best_score:
                best_score = score
                best_ckpt = f

    if best_ckpt is None:
        raise FileNotFoundError(
            colored(f"No valid checkpoint found in {checkpoint_dir} matching pattern 'epoch=...-test_mean_score=....ckpt'", "red")
        )
    
    return os.path.join(checkpoint_dir, best_ckpt)


def get_log_path(output_dir, infer_strategy, nfe, t_first):
    """
    Return the base path: 
    example: eval_log_L1Flow_t_0.5.yaml or eval_log_FM_n_10.yaml
    """
    base_dir = pathlib.Path(output_dir)

    if infer_strategy is None:
        return None
    if infer_strategy == "L1Flow":
        if t_first is None:
            raise ValueError("`t_first` must be provided for `L1Flow` inference strategy")
        stem = f"eval_log_{infer_strategy}_t_{t_first}"
    else:
        if nfe is None:
            raise ValueError("`nfe` must be provided for `FM` inference strategy")
        stem = f"eval_log_{infer_strategy}_n_{nfe}"

    return base_dir / (stem + ".yaml")


def to_python_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            # print("This is Tensor")
            return value.item()
        else:
            return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or (value.ndim == 1 and value.size == 1):
            # print("This is ndarray")
            return value.item()
        else:
            return None
    if np.isscalar(value):
        try:
            # print("This is isscalar")
            return float(value) if isinstance(value, (np.floating, float)) else int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float, bool)) or value is None:
        print("This is float")
        return value
    return None


@click.command()
@click.option('-c', '--checkpoint_dir', required=True, help="Directory containing .ckpt files")
@click.option('-i', '--infer_strategy', type=str)
@click.option('-n', '--nfe', type=int)
@click.option('-t', '--t_first', type=float)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint_dir, infer_strategy, nfe, t_first, device):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(colored(f"checkpoint_dir must be a directory: {checkpoint_dir}", "red"))

    output_dir = checkpoint_dir.parent / "eval_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Check if evaluation already exists ---
    base_log_path = get_log_path(output_dir, infer_strategy, nfe, t_first)
    if base_log_path is not None and base_log_path.exists():
        print(colored(f"[SKIP] Evaluation already exists at {base_log_path}. Skipping.", "yellow"))
        return  # Early exit

    # --- Proceed with evaluation ---
    checkpoint = find_best_checkpoint(str(checkpoint_dir))
    print(colored(f"[INFO] Using checkpoint: {checkpoint}", "cyan"))

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(cfg, False)
    
    if infer_strategy is not None:
        cfg.policy.infer_strategy = infer_strategy
        cfg.policy.t_first = t_first
        cfg.policy.nfe = nfe
    
    base_log_path = get_log_path(output_dir, cfg.policy.infer_strategy, cfg.policy.nfe, cfg.policy.t_first)
        
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy_model = workspace.model
    if cfg.training.use_ema:
        policy_model = workspace.ema_model

    device = torch.device(device)
    policy_model.to(device)
    policy_model.eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(output_dir))
    runner_log = env_runner.run(policy_model)
    

    test_mean_score = float(runner_log.get("test/mean_score", None))
    train_mean_score = float(runner_log.get("train/mean_score", None))

    cli_args = {
        "checkpoint": str(checkpoint),
        "infer_strategy": infer_strategy,
        "nfe": nfe,
        "t_first": t_first,
        "test_mean_score": test_mean_score,
        "train_mean_score": train_mean_score,
        "device": str(device)
    }

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    metrics = {}
    for key, value in runner_log.items():
        scalar_val = to_python_scalar(value)
        if scalar_val is not None:
            metrics[key] = scalar_val

    final_log = {
        "cli_args": cli_args,
        "config": cfg_dict,
        "metrics": metrics
    }

    # Save to base name
    out_path = str(base_log_path)
    with open(out_path, 'w') as f:
        yaml.dump(final_log, f, default_flow_style=False, indent=2, sort_keys=False)

    print(colored(f"[INFO] Evaluation log saved to {out_path}", "green"))


if __name__ == '__main__':
    main()