"""
Usage:
python eval.py --checkpoint_dir data/outputs4/pusht/flow_L1_10/run_4/checkpoints
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

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
from diffusion_policy.dataset.base_dataset import BaseImageDataset

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


def get_base_log_path(base_dir, policy, nfe):
    """Return the base path without version suffix: eval_log_p1_n2.yaml"""
    base_dir = pathlib.Path(base_dir)
    stem = f"eval_log_p{policy}_n{nfe}"
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
@click.option('-n', '--nfe', default=2, type=float)
@click.option('-p', '--policy', default=1, type=int)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint_dir, nfe, policy, device):
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(colored(f"checkpoint_dir must be a directory: {checkpoint_dir}", "red"))

    output_dir = checkpoint_dir.parent / "eval_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Check if evaluation already exists ---
    base_log_path = get_base_log_path(output_dir, policy, nfe)
    # if base_log_path.exists():
    #     print(colored(f"[SKIP] Evaluation already exists at {base_log_path}. Skipping.", "yellow"))
    #     return  # Early exit

    # --- Proceed with evaluation ---
    checkpoint = find_best_checkpoint(str(checkpoint_dir))
    print(colored(f"[INFO] Using checkpoint: {checkpoint}", "cyan"))

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    print(payload.keys())
    print("Action normalizer stats:", payload.normalizer['action']._stats)

    # configure dataset
    dataset: BaseImageDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    normalizer = dataset.get_normalizer()

    # self.model.set_normalizer(normalizer)
    # if cfg.training.use_ema:
    #     self.ema_model.set_normalizer(normalizer)
    
    # 然后传给 policy model（关键！）
    # policy_model.set_normalizer(normalizer)   # ← 必须手动调用！

    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(cfg, False)
    if policy == 1:
        # 使用原始 FM 的 inference
        cfg.policy._target_ = "diffusion_policy.policy.flow_test_NFE_origin.FlowUnetL1SampleHybridImagePolicy"
        log_policy = "flow_test_NFE_origin"
        # 转成 int
        cfg.policy.num_inference_steps = int(nfe)
    else:
        # 使用论文中的两步 inference
        cfg.policy._target_ = "diffusion_policy.policy.flow_test_NFE_ours.FlowUnetL1SampleHybridImagePolicy"
        log_policy = "flow_test_NFE_ours"
        # 这里是当做 first_t 用
        cfg.policy.num_inference_steps = nfe
    
    # 统一测 100 次
    # print(cfg.task.env_runner.n_test)
    cfg.task.env_runner.n_test = 100
    
    seed = cfg.training.seed  # 从checkpoint配置获取
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    


if __name__ == '__main__':
    main()