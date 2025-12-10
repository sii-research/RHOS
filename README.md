# L1Flow: L1 Sample Flow for Efficient Visuomotor Learning

[[Project page]](https://song-wx.github.io/l1flow.github.io/)
[[Paper]](https://arxiv.org/pdf/2511.17898)
[[Dataset]](https://diffusion-policy.cs.columbia.edu/data/training/)

[Weixi Song](https://scholar.google.com/citations?user=fvP8SGcAAAAJ)<sup>1,2,3</sup>,
[Zhetao Chen](https://scholar.google.com/citations?hl=zh-CN&user=1r_iQ9YAAAAJ)<sup>1,2</sup>,
[Tao Xu](https://openreview.net/profile?id=~Tao_Xu19)<sup>2</sup>,
[Xianchao Zeng](https://openreview.net/profile?id=~Xianchao_Zeng1)<sup>2</sup>,
[Xinyu Zhou](https://openreview.net/profile?id=~Xinyu_Zhou13)<sup>2</sup>,
[Lixin Yang](https://scholar.google.com/citations?user=Bm8p4JsAAAAJ)<sup>2,4</sup>,
[Donglin Wang](https://scholar.google.com/citations?user=-fo6wdwAAAAJ)<sup>3†</sup>,
[Cewu Lu](https://scholar.google.com/citations?user=QZVQEWAAAAAJ)<sup>2,4</sup>,
[Yonglu Li](https://scholar.google.com/citations?user=UExAaVgAAAAJ)<sup>2,4†</sup>

<sup>1</sup>Zhejiang University,
<sup>2</sup>Shanghai Innovation Institute,
<sup>3</sup>Westlake University,
<sup>4</sup>Shanghai Jiao Tong University

<sup>†</sup>Corresponding Author

<img src="media/outline.jpg" alt="drawing" width="60%"/>
<img src="media/algo.jpg" alt="drawing" width="50%"/>

## 🛠️ Environment Installation

To reproduce our simulation benchmark results, install our conda environment on a Linux machine with Nvidia GPU. First, you should install the following apt packages for `mujoco`:

```console
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

Then you can use `conda` or `mamba` as the package manager to create the environment:

```console
conda env create -f yamls/environment.yaml
```

This will create a conda environment named `robodiff`, which is mainly derived from [diffusion_policy](https://github.com/real-stanford/diffusion_policy).

**⚠️ Attention: Do not upgrade the package version arbitrarily, as the code strongly depends on `gym==0.21.0`**

## 🖥️ Training on Robomimic Benchmark

### 1. Download Training Data

You can run this script to download all datasets automatically from [here](https://diffusion-policy.cs.columbia.edu/data/training/).

```console
python download_dataset.py
```

Or you can execute them manually:

```console
[diffusion_policy]$ mkdir data && cd data
[data]$ wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
[data]$ wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip
[data]$ unzip pusht.zip
[data]$ unzip robomimic_image.zip
```

### 2. Train Single Task

Activate conda environment and login to [wandb](https://wandb.ai) (if you haven't already).

```console
conda activate robodiff
wandb login
```

Launch training with seed 42 on GPU 0.

```console
python train.py --config-dir=./yamls --config-name=pusht_flow training.device=cuda:0 hydra.run.dir=results/EXP1/pusht/run_0 logging.name=pusht1_L1Flow_0
```

This will create a directory `results/EXP1/pusht/run_0` where configs, logs and checkpoints are written to. The policy will be evaluated every 5 epochs with the success rate logged as `test/mean_score` on wandb, as well as videos for some rollouts.

The result directory `results/EXP1/pusht/run_0` structure:

```console
├── .hydra
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── checkpoints
│   ├── epoch=0090-test_mean_score=0.744.ckpt
│   ├── epoch=0140-test_mean_score=0.738.ckpt
│   └── epoch=0185-test_mean_score=0.758.ckpt
├── media
│   ├── xxx.mp4
│   └── ...
├── wandb
│   └── ...
├── logs.json.txt
└── train.log
```

### ⭐ 3. Generate Multi-task (Recommended)

For convenience, we provide a script `TASKS/generate_exp1.py` for generating multi-task training configurations. You can modify the task list and configuration options in this script. The detailed configuration options are shown in Sec4:configurations.

```console
python TASKS/generate_exp1.py
```

Run this command will generate `TASKS/EXP1.json`, which contains the configuration for multi-task training, which will adjust configuration through the override field. The format is as follows:

```json
{
    "task_id": "pusht1_L1Flow_0",
    "run_id": 0,
    "cmd": "python train.py --config-dir=./yamls --config-name=pusht_flow training.device=cuda:0 hydra.run.dir=results/EXP1/pusht/run_0 logging.name=pusht1_L1Flow_0 policy.infer_strategy=L1Flow policy.num_inference_steps=2 policy.t_first=0.5 policy.loss_type=l1 policy.loss_space=sample policy.timestep_sampler_type=mixed task.env_runner.n_test=100 training.num_epochs=200 optimizer.lr=1e-4 policy._target_=diffusion_policy.policy.L1Flow_unet_hybrid_image_policy.L1FlowUnetHybridImagePolicy",
    "output_dir": "results/EXP1/pusht/run_0"
}
```

You can then launch the training tasks using the commands below. The launcher supports multi-GPU and multi-node execution. It will automatically update `training.device=cuda:0` based on the assigned GPU. It also uses lock files(under `locks/EXP1/`) to prevent the same task from being started more than once.

```console
# Launch with multiple GPUs. You will be prompted with "Select EXP_id:".
# Enter a number, e.g., `1` corresponds to the tasks in `TASKS/EXP1.json`.
python task_worker.py --gpu_nums 2

# Launch on a specified single GPU. You will be prompted with "Select EXP_id:".
# Enter a number, e.g., `1` corresponds to the tasks in `TASKS/EXP1.json`.
python task_worker_single.py --gpu_id 0
```

### 4. Summary Results

We provide some scripts to summarize the results of multiple runs.

```py
# Enter a number to choose the EXP you want to summary, e.g., `1` corresponds to the results in `results/EXP1/`.
# It will summary all tasks under `results/<EXP>/`
python summary_exp.py

# 1. Enter a number to choose the EXP you want to summary
# 2. Enter a number to choose the task you want to summary
# It will summary the choosen task under `results/<EXP>/`
python summary_task.py

# Enter a number to choose the task you want to summary
# It will summary all runs under `results/<EXP>/<task>/`
python summary_task_all.py
```

### 5. Evaluate Pre-trained Checkpoints

**⚠️ We do _not recommend_ using this method to determine the performance of the policy, as there is a problem of inconsistent training and evaluation results, which can be seen in this [issue](https://github.com/real-stanford/diffusion_policy/issues/124).**

We provide pre-trained checkpoints for evaluation on the Robomimic benchmark, you can download them from [huggingface](https://huggingface.co/datasets/THyanNK/L1FLOW/tree/main/results).

#### Example

First, run the download script to download the checkpoint of `pusht/run_0`, which will be saved in `results/L1FLOW/pusht/run_0/`

```console
python download_ckpt.py
```

Then you can run the evaluation script with the inference strategy you want.

```console
python eval.py -c results/L1FLOW/pusht/run_0 -i L1Flow -n 2 -t 0.5 -d cuda:0
```

This will generate the following directory structure in `{ckpt_dir}/eval_logs/`

You can check `eval_log_L1Flow_t_0.5.json` to see the eval results. The format is as follows:

```console
{
cli_args:
  ...
  test_mean_score: 0.829236895631236
  train_mean_score: 0.7408510637545613
  ...
config:
  ...
metrics:
  train/sim_max_reward_0: 0.9882974097369663
  train/sim_max_reward_1: 1.0
  ...
}
```

We also provide scripts to generate multi eval tasks. You can use them just like the `3. Generate Multi-task`.

```console
python TASKS/generate_eval1.py
python eval_worker.py --gpu_nums 2
```

## 🗺️ Codebase Tutorial

You can find a detailed codebase tutorial in [TUTORIAL.md](TUTORIAL.md) to help you understand the implementation details.

## 🏷️ License

This repository is released under the MIT license. See [LICENSE](media/LICENSE) for additional details.

## 🙏 Acknowledgement

This code is mainly derived from [diffusion_policy](https://github.com/real-stanford/diffusion_policy), and we have made a series of modifications to adapt our algorithm and make it more user-friendly. We sincerely thank the authors for their excellent work.

Below is the original acknowledgement:

-   Our [`ConditionalUnet1D`](./diffusion_policy/model/diffusion/conditional_unet1d.py) implementation is adapted from [Planning with Diffusion](https://github.com/jannerm/diffuser).
-   Our [`TransformerForDiffusion`](./diffusion_policy/model/diffusion/transformer_for_diffusion.py) implementation is adapted from [MinGPT](https://github.com/karpathy/minGPT).
-   The [BET](./diffusion_policy/model/bet) baseline is adapted from [its original repo](https://github.com/notmahi/bet).
-   The [IBC](./diffusion_policy/policy/ibc_dfo_lowdim_policy.py) baseline is adapted from [Kevin Zakka's reimplementation](https://github.com/kevinzakka/ibc).
-   The [Robomimic](https://github.com/ARISE-Initiative/robomimic) tasks and [`ObservationEncoder`](https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/models/obs_nets.py) are used extensively in this project.
-   The [Push-T](./diffusion_policy/env/pusht) task is adapted from [IBC](https://github.com/google-research/ibc).
-   The [Block Pushing](./diffusion_policy/env/block_pushing) task is adapted from [BET](https://github.com/notmahi/bet) and [IBC](https://github.com/google-research/ibc).
-   The [Kitchen](./diffusion_policy/env/kitchen) task is adapted from [BET](https://github.com/notmahi/bet) and [Relay Policy Learning](https://github.com/google-research/relay-policy-learning).
-   Our [shared_memory](./diffusion_policy/shared_memory) data structures are heavily inspired by [shared-ndarray2](https://gitlab.com/osu-nrsg/shared-ndarray2).
