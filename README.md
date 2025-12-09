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

## 🛠️ Installation

### 🖥️ Simulation

To reproduce our simulation benchmark results, install our conda environment on a Linux machine with Nvidia GPU. First, you should install the following apt packages for `mujoco`:

```console
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

Then you can use `conda` or `mamba` as the package manager to create the environment:

```console
conda env create -f yamls/environment.yaml
```

This will create a conda environment named `robodiff`, which is mainly derived from [diffusion_policy](https://github.com/real-stanford/diffusion_policy).

**Attention**: Do not upgrade the package version arbitrarily, as the code strongly depends on `gym==0.21.0`

### 🦾 Real Robot

## 🖥️ Training on Robomimic Benchmark

### Download Training Data

You can run `python download_dataset.py` to download all datasets automatically from [here](https://diffusion-policy.cs.columbia.edu/data/training/).

Or you can execute them manually:

```console
[diffusion_policy]$ mkdir data && cd data
[data]$ wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
[data]$ wget https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip
[data]$ unzip pusht.zip
[data]$ unzip robomimic_image.zip
```

### Train Single Task

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

### Generate Multi-task(recommended)

For convenience, we provide a script for generating multi-task training configurations: TASKS/generate_exp1.py.
You can modify the task list and configuration options in this script to produce the multi-task training setups you need.

```console
python TASKS/generate_exp1.py
```

This command generates TASKS/EXP1.json, which contains the configuration for multi-task training. You can adjust parameters through the override field. The format is as follows:

```json
{
    "task_id": "pusht1_L1Flow_0",
    "run_id": 0,
    "cmd": "python train.py --config-dir=./yamls --config-name=pusht_flow training.device=cuda:0 hydra.run.dir=results/EXP1/pusht/run_0 logging.name=pusht1_L1Flow_0 policy.infer_strategy=L1Flow policy.num_inference_steps=2 policy.t_first=0.5 policy.loss_type=l1 policy.loss_space=sample policy.timestep_sampler_type=mixed task.env_runner.n_test=100 training.num_epochs=200 optimizer.lr=1e-4 policy._target_=diffusion_policy.policy.L1Flow_unet_hybrid_image_policy.L1FlowUnetHybridImagePolicy",
    "output_dir": "results/EXP1/pusht/run_0"
}
```

You can then launch the training tasks using the commands below. The launcher supports multi-GPU and multi-node execution. It will automatically update `training.device=cuda:0` based on the assigned GPU. It also uses lock files to prevent the same task from being started more than once, which you can find under `locks/EXP1/`.

```console
# Launch with multiple GPUs. You will be prompted with "Select EXP_id:".
# Enter a number, e.g., 1 corresponds to the tasks in `TASKS/EXP1.json`.
python task_worker.py --gpu_nums 2

# Launch on a specified single GPU
python task_worker_single.py --gpu_id 0
```

## Evaluate Pre-trained Checkpoints

We provide pre-trained checkpoints for evaluation on the Robomimic benchmark, you can download them from [huggingface](https://huggingface.co/datasets/THyanNK/L1FLOW/tree/main/results).

> We do **not recommend** using this method to determine the performance of the policy, as there is a problem of inconsistent training and evaluation results, which can be seen in this [issue](https://github.com/real-stanford/diffusion_policy/issues/124)

#### Example

First, run `python download_ckpt.py` to download the checkpoint for the checkpoint of `pusht/run_0`, which will be saved in `./results/L1FLOW/pusht/run_0/`

Run the evaluation script:

```console
python eval.py -c results/L1FLOW/pusht/run_0 -i L1Flow -n 2 -t 0.5 -d cuda:0
```

This will generate the following directory structure in `{ckpt_dir}/eval_logs/`

You can check `eval_log.json` to see the metrics that are logged to wandb during training:

```console
{
  "test/mean_score": 0.9150393806777066,
  "test/sim_max_reward_4300000": 1.0,
  "test/sim_max_reward_4300001": 0.9872969750774386,
...
  "train/sim_video_1": "data/pusht_eval_output//media/2fo4btlf.mp4"
}
```

Or you can generate multi eval tasks just like training:

```console
python TASKS/generate_eval1.py
python eval_worker.py --gpu_nums 2
```

## 🦾 Demo, Training and Eval on a Real Robot

Make sure your UR5 robot is running and accepting command from its network interface (emergency stop button within reach at all time), your RealSense cameras plugged in to your workstation (tested with `realsense-viewer`) and your SpaceMouse connected with the `spacenavd` daemon running (verify with `systemctl status spacenavd`).

Start the demonstration collection script. Press "C" to start recording. Use SpaceMouse to move the robot. Press "S" to stop recording.

```console
(robodiff)[diffusion_policy]$ python demo_real_robot.py -o data/demo_pusht_real --robot_ip 192.168.0.204
```

This should result in a demonstration dataset in `data/demo_pusht_real` with in the same structure as our example [real Push-T training dataset](https://diffusion-policy.cs.columbia.edu/data/training/pusht_real.zip).

To train a Diffusion Policy, launch training with config:

```console
(robodiff)[diffusion_policy]$ python train.py --config-name=train_diffusion_unet_real_image_workspace task.dataset_path=data/demo_pusht_real
```

Edit [`diffusion_policy/config/task/real_pusht_image.yaml`](./diffusion_policy/config/task/real_pusht_image.yaml) if your camera setup is different.

Assuming the training has finished and you have a checkpoint at `data/outputs/blah/checkpoints/latest.ckpt`, launch the evaluation script with:

```console
python eval_real_robot.py -i data/outputs/blah/checkpoints/latest.ckpt -o data/eval_pusht_real --robot_ip 192.168.0.204
```

Press "C" to start evaluation (handing control over to the policy). Press "S" to stop the current episode.

## 🗺️ Codebase Tutorial

You can find a detailed codebase tutorial in [TUTORIAL.md](TUTORIAL.md) to help you understand the implementation details.

## 🏷️ License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

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
