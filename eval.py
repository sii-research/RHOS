import argparse
import random
import shutil
from collections import OrderedDict
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from torch import optim
import os
import tqdm
import warnings
import json
import wandb
from torch.utils.tensorboard import SummaryWriter
from utils.utils import backup_config_file, create_logger, set_seed
from utils.optimizer import build_optimizer, build_lr_scheduler
from dataset import load_data_norm, get_preprocess_fn, get_collate_fn, contrastDataset, contrastEvalDataset
from engine import make_train_one_epoch, evaluate
from models import get_model, get_loss, get_metric
import time

if __name__ == '__main__':
    cli_conf = OmegaConf.from_cli()
    if not hasattr(cli_conf, 'config_path'):
        cli_conf.config_path = 'configs/naive.yml'
    config   = OmegaConf.merge(OmegaConf.load(cli_conf.config_path), cli_conf)
    seed     = config.get('seed', 42)
    if config.IGNORE_WARNINGS:
        warnings.filterwarnings("ignore")
    config.DEVICE_STR = f"cuda:{config.DEVICE}" if torch.cuda.is_available() else "cpu"
    device = torch.device(config.DEVICE_STR)
    

    if config.get('USE_WANDB', False):
        config_dict = {
            'model': config.MODEL.NAME,
            'optimizer': config.OPTIMIZER.TYPE,
            'learning_rate': config.OPTIMIZER.LR.base,
            'batch_size': config.TRAIN.BATCH_SIZE,
            'max_norm': config.TRAIN.max_norm
        }
        wandb.init(
            project='Molign',
            name=config.RUN_NAME + '_eval',
            id=config.get('RUN_ID', 't' + str(int(time.time() * 100))),
            resume='allow',
            config=config_dict
        )
    
    set_seed(seed)
    logger = create_logger(config)
    writer = SummaryWriter(config.RUN_PATH)
    backup_config_file(config)

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"RUN name: {config.RUN_NAME}")
    logger.info(f"Using device: {config.DEVICE_STR}")
    logger.info("Initializing dataset...")
    
    if config.DATASET.TEST.get('MODE', 'raw') == 'raw':
        test_dataset = contrastDataset(config.DATASET.TEST, split='test')
    elif config.DATASET.TEST.get('MODE', 'raw') in ['seq']:
        test_dataset = contrastEvalDataset(config.DATASET.TEST, split='test')
    else:
        raise NotImplementedError
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=config.TEST.NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=config.TEST.PREFETCH,
        collate_fn=get_collate_fn(config.DATASET.TEST),
    )
    test_preprocess_fn  = get_preprocess_fn(config.DATASET.TEST, device)
    
    
    logger.info("Initializing network...")
    network   = get_model(config).to(device)
    metric    = get_metric(config.METRIC)
    
    logger.info("Start evaluation...")
    eval_epoch = config.get('eval_epoch', 1000)
    if os.path.exists(os.path.join(config.RUN_PATH, f'epoch_{eval_epoch}.pt')):
        checkpoint = torch.load(os.path.join(config.RUN_PATH, f'epoch_{i}.pt'), map_location='cpu')
        network.load_state_dict(checkpoint['model_state'], strict=False)
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step'] + 1
    
        eval_stats = evaluate(network, metric, test_loader, test_preprocess_fn, writer, start_epoch)
        log_stats = {**{f'test_{k}': v for k, v in eval_stats.items()}, 'epoch': start_epoch,}
        logger.info(json.dumps(log_stats))
        wandb_log = {}
        if config.get('USE_WANDB', False):
            wandb_log.update({k: v for k, v in log_stats.items() if v != []})
            wandb.log(wandb_log)
    logger.info("Evaluation finished.")


