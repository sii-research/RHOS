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
from dataset import get_preprocess_fn, get_collate_fn, contrastDataset, contrastEvalDataset
from engine import make_train_one_epoch, evaluate
from models import get_model, get_loss, get_metric
import time

def train(config, network, criterion, metric, optimizer, scheduler, train_loader, train_preprocess_fn, test_loader, test_preprocess_fn,
          logger, writer, device, global_step, max_norm, start_ep):
    logger.info(f"Length of train loader: {len(train_loader)}")
    use_wandb = config.get('USE_WANDB', False)
    train_one_epoch = make_train_one_epoch(config)
    for epoch in range(start_ep, config.TRAIN.EPOCHS):
        
        train_stats, global_step = train_one_epoch(network, criterion, train_loader, train_preprocess_fn, 
                                                   optimizer, writer, device, epoch, max_norm, global_step)
        if scheduler is not None:
            scheduler.step()

        if use_wandb:
            wandb_log = {}

        if (epoch + 1) % config.log_interval == 0:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch + 1,}
            logger.info(json.dumps(log_stats))
            if use_wandb:
                wandb_log.update({k: v for k, v in log_stats.items() if v != []})
        
        if (epoch + 1) % config.eval_interval == 0:
            eval_stats = evaluate(network, metric, test_loader, test_preprocess_fn, writer, epoch)
            log_stats = {**{f'test_{k}': v for k, v in eval_stats.items()}, 'epoch': epoch + 1,}
            logger.info(json.dumps(log_stats))
            if use_wandb:
                wandb_log.update({k: v for k, v in log_stats.items() if v != []})

        if use_wandb:
            wandb.log(wandb_log)
        
        if (epoch + 1) % config.save_interval == 0:
            logger.info("Saving checkpoint...")
            checkpoint = {
                'epoch': epoch, # from 0
                'model_state': network.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
                'global_step': global_step,
            }
            if config.DEVICE_STR.startswith('cuda'):
                checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state(device=config.DEVICE_STR)
            ckpt_save_path = os.path.join(config.RUN_PATH, 'checkpoint.pth')
            torch.save(checkpoint, os.path.join(config.RUN_PATH, f'epoch_{epoch+1}.pt'))

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
    
    set_seed(seed)
    logger = create_logger(config)
    writer = SummaryWriter(config.RUN_PATH)
    backup_config_file(config)

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
            name=config.RUN_NAME,
            id=config.get('RUN_ID', 't' + str(int(time.time() * 100))),
            resume='allow',
            config=config_dict
        )

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"RUN name: {config.RUN_NAME}")
    logger.info(f"Using device: {config.DEVICE_STR}")
    logger.info("Initializing dataset...")
    
    
    # train_dataset = rawTorqueDataset(config.DATASET.TRAIN, split='train')
    if config.DATASET.TRAIN.get('MODE', 'raw') in ['raw', 'seq']:
        train_dataset = contrastDataset(config.DATASET.TRAIN, split='train')
    else:
        raise NotImplementedError
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True if not config.TRAIN.DEBUG else False,
        drop_last=True,
        num_workers=config.TRAIN.NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=config.TRAIN.PREFETCH,
        collate_fn=get_collate_fn(config.DATASET.TRAIN),
    )
    train_preprocess_fn = get_preprocess_fn(config.DATASET.TRAIN, device)
    if config.DATASET.TEST.get('MODE', 'raw') in ['raw']:
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
    criterion = get_loss(config.LOSS)
    metric    = get_metric(config.METRIC)

    logger.info("Initializing optimizer...")
    optimizer = build_optimizer(network, config)
    scheduler = build_lr_scheduler(config, optimizer)
    

    if config.get('RESUME', False):
        checkpoint = torch.load(config.RESUME.RESUME_CKPT, map_location='cpu', weights_only=False)
        network.load_state_dict(checkpoint['model_state'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler is not None and checkpoint['scheduler_state'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        torch.set_rng_state(checkpoint['rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['python_rng_state'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step'] + 1
        if config.DEVICE_STR.startswith('cuda'):
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'], device=config.DEVICE_STR)
            
        logger.info(f"Resuming from epoch {start_epoch+1}")
    else:
        start_epoch = 0
        global_step = 0
    
    logger.info("Start training...")
    train(config, network, criterion, metric, optimizer, scheduler, train_loader, train_preprocess_fn, test_loader, test_preprocess_fn, 
          logger, writer, device, global_step, config.TRAIN.max_norm, start_epoch)
    logger.info("Training finished.")


