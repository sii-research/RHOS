# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import tqdm
import joblib
from collections import defaultdict
from easydict import EasyDict as edict
import torch
import torch.utils.tensorboard
import torch.nn.functional as F
from utils.geometry import *
import utils.misc as utils

def debug_hook(name):
    def hook(grad):
        if grad.isnan().sum() > 0:
            print(name, grad.isnan().sum())
    return hook

def make_train_one_epoch(config):
    style = config.TRAIN.get('style', 'nce')
    if 'FD_scale' in style:
        return train_one_epoch_FD_scale
    else:
        raise NotImplementedError

def train_one_epoch_FD(model: torch.nn.Module, criterion: torch.nn.Module, 
                    data_loader: Iterable, preprocess_fn, optimizer: torch.optim.Optimizer, writer: torch.utils.tensorboard.SummaryWriter, 
                    device: torch.device, epoch: int, 
                    max_norm: float = 0, global_step: int = 0,):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        loss_dict = {}
        input_real, input_sim, input_smpl, input_mint, input_mia, input_glink  = preprocess_fn(batch)

        optimizer.zero_grad()
        if input_real is not None:
            output_real = model(input_real, 'real')
            loss_dict.update(criterion.loss_torque(input_real, output_real, 'btau', 'mar'))
            loss_dict.update(criterion.loss_torque(input_real, output_real, 'btau', 'bio'))
            loss_dict.update(criterion.loss_torque(input_real, output_real, 'btau', 'joi'))
            loss_dict.update(criterion.loss_acc(input_real, output_real, 'real'))
            loss_tmp = criterion.loss_nce(output_real, 'real')
            loss_dict.update(loss_tmp)
        if input_sim is not None:
            output_sim  = model(input_sim, 'sim')
            loss_dict.update(criterion.loss_torque(input_sim, output_sim, 'stau', 'mar'))
            loss_dict.update(criterion.loss_torque(input_sim, output_sim, 'stau', 'joi'))
            loss_dict.update(criterion.loss_torque(input_sim, output_sim, 'stau', 'smp'))
            loss_dict.update(criterion.loss_acc(input_sim, output_sim, 'sim'))
            loss_tmp = criterion.loss_nce(output_sim, 'sim')
            loss_dict.update(loss_tmp)
        if input_smpl is not None:
            output_smpl = model(input_smpl, 'smpl')
            loss_tmp = criterion.loss_nce(output_smpl, 'smpl')
            loss_dict.update(loss_tmp)
        if input_mint is not None:
            output_mint = model(input_mint, 'mint')
            loss_dict.update(criterion.loss_torque(input_mint, output_mint, 'mtau', 'mar'))
            loss_dict.update(criterion.loss_torque(input_mint, output_mint, 'mtau', 'joi'))
            loss_dict.update(criterion.loss_torque(input_mint, output_mint, 'mtau', 'smp'))
            loss_dict.update(criterion.loss_acc(input_mint, output_mint, 'mint'))
            loss_tmp = criterion.loss_nce(output_mint, 'mint')
            loss_dict.update(loss_tmp)
        if input_mia is not None:
            output_mia = model(input_mia, 'mia')
            loss_dict.update(criterion.loss_torque(input_mia, output_mia, 'etau', 'mar'))
            loss_dict.update(criterion.loss_torque(input_mia, output_mia, 'etau', 'joi'))
            loss_tmp = criterion.loss_nce(output_mia, 'mia')
            loss_dict.update(loss_tmp)
        if input_glink is not None:
            output_glink = model(input_glink, 'glink')
            loss_dict.update(criterion.loss_torque(input_glink, output_glink, 'grf', 'mar'))
            loss_dict.update(criterion.loss_torque(input_glink, output_glink, 'grf', 'joi'))
            loss_dict.update(criterion.loss_torque(input_glink, output_glink, 'grf', 'smp'))
            loss_tmp = criterion.loss_nce(output_glink, 'glink')
            loss_dict.update(loss_tmp)
        weight_dict   = criterion.weight_dict
        loss_dict     = {k: v.mean() * weight_dict.get(k, 1.) for k, v in loss_dict.items()}
        losses        = sum(loss_dict[k] for k in loss_dict.keys())
        loss_value    = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            for k, v in loss_dict.items():
                print(k, v)
            sys.exit(1)

        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        writer.add_scalars('losses_scaled', loss_dict, global_step)
        global_step += 1
        data_loader.dataset.update_msel()
    optimizer.zero_grad()
    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step    


def train_one_epoch_FD_scale(model: torch.nn.Module, criterion: torch.nn.Module, 
                    data_loader: Iterable, preprocess_fn, optimizer: torch.optim.Optimizer, writer: torch.utils.tensorboard.SummaryWriter, 
                    device: torch.device, epoch: int, 
                    max_norm: float = 0, global_step: int = 0,):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        loss_dict = {}
        input_real, input_sim, input_smpl, input_mint, input_mia, input_glink  = preprocess_fn(batch)

        optimizer.zero_grad()
        if input_real is not None:
            output_real = model(input_real, 'real')
            loss_dict.update(criterion.loss_torque(input_real, output_real, 'btau', 'bio'))
            loss_dict.update(criterion.loss_torque(input_real, output_real, 'btau', 'joi'))
            loss_dict.update(criterion.loss_torque(input_real, output_real, 'btau', 'mar'))
            loss_dict.update(criterion.loss_acc(input_real, output_real, 'real'))
            loss_dict.update(criterion.loss_jacc(input_real, output_real, 'real'))
            loss_dict.update(criterion.loss_nce(output_real, 'real'))
        if input_sim is not None:
            output_sim  = model(input_sim, 'sim')
            loss_dict.update(criterion.loss_torque(input_sim, output_sim, 'stau', 'joi'))
            loss_dict.update(criterion.loss_torque(input_sim, output_sim, 'stau', 'smp'))
            loss_dict.update(criterion.loss_torque(input_sim, output_sim, 'stau', 'mar'))
            loss_dict.update(criterion.loss_acc(input_sim, output_sim, 'sim'))
            loss_dict.update(criterion.loss_jacc(input_sim, output_sim, 'sim'))
            loss_dict.update(criterion.loss_nce(output_sim, 'sim'))
        if input_smpl is not None:
            pass
        if input_mint is not None:
            output_mint = model(input_mint, 'mint')
            loss_dict.update(criterion.loss_torque(input_mint, output_mint, 'mtau', 'joi'))
            loss_dict.update(criterion.loss_torque(input_mint, output_mint, 'mtau', 'smp'))
            loss_dict.update(criterion.loss_torque(input_mint, output_mint, 'mtau', 'mar'))
            loss_dict.update(criterion.loss_acc(input_mint, output_mint, 'mint'))
            loss_dict.update(criterion.loss_jacc(input_mint, output_mint, 'mint'))
            loss_dict.update(criterion.loss_nce(output_real, 'real'))
        if input_mia is not None:
            output_mia = model(input_mia, 'mia')
            loss_dict.update(criterion.loss_torque(input_mia, output_mia, 'etau', 'joi'))
            loss_dict.update(criterion.loss_torque(input_mia, output_mia, 'etau', 'mar'))
            loss_dict.update(criterion.loss_jacc(input_mia, output_mia, 'mia'))
            loss_dict.update(criterion.loss_nce(output_mia, 'mia'))
        weight_dict   = criterion.weight_dict
        loss_dict     = {k: v.mean() * weight_dict.get(k, 1.) for k, v in loss_dict.items()}
        losses        = sum(loss_dict[k] for k in loss_dict.keys())
        loss_value    = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            for k, v in loss_dict.items():
                print(k, v)
            sys.exit(1)

        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        metric_logger.update(loss=loss_value, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        writer.add_scalars('losses_scaled', loss_dict, global_step)
        global_step += 1
        data_loader.dataset.update_msel()
    optimizer.zero_grad()
    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step    

def evaluate(model: torch.nn.Module, metric: torch.nn.Module, data_loader: Iterable, preprocess_fn, writer: torch.utils.tensorboard.SummaryWriter, epoch: int):
    
    model.eval()
    # metric.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            input_real, input_sim, _, input_mint, input_mia, input_glink = preprocess_fn(batch)
            tmpres = {}
            if input_real is not None:
                output_real = model(input_real, 'real')
                tmpres.update(metric.L1_torque(input_real, output_real, 'btau'))
                metric_logger.update(**tmpres, cnt=output_real['btau_joi'].shape[0])
            tmpres = {}
            if input_sim is not None:
                output_sim  = model(input_sim, 'sim')
                tmpres.update(metric.imdy_mPJE(input_sim, output_sim, 'stau'))
                metric_logger.update(**tmpres, cnt=output_sim['stau_joi'].shape[0])
            tmpres = {}
            if input_mint is not None:
                output_mint  = model(input_mint, 'mint')
                tmpres.update(metric.RMSE(input_mint, output_mint, 'mtau'))
                tmpres.update(metric.PCC(input_mint, output_mint, 'mtau'))
                metric_logger.update(**tmpres, cnt=output_mint['mtau_joi'].shape[0])
            tmpres = {}
            if input_mia is not None:
                output_mia  = model(input_mia, 'mia')
                tmpres.update(metric.RMSE(input_mia, output_mia, 'etau'))
                tmpres.update(metric.PCC(input_mia, output_mia, 'etau'))
                metric_logger.update(**tmpres, cnt=output_mia['etau_joi'].shape[0])
            tmpres = {}
            if input_glink is not None:
                output_glink  = model(input_glink, 'glink')
                tmpres.update(metric.glink(input_glink, output_glink, 'grf'))
                metric_logger.update(**tmpres, cnt=output_glink['grf_joi'].shape[0])
    print("Averaged stats:", metric_logger)
    res = {}
    for k, meter in metric_logger.meters.items():
        res[k] = meter.global_avg
    writer.add_scalars('metrics', {k: v for k, v in res.items()}, epoch)
    return res
