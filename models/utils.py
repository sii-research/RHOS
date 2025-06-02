import torch.nn as nn
import torch.nn.functional as F
import torch


JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye",
    "right_eye",
]
joint_torque_limits = [2500., 2500., 2500., 2600., 2600., 2600., 2500., 2500., 2500., 2000., 2000., 1500., 2000., 2000., 1500., 2000., 2000., 2000., 2000., 1500., 1500., 1200., 1200.,]

acti_dict = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leakyrelu': nn.LeakyReLU,
}

norm_dict = {
    'layer': nn.LayerNorm,
    'batch': nn.BatchNorm1d,
    'rms': nn.RMSNorm,
}