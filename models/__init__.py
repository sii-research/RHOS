import torch
from .losses import lossLayer
from .metrics import metricLayer
from .motrastAEFD import MoTrastAEFD, MoTrastAEFD_Large
model_dict = {
    'MoTrastAEFD': MoTrastAEFD,
    'MoTrastAEFD_Large': MoTrastAEFD_Large,
}

def get_model(config):
    model = model_dict[config.MODEL.NAME](config.MODEL)
    if config.MODEL.get('ckpt', None):
        model.load_state_dict(torch.load(config.MODEL.ckpt, map_location='cpu', weights_only=False)['model_state'], strict=False)
    return model
    
def get_loss(config):
    return lossLayer(config)
    
def get_metric(config):
    return metricLayer(config)