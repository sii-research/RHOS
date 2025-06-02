import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.correspond import mint_group_idx


def normal_kl(mean1, var1, mean2, var2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = var1.log(), var2.log()

    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * var2)

class lossLayer:

    def __init__(self, config):
        self.nce_temp    = config.get('nce_temp', 1.)
        self.weight_dict = config.weight_dict
        self.mint_group_idx = list(mint_group_idx.values())
        self.adb_dof     = config.get('adb_dof', 23)
        
    def loss_commit(self, output, btype):
        losses = 0
        for k in output.keys():
            if 'l_' in k:
                losses += output[k]
        return {f'L_commit_{btype}': losses}
        
    def loss_vqprior(self, output, btype):
        loss_prior = 0
        for k in ['_mar', '_joi', '_bio', '_smp']:
            if f'z{k}' in output:
                loss_prior += F.mse_loss(output[f'z{k}'], output[f'z{k}S'])
        return {f'L_vqprior_{btype}': loss_prior / 3}
    
    def loss_mus(self, batch, output, tauType, inType):
        losses  = sum([F.l1_loss(batch[tauType][..., idx].mean(dim=-1), output[f'{tauType}_{inType}'][..., idx].mean(dim=-1)) for idx in self.mint_group_idx]) / len(self.mint_group_idx)
        losses +=  F.l1_loss(batch[tauType], output[f'{tauType}_{inType}'])
        return {f'L_{tauType}_{inType}': losses}
    
    def loss_torque(self, batch, output, tauType, inType):
        if batch['src'] == 'real':
            return {f'L_{tauType}_{inType}': F.l1_loss(batch[tauType][..., :17], output[f'{tauType}_{inType}'][..., :17])}
        elif batch['src'] == 'realarm':
            return {f'L_arm{tauType}_{inType}': F.l1_loss(batch[tauType], output[f'{tauType}_{inType}'])}
        else:
            return {f'L_{tauType}_{inType}': F.l1_loss(batch[tauType], output[f'{tauType}_{inType}'])}
        
    def loss_nce(self, batch, btype):
        zs = [F.normalize(v, dim=-1) for k, v in batch.items() if 'z_' in k]
        num_zs = len(zs)
        labels = torch.arange(len(zs[0]), device=zs[0].device)
        losses = 0
        for i in range(num_zs):
            for j in range(i, num_zs):
                tmp_logits = zs[i] @ zs[j].T
                losses += F.cross_entropy(tmp_logits / self.nce_temp, labels)
                losses += F.cross_entropy(tmp_logits.permute(1, 0) / self.nce_temp, labels)
        return {f'L_nce_{btype}': losses / num_zs ** 2}
        
    def loss_crossNCE(self, b1, b2, pre_logits, btype1, btype2):
        zs1 = [F.normalize(v, dim=-1) for k, v in b1.items() if 'z_' in k]
        zs2 = [F.normalize(v, dim=-1) for k, v in b2.items() if 'z_' in k]
        losses = 0
        for i in range(len(zs1)):
            for j in range(len(zs2)):
                logits = zs1[i] @ zs2[j].T # B1, B2
                logits1 = torch.cat((pre_logits[btype1][:, None], logits), dim=1)
                losses += F.cross_entropy(logits1 / self.nce_temp, torch.zeros(logits1.shape[0], device=logits1.device, dtype=torch.long))
                logits2 = torch.cat((pre_logits[btype2][None], logits), dim=0).permute(1, 0)
                losses += F.cross_entropy(logits2 / self.nce_temp, torch.zeros(logits2.shape[0], device=logits2.device, dtype=torch.long))
        return {f'L_nce_{btype1}_{btype2}': losses / (len(zs1) * len(zs2) * 2)}
    
    def loss_acc(self, batch, output, btype):
        losses = {}
        if btype in ['sim', 'mint']:
            for k in ['smp', 'mar', 'joi']:
                losses.update({f'L_sacc_{k}_{btype}': F.l1_loss(batch['smp'][..., 6:].flatten(-2), output[f'sacc_{k}'])})
        elif btype in ['real']:
            for k in ['mar', 'joi', 'bio']:
                losses.update({f'L_bacc_{k}_{btype}': F.l1_loss(batch['bio'][..., 46:], output[f'bacc_{k}'])})
        elif btype in ['smpl']:
            for k in ['sim', 'rea', 'mus', 'emg']:
                losses.update({f'L_sacc_{k}_{btype}': F.l1_loss(batch['smp'][..., 6:].flatten(-2), output[f'sacc_{k}'])})
        return losses
    
    def loss_jacc(self, batch, output, btype):
        losses = {}
        if btype in ['sim', 'mint']:
            for k in ['smp', 'mar', 'joi']:
                losses.update({f'L_jacc_{k}_{btype}': F.l1_loss(batch['joi'][..., :23, 6:].flatten(-2), output[f'jacc_{k}'])})
        elif btype in ['mia']:
            for k in ['mar', 'joi']:
                losses.update({f'L_jacc_{k}_{btype}': F.l1_loss(batch['joi'][..., :23, 6:].flatten(-2), output[f'jacc_{k}'])})
        return losses
