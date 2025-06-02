import torch
import torch.nn as nn
import torch.nn.functional as F

def corrcoef(pred, target):
    
    pred_centred = pred-torch.unsqueeze(torch.mean(pred, 1), 1)
    target_centred = target-torch.unsqueeze(torch.mean(target, 1), 1)
    
    pred_std = torch.unsqueeze(torch.sqrt(torch.mean(pred_centred**2, 1)), 1)
    target_std = torch.unsqueeze(torch.sqrt(torch.mean(target_centred**2, 1)), 1)
    
    bottom = pred_std * target_std
    top    = pred_centred * target_centred
    corr_torch = top / bottom
    # corr_torch[bottom.expand(-1, corr_torch.shape[1], -1) == 0] = 1.

    return corr_torch.mean()

class metricLayer:

    def __init__(self, config):
        pass
        
    def L1_torque(self, batch, output, tauType):
        res = {}
        tmp, cnt = 0, 0
        if batch['src'] == 'real':
            for inType in ['mar', 'smp', 'bio', 'joi']:
                if f'{tauType}_{inType}' in output:
                    tmp += output[f'{tauType}_{inType}']
                    cnt += 1
                    err  = torch.abs(batch[tauType][..., :17] - output[f'{tauType}_{inType}'][..., :17])
                    res.update({
                        f'Err_{tauType}_{inType}': err.mean(),
                    })
            tmp /= cnt
            err  = torch.abs(batch[tauType][..., :17] - tmp[..., :17])
            res.update({
                f'Err_{tauType}_avg': err.mean(),
            })
        elif batch['src'] == 'realarm':
            for inType in ['mar', 'smp', 'bio', 'joi']:
                if f'{tauType}_{inType}' in output:
                    tmp += output[f'{tauType}_{inType}']
                    cnt += 1
                    err  = torch.abs(batch[tauType] - output[f'{tauType}_{inType}'])
                    res.update({
                        f'Err_arm{tauType}_{inType}': err.mean(),
                    })
            tmp /= cnt
            err  = torch.abs(batch[tauType] - tmp)
            res.update({
                f'Err_arm{tauType}_avg': err.mean(),
            })
        else:
            for inType in ['mar', 'smp', 'bio', 'joi']:
                if f'{tauType}_{inType}' in output:
                    tmp += output[f'{tauType}_{inType}']
                    cnt += 1
                    err    = torch.abs(batch[tauType] - output[f'{tauType}_{inType}'])
                    res.update({
                        f'Err_{tauType}_{inType}': err.mean(),
                    })
            tmp /= cnt
            err    = torch.abs(batch[tauType] - tmp)
            res.update({
                f'Err_{tauType}_avg': err.mean(),
            })
        return res
        
    def glink(self, batch, output, tauType):
        res = {}
        tmp, cnt = 0, 0
        for inType in ['mar', 'smp', 'bio', 'joi']:
            if f'{tauType}_{inType}' in output:
                tmp += output[f'{tauType}_{inType}']
                cnt += 1
                err_lf    = torch.linalg.vector_norm(batch[tauType][..., 3:] - output[f'{tauType}_{inType}'][..., 3:], dim=-1)
                err_rf    = torch.linalg.vector_norm(batch[tauType][..., :3] - output[f'{tauType}_{inType}'][..., :3], dim=-1)
                res.update({
                    f'lfErr_{tauType}_{inType}': torch.mean(err_lf),
                    f'rfErr_{tauType}_{inType}': torch.mean(err_rf),
                })
        tmp /= cnt
        err_lf    = torch.linalg.vector_norm(batch[tauType][..., 3:] - tmp[..., 3:], dim=-1)
        err_rf    = torch.linalg.vector_norm(batch[tauType][..., :3] - tmp[..., :3], dim=-1)
        res.update({
            f'lfErr_{tauType}_avg': err_lf.mean(),
            f'rfErr_{tauType}_avg': err_rf.mean(),
        })
        return res
        
    def imdy_mPJE(self, batch, output, tauType):
        res = {}
        tmp, cnt = 0, 0
        for inType in ['mar', 'smp', 'bio', 'joi']:
            if f'{tauType}_{inType}' in output:
                tmp += output[f'{tauType}_{inType}']
                cnt += 1
                err  = batch[tauType] - output[f'{tauType}_{inType}']
                err  = torch.linalg.vector_norm(err.view(-1, 24, 3), dim=-1)
                res.update({
                    f'mPJE_{tauType}_{inType}': err.mean(),
                })
        tmp /= cnt
        err    = batch[tauType] - tmp
        err  = torch.linalg.vector_norm(err.view(-1, 24, 3), dim=-1)
        res.update({
            f'mPJE_{tauType}_avg': err.mean(),
        })
        return res
        
    def RMSE(self, batch, output, tauType):
        res = {}
        tmp, cnt = 0, 0
        for inType in ['mar', 'smp', 'bio', 'joi']:
            if f'{tauType}_{inType}' in output:
                tmp += output[f'{tauType}_{inType}']
                cnt += 1
                err    = torch.sqrt(F.mse_loss(batch[tauType], output[f'{tauType}_{inType}']))
                res.update({
                    f'RMSE_{tauType}_{inType}': err.mean(),
                })
        tmp /= cnt
        err    = torch.sqrt(F.mse_loss(batch[tauType], tmp))
        res.update({
            f'RMSE_{tauType}_avg': err.mean(),
        })
        return res
        
    def musRMSE(self, batch, output, tauType):
        res = {}
        tmp, cnt = 0, 0
        for inType in ['mar', 'smp', 'bio', 'joi']:
            if f'{tauType}_{inType}' in output:
                tmp += output[f'{tauType}_{inType}']
                cnt += 1
                lower_err    = torch.sqrt(F.mse_loss(batch[tauType][..., -80:], output[f'{tauType}_{inType}'][..., -80:]))
                upper_err    = torch.sqrt(F.mse_loss(batch[tauType][..., :-80], output[f'{tauType}_{inType}'][..., :-80]))
                res.update({
                    f'upperRMSE_{tauType}_{inType}': upper_err.mean(),
                    f'lowerRMSE_{tauType}_{inType}': lower_err.mean(),
                })
        tmp /= cnt
        lower_err    = torch.sqrt(F.mse_loss(batch[tauType][..., -80:], tmp[..., -80:]))
        upper_err    = torch.sqrt(F.mse_loss(batch[tauType][..., :-80], tmp[..., :-80]))
        res.update({
            f'upperRMSE_{tauType}_avg': upper_err.mean(),
            f'lowerRMSE_{tauType}_avg': lower_err.mean(),
        })
        return res
        
    def musPCC(self, batch, output, tauType):
        res = {}
        tmp, cnt = 0, 0
        for inType in ['mar', 'smp', 'bio', 'joi']:
            if f'{tauType}_{inType}' in output:
                tmp += output[f'{tauType}_{inType}']
                cnt += 1
                upper_err    = corrcoef(batch[tauType][..., 80:], output[f'{tauType}_{inType}'][..., 80:])
                lower_err    = corrcoef(batch[tauType][..., :80], output[f'{tauType}_{inType}'][..., :80])
                res.update({
                    f'upperpcc_{tauType}_{inType}': upper_err.mean(),
                    f'lowerpcc_{tauType}_{inType}': lower_err.mean(),
                })
        tmp /= cnt
        upper_err    = corrcoef(batch[tauType][..., 80:], tmp[..., 80:])
        lower_err    = corrcoef(batch[tauType][..., :80], tmp[..., :80])
        res.update({
            f'upperpcc_{tauType}_avg': upper_err.mean(),
            f'lowerpcc_{tauType}_avg': lower_err.mean(),
        })
        return res
        
    def PCC(self, batch, output, tauType):
        res = {}
        tmp, cnt = 0, 0
        for inType in ['mar', 'smp', 'bio', 'joi']:
            if f'{tauType}_{inType}' in output:
                tmp += output[f'{tauType}_{inType}']
                cnt += 1
                pcc    = corrcoef(batch[tauType].flatten(0, -2), output[f'{tauType}_{inType}'].flatten(0, -2))
                res.update({
                    f'pcc_{tauType}_{inType}': pcc.mean(),
                })
        tmp /= cnt
        pcc  = corrcoef(batch[tauType].flatten(0, -2), tmp.flatten(0, -2))
        res.update({
            f'pcc_{tauType}_avg': pcc.mean(),
        })
        return res
    
    def L1_acc(self, batch, output, btype):
        losses = {}
        if btype in ['sim', 'mint']:
            for k in ['smp', 'mar', 'joi']:
                losses.update({f'L_sacc_{k}_{btype}': F.l1_loss(batch['smp'][..., 6:].flatten(2), output[f'sacc_{k}'])})
                losses.update({f'L_jacc_{k}_{btype}': F.l1_loss(batch['joi'][..., 6:].flatten(2), output[f'jacc_{k}'])})
        elif btype in ['real']:
            for k in ['mar', 'joi', 'bio']:
                losses.update({f'L_bacc_{k}_{btype}': F.l1_loss(batch['bio'][..., 46:], output[f'bacc_{k}'])})
        elif btype in ['smpl']:
            for k in ['sim', 'rea', 'mus', 'emg']:
                losses.update({f'L_sacc_{k}_{btype}': F.l1_loss(batch['smp'][..., 6:].flatten(2), output[f'sacc_{k}'])})
                losses.update({f'L_jacc_{k}_{btype}': F.l1_loss(batch['joi'][..., 6:].flatten(2), output[f'jacc_{k}'])})
        elif btype in ['mia']:
            for k in ['emg']:
                losses.update({f'L_jacc_{k}_{btype}': F.l1_loss(batch['joi'][..., 6:].flatten(2), output[f'jacc_{k}'])})
        return losses