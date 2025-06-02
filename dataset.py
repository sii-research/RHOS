from collections import OrderedDict, defaultdict
import pickle
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import torch.nn.functional as F
import joblib
import tqdm

from utils.geometry import *

def normalize(data, mean, std):
    epsilon = 1e-8
    mask    = std.abs() < epsilon
    normed  = (data - mean) / std
    normed[..., mask] = 0.0
    return normed

def denormalize(data, mean, std):
    return data * std + mean

class contrastDataset(Dataset):
    def __init__(self, config: OmegaConf, split: str = 'train'):
        super().__init__()
        self.CONFIG  = config
        self.SPLIT   = split
        self.dpath   = config.dpath
        self.sample  = config.get('sample_type', 0) # 0 for random, 1 for consequtive, 2 for all
        self.nfsel   = config.get('nfsel', 100)
        self.nmsel   = config.get('nmsel', [30, 60])
        self.world   = config.get('world', False)
        self.cand_  = [item for item in joblib.load(config.cand) if item[1] >= self.nfsel]
        self.amassidx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 37]
        self.split_cand = defaultdict(list)
        for i, item in enumerate(self.cand_):
            self.split_cand[item[0].split('/')[0]].append(i)
        self.balanced = config.get('balanced', False)
        print(len(self.cand_))
        if self.balanced and split in ['train']:
            if config.get('balance_cnt', False):
                self.balance_cnt = config.balance_cnt
            else:
                self.balance_cnt = 999999999
                for k, v in self.split_cand.items():
                    self.balance_cnt = min(self.balance_cnt, len(v))
        else:
            self.cand = np.arange(len(self.cand_)).tolist()
        self.update_msel()

    def __len__(self):
        return len(self.cand)
    
    def update_msel(self):
        self.msel = np.random.randint(self.nmsel[0], self.nmsel[1])
        if self.balanced:
            self.cand = []
            for v in self.split_cand.values():
                self.cand += np.random.choice(v, self.balance_cnt, True).tolist()
    
    def get_amass(self, item, fid):
        # joblib.dump({
        #     'pose': poses[idx].numpy(),
        #     'pvel': pvel[idx].numpy(),
        #     'pacc': pacc[idx].numpy(),
        #     'joint': joints[idx].numpy(),
        #     'jvel': jvel[idx].numpy(),
        #     'jacc': jacc[idx].numpy(),
        #     'mkr': mkrs[idx].numpy(),
        #     'mvel': mvel[idx].numpy(),
        #     'macc': macc[idx].numpy(),
        #     'beta': betas.numpy(),
        # }, f'{outdir}/{cnt}.pkl')
        data = {
            'src': 'smpl',
            'jpos': torch.from_numpy(item['joint'][fid]), # T, J, 3
            'jvel': torch.from_numpy(item['jvel'][fid]),  # T, J, 3
            'jacc': torch.from_numpy(item['jacc'][fid]),  # T, J, 3
        }
        if self.world:
            data.update({
                'qpos': torch.from_numpy(item['gqpos'][fid]),  # T, Q, 3
                'qvel': torch.from_numpy(item['gqvel'][fid]),  # T, Q, 3
                'qacc': torch.from_numpy(item['gqacc'][fid]),  # T, Q, 3
            })
        else:
            data.update({
                'qpos': torch.from_numpy(item['pose'][fid])[:, self.amassidx],  # T, Q, 3
                'qvel': torch.from_numpy(item['pvel'][fid])[:, self.amassidx],  # T, Q, 3
                'qacc': torch.from_numpy(item['pacc'][fid])[:, self.amassidx],  # T, Q, 3
            })
        nf   = len(fid)
        nsel = self.msel
        nmkr = item['mkr'].shape[1]
        if nmkr < nsel:
            data.update({
                'mpos': torch.cat((torch.from_numpy(item['mkr'][fid]),  torch.zeros(nf, nsel - nmkr, 3)), dim=1),   # T, M, 3
                'mvel': torch.cat((torch.from_numpy(item['mvel'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'macc': torch.cat((torch.from_numpy(item['macc'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'beta': torch.from_numpy(item['beta'])[None].expand(data['qpos'].shape[0], -1),
            })
        else:
            mkrsel = torch.randperm(nmkr).tolist()[:nsel]
            data.update({
                'mpos':  torch.from_numpy(item['mkr'][fid])[:, mkrsel],  # T, M, 3
                'mvel': torch.from_numpy(item['mvel'][fid])[:, mkrsel], # T, M, 3
                'macc': torch.from_numpy(item['macc'][fid])[:, mkrsel], # T, M, 3
                # 'beta': torch.from_numpy(item['beta'])[None].expand(data['qpos'].shape[0], -1),
            })
        # data['weight'] = torch.tensor([item['weight']] * data['mpos'].shape[0])
        return data
    
    def get_glink(self, item, fid):
        # joblib.dump({
        #     'pose': poses[idx].numpy(),
        #     'pvel': pvel[idx].numpy(),
        #     'pacc': pacc[idx].numpy(),
        #     'joint': joints[idx].numpy(),
        #     'jvel': jvel[idx].numpy(),
        #     'jacc': jacc[idx].numpy(),
        #     'mkr': mkrs[idx].numpy(),
        #     'mvel': mvel[idx].numpy(),
        #     'macc': macc[idx].numpy(),
        #     'beta': betas.numpy(),
        # }, f'{outdir}/{cnt}.pkl')
        data = {
            'src': 'glink',
            'jpos': torch.from_numpy(item['jpos'][fid]), # T, J, 3
            'jvel': torch.from_numpy(item['jvel'][fid]),  # T, J, 3
            'jacc': torch.from_numpy(item['jacc'][fid]),  # T, J, 3
            'grf': torch.from_numpy(item['grf'][fid]), # T, 2, 3
        }
        if self.world:
            data.update({
                'qpos': torch.from_numpy(item['gqpos'][fid]),  # T, Q, 3
                'qvel': torch.from_numpy(item['gqvel'][fid]),  # T, Q, 3
                'qacc': torch.from_numpy(item['gqacc'][fid]),  # T, Q, 3
            })
        else:
            data.update({
                'qpos': torch.from_numpy(item['qpos'][fid])[:, self.amassidx],  # T, Q, 3
                'qvel': torch.from_numpy(item['qvel'][fid])[:, self.amassidx],  # T, Q, 3
                'qacc': torch.from_numpy(item['qacc'][fid])[:, self.amassidx],  # T, Q, 3
            })
        nf   = len(fid)
        nsel = self.msel
        nmkr = item['mpos'].shape[1]
        if nmkr < nsel:
            data.update({
                'mpos': torch.cat((torch.from_numpy(item['mpos'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),   # T, M, 3
                'mvel': torch.cat((torch.from_numpy(item['mvel'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'macc': torch.cat((torch.from_numpy(item['macc'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'beta': torch.from_numpy(item['beta'])[None].expand(data['qpos'].shape[0], -1),
            })
        else:
            mkrsel = torch.randperm(nmkr).tolist()[:nsel]
            data.update({
                'mpos':  torch.from_numpy(item['mpos'][fid])[:, mkrsel],  # T, M, 3
                'mvel': torch.from_numpy(item['mvel'][fid])[:, mkrsel], # T, M, 3
                'macc': torch.from_numpy(item['macc'][fid])[:, mkrsel], # T, M, 3
                # 'beta': torch.from_numpy(item['beta'])[None].expand(data['qpos'].shape[0], -1),
            })
        # data['weight'] = torch.tensor([item['weight']] * data['mpos'].shape[0])
        return data
    
    def get_mint(self, item, fid):
        # joblib.dump({
        #     'pose': poses[idx].numpy(),
        #     'pvel': pvel[idx].numpy(),
        #     'pacc': pacc[idx].numpy(),
        #     'joint': joints[idx].numpy(),
        #     'jvel': jvel[idx].numpy(),
        #     'jacc': jacc[idx].numpy(),
        #     'mkr': mkrs[idx].numpy(),
        #     'mvel': mvel[idx].numpy(),
        #     'macc': macc[idx].numpy(),
        #     'beta': betas.numpy(),
        # }, f'{outdir}/{cnt}.pkl')
        data = {
            'src': 'mint',
            'jpos': torch.from_numpy(item['jpos'][fid]), # T, J, 3
            'jvel': torch.from_numpy(item['jvel'][fid]),  # T, J, 3
            'jacc': torch.from_numpy(item['jacc'][fid]),  # T, J, 3
            'mtau': torch.from_numpy(item['mtau'][fid]), # T, 402
        }
        if self.world:
            data.update({
                'qpos': torch.from_numpy(item['gqpos'][fid]),  # T, Q, 3
                'qvel': torch.from_numpy(item['gqvel'][fid]),  # T, Q, 3
                'qacc': torch.from_numpy(item['gqacc'][fid]),  # T, Q, 3
            })
        else:
            data.update({
                'qpos': torch.from_numpy(item['qpos'][fid])[:, self.amassidx],  # T, Q, 3
                'qvel': torch.from_numpy(item['qvel'][fid])[:, self.amassidx],  # T, Q, 3
                'qacc': torch.from_numpy(item['qacc'][fid])[:, self.amassidx],  # T, Q, 3
            })
        nf   = len(fid)
        nsel = self.msel
        nmkr = item['mpos'].shape[1]
        if nmkr < nsel:
            data.update({
                'mpos': torch.cat((torch.from_numpy(item['mpos'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),   # T, M, 3
                'mvel': torch.cat((torch.from_numpy(item['mvel'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'macc': torch.cat((torch.from_numpy(item['macc'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'beta': torch.from_numpy(item['beta'])[None].expand(data['qpos'].shape[0], -1),
            })
        else:
            mkrsel = torch.randperm(nmkr).tolist()[:nsel]
            data.update({
                'mpos':  torch.from_numpy(item['mpos'][fid])[:, mkrsel],  # T, M, 3
                'mvel': torch.from_numpy(item['mvel'][fid])[:, mkrsel], # T, M, 3
                'macc': torch.from_numpy(item['macc'][fid])[:, mkrsel], # T, M, 3
                # 'beta': torch.from_numpy(item['beta'])[None].expand(data['qpos'].shape[0], -1),
            })
        # data['weight'] = torch.tensor([item['weight']] * data['mpos'].shape[0])
        return data
    
    def get_mia(self, item, fid):
        # joblib.dump({
        #     'pose': poses[idx].numpy(),
        #     'pvel': pvel[idx].numpy(),
        #     'pacc': pacc[idx].numpy(),
        #     'joint': joints[idx].numpy(),
        #     'jvel': jvel[idx].numpy(),
        #     'jacc': jacc[idx].numpy(),
        #     'mkr': mkrs[idx].numpy(),
        #     'mvel': mvel[idx].numpy(),
        #     'macc': macc[idx].numpy(),
        #     'beta': betas.numpy(),
        # }, f'{outdir}/{cnt}.pkl')
        data = {
            'src': 'mia',
            'jpos': torch.from_numpy(item['jpos'][fid]), # T, J, 3
            'jvel': torch.from_numpy(item['jvel'][fid]),  # T, J, 3
            'jacc': torch.from_numpy(item['jacc'][fid]),  # T, J, 3
            'etau': torch.from_numpy(item['etau'][fid]), # T, 8
        }
        nf   = len(fid)
        nsel = self.msel
        nmkr = item['mpos'].shape[1]
        if nmkr < nsel:
            data.update({
                'mpos': torch.cat((torch.from_numpy(item['mpos'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),   # T, M, 3
                'mvel': torch.cat((torch.from_numpy(item['mvel'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'macc': torch.cat((torch.from_numpy(item['macc'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'beta': torch.from_numpy(item['beta'])[None].expand(data['qpos'].shape[0], -1),
            })
        else:
            mkrsel = torch.randperm(nmkr).tolist()[:nsel]
            data.update({
                'mpos':  torch.from_numpy(item['mpos'][fid])[:, mkrsel],  # T, M, 3
                'mvel': torch.from_numpy(item['mvel'][fid])[:, mkrsel], # T, M, 3
                'macc': torch.from_numpy(item['macc'][fid])[:, mkrsel], # T, M, 3
                # 'beta': torch.from_numpy(item['beta'])[None].expand(data['qpos'].shape[0], -1),
            })
        # data['weight'] = torch.tensor([item['weight']] * data['mpos'].shape[0])
        return data
        
    def get_imdy(self, item, fid):
        # joblib.dump({
        #     'qpos': qpos.numpy(),
        #     'qvel': qvel.numpy(),
        #     'qacc': qacc.numpy(),
        #     'jpos': jpos.numpy(),
        #     'jvel': jvel.numpy(),
        #     'jacc': jacc.numpy(),
        #     'mpos': mpos.numpy(),
        #     'mvel': mvel.numpy(),
        #     'macc': macc.numpy(),
        #     'torque': torque[idx], 
        #     'grf': grf[idx],
        # }, f'{outdir}/{idx}.pkl')
        data = {
            'src': 'sim', 
            'jpos': torch.from_numpy(item['jpos'][fid]),  # T, J, 3
            'jvel': torch.from_numpy(item['jvel'][fid]),  # T, J, 3
            'jacc': torch.from_numpy(item['jacc'][fid]),  # T, J, 3
            'stau': torch.from_numpy(item['torque'][fid]), # T, Q-6
            'slam': torch.from_numpy(item['grf'][fid]), # T, 6
        }
        if self.world:
            data.update({
                'qpos': torch.from_numpy(item['gqpos'][fid]),  # T, Q, 3
                'qvel': torch.from_numpy(item['gqvel'][fid]),  # T, Q, 3
                'qacc': torch.from_numpy(item['gqacc'][fid]),  # T, Q, 3
            })
        else:
            data.update({
                'qpos': torch.from_numpy(item['qpos'][fid]),  # T, Q, 3
                'qvel': torch.from_numpy(item['qvel'][fid]),  # T, Q, 3
                'qacc': torch.from_numpy(item['qacc'][fid]),  # T, Q, 3
            })
        nf   = len(fid)
        nsel = self.msel
        nmkr = item['mpos'].shape[1]
        if nmkr < nsel:
            data.update({
                'mpos': torch.cat((torch.from_numpy(item['mpos'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),   # T, M, 3
                'mvel': torch.cat((torch.from_numpy(item['mvel'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'macc': torch.cat((torch.from_numpy(item['macc'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
            })
        else:
            mkrsel = torch.randperm(nmkr).tolist()[:nsel]
            data.update({
                'mpos': torch.from_numpy(item['mpos'][fid])[:, mkrsel],  # T, M, 3
                'mvel': torch.from_numpy(item['mvel'][fid])[:, mkrsel], # T, M, 3
                'macc': torch.from_numpy(item['macc'][fid])[:, mkrsel], # T, M, 3
            })
        # data['weight'] = torch.tensor([item['weight']] * data['mpos'].shape[0])
        return data
    
    def get_adb(self, item, fid):
        # joblib.dump({
        #     'key': (key, int(idx[0]), int(idx[-1])),
        #     'torque': tor_,
        #     'grf': grf_,
        #     'qpos': pos_,
        #     'qvel': vel_,
        #     'qacc': acc_,
        #     'jpos': fps_conversion(jpos, 'joint', fps).permute(2, 0, 1)[..., [0, 2, 1]].numpy(), # J, 3, T -> T, J, 3
        #     'jvel': fps_conversion(jvel, 'joint', fps).permute(2, 0, 1)[..., [0, 2, 1]].numpy(),
        #     'jacc': fps_conversion(jacc, 'joint', fps).permute(2, 0, 1)[..., [0, 2, 1]].numpy(),
        #     'mpos': fps_conversion(mpos, 'marker', fps).permute(2, 0, 1)[..., [0, 2, 1]].numpy(),
        #     'mvel': fps_conversion(mvel, 'marker', fps).permute(2, 0, 1)[..., [0, 2, 1]].numpy(),
        #     'macc': fps_conversion(macc, 'marker', fps).permute(2, 0, 1)[..., [0, 2, 1]].numpy(),
        #     'weight': subj[key.split('/')[0]]['weight'],
        # }, f'{output_path}/{cnt}.pkl')
        data = {
            'blam': torch.from_numpy(item['grf'][fid]), # T, 6
        }
        data['src'] = 'real'
        data.update({
            'bpos': torch.from_numpy(item['qpos'][fid]),  # T, Q
            'bvel': torch.from_numpy(item['qvel'][fid]),  # T, Q
            'bacc': torch.from_numpy(item['qacc'][fid]),  # T, Q
            'jpos': torch.from_numpy(item['jpos'][fid]),  # T, J, 3
            'jvel': torch.from_numpy(item['jvel'][fid]),  # T, J, 3
            'jacc': torch.from_numpy(item['jacc'][fid]),  # T, J, 3
            'btau': torch.from_numpy(item['torque'][fid]).clamp(-500, 500), # T, Q-6
        })
        nf   = len(fid)
        nsel = self.msel
        nmkr = item['mpos'].shape[1]
        if nmkr < nsel:
            data.update({
                'mpos': torch.cat((torch.from_numpy(item['mpos'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),   # T, M, 3
                'mvel': torch.cat((torch.from_numpy(item['mvel'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
                'macc': torch.cat((torch.from_numpy(item['macc'][fid]), torch.zeros(nf, nsel - nmkr, 3)), dim=1),  # T, M, 3
            })
        else:
            mkrsel = torch.randperm(nmkr).tolist()[:nsel]
            data.update({
                'mpos': torch.from_numpy(item['mpos'][fid])[:, mkrsel],  # T, M, 3
                'mvel': torch.from_numpy(item['mvel'][fid])[:, mkrsel], # T, M, 3
                'macc': torch.from_numpy(item['macc'][fid])[:, mkrsel], # T, M, 3
            })
        # data['weight'] = torch.tensor([item['weight']] * data['mpos'].shape[0])
        return data
    
    def __getitem__(self, i):
        # 1. frame sampling
        # 2. marker sampling
        idx    = self.cand_[self.cand[i]]
        data   = joblib.load(os.path.join(self.dpath, idx[0]))
        if self.sample == 0:
            fid    = torch.randperm(idx[1])[:self.nfsel] 
        elif self.sample == 1:
            if idx[1] <= self.nfsel:
                fid = torch.arange(idx[1])
            else:
                sid = np.random.randint(idx[1] - self.nfsel)
                fid = torch.arange(sid, sid + self.nfsel)
        elif self.sample == 2:
            fid = torch.arange(idx[1])
        if 'amass' in idx[0]:
            return self.get_amass(data, fid)
        elif 'imdy' in idx[0]:
            return self.get_imdy(data, fid)
        elif 'ADB' in idx[0]:
            return self.get_adb(data, fid)
        elif 'mint' in idx[0]:
            return self.get_mint(data, fid)
        elif 'mia' in idx[0]:
            return self.get_mia(data, fid)
        elif 'glink' in idx[0]:
            return self.get_glink(data, fid)
        else:
            raise NotImplementedError

class contrastEvalDataset(contrastDataset):
    def __init__(self, config: OmegaConf, split: str = 'testeval'):
        super().__init__(config, split)
        if split == 'test':
            self.cand_  = [item for item in joblib.load(config.cand) if item[1] >= self.nfsel]
        else:
            self.cand_  = [item for item in joblib.load(config.cand)]
        self.cand = []
        for item in self.cand_:
            if item[1] < self.nfsel:
                continue
            else:
                for i in range(0, item[1] - self.nfsel + 1, self.nfsel):
                    self.cand.append([item, i])
    
    def __getitem__(self, i):
        idx, sid = self.cand[i]
        data = joblib.load(os.path.join(self.dpath, idx[0]))
        fid  = torch.arange(sid, sid + self.nfsel)
        if 'amass' in idx[0]:
            return self.get_amass(data, fid)
        elif 'imdy' in idx[0]:
            return self.get_imdy(data, fid)
        elif 'ADB' in idx[0]:
            return self.get_adb(data, fid)
        elif 'mint' in idx[0]:
            return self.get_mint(data, fid)
        elif 'mia' in idx[0]:
            return self.get_mia(data, fid)
        elif 'glink' in idx[0]:
            return self.get_glink(data, fid)
        else:
            raise NotImplementedError

def get_preprocess_fn(config, device):
    if config.get('MODE', 'raw') in ['raw']:
        ori_augmentation = config.get('ori_augmentation', False)
        pelvis_frame     = config.get('pelvis_frame', False)
        def preprocess_fn(batch):
            # preprocess: 
            # 1. random XOY orientation augmentation, optional
            # 2. prepare input
            batch_real, batch_sim, batch_smpl, batch_mia, batch_mint, batch_glink, batch_realarm, batch_phc = batch
            if len(batch_real) > 0 and batch_real['mpos'].shape[0] > 0:
                for key in batch_real.keys():
                    batch_real[key] = batch_real[key].to(device)
                mpos, mvel, macc = batch_real['mpos'], batch_real['mvel'], batch_real['macc']
                mpos[:, :, :2] -= batch_real['jpos'][:, :1, :2]
                jpos, jvel, jacc = batch_real['jpos'], batch_real['jvel'], batch_real['jacc']
                jpos[:, :, :2] -= batch_real['jpos'][:, :1, :2]
                bpos, bvel, bacc = batch_real['bpos'], batch_real['bvel'], batch_real['bacc']
                bpos[:, :2] -= batch_real['jpos'][:, 0, :2]
                blam, btau = batch_real['blam'], batch_real['btau']
                if ori_augmentation:
                    mangle = 2 * np.pi * np.random.random()
                    maugaa = torch.tensor([[[0, 0, mangle]]], device=device) # 1, 1, 3
                    mpos   = rot_apply(maugaa.expand(mpos.shape[0], mpos.shape[1], -1), mpos, 'aa')
                    mvel   = rot_apply(maugaa.expand(mvel.shape[0], mvel.shape[1], -1), mvel, 'aa')
                    macc   = rot_apply(maugaa.expand(macc.shape[0], macc.shape[1], -1), macc, 'aa')
                    
                    jangle = 2 * np.pi * np.random.random()
                    jaugaa = torch.tensor([[[0, 0, jangle]]], device=device) # 1, 1, 3
                    jpos   = rot_apply(jaugaa.expand(jpos.shape[0], jpos.shape[1], -1), jpos, 'aa')
                    jvel   = rot_apply(jaugaa.expand(jvel.shape[0], jvel.shape[1], -1), jvel, 'aa')
                    jacc   = rot_apply(jaugaa.expand(jacc.shape[0], jacc.shape[1], -1), jacc, 'aa')
                    ## TODO 1 placeholder
                input_real = {
                    'src': 'real',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'bio': torch.cat((bpos, bvel, bacc), dim=-1).float(),
                    'btau': btau.float(),
                    'blam': btau.float(),
                }
                if pelvis_frame:
                    input_real['mar'][..., [0, 1, 3, 4, 6, 7]]   -= input_real['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_real['bio'][:, [0, 1, 23, 24, 46, 47]] -= input_real['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_real['joi'][..., [0, 1, 3, 4, 6, 7]]   -= input_real['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_real = None
            
            if len(batch_sim) > 0 and batch_sim['mpos'].shape[0] > 0:
                for key in batch_sim.keys():
                    batch_sim[key]  = batch_sim[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_sim['mpos'], batch_sim['mvel'], batch_sim['macc']
                mpos[:, :, :2] -= batch_sim['jpos'][:, :1, :2]
                jpos, jvel, jacc = batch_sim['jpos'], batch_sim['jvel'], batch_sim['jacc']
                jpos[:, :, :2] -= batch_sim['jpos'][:, :1, :2]
                qpos = torch.cat((batch_sim['qpos'], jpos[:, :1]), dim=1)
                qvel = torch.cat((batch_sim['qvel'], jvel[:, :1]), dim=1)
                qacc = torch.cat((batch_sim['qacc'], jacc[:, :1]), dim=1)
                if ori_augmentation:
                    mangle = 2 * np.pi * np.random.random()
                    maugaa = torch.tensor([[[0, 0, mangle]]], device=device) # 1, 1, 3
                    mpos   = rot_apply(maugaa.expand(mpos.shape[0], mpos.shape[1], -1), mpos, 'aa')
                    mvel   = rot_apply(maugaa.expand(mvel.shape[0], mvel.shape[1], -1), mvel, 'aa')
                    macc   = rot_apply(maugaa.expand(macc.shape[0], macc.shape[1], -1), macc, 'aa')
                    
                    qangle = 2 * np.pi * np.random.random()
                    qaugaa = torch.tensor([[[0, 0, qangle]]], device=device)
                    qpos[:, :1]  = rot_apply(qaugaa.expand(qpos.shape[0], -1, -1), qpos[:, :1], 'aa')
                    qpos[:, -1:] = rot_apply(qaugaa.expand(qpos.shape[0], -1, -1), qpos[:, -1:], 'aa')
                    qvel[:, :1]  = rot_apply(qaugaa.expand(qvel.shape[0], -1, -1), qvel[:, :1], 'aa')
                    qvel[:, -1:] = rot_apply(qaugaa.expand(qvel.shape[0], -1, -1), qvel[:, -1:], 'aa')
                    qacc[:, :1]  = rot_apply(qaugaa.expand(qacc.shape[0], -1, -1), qacc[:, :1], 'aa')
                    qacc[:, -1:] = rot_apply(qaugaa.expand(qacc.shape[0], -1, -1), qacc[:, -1:], 'aa')
                    
                    jangle = 2 * np.pi * np.random.random()
                    jaugaa = torch.tensor([[[0, 0, jangle]]], device=device) # 1, 1, 3
                    jpos   = rot_apply(jaugaa.expand(jpos.shape[0], jpos.shape[1], -1), jpos, 'aa')
                    jvel   = rot_apply(jaugaa.expand(jvel.shape[0], jvel.shape[1], -1), jvel, 'aa')
                    jacc   = rot_apply(jaugaa.expand(jacc.shape[0], jacc.shape[1], -1), jacc, 'aa')
                
                input_sim = {
                    'src': 'sim',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'smp': torch.cat((qpos, qvel, qacc), dim=-1).float(),
                    'stau': batch_sim['stau'].flatten(1).float(),
                    'slam': batch_sim['slam'].flatten(1).float(),
                }
                if pelvis_frame:
                    input_sim['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_sim['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_sim['smp'][:, 0, [0, 1, 3, 4, 6, 7]] -= input_sim['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_sim['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_sim['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_sim = None
            
            if len(batch_smpl) > 0 and batch_smpl['mpos'].shape[0] > 0:
                for key in batch_smpl.keys():
                    batch_smpl[key] = batch_smpl[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_smpl['mpos'], batch_smpl['mvel'], batch_smpl['macc']
                mpos[:, :, :2] -= batch_smpl['jpos'][:, :1, :2]
                jpos, jvel, jacc = batch_smpl['jpos'], batch_smpl['jvel'], batch_smpl['jacc']
                jpos[:, :, :2] -= batch_smpl['jpos'][:, :1, :2]
                qpos = torch.cat((batch_smpl['qpos'], jpos[:, :1]), dim=1)
                qvel = torch.cat((batch_smpl['qvel'], jvel[:, :1]), dim=1)
                qacc = torch.cat((batch_smpl['qacc'], jacc[:, :1]), dim=1)
                if ori_augmentation:
                    mangle = 2 * np.pi * np.random.random()
                    maugaa = torch.tensor([[[0, 0, mangle]]], device=device) # 1, 1, 3
                    mpos   = rot_apply(maugaa.expand(mpos.shape[0], mpos.shape[1], -1), mpos, 'aa')
                    mvel   = rot_apply(maugaa.expand(mvel.shape[0], mvel.shape[1], -1), mvel, 'aa')
                    macc   = rot_apply(maugaa.expand(macc.shape[0], macc.shape[1], -1), macc, 'aa')
                    
                    qangle = 2 * np.pi * np.random.random()
                    qaugaa = torch.tensor([[[0, 0, qangle]]], device=device)
                    qpos[:, :1]  = rot_apply(qaugaa.expand(qpos.shape[0], -1, -1), qpos[:, :1], 'aa')
                    qpos[:, -1:] = rot_apply(qaugaa.expand(qpos.shape[0], -1, -1), qpos[:, -1:], 'aa')
                    qvel[:, :1]  = rot_apply(qaugaa.expand(qvel.shape[0], -1, -1), qvel[:, :1], 'aa')
                    qvel[:, -1:] = rot_apply(qaugaa.expand(qvel.shape[0], -1, -1), qvel[:, -1:], 'aa')
                    qacc[:, :1]  = rot_apply(qaugaa.expand(qacc.shape[0], -1, -1), qacc[:, :1], 'aa')
                    qacc[:, -1:] = rot_apply(qaugaa.expand(qacc.shape[0], -1, -1), qacc[:, -1:], 'aa')
                    
                    jangle = 2 * np.pi * np.random.random()
                    jaugaa = torch.tensor([[[0, 0, jangle]]], device=device) # 1, 1, 3
                    jpos   = rot_apply(jaugaa.expand(jpos.shape[0], jpos.shape[1], -1), jpos, 'aa')
                    jvel   = rot_apply(jaugaa.expand(jvel.shape[0], jvel.shape[1], -1), jvel, 'aa')
                    jacc   = rot_apply(jaugaa.expand(jacc.shape[0], jacc.shape[1], -1), jacc, 'aa')
                input_smpl = {
                    'src': 'smpl',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'smp': torch.cat((qpos, qvel, qacc), dim=-1).float(),
                    # 'stau': batch_smpl['stau'],
                    # 'slam': batch_smpl['slam'],
                }
                if pelvis_frame:
                    input_smpl['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_smpl['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_smpl['smp'][:, 0, [0, 1, 3, 4, 6, 7]] -= input_smpl['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_smpl['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_smpl['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_smpl = None
            
            if len(batch_mint) > 0 and batch_mint['mpos'].shape[0] > 0:
                for key in batch_mint.keys():
                    batch_mint[key] = batch_mint[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_mint['mpos'], batch_mint['mvel'], batch_mint['macc']
                mpos[:, :, :2]  -= batch_mint['jpos'][:, :1, :2]
                jpos, jvel, jacc = batch_mint['jpos'], batch_mint['jvel'], batch_mint['jacc']
                jpos[:, :, :2]  -= batch_mint['jpos'][:, :1, :2]
                qpos = torch.cat((batch_mint['qpos'], jpos[:, :1]), dim=1)
                qvel = torch.cat((batch_mint['qvel'], jvel[:, :1]), dim=1)
                qacc = torch.cat((batch_mint['qacc'], jacc[:, :1]), dim=1)
                if ori_augmentation:
                    mangle = 2 * np.pi * np.random.random()
                    maugaa = torch.tensor([[[0, 0, mangle]]], device=device) # 1, 1, 3
                    mpos   = rot_apply(maugaa.expand(mpos.shape[0], mpos.shape[1], -1), mpos, 'aa')
                    mvel   = rot_apply(maugaa.expand(mvel.shape[0], mvel.shape[1], -1), mvel, 'aa')
                    macc   = rot_apply(maugaa.expand(macc.shape[0], macc.shape[1], -1), macc, 'aa')
                    
                    qangle = 2 * np.pi * np.random.random()
                    qaugaa = torch.tensor([[[0, 0, qangle]]], device=device)
                    qpos[:, :1]  = rot_apply(qaugaa.expand(qpos.shape[0], -1, -1), qpos[:, :1], 'aa')
                    qpos[:, -1:] = rot_apply(qaugaa.expand(qpos.shape[0], -1, -1), qpos[:, -1:], 'aa')
                    qvel[:, :1]  = rot_apply(qaugaa.expand(qvel.shape[0], -1, -1), qvel[:, :1], 'aa')
                    qvel[:, -1:] = rot_apply(qaugaa.expand(qvel.shape[0], -1, -1), qvel[:, -1:], 'aa')
                    qacc[:, :1]  = rot_apply(qaugaa.expand(qacc.shape[0], -1, -1), qacc[:, :1], 'aa')
                    qacc[:, -1:] = rot_apply(qaugaa.expand(qacc.shape[0], -1, -1), qacc[:, -1:], 'aa')
                    
                    jangle = 2 * np.pi * np.random.random()
                    jaugaa = torch.tensor([[[0, 0, jangle]]], device=device) # 1, 1, 3
                    jpos   = rot_apply(jaugaa.expand(jpos.shape[0], jpos.shape[1], -1), jpos, 'aa')
                    jvel   = rot_apply(jaugaa.expand(jvel.shape[0], jvel.shape[1], -1), jvel, 'aa')
                    jacc   = rot_apply(jaugaa.expand(jacc.shape[0], jacc.shape[1], -1), jacc, 'aa')
                input_mint = {
                    'src': 'mint',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'smp': torch.cat((qpos, qvel, qacc), dim=-1).float(),
                    'mtau': batch_mint['mtau'].float(),
                }
                if pelvis_frame:
                    input_mint['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_mint['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_mint['smp'][:, 0, [0, 1, 3, 4, 6, 7]] -= input_mint['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_mint['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_mint['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_mint = None
            
            if len(batch_mia) > 0 and batch_mia['mpos'].shape[0] > 0:
                for key in batch_mia.keys():
                    batch_mia[key] = batch_mia[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_mia['mpos'], batch_mia['mvel'], batch_mia['macc']
                mpos[:, :, :2]  -= batch_mia['jpos'][:, :1, :2]
                jpos, jvel, jacc = batch_mia['jpos'], batch_mia['jvel'], batch_mia['jacc']
                jpos[:, :, :2]  -= batch_mia['jpos'][:, :1, :2]
                if ori_augmentation:
                    mangle = 2 * np.pi * np.random.random()
                    maugaa = torch.tensor([[[0, 0, mangle]]], device=device) # 1, 1, 3
                    mpos   = rot_apply(maugaa.expand(mpos.shape[0], mpos.shape[1], -1), mpos, 'aa')
                    mvel   = rot_apply(maugaa.expand(mvel.shape[0], mvel.shape[1], -1), mvel, 'aa')
                    macc   = rot_apply(maugaa.expand(macc.shape[0], macc.shape[1], -1), macc, 'aa')
                    
                    jangle = 2 * np.pi * np.random.random()
                    jaugaa = torch.tensor([[[0, 0, jangle]]], device=device) # 1, 1, 3
                    jpos   = rot_apply(jaugaa.expand(jpos.shape[0], jpos.shape[1], -1), jpos, 'aa')
                    jvel   = rot_apply(jaugaa.expand(jvel.shape[0], jvel.shape[1], -1), jvel, 'aa')
                    jacc   = rot_apply(jaugaa.expand(jacc.shape[0], jacc.shape[1], -1), jacc, 'aa')
                input_mia = {
                    'src': 'mia',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'etau': batch_mia['etau'].float(),
                }
                if pelvis_frame:
                    input_mia['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_mia['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_mia['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_mia['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_mia = None
            
            if len(batch_glink) > 0 and batch_glink['mpos'].shape[0] > 0:
                for key in batch_glink.keys():
                    batch_glink[key] = batch_glink[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_glink['mpos'], batch_glink['mvel'], batch_glink['macc']
                mpos[:, :, :2]  -= batch_glink['jpos'][:, :1, :2]
                jpos, jvel, jacc = batch_glink['jpos'], batch_glink['jvel'], batch_glink['jacc']
                jpos[:, :, :2]  -= batch_glink['jpos'][:, :1, :2]
                qpos = torch.cat((batch_glink['qpos'], jpos[:, :1]), dim=1)
                qvel = torch.cat((batch_glink['qvel'], jvel[:, :1]), dim=1)
                qacc = torch.cat((batch_glink['qacc'], jacc[:, :1]), dim=1)
                if ori_augmentation:
                    mangle = 2 * np.pi * np.random.random()
                    maugaa = torch.tensor([[[0, 0, mangle]]], device=device) # 1, 1, 3
                    mpos   = rot_apply(maugaa.expand(mpos.shape[0], mpos.shape[1], -1), mpos, 'aa')
                    mvel   = rot_apply(maugaa.expand(mvel.shape[0], mvel.shape[1], -1), mvel, 'aa')
                    macc   = rot_apply(maugaa.expand(macc.shape[0], macc.shape[1], -1), macc, 'aa')
                    
                    qangle = 2 * np.pi * np.random.random()
                    qaugaa = torch.tensor([[[0, 0, qangle]]], device=device)
                    qpos[:, :1]  = rot_apply(qaugaa.expand(qpos.shape[0], -1, -1), qpos[:, :1], 'aa')
                    qpos[:, -1:] = rot_apply(qaugaa.expand(qpos.shape[0], -1, -1), qpos[:, -1:], 'aa')
                    qvel[:, :1]  = rot_apply(qaugaa.expand(qvel.shape[0], -1, -1), qvel[:, :1], 'aa')
                    qvel[:, -1:] = rot_apply(qaugaa.expand(qvel.shape[0], -1, -1), qvel[:, -1:], 'aa')
                    qacc[:, :1]  = rot_apply(qaugaa.expand(qacc.shape[0], -1, -1), qacc[:, :1], 'aa')
                    qacc[:, -1:] = rot_apply(qaugaa.expand(qacc.shape[0], -1, -1), qacc[:, -1:], 'aa')
                    
                    jangle = 2 * np.pi * np.random.random()
                    jaugaa = torch.tensor([[[0, 0, jangle]]], device=device) # 1, 1, 3
                    jpos   = rot_apply(jaugaa.expand(jpos.shape[0], jpos.shape[1], -1), jpos, 'aa')
                    jvel   = rot_apply(jaugaa.expand(jvel.shape[0], jvel.shape[1], -1), jvel, 'aa')
                    jacc   = rot_apply(jaugaa.expand(jacc.shape[0], jacc.shape[1], -1), jacc, 'aa')
                input_glink = {
                    'src': 'glink',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'smp': torch.cat((qpos, qvel, qacc), dim=-1).float(),
                    'grf': batch_glink['grf'].float().flatten(1),
                }
                if pelvis_frame:
                    input_glink['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_glink['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_glink['smp'][:, 0, [0, 1, 3, 4, 6, 7]] -= input_glink['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_glink['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_glink['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_glink = None
            
            return input_real, input_sim, input_smpl, input_mint, input_mia, input_glink
        return preprocess_fn
    elif config.get('MODE', 'raw') in ['seq']:
        ori_augmentation = config.get('ori_augmentation', False)
        pelvis_frame     = config.get('pelvis_frame', False)
        def preprocess_fn(batch):
            # preprocess: 
            # 1. random XOY orientation augmentation, optional
            # 2. prepare input
            batch_real, batch_sim, batch_smpl, batch_mia, batch_mint, batch_glink = batch
            if len(batch_real) > 0 and batch_real['mpos'].shape[0] > 0:
                for key in batch_real.keys():
                    batch_real[key] = batch_real[key].to(device)
                mpos, mvel, macc = batch_real['mpos'], batch_real['mvel'], batch_real['macc']
                mpos[:, :, :, :2] -= batch_real['jpos'][:, :, :1, :2]
                jpos, jvel, jacc = batch_real['jpos'], batch_real['jvel'], batch_real['jacc']
                jpos[:, :, :, :2] -= batch_real['jpos'][:, :, :1, :2]
                bpos, bvel, bacc = batch_real['bpos'], batch_real['bvel'], batch_real['bacc']
                bpos[:, :, :2] -= batch_real['jpos'][:, :, 0, :2]
                blam, btau = batch_real['blam'], batch_real['btau']
                input_real = {
                    'src': 'real',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'bio': torch.cat((bpos, bvel, bacc), dim=-1).float(),
                    'btau': btau.float(),
                    # 'blam': btau.float(),
                }
                if pelvis_frame:
                    input_real['mar'][..., [0, 1, 3, 4, 6, 7]]   -= input_real['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_real['bio'][:, [0, 1, 23, 24, 46, 47]] -= input_real['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_real['joi'][..., [0, 1, 3, 4, 6, 7]]   -= input_real['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_real = None
            
            if len(batch_sim) > 0 and batch_sim['mpos'].shape[0] > 0:
                for key in batch_sim.keys():
                    batch_sim[key]  = batch_sim[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_sim['mpos'], batch_sim['mvel'], batch_sim['macc']
                mpos[:, :, :, :2] -= batch_sim['jpos'][:, :, :1, :2]
                jpos, jvel, jacc = batch_sim['jpos'], batch_sim['jvel'], batch_sim['jacc']
                jpos[:, :, :, :2] -= batch_sim['jpos'][:, :, :1, :2]
                qpos = torch.cat((batch_sim['qpos'], jpos[:, :, :1]), dim=2)
                qvel = torch.cat((batch_sim['qvel'], jvel[:, :, :1]), dim=2)
                qacc = torch.cat((batch_sim['qacc'], jacc[:, :, :1]), dim=2)
                
                input_sim = {
                    'src': 'sim',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'smp': torch.cat((qpos, qvel, qacc), dim=-1).float(),
                    'stau': batch_sim['stau'].flatten(2).float(),
                    # 'slam': batch_sim['slam'].flatten(2).float(),
                }
                if pelvis_frame:
                    input_sim['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_sim['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_sim['smp'][:, 0, [0, 1, 3, 4, 6, 7]] -= input_sim['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_sim['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_sim['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_sim = None
            
            if len(batch_smpl) > 0 and batch_smpl['mpos'].shape[0] > 0:
                for key in batch_smpl.keys():
                    batch_smpl[key] = batch_smpl[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_smpl['mpos'], batch_smpl['mvel'], batch_smpl['macc']
                mpos[:, :, :, :2] -= batch_smpl['jpos'][:, :, :1, :2]
                jpos, jvel, jacc = batch_smpl['jpos'], batch_smpl['jvel'], batch_smpl['jacc']
                jpos[:, :, :, :2] -= batch_smpl['jpos'][:, :, :1, :2]
                qpos = torch.cat((batch_smpl['qpos'], jpos[:, :, :1]), dim=2)
                qvel = torch.cat((batch_smpl['qvel'], jvel[:, :, :1]), dim=2)
                qacc = torch.cat((batch_smpl['qacc'], jacc[:, :, :1]), dim=2)
                input_smpl = {
                    'src': 'smpl',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'smp': torch.cat((qpos, qvel, qacc), dim=-1).float(),
                    # 'stau': batch_smpl['stau'],
                    # 'slam': batch_smpl['slam'],
                }
                if pelvis_frame:
                    input_smpl['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_smpl['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_smpl['smp'][:, 0, [0, 1, 3, 4, 6, 7]] -= input_smpl['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_smpl['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_smpl['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_smpl = None
            
            if len(batch_mint) > 0 and batch_mint['mpos'].shape[0] > 0:
                for key in batch_mint.keys():
                    batch_mint[key] = batch_mint[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_mint['mpos'], batch_mint['mvel'], batch_mint['macc']
                mpos[:, :, :, :2]  -= batch_mint['jpos'][:, :, :1, :2]
                jpos, jvel, jacc = batch_mint['jpos'], batch_mint['jvel'], batch_mint['jacc']
                jpos[:, :, :, :2]  -= batch_mint['jpos'][:, :, :1, :2]
                qpos = torch.cat((batch_mint['qpos'], jpos[:, :, :1]), dim=2)
                qvel = torch.cat((batch_mint['qvel'], jvel[:, :, :1]), dim=2)
                qacc = torch.cat((batch_mint['qacc'], jacc[:, :, :1]), dim=2)
                input_mint = {
                    'src': 'mint',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'smp': torch.cat((qpos, qvel, qacc), dim=-1).float(),
                    'mtau': batch_mint['mtau'].float(),
                }
                if pelvis_frame:
                    input_mint['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_mint['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_mint['smp'][:, 0, [0, 1, 3, 4, 6, 7]] -= input_mint['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_mint['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_mint['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_mint = None
            
            if len(batch_mia) > 0 and batch_mia['mpos'].shape[0] > 0:
                for key in batch_mia.keys():
                    batch_mia[key] = batch_mia[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_mia['mpos'], batch_mia['mvel'], batch_mia['macc']
                mpos[:, :, :, :2]  -= batch_mia['jpos'][:, :, :1, :2]
                jpos, jvel, jacc = batch_mia['jpos'], batch_mia['jvel'], batch_mia['jacc']
                jpos[:, :, :, :2]  -= batch_mia['jpos'][:, :, :1, :2]
                input_mia = {
                    'src': 'mia',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'etau': batch_mia['etau'].float(),
                }
                if pelvis_frame:
                    input_mia['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_mia['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_mia['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_mia['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_mia = None
            
            if len(batch_glink) > 0 and batch_glink['mpos'].shape[0] > 0:
                for key in batch_glink.keys():
                    batch_glink[key] = batch_glink[key].to(device)
                ## TODO 1 placeholder
                mpos, mvel, macc = batch_glink['mpos'], batch_glink['mvel'], batch_glink['macc']
                mpos[:, :, :, :2]  -= batch_glink['jpos'][:, :, :1, :2]
                jpos, jvel, jacc = batch_glink['jpos'], batch_glink['jvel'], batch_glink['jacc']
                jpos[:, :, :, :2]  -= batch_glink['jpos'][:, :, :1, :2]
                qpos = torch.cat((batch_glink['qpos'], jpos[:, :, :1]), dim=2)
                qvel = torch.cat((batch_glink['qvel'], jvel[:, :, :1]), dim=2)
                qacc = torch.cat((batch_glink['qacc'], jacc[:, :, :1]), dim=2)
                input_glink = {
                    'src': 'glink',
                    'mar': torch.cat((mpos, mvel, macc), dim=-1).float(),
                    'joi': torch.cat((jpos, jvel, jacc), dim=-1).float(),
                    'smp': torch.cat((qpos, qvel, qacc), dim=-1).float(),
                    'grf': batch_glink['grf'].flatten(2).float(),
                }
                if pelvis_frame:
                    input_glink['mar'][..., [0, 1, 3, 4, 6, 7]]  -= input_glink['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
                    input_glink['smp'][:, 0, [0, 1, 3, 4, 6, 7]] -= input_glink['joi'][:, 0, [0, 1, 3, 4, 6, 7]]
                    input_glink['joi'][..., [0, 1, 3, 4, 6, 7]]  -= input_glink['joi'][:, :1, [0, 1, 3, 4, 6, 7]]
            else:
                input_glink = None
            
            return input_real, input_sim, input_smpl, input_mint, input_mia, input_glink
        return preprocess_fn
        
    else: 
        def preprocess_fn(batch):
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            return batch
        return preprocess_fn

def get_collate_fn(config):
    if config.get('MODE', None) in ['raw']:
        def collate_fn(data):
            batch_smpl = defaultdict(list)
            batch_sim  = defaultdict(list)
            batch_real = defaultdict(list)
            batch_mia  = defaultdict(list)
            batch_mint = defaultdict(list)
            batch_glink = defaultdict(list)
            b = len(data)
            for i in range(b):
                if data[i]['src'] == 'real':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_real[key].append(data[i][key])
                if data[i]['src'] == 'sim':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_sim[key].append(data[i][key])
                if data[i]['src'] == 'smpl':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_smpl[key].append(data[i][key])
                if data[i]['src'] == 'mia':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_mia[key].append(data[i][key])
                if data[i]['src'] == 'mint':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_mint[key].append(data[i][key])
                if data[i]['src'] == 'glink':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_glink[key].append(data[i][key])
            for key in batch_real.keys():
                batch_real[key] = torch.cat(batch_real[key])
            for key in batch_sim.keys():
                batch_sim[key]  = torch.cat(batch_sim[key])
            for key in batch_smpl.keys():
                batch_smpl[key] = torch.cat(batch_smpl[key])
            for key in batch_mia.keys():
                batch_mia[key] = torch.cat(batch_mia[key])
            for key in batch_mint.keys():
                batch_mint[key] = torch.cat(batch_mint[key])
            for key in batch_glink.keys():
                batch_glink[key] = torch.cat(batch_glink[key])
            return batch_real, batch_sim, batch_smpl, batch_mia, batch_mint, batch_glink
        return collate_fn
    elif config.get('MODE', None) in ['seq']:
        def collate_fn(data):
            batch_smpl = defaultdict(list)
            batch_sim  = defaultdict(list)
            batch_real = defaultdict(list)
            batch_mia  = defaultdict(list)
            batch_mint = defaultdict(list)
            batch_glink = defaultdict(list)
            b = len(data)
            for i in range(b):
                if data[i]['src'] == 'real':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_real[key].append(data[i][key][None])
                if data[i]['src'] == 'sim':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_sim[key].append(data[i][key][None])
                if data[i]['src'] == 'smpl':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_smpl[key].append(data[i][key][None])
                if data[i]['src'] == 'mia':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_mia[key].append(data[i][key][None])
                if data[i]['src'] == 'mint':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_mint[key].append(data[i][key][None])
                if data[i]['src'] == 'glink':
                    for key in data[i].keys():
                        if key != 'src':
                            batch_glink[key].append(data[i][key][None])
            for key in batch_real.keys():
                batch_real[key] = torch.cat(batch_real[key])
            for key in batch_sim.keys():
                batch_sim[key]  = torch.cat(batch_sim[key])
            for key in batch_smpl.keys():
                batch_smpl[key] = torch.cat(batch_smpl[key])
            for key in batch_mia.keys():
                batch_mia[key] = torch.cat(batch_mia[key])
            for key in batch_mint.keys():
                batch_mint[key] = torch.cat(batch_mint[key])
            for key in batch_glink.keys():
                batch_glink[key] = torch.cat(batch_glink[key])
            return batch_real, batch_sim, batch_smpl, batch_mia, batch_mint, batch_glink
        return collate_fn
    else:
        raise NotImplementedError
