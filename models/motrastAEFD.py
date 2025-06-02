import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .utils import *

class MoTrastAEFD(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config  = config
        
        self.mar_config  = config.mar_config # marker enc config, point transformer
        self.mar_inProj  = nn.Linear(9, self.config.dim)
        self.mar_query   = nn.Parameter(torch.randn(1, 1, self.config.dim).float())
        dec_layer        = nn.TransformerDecoderLayer(d_model=self.config.dim, nhead=self.mar_config.num_head, dim_feedforward=self.config.dim * 2)
        self.mar_enc     = nn.TransformerDecoder(dec_layer, num_layers=self.mar_config.num_layers, norm=norm_dict[self.mar_config.norm](self.mar_config.dim))
        self.marA_inProj = nn.Linear(6, self.config.dim)
        self.marA_query  = nn.Parameter(torch.randn(1, 1, self.config.dim).float())
        dec_layer        = nn.TransformerDecoderLayer(d_model=self.config.dim, nhead=self.mar_config.num_head, dim_feedforward=self.config.dim * 2)
        self.marA_enc    = nn.TransformerDecoder(dec_layer, num_layers=self.mar_config.num_layers, norm=norm_dict[self.mar_config.norm](self.mar_config.dim))
        
        self.joi_config  = config.joi_config # joint enc config, point trainsformer
        self.joi_inProj  = nn.Linear(9, self.config.dim)
        self.joi_query   = nn.Parameter(torch.randn(1, 1, self.config.dim).float())
        dec_layer        = nn.TransformerDecoderLayer(d_model=self.config.dim, nhead=self.joi_config.num_head, dim_feedforward=self.config.dim * 2)
        self.joi_enc     = nn.TransformerDecoder(dec_layer, num_layers=self.joi_config.num_layers, norm=norm_dict[self.joi_config.norm](self.joi_config.dim))
        self.joiA_inProj = nn.Linear(6, self.config.dim)
        self.joiA_query  = nn.Parameter(torch.randn(1, 1, self.config.dim).float())
        dec_layer        = nn.TransformerDecoderLayer(d_model=self.config.dim, nhead=self.joi_config.num_head, dim_feedforward=self.config.dim * 2)
        self.joiA_enc    = nn.TransformerDecoder(dec_layer, num_layers=self.joi_config.num_layers, norm=norm_dict[self.joi_config.norm](self.joi_config.dim))
        
        self.bio_config = config.bio_config # biomechanic enc config, simple mlp
        units  = self.bio_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.bio_config.acti]())
            layers.append(norm_dict[self.bio_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.config.dim))
        self.bio_enc = nn.Sequential(*layers)
        self.bioA_dim = units[0] * 2 // 3
        layers = []
        for i in range(1, len(units)):
            if i == 1:
                layers.append(nn.Linear(units[i - 1] * 2 // 3, units[i]))
            else:
                layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.bio_config.acti]())
            layers.append(norm_dict[self.bio_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.config.dim))
        self.bioA_enc = nn.Sequential(*layers)
        
        self.smp_config = config.smp_config # smpl(qpos) enc config, simple mlp
        units  = self.smp_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.smp_config.acti]())
            layers.append(norm_dict[self.smp_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.config.dim))
        self.smp_enc = nn.Sequential(*layers)
        layers = []
        for i in range(1, len(units)):
            if i == 1:
                layers.append(nn.Linear(units[i - 1] * 2 // 3, units[i]))
            else:
                layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.smp_config.acti]())
            layers.append(norm_dict[self.smp_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.config.dim))
        self.smpA_enc = nn.Sequential(*layers)
        
        self.sim_config = config.sim_config # sim_torque dec config, simple mlp
        units  = self.sim_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.sim_config.acti]())
            layers.append(norm_dict[self.sim_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.sim_config.outdim))
        self.sim_dec = nn.Sequential(*layers)
        
        self.rea_config = config.rea_config # real_torque dec config, simple mlp
        units  = self.rea_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.rea_config.acti]())
            layers.append(norm_dict[self.rea_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.rea_config.outdim))
        self.rea_dec = nn.Sequential(*layers)
        
        self.mus_config = config.mus_config # muscle_action dec config, simple mlp
        units  = self.mus_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.mus_config.acti]())
            layers.append(norm_dict[self.mus_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.mus_config.outdim))
        layers.append(nn.Sigmoid())
        self.mus_dec = nn.Sequential(*layers)
        
        self.emg_config = config.emg_config # emg dec config, simple mlp
        units  = self.emg_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.emg_config.acti]())
            layers.append(norm_dict[self.emg_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.emg_config.outdim))
        self.emg_dec = nn.Sequential(*layers)
        
        self.grf_config = config.get('grf_config', config.emg_config) # emg dec config, simple mlp
        units  = self.grf_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.grf_config.acti]())
            layers.append(norm_dict[self.grf_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], 6))
        self.grf_dec = nn.Sequential(*layers)
        
        self.smpang_config = config.smpang_config
        units = self.smpang_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.smpang_config.acti]())
            layers.append(norm_dict[self.smpang_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.smpang_config.outdim))
        self.smpang_dec = nn.Sequential(*layers)
        
        self.bioang_config = config.bioang_config
        units = self.bioang_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.bioang_config.acti]())
            layers.append(norm_dict[self.bioang_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.bioang_config.outdim))
        self.bioang_dec = nn.Sequential(*layers)
        
        self.joiang_config = config.joiang_config
        units = self.joiang_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.joiang_config.acti]())
            layers.append(norm_dict[self.joiang_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.joiang_config.outdim))
        self.joiang_dec = nn.Sequential(*layers)
        
        units  = self.sim_config.units
        layers = []
        layers.append(nn.Linear(self.sim_config.outdim, units[-1]))
        for i in range(1, len(units)):
            layers.append(acti_dict[self.sim_config.acti]())
            layers.append(norm_dict[self.sim_config.norm](units[len(units) - i]))
            layers.append(nn.Linear(units[len(units) - i], units[len(units) - i - 1]))
        self.sim_enc = nn.Sequential(*layers)
        
        units  = self.rea_config.units
        layers = []
        layers.append(nn.Linear(self.rea_config.outdim, units[-1]))
        for i in range(1, len(units)):
            layers.append(acti_dict[self.rea_config.acti]())
            layers.append(norm_dict[self.rea_config.norm](units[len(units) - i]))
            layers.append(nn.Linear(units[len(units) - i], units[len(units) - i - 1]))
        self.rea_enc = nn.Sequential(*layers)
        
        units  = self.mus_config.units
        layers = []
        layers.append(nn.Linear(self.mus_config.outdim, units[-1]))
        for i in range(1, len(units)):
            layers.append(acti_dict[self.mus_config.acti]())
            layers.append(norm_dict[self.mus_config.norm](units[len(units) - i]))
            layers.append(nn.Linear(units[len(units) - i], units[len(units) - i - 1]))
        self.mus_enc = nn.Sequential(*layers)
        
        units  = self.emg_config.units
        layers = []
        layers.append(nn.Linear(self.emg_config.outdim, units[-1]))
        for i in range(1, len(units)):
            layers.append(acti_dict[self.emg_config.acti]())
            layers.append(norm_dict[self.emg_config.norm](units[len(units) - i]))
            layers.append(nn.Linear(units[len(units) - i], units[len(units) - i - 1]))
        self.emg_enc = nn.Sequential(*layers)
        
        self.composer_config = config.composer_config
        units = self.composer_config.units
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            layers.append(acti_dict[self.composer_config.acti]())
            layers.append(norm_dict[self.composer_config.norm](units[i]))
        layers.append(nn.Linear(units[-1], self.composer_config.outdim))
        self.composer = nn.Sequential(*layers)

    def forward(self, batch, btype='smpl'):
        if btype == 'smpl':
            z_smp = self.smp_enc(batch['smp'].flatten(1))                        # B, C_smpl
            x_mar = self.mar_inProj(batch['mar']).permute(1, 0, 2)               # L, B, C_mark
            z_mar = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joi = self.joi_inProj(batch['joi']).permute(1, 0, 2)               # L, B, C_joi
            z_joi = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            output = {
                'z_smp': z_smp,
                'z_mar': z_mar,
                'z_joi': z_joi,
            }
        elif btype in ['sim']:
            z_smp = self.smp_enc(batch['smp'].flatten(1))          # B, C_smpl
            x_mar = self.mar_inProj(batch['mar']).permute(1, 0, 2) # L, B, C_mark
            z_mar = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joi = self.joi_inProj(batch['joi']).permute(1, 0, 2)                    # L, B, C_joi
            z_joi = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            nz    = z_smp.shape[0]
            z     = torch.cat((z_smp, z_mar, z_joi))
            z_smpA = self.smpA_enc(batch['smp'][..., :6].flatten(1))                        # B, C_smpl
            x_marA = self.marA_inProj(batch['mar'][..., :6]).permute(1, 0, 2)               # L, B, C_mark
            z_marA = self.marA_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joiA = self.joiA_inProj(batch['joi'][..., :6]).permute(1, 0, 2)               # L, B, C_joi
            z_joiA = self.joiA_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            z_A    = torch.cat((z_smpA, z_marA, z_joiA))
            
            stau  = self.sim_dec(z)
            z_sim  = self.sim_enc(batch['stau'].flatten(1))
            z_sim  = torch.tile(z_sim, (3, 1))
            z_D    = self.composer(torch.cat((z_A, z_sim), dim=-1))
            sacc   = self.smpang_dec(z_D)
            jacc   = self.joiang_dec(z_D)
            output = {
                'z_smp': z[:nz],
                'z_mar': z[nz:2 * nz],
                'z_joi': z[2 * nz:],
                'z_smpD': z_D[:nz],
                'z_marD': z_D[nz:2 * nz],
                'z_joiD': z_D[2 * nz:],
                'stau_smp': stau[:nz],
                'stau_mar': stau[nz:2 * nz],
                'stau_joi': stau[2 * nz:],
                'sacc_smp': sacc[:nz],
                'sacc_mar': sacc[nz:2 * nz],
                'sacc_joi': sacc[2 * nz:],
                'jacc_smp': jacc[:nz],
                'jacc_mar': jacc[nz:2 * nz],
                'jacc_joi': jacc[2 * nz:],
            }
        elif btype == 'real':
            z_bio = self.bio_enc(batch['bio'].flatten(1))          # B, C_smpl
            x_mar = self.mar_inProj(batch['mar']).permute(1, 0, 2) # L, B, C_mark
            z_mar = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joi = self.joi_inProj(batch['joi']).permute(1, 0, 2)                    # L, B, C_joi
            z_joi = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            nz    = z_bio.shape[0]
            z     = torch.cat((z_bio, z_mar, z_joi))
            btau  = self.rea_dec(z)
            z_bioA = self.bioA_enc(batch['bio'][..., :self.bioA_dim].flatten(1))            # B, C_smpl
            x_marA = self.marA_inProj(batch['mar'][..., :6]).permute(1, 0, 2)               # L, B, C_mark
            z_marA = self.marA_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joiA = self.joiA_inProj(batch['joi'][..., :6]).permute(1, 0, 2)               # L, B, C_joi
            z_joiA = self.joiA_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            z_A    = torch.cat((z_bioA, z_marA, z_joiA))
            z_rea  = self.rea_enc(batch['btau'].flatten(1))
            z_rea  = torch.tile(z_rea, (3, 1))
            z_D    = self.composer(torch.cat((z_A, z_rea), dim=-1))
            bacc   = self.bioang_dec(z_D)
            output = {
                'z_bio': z[:nz],
                'z_mar': z[nz:2 * nz],
                'z_joi': z[2 * nz:],
                'z_bioD': z_D[:nz],
                'z_marD': z_D[nz:2 * nz],
                'z_joiD': z_D[2 * nz:],
                'btau_bio': btau[:nz],
                'btau_mar': btau[nz:2 * nz],
                'btau_joi': btau[2 * nz:],
                'bacc_bio': bacc[:nz],
                'bacc_mar': bacc[nz:2 * nz],
                'bacc_joi': bacc[2 * nz:],
            }
        elif btype == 'mint':
            z_smp = self.smp_enc(batch['smp'].flatten(1))          # B, C_smpl
            x_mar = self.mar_inProj(batch['mar']).permute(1, 0, 2) # L, B, C_mark
            z_mar = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joi = self.joi_inProj(batch['joi']).permute(1, 0, 2)                    # L, B, C_joi
            z_joi = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            nz    = z_smp.shape[0]
            z     = torch.cat((z_smp, z_mar, z_joi))
            mtau  = self.mus_dec(z)
            z_smpA = self.smpA_enc(batch['smp'][..., :6].flatten(1))                        # B, C_smpl
            x_marA = self.marA_inProj(batch['mar'][..., :6]).permute(1, 0, 2)               # L, B, C_mark
            z_marA = self.marA_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joiA = self.joiA_inProj(batch['joi'][..., :6]).permute(1, 0, 2)               # L, B, C_joi
            z_joiA = self.joiA_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            z_mus  = self.mus_enc(batch['mtau'].flatten(1))
            z_mus  = torch.tile(z_mus, (3, 1))
            z_A    = torch.cat((z_smpA, z_marA, z_joiA))
            z_D    = self.composer((torch.cat((z_A, z_mus), dim=-1)))
            sacc   = self.smpang_dec(z_D)
            jacc   = self.joiang_dec(z_D)
            output = {
                'z_smp': z[:nz],
                'z_mar': z[nz:2 * nz],
                'z_joi': z[2 * nz:],
                'z_smpD': z_D[:nz],
                'z_marD': z_D[nz:2 * nz],
                'z_joiD': z_D[2 * nz:],
                'mtau_smp': mtau[:nz],
                'mtau_mar': mtau[nz:2 * nz],
                'mtau_joi': mtau[2 * nz:],
                'sacc_smp': sacc[:nz],
                'sacc_mar': sacc[nz:2 * nz],
                'sacc_joi': sacc[2 * nz:],
                'jacc_smp': jacc[:nz],
                'jacc_mar': jacc[nz:2 * nz],
                'jacc_joi': jacc[2 * nz:],
            }
        elif btype == 'mia':
            x_mar = self.mar_inProj(batch['mar']).permute(1, 0, 2) # L, B, C_mark
            z_mar = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joi = self.joi_inProj(batch['joi']).permute(1, 0, 2)                    # L, B, C_joi
            z_joi = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            etau_mar = self.emg_dec(z_mar)
            etau_joi = self.emg_dec(z_joi)
            x_marA = self.marA_inProj(batch['mar'][..., :6]).permute(1, 0, 2)               # L, B, C_mark
            z_marA = self.marA_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joiA = self.joiA_inProj(batch['joi'][..., :6]).permute(1, 0, 2)               # L, B, C_joi
            z_joiA = self.joiA_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            z_emg  = self.emg_enc(batch['etau'].flatten(1))
            z_marD = self.composer(torch.cat((z_marA, z_emg), dim=-1))
            z_joiD = self.composer(torch.cat((z_joiA, z_emg), dim=-1))
            jacc_mar = self.joiang_dec(z_marD)
            jacc_joi = self.joiang_dec(z_joiD)
            output = {
                'z_mar': z_mar,
                'z_joi': z_joi,
                'z_marD': z_marD,
                'z_joiD': z_joiD,
                'etau_mar': etau_mar,
                'etau_joi': etau_joi,
                'jacc_mar': jacc_mar,
                'jacc_joi': jacc_joi,
            }
        elif btype == 'glink':
            z_smp = self.smp_enc(batch['smp'].flatten(1))          # B, C_smpl
            x_mar = self.mar_inProj(batch['mar']).permute(1, 0, 2) # L, B, C_mark
            z_mar = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # 1, B, C_mark
            x_joi = self.joi_inProj(batch['joi']).permute(1, 0, 2)                    # L, B, C_joi
            z_joi = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # 1, B, C_joi
            nz    = z_smp.shape[0]
            z     = torch.cat((z_smp, z_mar, z_joi))
            grf   = self.grf_dec(z)
            output = {
                'z_smp': z[:nz],
                'z_mar': z[nz:2 * nz],
                'z_joi': z[2 * nz:],
                'grf_smp': grf[:nz],
                'grf_mar': grf[nz:2 * nz],
                'grf_joi': grf[2 * nz:],
            }
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        #pdb.set_trace()
        return self.dropout(token_embedding + self.pos_encoding[:, :token_embedding.shape[1]])
        
class MoTrastAEFD_Large(MoTrastAEFD):
    def __init__(self, config):
        
        super().__init__(config)
        
        self.positional_encoder = PositionalEncoding(
            dim_model=self.config.dim, dropout_p=0.1, max_len=30,
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.dim, nhead=self.config.num_head, dim_feedforward=self.config.dim * 4)
        self.temp_composer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        
    def forward(self, batch, btype='smpl'):
        if btype == 'smpl':
            z_smp_ = self.smp_enc(batch['smp'].flatten(2)) # B, L, C
            z_smp = self.positional_encoder(z_smp_)        # B, L, C
            z_smp = z_smp.permute(1, 0, 2)                 # L, B, C
            z_smp = self.temp_composer(z_smp)              # L, B, C
            stau_smp = self.sim_dec(z_smp)                 # L, B, stau
            stau_smp = stau_smp.permute(1, 0, 2)           # B, L, stau
            
            x_joi  = self.joi_inProj(batch['joi'])         # N, L, J, C
            N, L, J, C = x_joi.shape
            x_joi  = torch.cat(torch.split(x_joi, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_joi_ = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # NL, C
            z_joi_ = torch.stack(torch.split(z_joi_, L, 0))
            z_joi  = self.positional_encoder(z_joi_)  # B, L, C
            z_joi  = z_joi.permute(1, 0, 2)           # L, B, C
            z_joi  = self.temp_composer(z_joi)        # L, B, C
            stau_joi = self.sim_dec(z_joi) # L, B, stau
            stau_joi = stau_joi.permute(1, 0, 2)     # B, L, stau
            
            x_mar  = self.mar_inProj(batch['mar'])         # N, L, J, C
            N, L, J, C = x_mar.shape
            x_mar  = torch.cat(torch.split(x_mar, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_mar_ = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # NL, C
            z_mar_ = torch.stack(torch.split(z_mar_, L, 0))
            z_mar  = self.positional_encoder(z_mar_)  # B, L, C
            z_mar  = z_mar.permute(1, 0, 2)           # L, B, C
            z_mar  = self.temp_composer(z_mar)        # L, B, C
            stau_mar = self.sim_dec(z_mar) # L, B, stau
            stau_mar = stau_mar.permute(1, 0, 2)     # B, L, stau
            
            output = {
                'z_smp': torch.cat(torch.split(z_smp_, 1, 0), dim=1)[0],
                'z_joi': torch.cat(torch.split(z_joi_, 1, 0), dim=1)[0],
                'z_mar': torch.cat(torch.split(z_mar_, 1, 0), dim=1)[0],
                'stau_smp': stau_smp,
                'stau_joi': stau_joi,
                'stau_mar': stau_mar,
            }
        elif btype == 'sim':
            z_smp_ = self.smp_enc(batch['smp'].flatten(2)) # B, L, C
            z_smp = self.positional_encoder(z_smp_)        # B, L, C
            z_smp = z_smp.permute(1, 0, 2)                 # L, B, C
            z_smp = self.temp_composer(z_smp)              # L, B, C
            stau_smp = self.sim_dec(z_smp)       # L, B, stau
            stau_smp = stau_smp.permute(1, 0, 2)           # B, L, stau
            
            x_joi  = self.joi_inProj(batch['joi'])         # N, L, J, C
            N, L, J, C = x_joi.shape
            x_joi  = torch.cat(torch.split(x_joi, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_joi_ = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # NL, C
            z_joi_ = torch.stack(torch.split(z_joi_, L, 0))
            z_joi  = self.positional_encoder(z_joi_)  # B, L, C
            z_joi  = z_joi.permute(1, 0, 2)           # L, B, C
            z_joi  = self.temp_composer(z_joi)        # L, B, C
            stau_joi = self.sim_dec(z_joi) # L, B, stau
            stau_joi = stau_joi.permute(1, 0, 2)     # B, L, stau
            
            x_mar  = self.mar_inProj(batch['mar'])         # N, L, J, C
            N, L, J, C = x_mar.shape
            x_mar  = torch.cat(torch.split(x_mar, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_mar_ = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # NL, C
            z_mar_ = torch.stack(torch.split(z_mar_, L, 0))
            z_mar  = self.positional_encoder(z_mar_)  # B, L, C
            z_mar  = z_mar.permute(1, 0, 2)           # L, B, C
            z_mar  = self.temp_composer(z_mar)        # L, B, C
            stau_mar = self.sim_dec(z_mar) # L, B, stau
            stau_mar = stau_mar.permute(1, 0, 2)     # B, L, stau
            
            nz    = z_smp_.shape[0]
            
            z_smpA = self.smpA_enc(batch['smp'][..., :6].flatten(2))                        # B, L, C_smpl
            
            x_marA = self.marA_inProj(batch['mar'][..., :6])                                # N, L, J, C
            x_marA = torch.cat(torch.split(x_marA, 1, 0), dim=1)[0].permute(1, 0, 2)         # J, BL, C
            z_marA = self.marA_enc(self.mar_query.expand(-1, x_marA.shape[1], -1), x_marA)[0] # 1, BL, C
            z_marA = torch.stack(torch.split(z_marA, L, 0))                                 # B, L, C
            
            x_joiA = self.joiA_inProj(batch['joi'][..., :6])                                # N, L, J, C
            x_joiA = torch.cat(torch.split(x_joiA, 1, 0), dim=1)[0].permute(1, 0, 2)         # J, BL, C
            z_joiA = self.joiA_enc(self.joi_query.expand(-1, x_joiA.shape[1], -1), x_joiA)[0] # 1, BL, C
            z_joiA = torch.stack(torch.split(z_joiA, L, 0))                                 # B, L, C
            
            z_A    = torch.cat((z_smpA, z_marA, z_joiA), dim=0)
            z_sim  = self.sim_enc(batch['stau'].flatten(2)) # B, L, C
            z_sim  = torch.tile(z_sim, (3, 1, 1))           # 3B, L, C    
            z_D    = self.composer(torch.cat((z_A, z_sim), dim=-1))
            sacc   = self.smpang_dec(z_D)
            jacc   = self.joiang_dec(z_D)
            
            output = {
                'z_smp': torch.cat(torch.split(z_smp_, 1, 0), dim=1)[0],
                'z_joi': torch.cat(torch.split(z_joi_, 1, 0), dim=1)[0],
                'z_mar': torch.cat(torch.split(z_mar_, 1, 0), dim=1)[0],
                'z_smpD': torch.cat(torch.split(z_D[:nz], 1, 0), dim=1)[0],
                'z_marD': torch.cat(torch.split(z_D[nz:2 * nz], 1, 0), dim=1)[0],
                'z_joiD': torch.cat(torch.split(z_D[2 * nz:], 1, 0), dim=1)[0],
                'stau_smp': stau_smp,
                'stau_joi': stau_joi,
                'stau_mar': stau_mar,
                'sacc_smp': sacc[:nz],
                'sacc_mar': sacc[nz:2 * nz],
                'sacc_joi': sacc[2 * nz:],
                'jacc_smp': jacc[:nz],
                'jacc_mar': jacc[nz:2 * nz],
                'jacc_joi': jacc[2 * nz:],
            }
        elif btype == 'real':
            z_bio_ = self.bio_enc(batch['bio'].flatten(2)) # B, C_smpl
            z_bio = self.positional_encoder(z_bio_)        # B, L, C
            z_bio = z_bio.permute(1, 0, 2)                 # L, B, C
            z_bio = self.temp_composer(z_bio)              # L, B, C
            btau_bio = self.rea_dec(z_bio)                 # L, B, stau
            btau_bio = btau_bio.permute(1, 0, 2)           # B, L, stau
            
            x_joi  = self.joi_inProj(batch['joi'])         # N, L, J, C
            N, L, J, C = x_joi.shape
            x_joi  = torch.cat(torch.split(x_joi, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_joi_ = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # NL, C
            z_joi_ = torch.stack(torch.split(z_joi_, L, 0))
            z_joi    = self.positional_encoder(z_joi_)  # B, L, C
            z_joi    = z_joi.permute(1, 0, 2)           # L, B, C
            z_joi    = self.temp_composer(z_joi)        # L, B, C
            btau_joi = self.rea_dec(z_joi)              # L, B, btau
            btau_joi = btau_joi.permute(1, 0, 2)        # B, L, btau
            
            x_mar  = self.mar_inProj(batch['mar'])         # N, L, J, C
            N, L, J, C = x_mar.shape
            x_mar  = torch.cat(torch.split(x_mar, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_mar_ = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # NL, C
            z_mar_ = torch.stack(torch.split(z_mar_, L, 0))
            z_mar  = self.positional_encoder(z_mar_)  # B, L, C
            z_mar  = z_mar.permute(1, 0, 2)           # L, B, C
            z_mar  = self.temp_composer(z_mar)        # L, B, C
            btau_mar = self.rea_dec(z_mar) # L, B, stau
            btau_mar = btau_mar.permute(1, 0, 2)     # B, L, stau
            
            nz    = z_bio_.shape[0]
            
            z_bioA = self.bioA_enc(batch['bio'][..., :46])                       # B, L, C_smpl
            
            x_marA = self.marA_inProj(batch['mar'][..., :6])                                # N, L, J, C
            x_marA = torch.cat(torch.split(x_marA, 1, 0), dim=1)[0].permute(1, 0, 2)         # J, BL, C
            z_marA = self.marA_enc(self.mar_query.expand(-1, x_marA.shape[1], -1), x_marA)[0] # 1, BL, C
            z_marA = torch.stack(torch.split(z_marA, L, 0))                                 # B, L, C
            
            x_joiA = self.joiA_inProj(batch['joi'][..., :6])                                # N, L, J, C
            x_joiA = torch.cat(torch.split(x_joiA, 1, 0), dim=1)[0].permute(1, 0, 2)         # J, BL, C
            z_joiA = self.joiA_enc(self.joi_query.expand(-1, x_joiA.shape[1], -1), x_joiA)[0] # 1, BL, C
            z_joiA = torch.stack(torch.split(z_joiA, L, 0))                                 # B, L, C
            
            z_A    = torch.cat((z_bioA, z_marA, z_joiA), dim=0)
            z_rea  = self.rea_enc(batch['btau']) # B, L, C
            z_rea  = torch.tile(z_rea, (3, 1, 1))           # 3B, L, C    
            z_D    = self.composer(torch.cat((z_A, z_rea), dim=-1))
            bacc   = self.bioang_dec(z_D)
            jacc   = self.joiang_dec(z_D)
            
            output = {
                'z_bio':  torch.cat(torch.split(z_bio_, 1, 0), dim=1)[0],
                'z_joi':  torch.cat(torch.split(z_joi_, 1, 0), dim=1)[0],
                'z_mar':  torch.cat(torch.split(z_mar_, 1, 0), dim=1)[0],
                'z_bioD': torch.cat(torch.split(z_D[:nz], 1, 0), dim=1)[0],
                'z_marD': torch.cat(torch.split(z_D[nz:2 * nz], 1, 0), dim=1)[0],
                'z_joiD': torch.cat(torch.split(z_D[2 * nz:], 1, 0), dim=1)[0],
                'btau_bio': btau_bio,
                'btau_joi': btau_joi,
                'btau_mar': btau_mar,
                'bacc_bio': bacc[:nz],
                'bacc_mar': bacc[nz:2 * nz],
                'bacc_joi': bacc[2 * nz:],
                'jacc_bio': jacc[:nz],
                'jacc_mar': jacc[nz:2 * nz],
                'jacc_joi': jacc[2 * nz:],
            }
            
        elif btype == 'mint':
            z_smp_ = self.smp_enc(batch['smp'].flatten(2)) # B, L, C
            z_smp = self.positional_encoder(z_smp_)        # B, L, C
            z_smp = z_smp.permute(1, 0, 2)                 # L, B, C
            z_smp = self.temp_composer(z_smp)              # L, B, C
            mtau_smp = self.mus_dec(z_smp)       # L, B, stau
            mtau_smp = mtau_smp.permute(1, 0, 2)           # B, L, stau
            
            x_joi  = self.joi_inProj(batch['joi'])         # N, L, J, C
            N, L, J, C = x_joi.shape
            x_joi  = torch.cat(torch.split(x_joi, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_joi_ = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # NL, C
            z_joi_ = torch.stack(torch.split(z_joi_, L, 0))
            z_joi  = self.positional_encoder(z_joi_)  # B, L, C
            z_joi  = z_joi.permute(1, 0, 2)           # L, B, C
            z_joi  = self.temp_composer(z_joi)        # L, B, C
            mtau_joi = self.mus_dec(z_joi) # L, B, stau
            mtau_joi = mtau_joi.permute(1, 0, 2)     # B, L, stau
            
            x_mar  = self.mar_inProj(batch['mar'])         # N, L, J, C
            N, L, J, C = x_mar.shape
            x_mar  = torch.cat(torch.split(x_mar, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_mar_ = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # NL, C
            z_mar_ = torch.stack(torch.split(z_mar_, L, 0))
            z_mar  = self.positional_encoder(z_mar_)  # B, L, C
            z_mar  = z_mar.permute(1, 0, 2)           # L, B, C
            z_mar  = self.temp_composer(z_mar)        # L, B, C
            mtau_mar = self.mus_dec(z_mar)            # L, B, stau
            mtau_mar = mtau_mar.permute(1, 0, 2)      # B, L, stau
            
            nz    = z_smp_.shape[0]
            
            z_smpA = self.smpA_enc(batch['smp'][..., :6].flatten(2))                        # B, L, C_smpl
            
            x_marA = self.marA_inProj(batch['mar'][..., :6])                                # N, L, J, C
            x_marA = torch.cat(torch.split(x_marA, 1, 0), dim=1)[0].permute(1, 0, 2)         # J, BL, C
            z_marA = self.marA_enc(self.mar_query.expand(-1, x_marA.shape[1], -1), x_marA)[0] # 1, BL, C
            z_marA = torch.stack(torch.split(z_marA, L, 0))                                 # B, L, C
            
            x_joiA = self.joiA_inProj(batch['joi'][..., :6])                                # N, L, J, C
            x_joiA = torch.cat(torch.split(x_joiA, 1, 0), dim=1)[0].permute(1, 0, 2)         # J, BL, C
            z_joiA = self.joiA_enc(self.joi_query.expand(-1, x_joiA.shape[1], -1), x_joiA)[0] # 1, BL, C
            z_joiA = torch.stack(torch.split(z_joiA, L, 0))                                 # B, L, C
            
            z_A    = torch.cat((z_smpA, z_marA, z_joiA), dim=0)
            z_mus  = self.mus_enc(batch['mtau'].flatten(2)) # B, L, C
            z_mus  = torch.tile(z_mus, (3, 1, 1))           # 3B, L, C    
            z_D    = self.composer(torch.cat((z_A, z_mus), dim=-1))
            sacc   = self.smpang_dec(z_D)
            jacc   = self.joiang_dec(z_D)
            
            output = {
                'z_smp':  torch.cat(torch.split(z_smp_, 1, 0), dim=1)[0],
                'z_joi':  torch.cat(torch.split(z_joi_, 1, 0), dim=1)[0],
                'z_mar':  torch.cat(torch.split(z_mar_, 1, 0), dim=1)[0],
                'z_smpD': torch.cat(torch.split(z_D[:nz], 1, 0), dim=1)[0],
                'z_marD': torch.cat(torch.split(z_D[nz:2 * nz], 1, 0), dim=1)[0],
                'z_joiD': torch.cat(torch.split(z_D[2 * nz:], 1, 0), dim=1)[0],
                'mtau_smp': mtau_smp,
                'mtau_joi': mtau_joi,
                'mtau_mar': mtau_mar,
                'sacc_smp': sacc[:nz],
                'sacc_mar': sacc[nz:2 * nz],
                'sacc_joi': sacc[2 * nz:],
                'jacc_smp': jacc[:nz],
                'jacc_mar': jacc[nz:2 * nz],
                'jacc_joi': jacc[2 * nz:],
            }
        
        elif btype == 'mia':
            x_joi  = self.joi_inProj(batch['joi'])         # N, L, J, C
            N, L, J, C = x_joi.shape
            x_joi  = torch.cat(torch.split(x_joi, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_joi_ = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # NL, C
            z_joi_ = torch.stack(torch.split(z_joi_, L, 0))
            z_joi = self.positional_encoder(z_joi_)  # B, L, C
            z_joi  = z_joi.permute(1, 0, 2)           # L, B, C
            z_joi = self.temp_composer(z_joi)        # L, B, C
            etau_joi = self.emg_dec(z_joi) # L, B, stau
            etau_joi = etau_joi.permute(1, 0, 2)     # B, L, stau
            
            x_mar  = self.mar_inProj(batch['mar'])         # N, L, J, C
            N, L, J, C = x_mar.shape
            x_mar  = torch.cat(torch.split(x_mar, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_mar_ = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # NL, C
            z_mar_ = torch.stack(torch.split(z_mar_, L, 0))
            z_mar  = self.positional_encoder(z_mar_)  # B, L, C
            z_mar  = z_mar.permute(1, 0, 2)           # L, B, C
            z_mar  = self.temp_composer(z_mar)        # L, B, C
            etau_mar = self.emg_dec(z_mar)            # L, B, etau
            etau_mar = etau_mar.permute(1, 0, 2)      # B, L, etau
            
            nz    = z_joi_.shape[0]
            
            x_marA = self.marA_inProj(batch['mar'][..., :6])                                # N, L, J, C
            x_marA = torch.cat(torch.split(x_marA, 1, 0), dim=1)[0].permute(1, 0, 2)         # J, BL, C
            z_marA = self.marA_enc(self.mar_query.expand(-1, x_marA.shape[1], -1), x_marA)[0] # 1, BL, C
            z_marA = torch.stack(torch.split(z_marA, L, 0))                                 # B, L, C
            
            x_joiA = self.joiA_inProj(batch['joi'][..., :6])                                # N, L, J, C
            x_joiA = torch.cat(torch.split(x_joiA, 1, 0), dim=1)[0].permute(1, 0, 2)         # J, BL, C
            z_joiA = self.joiA_enc(self.joi_query.expand(-1, x_joiA.shape[1], -1), x_joiA)[0] # 1, BL, C
            z_joiA = torch.stack(torch.split(z_joiA, L, 0))                                 # B, L, C
            
            z_A    = torch.cat((z_marA, z_joiA), dim=0)
            z_emg  = self.emg_enc(batch['etau'].flatten(2)) # B, L, C
            z_emg  = torch.tile(z_emg, (2, 1, 1))           # 2B, L, C    
            z_D    = self.composer(torch.cat((z_A, z_emg), dim=-1))
            jacc   = self.joiang_dec(z_D)
            
            output = {
                'z_joi':  torch.cat(torch.split(z_joi_, 1, 0), dim=1)[0],
                'z_mar':  torch.cat(torch.split(z_mar_, 1, 0), dim=1)[0],
                'z_marD': torch.cat(torch.split(z_D[:nz], 1, 0), dim=1)[0],
                'z_joiD': torch.cat(torch.split(z_D[nz:], 1, 0), dim=1)[0],
                'etau_joi': etau_joi,
                'etau_mar': etau_mar,
                'jacc_mar': jacc[:nz],
                'jacc_joi': jacc[nz:],
            }
            
        elif btype == 'glink':
            z_smp_ = self.smp_enc(batch['smp'].flatten(2)) # B, L, C
            z_smp = self.positional_encoder(z_smp_)        # B, L, C
            z_smp = z_smp.permute(1, 0, 2)                 # L, B, C
            z_smp = self.temp_composer(z_smp)              # L, B, C
            mtau_smp = self.grf_dec(z_smp)                 # L, B, stau
            mtau_smp = mtau_smp.permute(1, 0, 2)           # B, L, stau
            
            x_joi  = self.joi_inProj(batch['joi'])         # N, L, J, C
            N, L, J, C = x_joi.shape
            x_joi  = torch.cat(torch.split(x_joi, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_joi_ = self.joi_enc(self.joi_query.expand(-1, x_joi.shape[1], -1), x_joi)[0] # NL, C
            z_joi_ = torch.stack(torch.split(z_joi_, L, 0))
            z_joi  = self.positional_encoder(z_joi_)  # B, L, C
            z_joi  = z_joi.permute(1, 0, 2)           # L, B, C
            z_joi = self.temp_composer(z_joi)        # L, B, C
            mtau_joi = self.grf_dec(z_joi) # L, B, stau
            mtau_joi = mtau_joi.permute(1, 0, 2)     # B, L, stau
            
            x_mar  = self.mar_inProj(batch['mar'])         # N, L, J, C
            N, L, J, C = x_mar.shape
            x_mar  = torch.cat(torch.split(x_mar, 1, 0), dim=1)[0].permute(1, 0, 2) # J, NL, C
            z_mar_ = self.mar_enc(self.mar_query.expand(-1, x_mar.shape[1], -1), x_mar)[0] # NL, C
            z_mar_ = torch.stack(torch.split(z_mar_, L, 0))
            z_mar  = self.positional_encoder(z_mar_)  # B, L, C
            z_mar  = z_mar.permute(1, 0, 2)           # L, B, C
            z_mar  = self.temp_composer(z_mar)        # L, B, C
            grf_mar = self.grf_dec(z_mar)             # L, B, etau
            grf_mar = grf_mar.permute(1, 0, 2)        # B, L, etau
            output = {
                'z_smp': z_smp_,
                'z_joi': z_joi_,
                'z_mar': z_mar_,
                'grf_smp': grf_smp,
                'grf_joi': grf_joi,
                'grf_mar': grf_mar,
            }
        return output
        