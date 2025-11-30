from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LogisticNormal, Beta
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
try:
    import robomimic.models.base_nets as rmbn
    if not hasattr(rmbn, 'CropRandomizer'):
        raise ImportError("CropRandomizer is not in robomimic.models.base_nets")
except ImportError:
    import robomimic.models.obs_core as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.vision.rot_randomizer import RotRandomizer


class L1FlowUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            infer_strategy = "L1Flow",
            num_inference_steps = 2,
            t_first = 0.5,
            loss_type = 'l1',
            loss_space = 'sample',
            timestep_sampler_type = 'mixed',
            obs_as_global_cond = True,
            crop_shape = (76, 76),
            diffusion_step_embed_dim = 256,
            down_dims = (256,512,1024),
            kernel_size = 5,
            n_groups = 8,
            cond_predict_scale = True,
            obs_encoder_group_norm = False,
            eval_fixed_crop = False,
            rot_aug = False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()
        self.beta_dist = Beta(concentration1=1, concentration0=1.5)
        self.logisticnormal_dist = LogisticNormal(0, 1)

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        #-----------------------------------------------------------------------------------------------
        # Configuration Options for L1Flow Training and Inference
        #-----------------------------------------------------------------------------------------------

        # infer_strategy: Inference strategy.
        #   - "L1Flow": (recommended) Our proposed two-step inference method.
        #   - "FM":     Standard flow-matching inference, i.e., Euler integration over [0,1].
        allowed_infer_strategy = {"L1Flow", "FM"}
        if infer_strategy not in allowed_infer_strategy:
            raise ValueError(
                f"infer_strategy must be one of {sorted(allowed_infer_strategy)}, "
                f"but now got {infer_strategy!r}"
            )
        self.infer_strategy = infer_strategy

        # num_inference_steps: Number of inference steps.
        #   - Only effective for in `FM`.
        #   - Ignored in "L1Flow", which uses a fixed two-step inference process.
        self.num_inference_steps = num_inference_steps

        # t_first: Initial time point for the first inference step.
        #   - Only used in "L1Flow".
        #   - Recommended value: 0.5.
        self.t_first = t_first
        
        # loss_type: Type of loss function.
        #   - Options: "l1" (recommended) or "mse".
        allowed_loss_type = {"l1", "mse"}
        if loss_type not in allowed_loss_type:
            raise ValueError(
                f"loss_type must be one of {sorted(allowed_loss_type)}, "
                f"but now got {loss_type!r}"
            )
        self.loss_type = loss_type

        # loss_space: Target loss space for supervision.
        #   - Options: "sample" (default) or "velocity".
        allowed_loss_space = {"velocity", "sample"}
        if loss_space not in allowed_loss_space:
            raise ValueError(
                f"loss_space must be one of {sorted(allowed_loss_space)}, "
                f"but now got {loss_space!r}"
            )
        self.loss_space = loss_space

        # timestep_sampler_type: timesteps sampling strategy
        #   - Options: "uniform", "beta", or "mixed" (recommended for balanced coverage).
        allowed_timestep_sampler_type = {"uniform", "beta", "mixed"}
        if timestep_sampler_type not in allowed_timestep_sampler_type:
            raise ValueError(
                f"timestep_sampler_type must be one of {sorted(allowed_timestep_sampler_type)}, "
                f"but now got {timestep_sampler_type!r}"
            )
        self.timestep_sampler_type = timestep_sampler_type

        #-----------------------------------------------------------------------------------------------

        print("Flow params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model

        x_t = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        t = torch.zeros(1, device=condition_data.device)

        # use our proposed two-step inference strategy
        if self.infer_strategy == "L1Flow":
            dt = self.t_first
            
            # 1. first step: x_0 -> x_0.5
            a_pred = model(x_t, t, 
                local_cond=local_cond, global_cond=global_cond)
            
            # ODE integration: x_t -> x_{t + dt}
            v_t = (a_pred - x_t)/(1-t)
            x_t = x_t + dt*v_t
            t = t + dt
        
            # 2. second step: x_0.5 -> x_1
            a_pred = model(x_t, t, 
                local_cond=local_cond, global_cond=global_cond)   
            
            return a_pred
        
        # use the standard inference strategy in flow matching
        elif self.infer_strategy == "FM":
            dt = 1/self.num_inference_steps
            
            for i in range(self.num_inference_steps):
                # predict model output
                a_pred = model(x_t, t, 
                    local_cond=local_cond, global_cond=global_cond)
                v_t = (a_pred - x_t)/(1 - t + 1e-8)

                # compute previous image: x_t -> x_{t + dt}
                x_t = x_t + dt*v_t
                t = t + dt

            return x_t 

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.rot_aug:
            nobs, nactions = self.rot_randomizer(nobs, nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image

        timesteps = None

        if self.timestep_sampler_type == 'uniform':
            timesteps = torch.rand((bsz,), device=trajectory.device)
        elif self.timestep_sampler_type == 'beta':
            timesteps = (self.beta_dist.sample((bsz,))).to(trajectory.device)
        elif self.timestep_sampler_type == 'mixed':
            # default: logistic + uniform
            timesteps = self.logisticnormal_dist.sample((bsz,))[:,0].to(trajectory.device)
            uni_timesteps = torch.rand_like(timesteps)
            mask = torch.rand_like(uni_timesteps) < 0.01
            timesteps[mask] = uni_timesteps[mask]
            
        timesteps_expand = timesteps[...,None,None]

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = timesteps_expand * trajectory + (1 - timesteps_expand) * noise
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the x
        a_pred = self.model(noisy_trajectory, timesteps,
                            local_cond=local_cond, global_cond=global_cond)
        loss = None

        # use the v-loss
        if self.loss_space == "velocity":
            # Set denominator truncation to avoid numerical overflow
            eps = 0.05
            v_t = (a_pred - noisy_trajectory) / (1 - timesteps_expand).clamp(min=eps)  
            # v_truth = trajectory - noise
            v_truth = (trajectory - noisy_trajectory) / (1 - timesteps_expand).clamp(min=eps)  
            if self.loss_type == "l1":
                loss = F.l1_loss(v_t, v_truth, reduction='none')
            elif self.loss_type == "mse":
                loss = F.mse_loss(v_t, v_truth, reduction='none')

        # use the x-loss
        elif self.loss_space == "sample":
            # Set denominator truncation to avoid numerical overflow 
            if self.loss_type == "l1":
                loss = F.l1_loss(a_pred, trajectory, reduction='none')
            elif self.loss_type == "mse":
                loss = F.mse_loss(a_pred, trajectory, reduction='none')
        
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
