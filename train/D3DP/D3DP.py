from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MixSTE2
from .utils import *

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class D3DP(nn.Module):
    """
    Implement D3DP
    """

    def __init__(self, 
                 number_of_frames,
                 test_time_augmentation, 
                 timesteps,
                 timesteps_eval,
                 scale,
                 dim_rep,
                 joints_left, 
                 joints_right, 
                 train=True, 
                 num_proposals=1, 
                 sampling_timesteps=1, 
                 num_joints=17, 
                 pose_estimator=None
                ):
        super().__init__()

        self.num_joints = num_joints

        self.frames = number_of_frames
        self.num_proposals = num_proposals
        self.flip = test_time_augmentation
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.is_train = train
        
        # build diffusion
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(torch.float32)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.num_timesteps_eval = int(timesteps_eval)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 0.
        self.self_condition = False
        self.scale = scale
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        #self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        
        self.condition_proj = nn.Linear(dim_rep, dim_rep)
        self.concat_proj = nn.Linear(dim_rep * 2, dim_rep)
        self.time_embed = nn.Embedding(self.num_timesteps, dim_rep)
        self.pose_estimator = pose_estimator

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
                x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_start
        ) / extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    def _build_condition(self, features, x_t, t):
        B = features.shape[0]

        x_cond = features[:, 0:1]
        x_cond = self.condition_proj(
            x_cond.reshape(B * self.num_joints, -1)
        ).reshape(B, 1, self.num_joints, -1)
        x_cond = x_cond / self.scale

        x_t_norm = x_t / self.scale

        time_emb = self.time_embed(t)
        time_emb = time_emb[:, None, None, :]

        x_t_time = x_t_norm + time_emb
        x_cond_time = x_cond + time_emb
        x_cond_time = x_cond_time.expand(B, self.frames, self.num_joints, -1)

        concat = torch.cat([x_t_time, x_cond_time], dim=-1)
        cond = self.concat_proj(concat.reshape(-1, concat.shape[-1]))
        cond = cond.reshape(B, self.frames, self.num_joints, -1)

        return cond

    def model_predictions(self, x, t, x_cond):
        x_t = torch.clamp(x, min=-1.1 * self.scale, max=1.1 * self.scale)
        x_t = x_t / self.scale

        pred_x_start = self.pose_estimator(x_t, t, x_cond)
        pred_x_start = torch.clamp(pred_x_start, min=-1.1, max=1.1)

        pred_noise = self.predict_noise_from_start(x_t, t, pred_x_start)
        return ModelPrediction(pred_noise, pred_x_start)

    def ddim_sample(self, inputs_3d):
        batch = inputs_3d.shape[0]
        total_timesteps, sampling_timesteps, eta = self.num_timesteps_eval, self.sampling_timesteps, self.ddim_sampling_eta

        t_init = torch.full((batch,), self.num_timesteps - 1, device=inputs_3d.device, dtype=torch.long)
        x_start = inputs_3d * self.scale
        img = self.q_sample(x_start=x_start, t=t_init, noise=torch.randn_like(x_start))
        img = torch.clamp(img, min=-1.1 * self.scale, max=1.1 * self.scale)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=inputs_3d.device, dtype=torch.long)
            cond = self._build_condition(inputs_3d, img, time_cond)
            preds = self.model_predictions(img, time_cond, cond)
            pred_noise, pred_x_start = preds.pred_noise, preds.pred_x_start

            if time_next < 0:
                img = pred_x_start
                break

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = pred_x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward_diffusion(self, features):
        targets_norm = features / self.scale
        noisy_features, _, t = self.prepare_targets(features)
        cond = self._build_condition(features, noisy_features, t)
        preds = self.model_predictions(noisy_features, t, cond)
        return targets_norm, preds.pred_x_start

    def train(self, mode: bool = True):
        super().train(mode)
        self.pose_estimator.train()
        self.is_train = True

        return self

    def eval(self):
        super().eval()
        self.pose_estimator.eval()
        self.is_train = False

        return self

    def forward(self, features):
        if self.is_train:
            noisy_features, targets_norm, t = self.prepare_targets(features)
            cond = self._build_condition(features, noisy_features, t)
            preds = self.model_predictions(noisy_features, t, cond)
            return targets_norm, preds.pred_x_start

        return self.ddim_sample(features)

    def prepare_targets(self, targets):
        B = targets.shape[0]
        device = targets.device

        # Sample one timestep per sample
        t = torch.randint(
            0,
            self.num_timesteps,
            (B,),
            device=device
        ).long()

        x_start = targets * self.scale
        noise = torch.randn_like(x_start)

        x = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise
        )

        x = torch.clamp(
            x,
            min=-1.1 * self.scale,
            max=1.1 * self.scale
        )

        x = x / self.scale
        targets_norm = targets / self.scale

        return x, targets_norm, t