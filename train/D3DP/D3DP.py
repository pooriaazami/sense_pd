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
        self.ddim_sampling_eta = 1.
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
        self.pose_estimator = pose_estimator

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, x_cond):
        x_t = torch.clamp(x, min=-1.1 * self.scale, max=1.1*self.scale)
        x_t = x_t / self.scale
        pred_pose = self.pose_estimator(x_t, t, x_cond)

        x_start = pred_pose
        x_start = x_start * self.scale
        x_start = torch.clamp(x_start, min=-1.1 * self.scale, max=1.1*self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, inputs_3d, input_cond):
        batch = inputs_3d.shape[0]
        shape = (batch, self.num_proposals, self.frames, 17, 512)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps_eval, self.sampling_timesteps, self.ddim_sampling_eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=inputs_3d.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        preds_all=[]
        for time, time_next in time_pairs:
            
            time_cond = torch.full((batch,), time, device=inputs_3d.device, dtype=torch.long)
            #self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(img, time_cond, input_cond)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
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
        B = features.shape[0]

        # Prepare condition: [B, F, J, C] -> first frame -> projection
        x_cond = features[:, 0]
        cond = self.condition_proj(
            x_cond.reshape(B * self.num_joints, -1)
        ).reshape(B, 1, self.num_joints, -1)

        if self.is_train:
            # x_t (noisy), noise, timestep
            noisy_features, noise, t = self.prepare_targets(features)
            noisy_features = noisy_features.float()
            t = t.squeeze(-1)

            # predict x_0
            predicted_noise = self.pose_estimator(
                noisy_features,
                t,
                cond
            )

            # Return everything needed for training
            return noise, predicted_noise

        else:
            predicted_features = self.ddim_sample(
                features,
                cond
            )

            # Return the original clean embeddings and the denoised predictions
            return predicted_features

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

        # Sample noise with same shape as input
        noise = torch.randn_like(targets)

        # Scale clean data
        x_start = targets * self.scale

        # Forward diffusion
        x = self.q_sample(
            x_start=x_start,
            t=t,
            noise=noise
        )

        # Same normalization as original code
        x = torch.clamp(
            x,
            min=-1.1 * self.scale,
            max=1.1 * self.scale
        )

        x = x / self.scale

        return x, noise, t