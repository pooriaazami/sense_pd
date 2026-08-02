import os

import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils import mpjpe
from .utils import transform_embedding

def diffusion_loss(noise, predicted_noise, seq, predicted_joints, weights=None):
    """Compute diffusion loss with configurable term weights.

    weights: dict-like with keys 'recon', 'noise', 'first_frame'.
    Returns: (total, recon, noise_term, first_frame_term)
    """
    if weights is None:
        weights = {'recon': 1.0, 'noise': 1.0, 'first_frame': 1.0}

    term_recon = mpjpe(predicted_joints, seq)
    term_noise = F.l1_loss(noise, predicted_noise)
    term_first = F.l1_loss(seq[:, 0, :, :], predicted_joints[:, 0, :, :])

    total = weights.get('recon', .1) * term_recon + \
            weights.get('noise', 1.0) * term_noise + \
            weights.get('first_frame', .1) * term_first

    return total, term_recon, term_noise, term_first

def train_epoch(backbone, regressor, d3dp, dataloader, optimizer, device, loss_weights=None):
    backbone.train()
    d3dp.train()

    total_loss, total_mjpe, total_noise_prediction, total_first_frame = [0] * 4
    for seq, mask in tqdm(dataloader):
        optimizer.zero_grad()
        
        seq = seq.float().to(device)
        mask = mask.float().to(device)

        embeddings = backbone(seq)
        embeddings = transform_embedding(embeddings, mask)
        noise, predicted_noise, predicted_embeddings = d3dp(embeddings)
        predicted_joints = regressor(predicted_embeddings)

        loss, mjpe_val, noise_prediction_val, first_frame_val = diffusion_loss(
            noise, predicted_noise, seq, predicted_joints, weights=loss_weights
        )

        loss.backward()
        # Clip gradients on the diffusion model to stabilize training
        try:
            torch.nn.utils.clip_grad_norm_(d3dp.parameters(), max_norm=1.0)
        except Exception:
            pass
        optimizer.step()

        total_loss += loss.detach().cpu().numpy().item()
        total_mjpe += mjpe_val.detach().cpu().numpy().item()
        total_noise_prediction += noise_prediction_val.detach().cpu().numpy().item()
        total_first_frame += first_frame_val.detach().cpu().numpy().item()

    return {
        'total_loss': total_loss,
        'mpje': total_mjpe,
        'noise_prediction': total_noise_prediction,
        'first_frame': total_first_frame
    }

def update_log(writer, train_log, step):
    writer.add_scalar(f'Diffusion/TotalLoss/', train_log['total_loss'], step)
    writer.add_scalar(f'Diffusion/MPJE/', train_log['mpje'], step)
    writer.add_scalar(f'Diffusion/NoisePrediction/', train_log['noise_prediction'], step)
    writer.add_scalar(f'Diffusion/FirstFrame/', train_log['first_frame'], step)

def train_diffusion_model(backbone, 
                          regressor,
                          d3dp, 
                          train_dataloader, 
                          optimizer, 
                          epochs, 
                          device, 
                          writer,
                          save_freq,
                          save_path_root,
                          loss_weights=None):
    
    for epoch in range(1, epochs+1):
        train_log = train_epoch(
            backbone, regressor, d3dp, train_dataloader, optimizer, device, loss_weights=loss_weights
        )

        update_log(writer, train_log, epoch)

        if epoch % save_freq == 0:
            torch.save(d3dp.state_dict(), os.path.join(save_path_root, f'd3dp_epoch_{epoch}.pth'))

