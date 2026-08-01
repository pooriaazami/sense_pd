import os

import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils import mpjpe
from .utils import transform_embedding

def diffusion_loss(noise, predicted_noise, seq, predicted_joints):
    loss = mpjpe(predicted_joints, seq)
    loss += F.mse_loss(noise, predicted_noise)
    loss += 1000 * F.mse_loss(seq[:, 0, :, :], predicted_joints[:, 0, :, :])

    return loss

def train_epoch(backbone, regressor, d3dp, dataloader, optimizer, device):
    backbone.train()
    d3dp.train()

    total_loss = 0
    for seq, mask in tqdm(dataloader):
        optimizer.zero_grad()
        
        seq = seq.float().to(device)
        mask = mask.float().to(device)

        embeddings = backbone(seq)
        embeddings = transform_embedding(embeddings, mask)
        noise, predicted_noise = d3dp(embeddings)
        predicted_embeddings = embeddings - predicted_noise
        predicted_joints = regressor(predicted_embeddings)

        loss = diffusion_loss(noise, predicted_noise, seq, predicted_joints)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().numpy().item()

    return {'loss': total_loss}

def update_log(writer, train_log, step):
    writer.add_scalar(f'Diffusion/Loss/', train_log['loss'], step)

def train_diffusion_model(backbone, 
                          regressor,
                          d3dp, 
                          train_dataloader, 
                          optimizer, 
                          epochs, 
                          device, 
                          writer,
                          save_freq,
                          save_path_root):
    
    for epoch in range(1, epochs+1):
        train_log = train_epoch(backbone, regressor, d3dp, train_dataloader, optimizer, device)

        update_log(writer, train_log, epoch)

        if epoch % save_freq == 0:
            torch.save(d3dp.state_dict(), os.path.join(save_path_root, f'd3dp_epoch_{epoch}.pth'))

