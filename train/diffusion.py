import os

import torch
import torch.nn.functional as F

from tqdm import tqdm

from utils import mpjpe
from .utils import transform_embedding

def diffusion_loss(predicted_emneddings, embeddings, predicted_joints, seq):
    loss = mpjpe(predicted_joints, seq)
    loss += 1000 * F.mse_loss(predicted_emneddings[:, :, 0], embeddings[:, :, 0])

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
        predicted_emneddings = d3dp(embeddings).squeeze(1)
        predicted_joints = regressor(embeddings)

        loss = diffusion_loss(predicted_emneddings, embeddings, predicted_joints, seq)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().numpy().item()

    return {'loss': total_loss}

def validation_epoch(backbone, regressor, d3dp, dataloader, device):
    backbone.eval()
    d3dp.eval()

    total_loss = 0
    with torch.no_grad():
        for seq, mask in tqdm(dataloader):
            seq = seq.float().to(device)
            mask = mask.float().to(device)

            embeddings = backbone(seq)
            embeddings = transform_embedding(embeddings, mask)
            predicted_emneddings = d3dp(embeddings).squeeze(1)
            predicted_joints = regressor(embeddings)

            loss = diffusion_loss(predicted_emneddings, embeddings, predicted_joints, seq)

            total_loss += loss.detach().cpu().numpy().item()

    return {'loss': total_loss}

def update_log(writer, train_log, val_log, step):
    writer.add_scalars(f'Diffusion/Loss/', {
        'train': train_log['loss'],
        'val': val_log['loss']
    }, step)

def train_diffusion_model(backbone, 
                          regressor,
                          d3dp, 
                          train_dataloader, 
                          val_dataloader, 
                          optimizer, 
                          epochs, 
                          device, 
                          writer,
                          save_freq):
    for epoch in range(1, epochs+1):
        train_log = train_epoch(backbone, regressor, d3dp, train_dataloader, optimizer, device)
        val_log = validation_epoch(backbone, regressor, d3dp, val_dataloader, device)

        update_log(writer, train_log, val_log, epoch)

        if epoch % save_freq == 0:
            torch.save(d3dp.state_dict(), os.path.join('assets', 'checkpoints', f'd3dp_epoch_{epoch}.pth'))

