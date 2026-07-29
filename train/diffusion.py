import os

import torch

from tqdm import tqdm

from utils import loss_mpjpe
from .utils import transform_embedding

def train_epoch(backbone, d3dp, dataloader, optimizer, device):
    backbone.train()
    d3dp.train()

    total_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        
        seq = data['seq'].to(device)
        mask = data['mask'].to(device)

        embeddings = backbone(seq)
        embeddings = transform_embedding(embeddings, mask)
        predicted_joints = d3dp(embeddings)

        loss = loss_mpjpe(predicted_joints, seq)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().numpy().item()

    return {'loss': total_loss}

def validation_epoch(backbone, d3dp, dataloader, device):
    backbone.eval()
    d3dp.train()

    total_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            seq = data['seq'].to(device)
            mask = data['mask'].to(device)

            embeddings = backbone(seq)
            embeddings = transform_embedding(embeddings, mask)
            predicted_joints = d3dp(embeddings)

            loss = loss_mpjpe(predicted_joints, seq)

            total_loss += loss.detach().cpu().numpy().item()

    return {'loss': total_loss}

def update_log(writer, train_log, val_log, step):
    writer.add_scalars(f'Loss/', {
        'train': train_log['loss'],
        'val': val_log['loss']
    }, step)

def train_diffusion_model(backbone, 
                          d3dp, 
                          train_dataloader, 
                          val_dataloader, 
                          optimizer, 
                          epochs, 
                          device, 
                          writer,
                          save_freq):
    for epoch in range(1, epochs+1):
        train_log = train_epoch(backbone, d3dp, train_dataloader, optimizer, device)
        val_log = validation_epoch(backbone, d3dp, val_dataloader, device)

        update_log(writer, train_log, val_log, epoch)

        if epoch % save_freq == 0:
            torch.save(d3dp.state_dict(), os.path.join('assets', 'checkpoints', f'd3dp_epoch_{epoch}.pth'))

