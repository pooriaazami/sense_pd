import torch

from tqdm import tqdm

from utils import loss_mpjpe
from .utils import transform_embedding

def train_epoch(backbone, regressor, d3dp, dataloader, optimizer, device):
    backbone.train()
    regressor.train()
    d3dp.train()

    total_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        
        seq = data['seq'].to(device)
        mask = data['mask'].to(device)

        embeddings = backbone(seq)
        embeddings = transform_embedding(embeddings, mask)
        predicted_embedding = d3dp(embeddings)
        predicted_joints = regressor(predicted_embedding)

        loss = loss_mpjpe(predicted_joints, seq)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().numpy().item()

    return {'loss': total_loss}

def validation_epoch(backbone, regressor, d3dp, dataloader, device):
    backbone.eval()
    regressor.eval()
    d3dp.train()

    total_loss = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            seq = data['seq'].to(device)
            mask = data['mask'].to(device)

            embeddings = backbone(seq)
            embeddings = transform_embedding(embeddings, mask)
            predicted_embedding = d3dp(embeddings)
            predicted_joints = regressor(predicted_embedding)

            loss = loss_mpjpe(predicted_joints, seq)

            total_loss += loss.detach().cpu().numpy().item()

    return {'loss': total_loss}

def update_log(writer, log, step):
    writer.add_scalar(f'Loss/', log['loss'], step)

def train_diffusion(backbone, regressor, d3dp, dataloader, optimizer, epochs, device, writer):
    for epoch in range(1, epochs+1):
        log = train_epoch(backbone, regressor, d3dp, dataloader, optimizer, device)
        update_log(writer, log, epoch)

