from tqdm import tqdm

from utils import loss_mpjpe

def transform_embedding(embeddings, mask):
    embeddings = embeddings.permute(0, 2, 3, 1)
    mask = mask.unsqueeze(1).unsqueeze(1).float().to(embeddings.device)
    embeddings = (embeddings * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
    embeddings = embeddings.flatten(1)

    return embeddings

def train_epoch(backbone, regressor, d3dp, dataloader, optimizer, device):
    total_loss = 0

    d3dp.train()
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

def update_log(writer, log, step):
    writer.add_scalar(f'Loss/', log['loss'], step)

def train_diffusion(backbone, regressor, d3dp, dataloader, optimizer, epochs, device, writer):
    for epoch in range(1, epochs+1):
        log = train_epoch(backbone, regressor, d3dp, dataloader, optimizer, device)
        update_log(writer, log, epoch)

