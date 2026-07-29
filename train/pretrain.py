import os
import torch
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from .utils import transform_embedding

def train_epoch(
        backbone, 
        regressor, 
        dataloader, 
        loss_fn, 
        optimizer, 
        random_joint_mask_fn, 
        random_frame_mask_fn, 
        device, 
    ):

    total_joint_masked_loss, total_joint_masked_temporal_loss, total_joint_masked_reconstruction_loss, \
    total_frame_masked_loss, total_frame_masked_temporal_loss, total_frame_masked_reconstruction_loss, \
    total_loss = [0] * 7

    for data, mask in tqdm(dataloader):
        data = data.float().to(device)
        mask = mask.float().to(device)

        optimizer.zero_grad()

        # Joint Masked
        data_joint_masked = random_joint_mask_fn(data)
        joint_masked_embeddings = backbone(data_joint_masked)
        joint_masked_embeddings = transform_embedding(joint_masked_embeddings, mask)
        predicted_masked_joints = regressor(joint_masked_embeddings)
    
        joint_masked_loss, joint_masked_temporal_loss, joint_masked_reconstruction_loss = loss_fn(predicted_masked_joints, data, mask)
        
        # Frame Masked
        data_frame_masked = random_frame_mask_fn(data)
        frame_masked_embeddings = backbone(data_frame_masked)
        frame_masked_embeddings = transform_embedding(frame_masked_embeddings, mask)
        predicted_masked_frames = regressor(frame_masked_embeddings)
    
        frame_masked_loss, frame_masked_temporal_loss, frame_masked_reconstruction_loss = loss_fn(predicted_masked_frames, data, mask)

        loss = joint_masked_loss + frame_masked_loss
        loss.backward()
        optimizer.step()

        total_joint_masked_temporal_loss += joint_masked_temporal_loss.detach().cpu().numpy().item()
        total_joint_masked_reconstruction_loss += joint_masked_reconstruction_loss.detach().cpu().numpy().item()
        total_joint_masked_loss += joint_masked_loss.detach().cpu().numpy().item()

        total_frame_masked_temporal_loss = frame_masked_temporal_loss.detach().cpu().numpy().item()
        total_frame_masked_reconstruction_loss = frame_masked_reconstruction_loss.detach().cpu().numpy().item()
        total_frame_masked_loss += frame_masked_loss.detach().cpu().numpy().item()

        total_loss += loss.detach().cpu().numpy().item()
        
    return {
        'frame_masked': {
            'temporal_loss': total_frame_masked_temporal_loss,
            'reconstruction_loss': total_frame_masked_reconstruction_loss
        },
        'joint_masked': {
            'temporal_loss': total_joint_masked_temporal_loss,
            'reconstruction_loss': total_joint_masked_reconstruction_loss
        },
        'total': total_loss
    }

def validation_epoch(
        backbone, 
        regressor, 
        dataloader, 
        loss_fn, 
        random_joint_mask_fn, 
        random_frame_mask_fn, 
        device, 
    ):

    total_joint_masked_loss, total_joint_masked_temporal_loss, total_joint_masked_reconstruction_loss, \
    total_frame_masked_loss, total_frame_masked_temporal_loss, total_frame_masked_reconstruction_loss, \
    total_loss = [0] * 7

    with torch.no_grad():
        for data, mask in tqdm(dataloader):
            data = data.float().to(device)
            mask = mask.float().to(device)

            # Joint Masked
            data_joint_masked = random_joint_mask_fn(data)
            joint_masked_embeddings = backbone(data_joint_masked)
            joint_masked_embeddings = transform_embedding(joint_masked_embeddings, mask)
            predicted_masked_joints = regressor(joint_masked_embeddings)
        
            joint_masked_loss, joint_masked_temporal_loss, joint_masked_reconstruction_loss = loss_fn(predicted_masked_joints, data, mask)
            
            # Frame Masked
            data_frame_masked = random_frame_mask_fn(data)
            frame_masked_embeddings = backbone(data_frame_masked)
            frame_masked_embeddings = transform_embedding(frame_masked_embeddings, mask)
            predicted_masked_frames = regressor(frame_masked_embeddings)
        
            frame_masked_loss, frame_masked_temporal_loss, frame_masked_reconstruction_loss = loss_fn(predicted_masked_frames, data, mask)

            loss = joint_masked_loss + frame_masked_loss

            total_joint_masked_temporal_loss += joint_masked_temporal_loss.detach().cpu().numpy().item()
            total_joint_masked_reconstruction_loss += joint_masked_reconstruction_loss.detach().cpu().numpy().item()
            total_joint_masked_loss += joint_masked_loss.detach().cpu().numpy().item()

            total_frame_masked_temporal_loss = frame_masked_temporal_loss.detach().cpu().numpy().item()
            total_frame_masked_reconstruction_loss = frame_masked_reconstruction_loss.detach().cpu().numpy().item()
            total_frame_masked_loss += frame_masked_loss.detach().cpu().numpy().item()

            total_loss += loss.detach().cpu().numpy().item()
        
    return {
        'frame_masked': {
            'temporal_loss': total_frame_masked_temporal_loss,
            'reconstruction_loss': total_frame_masked_reconstruction_loss
        },
        'joint_masked': {
            'temporal_loss': total_joint_masked_temporal_loss,
            'reconstruction_loss': total_joint_masked_reconstruction_loss
        },
        'total': total_loss
    }

def log_to_tensorboard(writer, step, train_log, val_log):
    # Frame Masked
    writer.add_scalar('Pretrain/Loss/FrameMasked/temporal_loss', {
        'train': train_log['frame_masked']['temporal_loss'],
        'val': val_log['frame_masked']['temporal_loss']
    }, step)
    writer.add_scalar('Pretrain/Loss/FrameMasked/', {
        'train': train_log['frame_masked']['reconstruction_loss'],
        'val': val_log['frame_masked']['reconstruction_loss']
        }, step)
    writer.add_scalar('Pretrain/Loss/FrameMasked/', {
        'train': train_log['frame_masked']['temporal_loss'] + train_log['frame_masked']['reconstruction_loss'],
        'val': val_log['frame_masked']['temporal_loss'] + val_log['frame_masked']['reconstruction_loss']
        }, step)

    # Joint Masked
    writer.add_scalar('Pretrain/Loss/JointMasked/temporal_loss', {
        'train': train_log['joint_masked']['temporal_loss'],
        'val': val_log['joint_masked']['temporal_loss']
    }, step)
    writer.add_scalar('Pretrain/Loss/JointMasked/', {
        'train': train_log['joint_masked']['reconstruction_loss'],
        'val': val_log['joint_masked']['reconstruction_loss']
        }, step)
    writer.add_scalar('Pretrain/Loss/JointMasked/', {
        'train': train_log['joint_masked']['temporal_loss'] + train_log['joint_masked']['reconstruction_loss'],
        'val': val_log['joint_masked']['temporal_loss'] + val_log['joint_masked']['reconstruction_loss']
        }, step)

    writer.add_scalar('Pretrain/Loss/Total/', {
            'train': train_log['total'],
            'val': val_log['total']
        }, step)

def pretrain(
        backbone, 
        regressor, 
        train_dataloader,
        val_dataloader, 
        loss_fn, 
        optimizer, 
        random_joint_mask_fn, 
        random_frame_mask_fn, 
        epochs, 
        device, 
        exp_name,
        save_freq
    ):

    dt = datetime.now()
    name = f'{exp_name}__{dt.month}_{dt.day}_{dt.hour}_{dt.minute}'
    writer = SummaryWriter(os.path.join('assets', 'logs', name))

    for epoch in range(1, epochs + 1):
        train_log = train_epoch(
            backbone=backbone,
            regressor=regressor,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            random_joint_mask_fn=random_joint_mask_fn,
            random_frame_mask_fn=random_frame_mask_fn,
            device=device,
        )

        val_log = validation_epoch(
            backbone=backbone,
            regressor=regressor,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            random_joint_mask_fn=random_joint_mask_fn,
            random_frame_mask_fn=random_frame_mask_fn,
            device=device,
        )

        log_to_tensorboard(
            writer=writer,
            step=epoch,
            train_log=train_log,
            val_log=val_log
        )

        if epoch % save_freq == 0:
            torch.save(backbone.state_dict, os.path.join('assets', 'checkpoints', f'epoch_{epoch}.pth'))