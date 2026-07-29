import os
import torch
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from .utils import transform_embedding

def train_one_epoch(
        backbone, 
        regressor, 
        dataloader, 
        loss_fn, 
        optimizer, 
        random_joint_mask_fn, 
        random_frame_mask_fn, 
        device, 
        writer, 
        epoch
    ):

    total_joint_masked_motion_loss, total_joint_masked_rec_loss, total_joint_masked_loss, \
    total_frame_masked_motion_loss, total_frame_masked_rec_loss , total_frame_masked_loss, \
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
    
        total_loss_joint_masked, motion_loss_joint_masked, rec_loss_joint_masked = loss_fn(predicted_masked_joints, data, mask)
        
        # Frame Masked
        data_frame_masked = random_frame_mask_fn(data)
        frame_masked_embeddings = backbone(data_frame_masked)
        frame_masked_embeddings = transform_embedding(frame_masked_embeddings, mask)
        predicted_masked_frames = regressor(frame_masked_embeddings)
    
        total_loss_frame_masked, motion_loss_frame_masked, rec_loss_frame_masked = loss_fn(predicted_masked_frames, data, mask)

        loss = total_loss_joint_masked + total_loss_frame_masked
        loss.backward()
        optimizer.step()

        total_joint_masked_motion_loss += motion_loss_joint_masked.detach().cpu().numpy().item()
        total_joint_masked_rec_loss += rec_loss_joint_masked.detach().cpu().numpy().item()
        total_joint_masked_loss += total_loss_joint_masked.detach().cpu().numpy().item()

        total_frame_masked_motion_loss = motion_loss_frame_masked.detach().cpu().numpy().item()
        total_frame_masked_rec_loss = rec_loss_frame_masked.detach().cpu().numpy().item()
        total_frame_masked_loss += total_loss_frame_masked.detach().cpu().numpy().item()

        total_loss += loss.detach().cpu().numpy().item()
        
    writer.add_scalar('Loss/total_joint_masked_motion_loss', total_joint_masked_motion_loss, epoch)
    writer.add_scalar('Loss/total_joint_masked_rec_loss', total_joint_masked_rec_loss, epoch)
    writer.add_scalar('Loss/total_joint_masked_loss', total_joint_masked_loss, epoch)
    
    writer.add_scalar('Loss/total_frame_masked_motion_loss', total_frame_masked_motion_loss, epoch)
    writer.add_scalar('Loss/total_frame_masked_rec_loss', total_frame_masked_rec_loss, epoch)
    writer.add_scalar('Loss/total_frame_masked_loss', total_frame_masked_loss, epoch)

    writer.add_scalar('Loss/total_loss', total_loss, epoch)

    return total_loss, total_joint_masked_loss, total_frame_masked_loss

def pretrain(
        backbone, 
        regressor, 
        dataloader, 
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
        total_loss, total_motion_loss, total_frames_loss = train_one_epoch(
            backbone,
            regressor,
            dataloader,
            loss_fn,
            optimizer,
            random_joint_mask_fn,
            random_frame_mask_fn,
            device,
            writer,
            epoch
        )

        print(f'Epoch {epoch} / {epochs}\n\ttotal_loss: {total_loss:.2f}, total_motion_loss: {total_motion_loss:.2f}, total_frames_loss: {total_frames_loss:.2f}')

        if epoch % save_freq == 0:
            torch.save(backbone.state_dict, os.path.join('assets', 'checkpoints', f'epoch_{epoch}.pth'))