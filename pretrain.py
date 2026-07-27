import os
import sys
import argparse
from datetime import datetime
from functools import partial
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_config
from models import MotionAGFormer
from data import Motion3DDataset
from train import motion_loss_fn, joints_loss_fn, RandomFrameMask, RandomJointMask

def pretext_loss(predicted_joints, ground_truth, mask, lambd):
    motion_loss = motion_loss_fn(predicted_joints, mask)
    joints_loss = joints_loss_fn(predicted_joints, ground_truth, mask)
    
    return lambd * motion_loss + joints_loss, motion_loss, joints_loss

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

    total_loss, total_motion_loss, total_frames_loss = [0] * 3
    for data, mask in dataloader:
        data = data.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        # Joint Masked
        data_joint_masked, _ = random_joint_mask_fn(data)
        joint_masked_embeddings = backbone(data_joint_masked)
        predicted_masked_joints = regressor(joint_masked_embeddings)
    
        total_loss_joint_masked, motion_loss_joint_masked, joints_loss_joint_masked = loss_fn(predicted_masked_joints, data, mask)
        
        # Frame Masked
        data_frame_masked, _ = random_frame_mask_fn(data)
        frame_masked_embeddings = backbone(data_frame_masked)
        predicted_masked_frames = regressor(frame_masked_embeddings)
    
        total_loss_frame_masked, motion_loss_frame_masked, joints_loss_frame_masked = loss_fn(predicted_masked_frames, data, mask)

        loss = total_loss_joint_masked + total_loss_frame_masked
        loss.backward()
        optimizer.step()

        total_loss += total_loss_joint_masked.detach().cpu().numpy().item() + total_loss_frame_masked.detach().cpu().numpy().item()
        total_motion_loss += motion_loss_joint_masked.detach().cpu().numpy().item() + motion_loss_frame_masked.detach().cpu().numpy().item()
        total_frames_loss += joints_loss_joint_masked.detach().cpu().numpy().item() + joints_loss_frame_masked.detach().cpu().numpy().item()

    writer.add_scalar('Loss/total_loss', total_loss, epoch)
    writer.add_scalar('Loss/total_motion_loss', total_motion_loss, epoch)
    writer.add_scalar('Loss/total_frames_loss', total_frames_loss, epoch)

    return total_loss, total_motion_loss, total_frames_loss

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
    name = f'{exp_name}__{dt.month}_{dt.day}_{dt.hour}'
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
            epoch,
            device,
            writer
        )

        print(f'Epoch {epoch} / {epochs}\n\ttotal_loss: {total_loss:.2f}, total_motion_loss: {total_motion_loss:.2f}, total_frames_loss: {total_frames_loss:.2f}')

        if epoch % save_freq == 0:
            torch.save(backbone.state_dict, os.path.join('assets', 'checkpoints', f'epoch_{epoch}.pth'))

def convert_params(params):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU
    }

    params.act_layer = act_mapper[params.act_layer]
    return params

def main(args):
    parser = argparse.ArgumentParser(description='This module trains the 3D encoder model.')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Config's path"
    )

    args = parser.parse_args()
    config = get_config(args.config)

    loss_fn = partial(pretext_loss, lambd=config.training.lambda_motion)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = MotionAGFormer(**convert_params(config.model)).to(device)
    regressor = nn.Linear(
        in_features=config.model.dim_rep,
        out_features=3
    )

    optimizer = optim.AdamW(
        chain(backbone.parameters(), regressor.parameters()),
        lr=config.training.lr
    )

    dataset = Motion3DDataset(config.dataset)
    dataloader = DataLoader(dataset, 
                            batch_size=config.training.batch_size, 
                            shuffle=True)

    random_joint_mask_fn = RandomJointMask(config.training.joint_mask_ratio)
    random_frame_mask_fn = RandomFrameMask(config.training.frame_mask_ratio)

    pretrain(
        backbone, 
        regressor, 
        dataloader, 
        loss_fn, 
        optimizer, 
        random_joint_mask_fn, 
        random_frame_mask_fn, 
        config.training.epochs, 
        device, 
        config.experiment_name,
        config.training.save_freq
    )


if __name__ == '__main__':
    main(sys.argv)