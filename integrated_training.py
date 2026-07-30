import os
import glob
import random
import argparse
from datetime import datetime
from functools import partial
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils import get_config
from models import MotionAGFormer, MixSTE2
from data import Motion3DDataset
from train import RandomFrameMask, RandomJointMask, D3DP
from utils import motion_loss_fn, joints_loss_fn

from train.pretrain import pretrain
from train.diffusion import train_diffusion_model

def parse_args():
    parser = argparse.ArgumentParser(description='This module trains the classifier model.')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Config's path"
    )

    parser.add_argument(
        '--pretrain',
        action='store_true',
    )

    parser.add_argument(
        '--diffusion',
        action='store_true',
    )

    return parser.parse_args()

def pretext_loss(predicted_joints, ground_truth, lambd):
    motion_loss = motion_loss_fn(predicted_joints, ground_truth)
    joints_loss = joints_loss_fn(predicted_joints, ground_truth)
    
    return lambd * motion_loss + joints_loss, motion_loss, joints_loss

def initiate_writer(config):
    dt = datetime.now()
    name = f'{config.experiment_name}__{dt.month}_{dt.day}_{dt.hour}_{dt.minute}'

    writer = SummaryWriter(os.path.join('assets', 'logs', name))

    writer.add_text(
        'Config Details',
        str(config)
    )

    writer.add_text(
        'Description',
        config.description
    )

    return writer

def split_motion_files(dataset_root, test_size):
    files = glob.glob(os.path.join(dataset_root, '*.pkl'))
    random.shuffle(files)

    num_train_examples = int(len(files) * (1 - test_size))

    train_files = files[:num_train_examples]
    val_files = files[num_train_examples:]

    return train_files, val_files

def pretrain_model(config, backbone, regressor, device, train_dataloader, val_dataloader, writer):
    loss_fn = partial(pretext_loss, lambd=config.training.pretraining.lambda_motion)
    optimizer = optim.AdamW(
        chain(backbone.parameters(), regressor.parameters()),
        lr=config.training.pretraining.lr
    )

    random_joint_mask_fn = RandomJointMask(config.training.pretraining.joint_mask_ratio)
    random_frame_mask_fn = RandomFrameMask(config.training.pretraining.frame_mask_ratio, 
                                           device=device)

    pretrain(
        backbone=backbone,
        regressor=regressor,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        random_frame_mask_fn=random_frame_mask_fn,
        random_joint_mask_fn=random_joint_mask_fn,
        loss_fn=loss_fn,
        device=device,
        save_freq=config.training.pretraining.save_freq,
        writer=writer,
        epochs=config.training.pretraining.epochs
    )

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)

def train_diffusion(config, 
                    backbone,
                    dim_rep, 
                    joints_left, 
                    joints_right, 
                    pose_estimator,
                    train_dataloader,
                    val_dataloader,
                    writer,
                    device,
                    save_freq):
    freeze_model(backbone)

    d3dp = D3DP(
        **config.training.d3dp,
        joints_left=joints_left, 
        joints_right=joints_right, 
        num_proposals=1, 
        sampling_timesteps=1, 
        dim_rep=dim_rep,
        num_joints=config.dataset.num_joints, 
        pose_estimator=pose_estimator,
    ).to(device)

    optimizer = optim.AdamW(d3dp.parameters(), lr=config.training.diffusion.lr)

    train_diffusion_model(
        backbone=backbone,
        d3dp=d3dp,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        epochs=config.training.diffusion.epochs,
        writer=writer,
        device=device,
        save_freq=save_freq
    )

def convert_params(params):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU
    }

    params.act_layer = act_mapper[params.act_layer]
    return params

def main():
    args = parse_args()
    config = get_config(args.config)
    writer = initiate_writer(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    convert_params(config.model)
    backbone = MotionAGFormer(**config.model).to(device)

    if config.checkpoints.backbone:
        print('Loading pretrained backbone')
        backbone.load_state_dict(torch.load(config.checkpoints.backbone))

    train_files, val_files = split_motion_files(
        config.dataset.pretraining.path, config.dataset.pretraining.test_size
    )

    train_dataset = Motion3DDataset(train_files)
    train_dataloader = DataLoader(
                        train_dataset, 
                        batch_size=config.training.batch_size, 
                        shuffle=True
                    )

    val_dataset = Motion3DDataset(val_files)
    val_dataloader = DataLoader(
                        val_dataset, 
                        batch_size=config.training.batch_size, 
                        shuffle=True
                    )

    regressor = nn.Linear(
        in_features=config.model.dim_rep,
        out_features=3
    ).to(device)

    pose_estimator = MixSTE2(
        num_frame=config.model.n_frames, 
        num_joints=config.dataset.num_joints, 
        in_chans=config.model.dim_rep, 
        embed_dim_ratio=config.mix2set.cs, 
        depth=config.mix2set.dep, 
        num_heads=config.model.num_heads, 
        mlp_ratio=config.model.mlp_ratio, 
        qkv_bias=config.model.qkv_bias, 
        qk_scale=config.model.qkv_scale,
    ).to(device)

    if args.pretrain:
        print('Training the backbone')
        pretrain_model(
            config=config, 
            backbone=backbone, 
            regressor=regressor, 
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            writer=writer
        )

    if args.diffusion:
        print('Traing the diffusion model')
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        train_diffusion(
            config=config, 
            backbone=backbone,
            dim_rep=config.model.dim_rep, 
            joints_left=joints_left, 
            joints_right=joints_right, 
            pose_estimator=pose_estimator,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            writer=writer,
            device=device,
            save_freq=config.training.diffusion.save_freq
        )

if __name__ == '__main__':
    main()