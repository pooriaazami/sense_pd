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
from models import MotionAGFormer
from data import Motion3DDataset
from train import RandomFrameMask, RandomJointMask
from utils import motion_loss_fn, joints_loss_fn

from train.pretrain import pretrain

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
        default=True
    )

    parser.add_argument(
        '--diffusion',
        action='store_true',
        default=True
    )

    return parser.parse_args()

def pretext_loss(predicted_joints, ground_truth, mask, lambd):
    motion_loss = motion_loss_fn(predicted_joints, ground_truth, mask)
    joints_loss = joints_loss_fn(predicted_joints, ground_truth, mask)
    
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

def pretrain_model(config, backbone, regressor, device, train_dataloader, val_dataloader):
    loss_fn = partial(pretext_loss, lambd=config.training.pretraining.lambda_motion)
    optimizer = optim.AdamW(
        chain(backbone.parameters(), regressor.parameters()),
        lr=config.training.lr
    )

    random_joint_mask_fn = RandomJointMask(config.training.pretraining.joint_mask_ratio)
    random_frame_mask_fn = RandomFrameMask(config.training.pretraining.frame_mask_ratio, 
                                           device=device)

    pretrain(
        exp_name=config.experiment_name,
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
        epochs=config.training.pretraining.epochs
    )

def train_diffusion():
    ...

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
    initiate_writer(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    convert_params(config.model)
    backbone = MotionAGFormer(**config.model).to(device)

    train_files, val_files = split_motion_files(
        config.dataset.pretraining.path, config.dataset.pretraining.test_size
    )

    train_dataset = Motion3DDataset(train_files)
    train_dataloader = DataLoader(train_dataset, 
                            batch_size=config.training.batch_size, 
                            shuffle=True)

    val_dataset = Motion3DDataset(val_files)
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=config.training.batch_size, 
                            shuffle=True)

    regressor = nn.Linear(
        in_features=config.model.dim_rep,
        out_features=3
    ).to(device)

    if args.pretrain:
        pretrain_model(
            config=config, 
            backbone=backbone, 
            regressor=regressor, 
            device=device,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )

    if args.diffusion:
        train_diffusion()

if __name__ == '__main__':
    main()