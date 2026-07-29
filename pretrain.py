import sys
import argparse
from functools import partial
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from utils import get_config
from models import MotionAGFormer
from data import Motion3DDataset
from train import RandomFrameMask, RandomJointMask
from train.pretrain import *
from utils import motion_loss_fn, joints_loss_fn

def pretext_loss(predicted_joints, ground_truth, mask, lambd):
    motion_loss = motion_loss_fn(predicted_joints, ground_truth, mask)
    joints_loss = joints_loss_fn(predicted_joints, ground_truth, mask)
    
    return lambd * motion_loss + joints_loss, motion_loss, joints_loss


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
    ).to(device)

    optimizer = optim.AdamW(
        chain(backbone.parameters(), regressor.parameters()),
        lr=config.training.lr
    )

    dataset = Motion3DDataset(config.dataset)
    dataloader = DataLoader(dataset, 
                            batch_size=config.training.batch_size, 
                            shuffle=True)

    random_joint_mask_fn = RandomJointMask(config.training.joint_mask_ratio)
    random_frame_mask_fn = RandomFrameMask(config.training.frame_mask_ratio, device=device)

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