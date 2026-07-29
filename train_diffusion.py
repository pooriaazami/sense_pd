import os
import argparse
from itertools import chain
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import get_config
from train.diffusion import train_diffusion
from models import MotionAGFormer

def convert_params(params):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU
    }

    params.act_layer = act_mapper[params.act_layer]
    return params

def initiate_writer(config):
    dt = datetime.now()
    name = f'{config.experiment_name}__{dt.month}_{dt.day}_{dt.hour}_{dt.minute}'

    writer = SummaryWriter(os.path.join('assets', 'logs', name))

    writer.add_text(
        'Config Details',
        str(config)
    )

    return writer

def main():
    parser = argparse.ArgumentParser(description='This module trains the classifier model.')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Config's path"
    )

    args = parser.parse_args()
    config = get_config(args.config)
    writer = initiate_writer(config)

    convert_params(config.model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = MotionAGFormer(**config.model).to(device)
    classifier = nn.Linear(config.model.dim_rep * config.data.num_joints, NUM_CLASSES).to(device)

    backbone.load_state_dict(torch.load(config.backbone_path)())
    classifier.load_state_dict(torch.load(config.classifier_path)())

    dataset = Motion3DDataset(config.dataset)
    dataloader = DataLoader(dataset, 
                            batch_size=config.training.batch_size, 
                            shuffle=True)

    d3dp = ...
    optimizer = optimizer = optim.AdamW(chain(backbone.parameters(), classifier.parameters()), lr=config.training.lr)

    train_diffusion(backbone, classifier, d3dp, dataloader, optimizer, config.training.epochs, device, writer)

if __name__ == '__main__':
    main()