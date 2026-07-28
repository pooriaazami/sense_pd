import os
from datetime import datetime
import argparse
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import MotionAGFormer
from data import CarePDDataset
from utils import CategoricalOrdinalFocalLoss
from train.classification import train_model
from utils import get_config

NUM_CLASSES = 4
DATASETS = ['3DGait', 'BMCLab', 'PD-GaM', 'T-SDU-PD']

def convert_params(params):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU
    }

    params.act_layer = act_mapper[params.act_layer]
    return params

def collate_fn(batch):
    """
    Collate function to stack data from the SlicedPDDataset into batches.
    
    Args:
        batch: A list of dictionaries returned by __getitem__.
    """
    # Use default_collate logic, but explicitly handle the dictionary keys
    # print(batch)
    return {
        'seq': torch.stack([item['seq'].detach() for item in batch]),
        'label': torch.stack([item['label'].detach() for item in batch]),
        'mask': torch.stack([item['mask'].detach() for item in batch])
    }

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

    dt = datetime.now()
    name = f'{config.experiment_name}__{dt.month}_{dt.day}_{dt.hour}_{dt.minute}'
    writer = SummaryWriter(os.path.join('assets', 'logs', name))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = MotionAGFormer(**convert_params(config.model)).to(device)
    classifier = nn.Linear(config.model.dim_rep * config.data.num_joints, NUM_CLASSES).to(device)

    optimizer = optim.AdamW(chain(backbone.parameters(), classifier.parameters()), lr=config.training.lr)

    loss_fn = CategoricalOrdinalFocalLoss()

    if config.backbone_path:
        backbone.load_state_dict(torch.load(config.backbone_path)())

    for val_dataset_name in DATASETS:
        train_datasets = DATASETS.copy()
        train_datasets.remove(val_dataset_name)

        train_dataset = CarePDDataset(config.dataset, train_datasets)
        val_dataset = CarePDDataset(config.dataset, [val_dataset_name])

        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=config.training.batch_size, 
                                    shuffle=True,
                                    collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=config.training.batch_size, 
                                    shuffle=True,
                                    collate_fn=collate_fn)

        train_model(
            train_dataloader,
            val_dataloader,
            backbone,
            classifier,
            loss_fn,
            optimizer,
            config.training.epochs,
            writer,
            device,
            val_dataset_name,
        )

if __name__ == '__main__':
    main()
