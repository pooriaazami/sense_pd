import os
import random
import argparse
from functools import partial
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from utils import get_config
from models import MotionAGFormer, CBAMClassificationHead
from data import CarePDDataset
from utils import CategoricalOrdinalFocalLoss, OrdinalFocalLoss, WeightedOrdinalFocalLoss
from train.cbam_classifier import train_model as train_classifier


NUM_CLASSES = 4
DATASETS = ['3DGait', 'BMCLab', 'PD-GaM', 'T-SDU-PD']
TEST_SIZE = .1

def train_test_splitter(files, test_size, split):
    if not hasattr(train_test_splitter, 'df'):
        df = []
        for path in files:
            file = pd.read_pickle(path)
            label = torch.tensor([file['label']])
            df.append({
                'path': path,
                'label': label
            })

        df = pd.DataFrame.from_dict(df)
        train, val = train_test_split(df, stratify=df['label'], test_size=test_size)

        train_test_splitter.train = train
        train_test_splitter.val = val

    if split == 'train':
        return train_test_splitter.train['path'].to_list()
    if split == 'val':
        return train_test_splitter.val['path'].to_list()

        

def parse_args():
    parser = argparse.ArgumentParser(description='This module trains the classifier model.')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Config's path"
    )

    parser.add_argument(
        '--classifier',
        action='store_true',
    )

    return parser.parse_args()

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

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)

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

def calculate_loss_weights(weights, beta):
    return (1 - beta) / (1 - beta ** weights)

def train_vanilla_classifier(
                    config,
                    backbone, 
                    writer,
                    device
                ):

    if config.training.classification.freeze_backend:
        print('Freezing the backbone')
        freeze_model(backbone)

    classifier = CBAMClassificationHead(
            in_channels=config.model.dim_rep,
            r=config.cbam.r,
            conv_channels=config.classifier.conv_channels,
            num_joints=config.dataset.num_joints,
            seq_length=config.model.n_frames,
            num_classes=config.dataset.num_classes
        ).to(device)

    if not config.training.classification.freeze_backend:
        backbone = MotionAGFormer(**config.model).to(device)

        if hasattr(config, 'checkpoints'):
            if hasattr(config.checkpoints, 'backbone'):
                print('Loading pretrained backbone')
                backbone.load_state_dict(torch.load(config.checkpoints.backbone))

    
        optimizer = optim.AdamW([{
            'params': classifier.parameters(),
            'lr': config.training.classification.lr,
            'weight_decay': 0.1
        }, {
            'params': backbone.parameters(),
            'lr': config.training.classification.lr * config.training.classification.backbone_lr_factor,
            'weight_decay': config.training.classification.weight_decay
        }],
            )
    else:
        optimizer = optim.AdamW(classifier.parameters(), 
                                lr=config.training.classification.lr,
                                weight_decay=config.training.classification.weight_decay)

    splitter = partial(train_test_splitter, test_size=TEST_SIZE)
    train_dataset = CarePDDataset(config.dataset.classification, 
                                  DATASETS, 
                                  selector=splitter,
                                  split='train')
    val_dataset = CarePDDataset(config.dataset.classification,
                                 DATASETS,
                                 selector=splitter,
                                 split='val')

    print(f'Using {len(train_dataset)} samples for training and {len(val_dataset)} samples for validation')
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=config.training.batch_size, 
                                shuffle=True,
                                collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=config.training.batch_size, 
                                shuffle=True,
                                collate_fn=collate_fn)

    if config.training.loss == 'ccf':
        loss_fn = CategoricalOrdinalFocalLoss()
    elif config.training.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif config.training.loss == 'of':
        loss_fn = OrdinalFocalLoss()
    elif config.training.loss == 'wof':
        class_weghts = np.array(config.training.weights)
        class_weghts = calculate_loss_weights(class_weghts, config.training.class_weight_beta)

        loss_fn = WeightedOrdinalFocalLoss(
            class_weights=class_weghts
        ).to(device)

    train_classifier(
        backbone=backbone, 
        classifier=classifier, 
        optimizer=optimizer, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        log_writer=writer,
        epochs=config.training.classification.epochs,
        device=device,
        dataset_name='combined',
        freeze_backbone=config.training.classification.freeze_backend,
        save_best=True,
        save_path=config.save_path_root
    )

def log_final_report(logs, writer):
    accuracy, f1, recall, precision = [0] * 4

    for key in logs.keys():
        accuracy += logs[key]['accuracy']
        f1 += logs[key]['f1']
        recall += logs[key]['recall']
        precision += logs[key]['precision']

    num_datasets = len(logs.keys())
    accuracy /= num_datasets
    f1 /= num_datasets
    recall /= num_datasets
    precision /= num_datasets

    report = F'Aggregation: f1: {f1:.2f}, accuracy: {accuracy:.2f}, recall: {recall:.2f}, precision: {precision:.2f}'
    writer.add_text('Final Report', report)

def main():
    args = parse_args()
    config = get_config(args.config)
    writer = initiate_writer(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    convert_params(config.model)
    os.makedirs(config.save_path_root, exist_ok=True)   

    backbone = MotionAGFormer(**config.model).to(device)

    if hasattr(config, 'checkpoints'):
        if hasattr(config.checkpoints, 'backbone'):
            print('Loading pretrained backbone')
            backbone.load_state_dict(torch.load(config.checkpoints.backbone))

    if args.classifier:
        print('Training the classifier')

        train_vanilla_classifier(
            config=config,
            backbone=backbone, 
            writer=writer,
            device=device
        )

if __name__ == '__main__':
    main()