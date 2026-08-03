import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils import get_config
from models import MotionAGFormer, CBAMClassificationHead
from data import CarePDDataset
from utils import CategoricalOrdinalFocalLoss, OrdinalFocalLoss
from train.cbam_classifier import train_model as train_classifier
from train.utils import transform_embedding


NUM_CLASSES = 4
DATASETS = ['3DGait', 'BMCLab', 'PD-GaM', 'T-SDU-PD']
NUM_EPOCHS = 140

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
    name = f'{config.experiment_name}__lr__{dt.month}_{dt.day}_{dt.hour}_{dt.minute}'

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

def draw_lr_plots(
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
            'lr': 1e-8,
            'weight_decay': 0.1
        }, {
            'params': backbone.parameters(),
            'lr': 1e-8 * config.training.classification.backbone_lr_factor,
            'weight_decay': config.training.classification.weight_decay
        }],
            )
    else:
        optimizer = optim.AdamW(classifier.parameters(), 
                                lr=1e-8,
                                weight_decay=config.training.classification.weight_decay)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 10 ** (epoch / 20)
    )

    dataset = CarePDDataset(config.dataset.classification, DATASETS)
    dataloader = DataLoader(dataset, 
                            batch_size=config.training.batch_size, 
                            shuffle=True,
                            collate_fn=collate_fn)

    if config.training.loss == 'ccf':
        loss_fn = CategoricalOrdinalFocalLoss()
    elif config.training.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif config.training.loss == 'of':
        loss_fn = OrdinalFocalLoss()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for data in tqdm(dataloader):
            optimizer.zero_grad()

            seq = data['seq'].to(device)
            mask = data['mask'].to(device)
            label = data['label'].to(device).squeeze()

            if config.training.classification.freeze_backend:
                with torch.no_grad():
                    embeddings = backbone(seq)
                    embeddings = transform_embedding(embeddings, mask)
            else:
                embeddings = backbone(seq)
                embeddings = transform_embedding(embeddings, mask)

            preds = classifier(embeddings)
            loss = loss_fn(preds, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().numpy().item()

        writer.add_scalar(
            "LearningRateTuner/Loss",
            total_loss,
            epoch
        )
        writer.add_scalar(
            "LearningRateTuner/LearningRate",
            optimizer.param_groups[0]["lr"],
            epoch
        )

        scheduler.step()

def main():
    args = parse_args()
    config = get_config(args.config)
    writer = initiate_writer(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    convert_params(config.model)
    backbone = MotionAGFormer(**config.model).to(device)

    if hasattr(config, 'checkpoints'):
        if hasattr(config.checkpoints, 'backbone'):
            print('Loading pretrained backbone')
            backbone.load_state_dict(torch.load(config.checkpoints.backbone))

    if args.classifier:
        print('Training the classifier')

        draw_lr_plots(
            config=config,
            backbone=backbone,
            writer=writer,
            device=device
        )

if __name__ == '__main__':
    main()
