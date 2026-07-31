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
from models import MotionAGFormer, MixSTE2, Classifier
from data import Motion3DDataset, CarePDDataset
from train import RandomFrameMask, RandomJointMask, D3DP
from utils import motion_loss_fn, joints_loss_fn, CategoricalOrdinalFocalLoss

from train.pretrain import pretrain
from train.diffusion import train_diffusion_model
from train.classification import train_model as train_classifier

NUM_CLASSES = 4
DATASETS = ['3DGait', 'BMCLab', 'PD-GaM', 'T-SDU-PD']

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

    parser.add_argument(
        '--classifier',
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
                    regressor,
                    d3dp,
                    train_dataloader,
                    val_dataloader,
                    writer,
                    device
                ):
    
    freeze_model(backbone)

    optimizer = optim.AdamW(d3dp.parameters(), 
                            lr=config.training.diffusion.lr, 
                            weight_decay=config.training.diffusion.weight_decay
                        )

    train_diffusion_model(
        backbone=backbone,
        d3dp=d3dp,
        regressor=regressor,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        epochs=config.training.diffusion.epochs,
        writer=writer,
        device=device,
        save_freq=config.training.diffusion.save_freq
    )

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

def train_LODO_classifier(
                    config,
                    backbone, 
                    d3dp, 
                    writer,
                    device
                ):

    freeze_model(backbone)
    freeze_model(d3dp)

    logs = {}
    for val_dataset_name in DATASETS:
        print(f'Val Dataset: {val_dataset_name}')

        classifier = Classifier(
            num_joints=config.dataset.num_joints,
            seq_length=config.model.n_frames,
            rep_dim=config.model.dim_rep,
            hidden_dim=config.classifier.hidden_dim,
            num_classes=config.dataset.num_classes,
            dropout_rate=config.classifier.dropout_rate,
        ).to(device)

        optimizer = optim.AdamW(classifier.parameters(), 
                                lr=config.training.classification.lr)

        train_datasets = DATASETS.copy()
        train_datasets.remove(val_dataset_name)

        train_dataset = CarePDDataset(config.dataset.classification, train_datasets)
        val_dataset = CarePDDataset(config.dataset.classification, [val_dataset_name])

        train_dataloader = DataLoader(train_dataset, 
                                    batch_size=config.training.batch_size, 
                                    shuffle=True,
                                    collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, 
                                    batch_size=config.training.batch_size, 
                                    shuffle=True,
                                    collate_fn=collate_fn)

        loss_fn = CategoricalOrdinalFocalLoss()

        val_log = train_classifier(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            backbone=backbone,
            d3dp=d3dp,
            classifier=classifier,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=config.training.classification.epochs,
            log_writer=writer,
            device=device,
            dataset_name=val_dataset_name,
        )

        logs[val_dataset_name] = val_log

    return logs

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
    backbone = MotionAGFormer(**config.model).to(device)

    regressor = nn.Linear(
            in_features=config.model.dim_rep,
            out_features=3
        ).to(device)

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

    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    d3dp = D3DP(
        **config.training.d3dp,
        joints_left=joints_left, 
        joints_right=joints_right, 
        num_proposals=1, 
        sampling_timesteps=1, 
        dim_rep=config.model.dim_rep,
        num_joints=config.dataset.num_joints, 
        pose_estimator=pose_estimator,
    ).to(device)

    if hasattr(config, 'checkpoints'):
        if hasattr(config.checkpoints, 'backbone'):
            print('Loading pretrained backbone')
            backbone.load_state_dict(torch.load(config.checkpoints.backbone))

        if hasattr(config.checkpoints, 'regressor'):
            print('Loading pretrained regressor')
            regressor.load_state_dict(torch.load(config.checkpoints.regressor))

        if hasattr(config.checkpoints, 'd3dp'):
            print('Loading pretrained d3dp')
            d3dp.load_state_dict(torch.load(config.checkpoints.d3dp))

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

        train_diffusion(
            config=config, 
            backbone=backbone,
            regressor=regressor,
            d3dp=d3dp,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            writer=writer,
            device=device,
        )

    if args.classifier:
        print('Training the classifier')

        logs = train_LODO_classifier(
            config=config,
            backbone=backbone, 
            d3dp=d3dp, 
            writer=writer,
            device=device
        )

        log_final_report(logs, writer)

if __name__ == '__main__':
    main()