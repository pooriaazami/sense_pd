from __future__ import annotations

import os
import glob
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from utils import get_config
from models import MotionAGFormer, CBAMClassificationHead
from data import CarePDDataset
from utils import CategoricalOrdinalFocalLoss, OrdinalFocalLoss, WeightedOrdinalFocalLoss
from train.cbam_classifier import train_model as train_classifier
from train.utils import transform_embedding
from utils.smplx.body_models import SMPL
from utils.smplx.lbs import vertices2joints
from data.preprocessing import generate_smpl_in_world

from data import Normalizer

def convert_params(params):
    act_mapper = {
        "gelu": nn.GELU,
        'relu': nn.ReLU
    }

    params.act_layer = act_mapper[params.act_layer]
    return params

FPS = 25
NUM_FRAMES = 81
STRIDE = 20

PATHS = [
    [10, 9, 8, 7, 0, 1, 2, 3],  # Head -> Torso -> Pelvis -> Right Leg
    [0, 4, 5, 6],               # Pelvis -> Left Leg
    [8, 11, 12, 13],            # Thorax -> Left Arm
    [8, 14, 15, 16]             # Thorax -> Right Arm
]

config = get_config(os.path.join('configs', 'cbam.yaml'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

_ = convert_params(config.model)

normalizer = Normalizer(PATHS)

smpl_model_path = os.path.join('.', 'assets', 'SMPL_NEUTRAL.pkl')
regressor = np.load(os.path.join('.', 'assets', 'J_regressor_h36m.npy'))
h36m_regressor = torch.tensor(regressor, dtype=torch.float32).to(device)

smpl_model = SMPL(model_path=smpl_model_path, num_betas=10).to(device)

def sliding_windows(seq, L, stride):
    T = seq.shape[0]

    if T < L:
        pad = L - T
    else:
        remainder = (T - L) % stride
        pad = (stride - remainder) % stride

    seq = F.pad(seq, (0, 0, 0, 0, 0, pad))

    mask = torch.ones(T + pad, dtype=torch.bool, device=seq.device)
    mask[T:] = False

    windows = seq.unfold(0, L, stride).permute(0, 3, 1, 2)
    masks = mask.unfold(0, L, stride)

    return windows, masks

def resample_sequence(seq, old_fps, new_fps):
    """
    Resample a sequence to a new FPS.

    Args:
        seq: (T, J, C) tensor
        old_fps: original FPS
        new_fps: target FPS

    Returns:
        (T_new, J, C) tensor
    """
    T, J, C = seq.shape
    T_new = max(1, round(T * new_fps / old_fps))

    # (1, J*C, T)
    x = seq.reshape(T, J * C).T.unsqueeze(0)

    x = F.interpolate(
        x,
        size=T_new,
        mode="linear",
        align_corners=False,
    )

    return x.squeeze(0).T.reshape(T_new, J, C)

def convert(dataset_path, save_path, normalizer, smpl_model, h36m_regressor):
    dataset_name = dataset_path.split('/')[-1].replace('_canonical.pkl', '')
    dataset = pd.read_pickle(dataset_path)

    for key in dataset.keys():
        for seq_key in dataset[key].keys():
            seq = dataset[key][seq_key]
            label = dataset[key][seq_key]['UPDRS_GAIT']
            fps = dataset[key][seq_key]['fps']

            convert_seq(seq, key, seq_key, normalizer, smpl_model, h36m_regressor, fps, label, save_path, dataset_name)

def get_model():
    backbone = MotionAGFormer(**config.model).to(device)
    
    classifier = CBAMClassificationHead(
            in_channels=config.model.dim_rep,
            r=config.cbam.r,
            conv_channels=config.classifier.conv_channels,
            num_joints=config.dataset.num_joints,
            seq_length=config.model.n_frames,
            num_classes=config.dataset.num_classes
        ).to(device)

    backbone.load_state_dict(torch.load(os.path.join('assets', 'backbone_epoch_60.pth', map_location=device)))
    classifier.load_state_dict(torch.load(os.path.join('assets', 'best_cbam_classifier_epoch_4.pth', map_location=device)))

    return backbone, classifier

def predict(data: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, int]]:
    """Return predictions[subject_id][walk_id] = UPDRS class in {0, 1, 2, 3}."""
    backbone, classifier = get_model()

    backbone.eval()
    classifier.eval()
    
    predictions: dict[str, dict[str, int]] = {}

    for subject_id, walks in data.items():
        subject_key = str(subject_id)
        predictions[subject_key] = {}
        for walk_id, sample in walks.items():
            out_world, _ = generate_smpl_in_world(smpl_model, sample)
            vertices_world = out_world.vertices
            
            fps = sample['fps']
            seq = vertices2joints(h36m_regressor, vertices_world)
            seq = seq.detach()
            seq = normalizer(seq)
            seq = resample_sequence(seq, fps, FPS)
            windows, masks = sliding_windows(seq, NUM_FRAMES, STRIDE)

            with torch.no_grad():
                embeddings = backbone(windows)
                embeddings = transform_embedding(embeddings, masks)
                preds = classifier(embeddings)
                preds = preds.softmax(1).argmax(1).detach().cpu().numpy().tolist()

            pred = Counter(preds).most_common()[0][0]
            
            predictions[subject_key][str(walk_id)] = pred

    return predictions

def main() -> None:
    """Optional local runner; CodaBench uses predict(data) directly."""
    parser = argparse.ArgumentParser(description="Run local MoCha baseline inference.")
    parser.add_argument("--input", required=True, type=Path, help="Challenge input .pkl file")
    parser.add_argument("--output", required=True, type=Path, help="Where to save predictions.json")
    args = parser.parse_args()

    with args.input.open("rb") as f:
        data = pickle.load(f)
    predictions = predict(data)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"Saved predictions for {sum(len(walks) for walks in predictions.values())} samples.")

if __name__ == '__main__':
    main()