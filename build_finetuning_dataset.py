import os
import glob
import pickle
import warnings

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from utils.smplx.body_models import SMPL
from utils.smplx.lbs import vertices2joints
from data.preprocessing import generate_smpl_in_world

from data import Normalizer

warnings.filterwarnings('ignore')

ROOT = os.path.join('.', 'datasets', 'CARE-PD', 'Canonicalized_SMPL_pickles')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FPS = 25
NUM_FRAMES = 81
STRIDE = 20

PATHS = [
    [10, 9, 8, 7, 0, 1, 2, 3],  # Head -> Torso -> Pelvis -> Right Leg
    [0, 4, 5, 6],               # Pelvis -> Left Leg
    [8, 11, 12, 13],            # Thorax -> Left Arm
    [8, 14, 15, 16]             # Thorax -> Right Arm
]

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

def convert_seq(seq, normalizer, smpl_model, h36m_regressor, fps, label, root_path, dataset_name):
    out_world, _ = generate_smpl_in_world(smpl_model, seq)
    vertices_world = out_world.vertices

    seq = vertices2joints(h36m_regressor, vertices_world)
    seq = seq.detach()
    seq = normalizer(seq)
    seq = resample_sequence(seq, fps, FPS)

    for i, data in enumerate(seq):
        print('\tSeq {i + 1}')
        destination = os.path.join(
            root_path, f'{dataset_name}_seq_{i + 1}.pkl'
        )

        data = {
            'seq': data.cpu().numpy(),
            'label': label
        }
        
        with open(destination, 'wb') as file:
            pickle.dump(data, file)

def convert(dataset_path, save_path, normalizer, smpl_model, h36m_regressor):
    dataset_name = dataset_path.split('/')[-1].replace('_canonical.pkl', '')
    dataset = pd.read_pickle(dataset_path)

    for key in dataset.keys():
        for seq_key in dataset[key].keys():
            seq = dataset[key][seq_key]
            label = dataset[key][seq_key]['UPDRS_GAIT']
            fps = dataset[key][seq_key]['fps']

            convert_seq(seq, normalizer, smpl_model, h36m_regressor, fps, label, save_path, dataset_name)

def main():
    normalizer = Normalizer(PATHS)

    smpl_model_path = os.path.join('.', 'assets', 'SMPL_NEUTRAL.pkl')
    regressor = np.load(os.path.join('.', 'assets', 'J_regressor_h36m.npy'))
    h36m_regressor = torch.tensor(regressor, dtype=torch.float32).to(DEVICE)

    smpl_model = SMPL(model_path=smpl_model_path, num_betas=10).to(DEVICE)

    datasets = ['PD-GaM', 'BMCLab', '3DGait', 'T-SDU-PD']
    files = [os.path.join(ROOT, dataset + '_canonical' + '.pkl') for dataset in datasets]

    for file in files:
        print(f'[START] {file}')

        convert(
            file, 
            os.path.join('datasets', 'CARE-PD', 'preprocessed'),
            normalizer,
            smpl_model,
            h36m_regressor
        )

    print('[DONE]')    


if __name__ == '__main__':
    main()
