import os
import glob
import pickle

import torch
import torch.nn.functional as F

import numpy as np

from data import Normalizer

AIST = glob.glob(os.path.join('datasets', 'AIST', 'converted', '*.npz'))
MPIINF = glob.glob(os.path.join('datasets', 'MPIINF', 'converted', '*.npz'))
DATASET_ROOT = os.path.join('datasets', 'preprocessed')

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

def squeeze_dataset(data):
    key = list(data.keys())[0]
    seqs = data[key][0]
    fps = data[key][1]

    dataset = []
    cameras = list(seqs.keys())
    for camera in cameras:
        dataset.append(seqs[camera]['data_3d'])

    return dataset, fps

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

def convert_seq(seq, normalizer, fps, root):
    mpi_data = normalizer(seq)

    if fps != FPS:
        mpi_data = resample_sequence(mpi_data, fps, FPS)

    windows, masks = sliding_windows(mpi_data, NUM_FRAMES, STRIDE)

    for i, (window, mask) in enumerate(zip(windows, masks)):
        print(f'\tslice {i + 1}')

        windows = window.cpu().numpy()
        mask = mask.cpu().numpy()

        data = {'data': windows, 'mask': mask}

        destination = os.path.join(
            DATASET_ROOT,
            root.split('/')[-1].replace('.npz', '') + '.pkl'
        )

        with open(destination, 'wb') as file:
            pickle.dump(data, file)

def convert_mpiinf_item(files, normalizer):
    for file_path in files:
        print(f'[START]: {file_path}')

        with open(file_path, 'rb') as file:
            mpi_data = np.load(file, allow_pickle=True)['data'].item()

        mpi_data, fps = squeeze_dataset(mpi_data)

        for dataset in mpi_data:
            dataset = torch.from_numpy(dataset)
            convert_seq(dataset, normalizer, fps, file_path)

        print(f'[DONE]')

def convert_aist_item(files, normalizer):
    for file_path in files:
        print(f'[START]: {file_path}')

        with open(file_path, 'rb') as file:
            aist_data = np.load(file)['data'].item()

        aist_data = torch.from_numpy(aist_data)
        aist_data, fps = squeeze_dataset(aist_data)
        convert_seq(aist_data, normalizer, fps, file_path)

        print(f'[DONE]')

def main():
    normalizer = Normalizer(PATHS)

    print('Processing MPI-INF')
    print(MPIINF)
    convert_mpiinf_item(MPIINF, normalizer)

    print('Processing AIST++')
    print(AIST)
    convert_aist_item(AIST, normalizer)

if __name__ == '__main__':
    main()