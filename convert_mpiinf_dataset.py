# Inspired from https://github.com/TaatiTeam/MotionAGFormer/blob/master/data/preprocess/data_to_npz_3dhp.py

import os
import glob
import pickle

import torch

import numpy as np
from scipy.io import loadmat

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MPIINF_DATASET = glob.glob(os.path.join('datasets', 'MPIINF', '*', '*', 'annot.mat'))

def mpii_get_sequence_info(subject_id, sequence):
    switcher = {
        "1 1": [6416,25],
        "1 2": [12430,50],
        "2 1": [6502,25],
        "2 2": [6081,25],
        "3 1": [12488,50],
        "3 2": [12283,50],
        "4 1": [6171,25],
        "4 2": [6675,25],
        "5 1": [12820,50],
        "5 2": [12312,50],
        "6 1": [6188,25],
        "6 2": [6145,25],
        "7 1": [6239,25],
        "7 2": [6320,25],
        "8 1": [6468,25],
        "8 2": [6054,25],

    }
    return switcher.get(subject_id+" "+sequence)

def convert_mat_to_h36m_joint(mat_path, s, seq):
    cam_set = [0, 1, 2, 4, 5, 6, 7, 8]
    joint_set = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]
    joint_moves = [14, 8, 9, 10, 11, 12, 13, 15, 1, 16, 0, 2, 3, 4, 5, 6, 7]

    data = loadmat(mat_path)

    dic_seq={}
    frames, fps = mpii_get_sequence_info(s, seq)

    cameras = data['cameras'][0]
    for cam_idx in range(len(cameras)):
        assert cameras[cam_idx] == cam_idx

    data_3d = data['univ_annot3'][cam_set]

    dic_cam = {}
    for cam_idx in range(len(data_3d)):
        data_3d_cam = data_3d[cam_idx][0]

        data_3d_cam = data_3d_cam.reshape(data_3d_cam.shape[0], 28, 3)

        data_3d_select = data_3d_cam[:frames, joint_set]
        data_3d_select[:, :, 1] *= -1
        data_3d_select = data_3d_select[joint_moves]

        dic_data = {"data_3d": data_3d_select}

        dic_cam.update({str(cam_set[cam_idx]):dic_data})

    dic_seq.update({s + ' ' + seq:[dic_cam, fps]})

    return dic_seq

def convert(files):
    destination_root = os.path.join('datasets', 'MPIINF', 'converted')

    for file in files:
        print(f'[START]: {file}')
        x = file.replace('Seq', '').replace('S', '').split('/')
        s, seq = x[2], x[3]
        joints = convert_mat_to_h36m_joint(file, s, seq)

        destination = os.path.join(destination_root, s + '_' + seq + '.pkl')
        np.savez_compressed(destination, data=joints)

        print('[DONE]')

if __name__ == '__main__':
    convert(MPIINF_DATASET)
        