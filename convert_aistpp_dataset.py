import os
import glob

import torch

import numpy as np
import pandas as pd

from utils.smplx.body_models import SMPL
from utils.smplx.lbs import vertices2joints
from data.preprocessing import generate_smpl_in_world

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AIST_FILES = glob.glob(os.path.join('datasets', 'AIST', '*.pkl'))

def convert_pickle_to_h36m_joint(pickle_path, smpl_model, h36m_regressor):
    pickle = pd.read_pickle(pickle_path)

    out_world, pose_world = generate_smpl_in_world(smpl_model, {
        'pose': pickle['smpl_poses'], 
        'trans': pickle['smpl_trans'], 
        'beta': torch.zeros((10,))})
    
    vertices_world = out_world.vertices
    h36m_joints_world = vertices2joints(h36m_regressor, vertices_world)

    return h36m_joints_world

def convert(files, smpl_model_path, regressor_path):
    smpl_model = SMPL(model_path=smpl_model_path, num_betas=10).to(DEVICE)
    destination_root = os.path.join('datasets', 'AIST', 'converted')
    regressor = np.load(regressor_path)
    h36m_regressor = torch.tensor(regressor, dtype=torch.float32).to(DEVICE)

    for file in files:
        print(f'[START]: {file}')
        joints = convert_pickle_to_h36m_joint(file, smpl_model, h36m_regressor).detach().cpu().numpy()

        destination = os.path.join(destination_root, file.split('/')[-1]).replace('.pkl', '')
        np.savez_compressed(destination, data=joints)
        
        print('[DONE]')

if __name__ == '__main__':
    convert(
        glob.glob(os.path.join('datasets', 'AIST', '*.pkl')),
        os.path.join('assets', 'SMPL_NEUTRAL.pkl'),
        os.path.join('assets', 'J_regressor_h36m.npy')
    )
        