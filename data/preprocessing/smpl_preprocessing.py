import torch

import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_smpl_in_world(smpl_model, sequence):
    frame_number = sequence['pose'].shape[0]

    if sequence['beta'].shape[0] != frame_number:
        sequence['beta'] = np.tile(sequence['beta'], (frame_number, 1))
    
    pose_world    = sequence['pose'].reshape(-1, 24, 3)  # (num_frames, 24, 3)
    betas         = sequence['beta']  # (num_frames, 10)
    world_trans   = sequence['trans']  # (num_frames, 3)
    
    # pose_world    = pose_world[down::down_sample_rate,...]  # (num_frames, 24, 3)  # start from down and they select every down_sample_rate
    pose_world_out = pose_world.copy()
    # betas         = betas[down::down_sample_rate,...]  # (num_frames, 10)
    # world_trans   = world_trans[down::down_sample_rate,...] 
    
    frame_number = pose_world.shape[0]
    
    # Extract global orientation (index 0) and body pose (indices 1-23)
    global_orient = torch.tensor(pose_world[:, 0:1, :], dtype=torch.float32)  # (num_frames, 1, 3)
    body_pose     = torch.tensor(pose_world[:, 1:24, :], dtype=torch.float32)  # (num_frames, 23, 3)
    betas         = torch.tensor(betas, dtype=torch.float32)  # (num_frames, 10)

    # Ensure everything is on the same device
    global_orient = global_orient.reshape(frame_number, -1).to(DEVICE)
    body_pose = body_pose.reshape(frame_number, -1).to(DEVICE)
    betas = betas.reshape(frame_number, -1).to(DEVICE)
    world_trans = torch.tensor(world_trans, dtype=torch.float32).to(DEVICE)  # Ensure on same device

    # Zero values for face, hands, and expression
    zero_pose = torch.zeros((frame_number, 3), dtype=torch.float32).to(DEVICE)
    zero_hand_pose = torch.zeros((frame_number, 15, 3), dtype=torch.float32).to(DEVICE)
    zero_expression = torch.zeros((frame_number, 10), dtype=torch.float32).to(DEVICE)

    # Generate SMPL output
    out = smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, 
                     jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                     left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose,
                     expression=zero_expression)

    # Apply global translation (world_trans) to the output vertices
    out.vertices += world_trans[:, None, :]  # Broadcasting (num_frames, 1, 3) to (num_frames, num_vertices, 3)

    return out, pose_world_out