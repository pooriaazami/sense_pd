import torch
import torch.nn.functional as F

def motion_loss_fn(predicted_joints, ground_truth):
    # mask = mask.unsqueeze(-1).unsqueeze(-1)

    # predicted_joints = predicted_joints * mask
    predicted_motion = predicted_joints[:, :-1, :, :] - predicted_joints[:, 1:, :, :]
    predicted_motion = torch.linalg.norm(predicted_motion, dim=-1)

    # ground_truth = ground_truth * mask
    real_motion = ground_truth[:, :-1, :, :] - ground_truth[:, 1:, :, :]
    real_motion = torch.linalg.norm(real_motion, dim=-1)

    motion_loss = F.mse_loss(predicted_motion, real_motion)
    
    return motion_loss

def joints_loss_fn(predicted_joints, ground_truth):
    # mask = mask.unsqueeze(-1)
    return (torch.linalg.norm(predicted_joints - ground_truth, dim=-1)).mean()