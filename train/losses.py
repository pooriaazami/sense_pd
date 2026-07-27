import torch

def motion_loss_fn(predicted_joints, mask):
    mask = mask.unsqueeze(-1).unsqueeze(-1)
    x = predicted_joints * mask
    x = x[:, :-1, :, :] - x[:, 1:, :, :]
    x = torch.linalg.norm(x, dim=-1)
    return x.sum()

def joints_loss_fn(predicted_joints, ground_truth, mask):
    mask = mask.unsqueeze(-1)
    return (torch.linalg.norm(predicted_joints - ground_truth, dim=-1) * mask).sum()