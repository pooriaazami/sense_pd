import torch

class RandomJointMask:
    def __init__(self, mask_ratio):
        self.ratio = mask_ratio

    def __call__(self, x):
        mask = torch.rand_like(x) >= self.ratio
        return x * mask

class RandomFrameMask:
    def __init__(self, mask_ratio):
        self.ratio = mask_ratio

    def __call__(self, x):
        mask = (torch.rand(x.shape[0]) >= self.ratio).unsqueeze(1).unsqueeze(1)
        return x * mask