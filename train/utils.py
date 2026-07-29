import torch

class RandomJointMask:
    def __init__(self, mask_ratio):
        self.ratio = mask_ratio

    def __call__(self, x):
        mask = torch.rand_like(x) >= self.ratio
        return x * mask

class RandomFrameMask:
    def __init__(self, mask_ratio, device):
        self.ratio = mask_ratio
        self.device = device

    def __call__(self, x):
        mask = (torch.rand(x.shape[1]) >= self.ratio).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(self.device)
        return x * mask

def transform_embedding(embeddings, mask):
    embeddings = embeddings.permute(0, 2, 3, 1)
    mask = mask.unsqueeze(1).unsqueeze(1).float().to(embeddings.device)
    embeddings = (embeddings * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)

    return embeddings