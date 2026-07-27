import torch

import numpy as np

class Normalizer:
    @staticmethod
    def compute_bones(paths):
        bones = []
        for path in paths:
            l = len(path)
            for i, j in zip(range(l - 1), range(1, l)):
                bones.append([path[i], path[j]])

        return bones
    
    def __init__(self, paths):
        self.bones_idx = Normalizer.compute_bones(paths)

    def __call__(self, pose: torch.Tensor):
        """
        pose: Input sequence with shape (T, J, 3)
        """

        pose = pose - pose[:, 0, :].unsqueeze(1)
        
        bones = pose[:, self.bones_idx, :]
        bone_lengths = bones[:, :, 0, :] - bones[:, :, 1, :]
        s = np.linalg.norm(bone_lengths, axis=1).mean()
        pose = pose / s

        pose = pose - pose[0, 0, 2]
        mins = torch.amin(pose, dim=(0, 1))
        maxs = torch.amax(pose, dim=(0, 1))

        pose = 2 * (pose - mins) / (maxs - mins + 1e-8) - 1
        
        return pose
