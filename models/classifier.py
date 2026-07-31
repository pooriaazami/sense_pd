import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_joints, seq_length, rep_dim, hidden_dim, num_classes, dropout_rate):
        super().__init__()

        self.backbone_head = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.SiLU()
        )
        self.diffusion_head = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.SiLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * seq_length * num_joints, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, backbone_output, diffusion_output):
        backbone_output = self.backbone_head(backbone_output)
        diffusion_output = self.diffusion_head(diffusion_output)

        x = torch.concat([backbone_output, diffusion_output], dim=-1)
        x = x.flatten(1)

        x = self.classifier(x)

        return x
