import torch
import torch.nn as nn
import torch.nn.functional as F

class CategoricalOrdinalFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, beta=0.2):
        """
        Categorical focal loss defined in https://arxiv.org/pdf/2007.08920v1.pdf
        
        Parameters:
          alpha -- weighing factor for focal loss
          gamma -- focusing parameter for modulating factor (1-p)
          beta  -- weighting factor for the ordinal penalty component
        """
        super(CategoricalOrdinalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-7

    def forward(self, y_pred, y_true):
        """
        :param y_pred: Logits tensor of shape (batch_size, num_classes)
        :param y_true: Target integer labels of shape (batch_size,)
        :return: Scalar loss (mean over the batch)
        """
        num_classes = y_pred.shape[1]
        
        # 1. Convert logits to probabilities via softmax
        probs = F.softmax(y_pred, dim=-1)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        
        # 2. Compute standard cross entropy component manually per class
        # Convert integer labels to one-hot encoding for element-wise math
        y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).float()
        cross_entropy = -y_true_one_hot * torch.log(probs)
        
        # 3. Ordinal Distance Penalty
        # Get predicted class indices
        pred_labels = torch.argmax(probs, dim=1)
        
        # Calculate absolute difference between true and predicted classes
        ordinal_dist = torch.abs(y_true - pred_labels).float()
        
        # Normalize the distance by (num_classes - 1)
        weights = ordinal_dist / (num_classes - 1)
        
        # Expand weights to match (batch_size, num_classes)
        weights_expanded = weights.unsqueeze(1).repeat(1, num_classes)
        
        # 4. Focal Loss Component
        focal_loss = self.alpha * torch.pow(1.0 - probs, self.gamma)
        
        # 5. Combine and reduce
        combined_loss = (self.beta * weights_expanded + focal_loss) * cross_entropy
        
        # Sum across classes, then mean across the batch
        return torch.mean(torch.sum(combined_loss, dim=1))