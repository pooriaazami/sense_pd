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

class OrdinalFocalLoss(nn.Module):
    """
    Differentiable Ordinal Focal Loss.

    L = Focal + beta * Ordinal

    Focal:
        -alpha * (1-pt)^gamma * log(pt)

    Ordinal:
        ((1 + |E[y]-y|) / C) * (-log(pt))

    where
        E[y] = sum_c c * p_c
    """

    def __init__(self, alpha=0.25, gamma=2.0, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, logits, targets):

        num_classes = logits.size(1)

        probs = F.softmax(logits, dim=1)

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.clamp(1e-8, 1.0)

        ce = -torch.log(pt)

        # -------------------------
        # Focal loss
        # -------------------------
        focal = self.alpha * (1.0 - pt).pow(self.gamma) * ce

        # -------------------------
        # Differentiable ordinal loss
        # -------------------------
        class_ids = torch.arange(
            num_classes,
            device=logits.device,
            dtype=probs.dtype
        )

        expected_class = (probs * class_ids).sum(dim=1)

        distance = torch.abs(expected_class - targets.float())

        ordinal = ((1.0 + distance) / num_classes) * ce

        loss = focal + self.beta * ordinal

        return loss.mean()

class WeightedOrdinalFocalLoss(nn.Module):
    """
    Differentiable Ordinal Focal Loss with class weighting.

    L = Focal + beta * Ordinal

    Focal:
        alpha * (1-pt)^gamma * w_y * (-log(pt))

    Ordinal:
        ((1 + |E[y]-y|) / C) * w_y * (-log(pt))
    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        beta=0.2,
        class_weights=None,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float)
            )
        else:
            self.class_weights = None

    def forward(self, logits, targets):

        num_classes = logits.size(1)

        probs = F.softmax(logits, dim=1)

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.clamp(min=1e-8)

        ce = -torch.log(pt)

        # -------------------------
        # Apply class weights
        # -------------------------
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            ce = ce * weights

        # -------------------------
        # Focal loss
        # -------------------------
        focal = self.alpha * (1.0 - pt).pow(self.gamma) * ce

        # -------------------------
        # Differentiable ordinal loss
        # -------------------------
        class_ids = torch.arange(
            num_classes,
            device=logits.device,
            dtype=probs.dtype,
        )

        expected_class = (probs * class_ids).sum(dim=1)

        distance = torch.abs(expected_class - targets.float())

        ordinal = ((1.0 + distance) / num_classes) * ce

        loss = focal + self.beta * ordinal

        return loss.mean()