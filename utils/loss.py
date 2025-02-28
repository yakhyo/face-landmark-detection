import torch
from torch import nn


class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='none')  # Element-wise L2 loss

    def forward(
        self,
        attribute_gt: torch.Tensor,
        landmark_gt: torch.Tensor,
        euler_angle_gt: torch.Tensor,
        angle: torch.Tensor,
        landmarks: torch.Tensor,
        train_batchsize: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Compute angle-based weighting
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), dim=1)

        # Compute attribute-based weighting
        attributes_w_n = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes_w_n, dim=0)

        # Avoid division by zero
        mat_ratio = torch.reciprocal(torch.clamp_min(mat_ratio, 1e-6))  # Ensure non-zero denominator
        weight_attribute = torch.sum(attributes_w_n * mat_ratio, dim=1)

        # Compute L2 loss
        l2_distance = torch.sum(self.l2_loss(landmarks, landmark_gt), dim=1)

        # Compute final loss
        weighted_loss = torch.mean(weight_angle * weight_attribute * l2_distance)
        mean_l2_loss = torch.mean(l2_distance)

        return weighted_loss, mean_l2_loss


def smoothL1(y_true: torch.Tensor, y_pred: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Smooth L1 Loss with adjustable beta parameter"""
    mae = torch.abs(y_true - y_pred)
    loss = torch.where(mae > beta, mae - 0.5 * beta, 0.5 * mae.pow(2) / beta)
    return loss.mean()


def wing_loss(y_true: torch.Tensor, y_pred: torch.Tensor, w: float = 10.0, epsilon: float = 2.0, n_landmark: int = 106) -> torch.Tensor:
    """Wing loss for landmark localization"""
    y_pred = y_pred.view(-1, n_landmark, 2)
    y_true = y_true.view(-1, n_landmark, 2)

    x = y_true - y_pred
    c = w * (1 - torch.log1p(w / epsilon))  # log1p(x) is numerically stable for log(1 + x)
    
    abs_x = torch.abs(x)
    losses = torch.where(abs_x < w, w * torch.log1p(abs_x / epsilon), abs_x - c)
    
    return losses.sum(dim=[1, 2]).mean()
