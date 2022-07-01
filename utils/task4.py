import torch
import torch.nn as nn
from utils.task1 import label2corners
import numpy as np

# def huber_loss(error, delta):
#     abs_error = torch.abs(error)
#     quadratic = np.minimum(abs_error, delta)
#     linear = (abs_error - quadratic)
#     losses = 0.5 * quadratic**2 + delta * linear
#     return losses


class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()
        self.huber_loss = nn.HuberLoss(reduction='mean', delta = self.config['huber_delta'])
        self.dir_loss_func = nn.BCELoss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        
        mask = iou >= self.config['positive_reg_lb']
        flag = torch.sum(mask) != 0
        if flag:
            pred_valid = pred['box'][mask]
            target_valid = target[mask]
        else:
            pred_valid = 0 * pred['box']
            target_valid = 0 * target

        loss_loc = self.loss(pred_valid[:,:3], target_valid[:,:3])
        loss_size = self.loss(pred_valid[:,3:6], target_valid[:,3:6])

        if self.config['use_dir_cls']:
            pred_sin, target_sin = self.add_sin_difference(pred_valid, target_valid)
            loss_rot = self.loss(pred_sin[:,6], target_sin[:,6])

            # Calculate direction classification loss
            pred_dir = pred['dir']
            dir_targets = get_direction_target(target_valid)
            weights = mask.type_as(pred_dir)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            loss_dir = self.dir_loss_func(pred_dir, dir_targets, weights=weights)

            loss = loss_loc + 3*loss_size + loss_rot + 0.5*loss_dir

        else:
            loss_rot = self.loss(pred_valid[:,6], target_valid[:,6])
            loss = loss_loc + 3*loss_size + loss_rot

        if self.config['use_corner_loss']:
            pred_corners = label2corners(pred_valid)
            target_corners = label2corners(target_valid)
            loss_corner = self.huber_loss(pred_corners - target_corners)
            loss = loss + loss_corner
        # print("task4 reg_loss", loss)
        return loss

class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,1) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        N = pred.size(0)
        target = torch.zeros((N,1), device=pred.device)
        
        mask_pos = iou >= self.config['positive_cls_lb']
        mask_neg = iou <= self.config['negative_cls_ub']
        mask = mask_pos | mask_neg

        target[mask_pos] = 1
        target = target[mask]
        pred = pred[mask]

        loss = self.loss(pred, target)
        return loss


def add_sin_difference(boxes1, boxes2):
    """Convert the rotation difference to difference in sine function.

    Args:
        boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
            and the 7th dimension is rotation dimension.
        boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
            the 7th dimension is rotation dimension.

    Returns:
        tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
            dimensions are changed.
    """
    rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
        boxes2[..., 6:7])
    rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                        6:7])
    boxes1 = torch.cat(
        [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
    boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                        dim=-1)
    return boxes1, boxes2

def get_direction_target(reg_targets, dir_offset=0, dir_limit_offset=0.0, num_bins=2, one_hot=True):
    """Encode direction to 0 ~ num_bins-1.

    Args:
        reg_targets (torch.Tensor): Bbox regression targets.
        dir_offset (int, optional): Direction offset. Default to 0.
        dir_limit_offset (float, optional): Offset to set the direction
            range. Default to 0.0.
        num_bins (int, optional): Number of bins to divide 2*PI.
            Default to 2.
        one_hot (bool, optional): Whether to encode as one hot.
            Default to True.

    Returns:
        torch.Tensor: Encoded direction targets.
    """
    rot_gt = reg_targets[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, dir_limit_offset, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if one_hot:
        dir_targets = torch.zeros(
            *list(dir_cls_targets.shape),
            num_bins,
            dtype=reg_targets.dtype,
            device=dir_cls_targets.device)
        dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        dir_cls_targets = dir_targets
    return dir_cls_targets

def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor | np.ndarray): The value to be converted.
        offset (float, optional): Offset to set the value range.
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        (torch.Tensor | np.ndarray): Value in the range of
            [-offset * period, (1-offset) * period]
    """
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val