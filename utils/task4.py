import torch
import torch.nn as nn
from utils.task1 import label2corners
import numpy as np
import torch.nn.functional as F

def box2corners(label):
    '''
    Task 1
    input
        label (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
    output
        corners (N,8,3) corner coordinates in the rectified reference frame
    '''
    corners = []
    for bbox in label:
        h = bbox[3]
        w = bbox[4]
        l = bbox[5]
        x = bbox[0]
        y = bbox[1]
        z = bbox[2]
        ry = bbox[6]

        '''
              1 -------- 0
             /|         /|
            2 -------- 3 .
            | |        | |
            . 5 -------- 4
            |/         |/
            6 -------- 7
        ''' 
        corner_3d = torch.Tensor([[l/2, -h,  w/2],
                    [-l/2, -h,  w/2],
                    [-l/2, -h, -w/2],
                    [ l/2, -h, -w/2],
                    [ l/2,  0,  w/2],
                    [-l/2,  0,  w/2],
                    [-l/2,  0, -w/2],
                    [ l/2,  0, -w/2]])

        corner_3d = torch.cat((corner_3d, torch.ones((8,1))), dim = 1)
        corner_3d = corner_3d.t()

        cos_ry = torch.cos(ry)
        sin_ry = torch.sin(ry)
        T = torch.Tensor([[ cos_ry, 0, sin_ry, x],
            [0,       1, 0,      y],
            [-sin_ry, 0, cos_ry, z],
            [0,       0, 0,      1]])

        corner_3d = torch.matmul(T, corner_3d)
        corner_3d = corner_3d.t()
        corner_3d = corner_3d[:, :3]
        corners.append(corner_3d)
    corners = torch.stack(corners, dim = 0)
    assert corners.shape == (len(label), 8, 3)

    return corners

# def huber_loss(error, delta):
#     abs_error = torch.abs(error)
#     quadratic = torch.minimum(abs_error, delta)
#     linear = (abs_error - quadratic)
#     losses = 0.5 * quadratic**2 + delta * linear
#     return losses

class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.
        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        #input = input.permute(1, 0)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        loss = loss.sum()/weights.sum()

        return loss

class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()
        #self.huber_loss = nn.HuberLoss(reduction='mean', delta = self.config['huber_delta'])
        self.wce = WeightedCrossEntropyLoss()

    def forward(self, pred, target, iou, anchor=None):
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
            pred_sin, target_sin = add_sin_difference(pred_valid, target_valid)
            loss_rot = self.loss(pred_sin[:,6], target_sin[:,6])

            # Calculate direction classification loss
            pred_dir = pred['dir']
            dir_targets = get_direction_target(target, anchor)
            weights = mask.type_as(pred_dir)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            #self.dir_loss_func = nn.CrossEntropyLoss(weight = weights)
            loss_dir = self.wce(pred_dir, dir_targets, weights)

            # Check for corner case
            if not flag:
                loss_dir = 0

            loss = loss_loc + 3*loss_size + loss_rot + 0.5*loss_dir

        else:
            loss_rot = self.loss(pred_valid[:,6], target_valid[:,6])
            loss = loss_loc + 3*loss_size + loss_rot

        if self.config['use_corner_loss']:
            pred_corners = box2corners(pred_valid)
            target_corners = box2corners(target_valid)
            loss_corner = self.loss(pred_corners, target_corners)
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

def get_direction_target(reg_targets, anchors=None, dir_offset=0, dir_limit_offset=0.0, num_bins=2, one_hot=True):
    """Encode direction to 0 ~ num_bins-1.
    Args:
        anchors (torch.Tensor): Bbox proposals from stage 1.
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
    if anchors is not None:
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
    else:
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
        dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        dir_cls_targets = dir_targets
    return dir_cls_targets.long()

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