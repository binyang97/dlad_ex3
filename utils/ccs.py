import numpy as np
import torch

from utils.task4 import limit_period

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_y(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along y-axis, angle increases z ==> x
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  zeros, -sina,
        zeros, ones, zeros,
        sina, zeros, cosa
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def global2canonical(points, anchors):
    """
    Args:
        points: (B, N, 3 + C)
        anchors: (B, 7), boxes
    Returns:
    """
    ry = anchors[:,6].reshape(-1)
    xyz = anchors[:,0:3].reshape(-1, 1, 3)

    points[:, :, 0:3] -= xyz
    points = rotate_points_along_y(points, -ry)

    return points

def canonical2global(boxes, anchors):
    """
    Args:
        boxes: (B, 3 + C)
        anchors: (B, 7), boxes from stage 1
    Returns:
    """
    ry = anchors[:, 6].reshape(-1)
    xyz = anchors[:, 0:3].reshape(-1, 1, 3)
    xyz, _ = check_numpy_to_torch(xyz)

    boxes = rotate_points_along_y(boxes.unsqueeze(dim=1), ry).squeeze(dim=1)
    boxes[:, 0:3] += xyz

    return boxes

def modify_ry(pred, dir_offset=0, dir_limit_offset=0.0, num_bins=2):
    """
    Args:
        pred:
        box: (N,7)
        dir: (N,2)
    Returns:
    """
    dir_labels = torch.max(pred['dir'], dim=-1)[1]
    box = pred['box']

    period = (2 * np.pi / num_bins)
    dir_rot = limit_period(box[:, 6] - dir_offset, dir_limit_offset, period)
    box[:, 6] = dir_rot + dir_offset + period * dir_labels.to(box.dtype)

    return box