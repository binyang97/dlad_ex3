from msilib.schema import Error
from turtle import dot
import numpy as np
from pyro import sample
from .task1 import label2corners, box3d_vol
from timeit import default_timer as timer

from numba import njit

import cupy as cp


# def expand_label(pred, delta = 1.0):
#     labels = pred.copy()
#     expand_labels = []
#     for label in labels:
#         #label[0] = label[0] + delta 
#         #label[1] = label[1] + delta 
#         label[2] = label[2] + delta 

#         label[3] = label[3] + delta * 2
#         label[4] = label[4] + delta * 2
#         label[5] = label[5] + delta * 2

#         expand_labels.append(label)
    
#     return expand_labels
#@njit
#@njit
def dot_product(v1, v2):
    #assert len(v1) == len(v2)
    out = 0
    for k in range(len(v1)):
        out += v1[k] * v2[k]
    return out


def create_mask(expand_box, xyz):
    o, a, b, c = expand_box[0], expand_box[1], expand_box[3], expand_box[4]

    oa = a - o
    ob = b - o
    oc = c - o
    xyz_oa = np.dot(xyz, oa)
    xyz_ob = np.dot(xyz, ob)
    xyz_oc = np.dot(xyz, oc)

    mask = (xyz_oa > np.dot(oa, o)) & (xyz_oa < np.dot(oa, a)) & \
            (xyz_ob > np.dot(ob, o)) & (xyz_ob < np.dot(ob, b)) & \
                (xyz_oc > np.dot(oc, o)) & (xyz_oc < np.dot(oc, c))

    return mask 

def roi_pool(pred, xyz, feat, config):
    '''
    Task 2
    a. Enlarge predicted 3D bounding boxes by delta=1.0 meters in all directions.
       As our inputs consist of coarse detection results from the stage-1 network,
       the second stage will benefit from the knowledge of surrounding points to
       better refine the initial prediction.
    b. Form ROI's by finding all points and their corresponding features that lie 
       in each enlarged bounding box. Each ROI should contain exactly 512 points.
       If there are more points within a bounding box, randomly sample until 512.
       If there are less points within a bounding box, randomly repeat points until
       512. If there are no points within a bounding box, the box should be discarded.
    input
        pred (N,7) bounding box labels
        xyz (N,3) point cloud
        feat (N,C) features
        config (dict) data config
    output
        valid_pred (K',7)
        pooled_xyz (K',M,3)
        pooled_feat (K',M,C)
            with K' indicating the number of valid bounding boxes that contain at least
            one point
    useful config hyperparameters
        config['delta'] extend the bounding box by delta on all sides (in meters)
        config['max_points'] number of points in the final sampled ROI
    '''

    N = feat.shape[0]
    expand_corners_pred = label2corners(pred, expand = True, delta = config['delta'])

    rng = np.random.default_rng(seed = 12345)


    valid_pred = []
    pooled = []

    indices = np.arange(N)

    validity = np.ones(len(pred), dtype = bool)


    for (i, expand_box) in enumerate(expand_corners_pred):
        
        mask = create_mask(expand_box = expand_box, xyz = xyz)

        validity[i] = np.any(mask)
        
        valid_indices = indices[mask]
        num_max_points = config['max_points']
        valid_xyz = xyz[valid_indices]
        
        if len(valid_xyz) == num_max_points:
            
            valid_features = feat[valid_indices]

        elif len(valid_xyz) > num_max_points:
            delete_point_num = len(valid_xyz) - num_max_points
            sample_indices = rng.choice(valid_indices, size = delete_point_num ,replace = False)
            mask[sample_indices] = False
            valid_xyz = xyz[mask]
            valid_features = feat[mask]
            assert len(valid_xyz) == num_max_points

        elif len(valid_xyz) < num_max_points and len(valid_xyz) > 0:
            valid_features = feat[valid_indices]
            add_point_num = num_max_points - len(valid_xyz)
            sample_indices = rng.choice(valid_indices, size = add_point_num ,replace = True)

            valid_xyz = np.vstack((valid_xyz, xyz[sample_indices]))
            valid_features = np.vstack((valid_features, feat[sample_indices]))

            assert len(valid_xyz) == num_max_points
        if len(valid_xyz) != 0:
            pooled.append([valid_xyz, valid_features])

    

    valid_pred = pred[validity]
    pooled_xyz, pooled_feat = zip(*pooled)

    pooled_xyz = np.array(pooled_xyz)
    pooled_feat = np.array(pooled_feat)

    
    return valid_pred, pooled_xyz, pooled_feat