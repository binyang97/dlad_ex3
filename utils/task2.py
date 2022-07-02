import numpy as np
from timeit import default_timer as timer

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
        pred (K,7) bounding box labels
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
    enlarged_pred = enlarge_box(pred, config['delta'])
    valid_indices, valid = points_in_boxes(xyz, enlarged_pred, config['max_points'])
    valid_pred = pred[valid]
    pooled_xyz = xyz[valid_indices]

    pooled_feat = feat[valid_indices]
    
    assert valid_pred.shape == (valid.size, 7)
    assert pooled_xyz.shape == (valid.size, config['max_points'], 3)
    assert pooled_feat.shape == (valid.size, config['max_points'], feat.shape[1])

    return valid_pred, pooled_xyz, pooled_feat

def enlarge_box(box, extention):
    '''
    input
        box (N,7) 3D bounding box label (x,y,z,h,w,l,ry)
        extention (float)  extend the bounding box by extension on all sides in meters
    output
        extended_box (N,7) extended bounding box label
    '''
    enlarged_box = box.copy()
    enlarged_box[:, 3:6] += 2 * extention

    enlarged_box[:, 1] += extention
    return enlarged_box

def points_in_box(xyz, box):
    '''
    input
        xyz (N,3) point coordinates in rectified reference frame
        box (7,) 3d bounding box label (x,y,z,h,w,l,ry)
    output
        xyz_index (# of valid xyz,) indices of points that are in each k' bounding box
    '''
    h = box[3]
    w = box[4]
    l = box[5]
    x = box[0]
    y = box[1]
    z = box[2]
    ry = box[6]
    d = np.sqrt(l**2+w**2)

    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)
    T = [[cos_ry, 0, sin_ry, x],
        [0,       1, 0,      y],
        [-sin_ry, 0, cos_ry, z]]
    T = np.array(T)
    C = T[:3, :3]
    r = T[:3,  3]
    T_inv = np.zeros((3,4))
    T_inv[:,:3] = C.T
    T_inv[:, 3] = np.dot(-C.T, r)

    xyz_sub_mask =  (xyz[:,0] >= x-d/2) & (xyz[:,0] <= x+d/2) & \
                    (xyz[:,1] <= y) & (xyz[:,1] >= y-h) & \
                    (xyz[:,2] >= z-d/2) & (xyz[:,2] <= z+d/2)

    xyz_sub_index = np.where(xyz_sub_mask)[0]

    xyz_sub = xyz[xyz_sub_index]
    xyz_sub = np.hstack([xyz_sub, np.ones((len(xyz_sub),1))]).T

    xyz_prime = np.dot(T_inv, xyz_sub).T

    xyz_prime_mask = (xyz_prime[:,0] >= -l/2) & (xyz_prime[:,0] <= l/2) & \
                     (xyz_prime[:,1] <= 0) & (xyz_prime[:,1] >= -h) & \
                     (xyz_prime[:,2] >= -w/2) & (xyz_prime[:,2] <= w/2)

    xyz_index = xyz_sub_index[xyz_prime_mask]
   #xyz_local = xyz_prime[xyz_prime_mask]

    return xyz_index


def points_in_boxes(xyz, boxes, max_points):
    '''
    input
        xyz (N,3) points in rectified reference frame
        boxes (K,7) 3d bounding box labels (x,y,z,h,w,l,ry)
    output
        valid_indices (K',M) indices of points that are in each k' bounding box
        valid (K') index vector showing valid bounding boxes, i.e. with at least
                   one point within the box
    '''
    valid_indices = []
    valid = []
    for (i, box) in enumerate(boxes):
        xyz_index, inv_matrix = points_in_box(xyz, box)
        if len(xyz_index) > 0:
            valid_index = sample_w_padding(xyz_index, max_points)
            valid_indices.append(valid_index)
            valid.append(i)
    valid_indices = np.array(valid_indices).reshape(-1, max_points)
    valid = np.array(valid)

    return valid_indices, valid

def sample_w_padding(indices, num_needed):
    num_elements = indices.size
    if num_elements >= num_needed:
        return np.random.choice(indices, size=num_needed, replace=False)
    else:
        choice = indices
        repeat = np.random.choice(indices, size=num_needed-num_elements, replace=True)
        return np.concatenate((choice, repeat))