from msilib.schema import Error
import numpy as np
from pyro import sample
from .task1 import label2corners, box3d_vol
from timeit import default_timer as timer


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
    #N = len(xyz)
    N = feat.shape[0]
    #start1= timer()
    expand_corners_pred = label2corners(pred, expand = True, delta = config['delta'])
    #time1 = timer() - start1
    #expand_pred = expand_label(pred, delta = config['delta'])

    rng = np.random.default_rng(seed = 12345)

    #expand_corners_pred = expand_bbox(corners_pred, delta = config['delta'])

    #masks_box = []
    #valid_pred = np.array([np.ones(7)])
    #pooled_xyz = np.array([np.ones(3)])
    #pooled_feat = np.array([np.ones(C)])

    valid_pred = []
    pooled_xyz = []
    pooled_feat = []

    indices = np.arange(N)

    #print(pred.shape)
    #start2= timer()
    for (label, expand_box) in zip(pred, expand_corners_pred):

        # Create a binary mask for each bounding box regarding to the location of each point
        min_x = np.min(expand_box[:,0])
        max_x = np.max(expand_box[:,0])
        min_y = np.min(expand_box[:,1])
        max_y = np.max(expand_box[:,1])
        min_z = np.min(expand_box[:,2])
        max_z = np.max(expand_box[:,2])

        mask_x = np.logical_and((xyz[:, 0] >= min_x) , (xyz[:, 0] <= max_x))
        mask_y = np.logical_and((xyz[:, 1] >= min_y) , (xyz[:, 1] <= max_y))
        mask_z = np.logical_and((xyz[:, 2] >= min_z) , (xyz[:, 2] <= max_z))

        mask = mask_x & mask_y & mask_z

        #print(np.any(mask))
        #print(np.any(mask))
        # if no points are located in the box, continue to the next box
        if np.any(mask) == False:
            continue
        
        #masks_box.append(mask)

        valid_xyz = xyz[mask]

        #print(valid_xyz)
        #valid_features = feat[mask]
        valid_indices = indices[mask]

        valid_pred.append(label)

        num_max_points = config['max_points']
        if len(valid_xyz) == num_max_points:
            
            valid_features = feat[mask]
            #pooled_xyz.append(valid_xyz)
           # pooled_feat.append(valid_features)
            #pooled_feat.append(feat[mask])

        elif len(valid_xyz) > num_max_points:
            delete_point_num = len(valid_xyz) - num_max_points
            sample_indices = rng.choice(valid_indices, size = delete_point_num ,replace = False)
            mask[sample_indices] = False
            valid_xyz = xyz[mask]
            valid_features = feat[mask]
            assert len(valid_xyz) == num_max_points
            #pooled_xyz = np.vstack([pooled_xyz, valid_xyz])
            #pooled_feat = np.vstack([pooled_feat, valid_features])

            #pooled_xyz.append(valid_xyz)
            #pooled_feat.append(valid_features)
        elif len(valid_xyz) < num_max_points:
            valid_features = feat[mask]
            add_point_num = num_max_points - len(valid_xyz)
            sample_indices = rng.choice(valid_indices, size = add_point_num ,replace = True)

            valid_xyz = np.vstack((valid_xyz, xyz[sample_indices]))
            valid_features = np.vstack((valid_features, feat[sample_indices]))

            assert len(valid_xyz) == num_max_points
            #pooled_xyz.append(valid_xyz)
            #pooled_feat.append(valid_features)
        else:
            raise Error
    #time2 = timer() - start2
    pooled_xyz.append(valid_xyz)
    pooled_feat.append(valid_features)
    #print(time2)
    #valid_pred = valid_pred[1:]
    #pooled_xyz = pooled_xyz[1:]
    #pooled_feat = pooled_feat[1:]
    return valid_pred, pooled_xyz, pooled_feat