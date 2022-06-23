from msilib.schema import Error
import numpy as np
from pyro import sample
from .task1 import label2corners, box3d_vol
from timeit import default_timer as timer

from numba import njit


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

@njit
def vstack(arr1, arr2):
    return np.vstack((arr1, arr2))

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
    #pooled_xyz = []
    #pooled_feat = []

    pooled = []

    indices = np.arange(N)

    #print(pred.shape)
    #start2= timer()
    validity = np.ones(len(pred), dtype = bool)


    for (i, expand_box) in enumerate(expand_corners_pred):

        # Create a binary mask for each bounding box regarding to the location of each point
        o, a, b, c = expand_box[0], expand_box[1], expand_box[3], expand_box[4]

        oa = a - o
        ob = b - o
        oc = c - o


        mask_a = (np.dot(xyz, oa) > np.dot(oa, o)) & (np.dot(xyz, oa) < np.dot(oa, a))
        mask_b = (np.dot(xyz, ob) > np.dot(ob, o)) & (np.dot(xyz, ob) < np.dot(ob, b))
        mask_c = (np.dot(xyz, oc) > np.dot(oc, o)) & (np.dot(xyz, oc) < np.dot(oc, c))

        mask = mask_a & mask_b & mask_c

        #print(np.any(mask))
        #print(np.any(mask))
        # if no points are located in the box, continue to the next box
        #if np.any(mask) == False:
            #continue
        
        #masks_box.append(mask)

        validity[i] = np.any(mask)
        valid_xyz = xyz[mask]

        #valid_features = feat[mask]
        valid_indices = indices[mask]

        #valid_pred.append(label)

        num_max_points = config['max_points']
        #print(len(valid_xyz))
        
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
        elif len(valid_xyz) < num_max_points and len(valid_xyz) > 0:
            valid_features = feat[mask]
            add_point_num = num_max_points - len(valid_xyz)
            sample_indices = rng.choice(valid_indices, size = add_point_num ,replace = True)

            valid_xyz = np.vstack((valid_xyz, xyz[sample_indices]))
            valid_features = np.vstack((valid_features, feat[sample_indices]))
           # valid_xyz = vstack(valid_xyz, xyz[sample_indices])
            #valid_features = vstack(valid_features, feat[sample_indices])

            assert len(valid_xyz) == num_max_points
            #pooled_xyz.append(valid_xyz)
            #pooled_feat.append(valid_features)
        #else:
            #raise AssertionError
    #time2 = timer() - start2
        #pooled_xyz.append(valid_xyz)
        #pooled_feat.append(valid_features)
        #print(len(np.unique(valid_xyz[:,1])))
        if len(valid_xyz) != 0:
            pooled.append([valid_xyz, valid_features])

    #print(np.unique(valid_xyz[:,1]))

    #print(validity)
    #print(pooled[0][0].shape)
    #pooled = np.array(pooled)
    

    valid_pred = pred[validity]
    pooled_xyz, pooled_feat = zip(*pooled)

    pooled_xyz = np.array(pooled_xyz)
    pooled_feat = np.array(pooled_feat)

    #print(pooled_feat.shape)
    #pooled_xyz = np.array([ele[0] for ele in pooled])
    #pooled_feat = np.array([ele[1] for ele in pooled])
    
    return valid_pred, pooled_xyz, pooled_feat