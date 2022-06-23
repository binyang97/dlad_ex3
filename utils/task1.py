from cmath import exp
import numpy as np
from scipy.spatial import ConvexHull
from shapely import geometry

def expand_bbox(box, delta):
    '''
    expand bounding box with given distance in all directions
    input (8, 3) corners coordinates
    '''

    direction = np.sign(box)

    direction[direction == 0] = 1
    
    newbox = box + direction * delta

    return newbox

def label2corners(label, expand = False, delta = 1):
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
        corner_3d = np.array([[l/2, -h,  w/2],
                    [-l/2, -h,  w/2],
                    [-l/2, -h, -w/2],
                    [ l/2, -h, -w/2],
                    [ l/2,  0,  w/2],
                    [-l/2,  0,  w/2],
                    [-l/2,  0, -w/2],
                    [ l/2,  0, -w/2]])

        
        #corner_3d = np.array(corner_3d)

        if expand:
            corner_3d = expand_bbox(corner_3d, delta = delta)
        corner_3d = np.hstack([corner_3d, np.ones((8,1))])
        corner_3d = corner_3d.T

        cos_ry = np.cos(ry)
        sin_ry = np.sin(ry)
        T = [[ cos_ry, 0, sin_ry, x],
            [0,       1, 0,      y],
            [-sin_ry, 0, cos_ry, z],
            [0,       0, 0,      1]]
        T = np.array(T)

        corner_3d = np.dot(T, corner_3d)
        corner_3d = corner_3d.T
        corner_3d = corner_3d[:, :3]
        corners.append(corner_3d)
    corners = np.array(corners)
    assert corners.shape == (len(label), 8, 3)

    return corners


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (M,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''

    corners_pred = label2corners(pred)
    corners_target = label2corners(target)

    IOU = []

    for corner_pred in corners_pred:
        iou_row = []
        for corner_target in corners_target:
            rect1 = [(corner_pred[i,0], corner_pred[i,2]) for i in range(3,-1,-1)]
            rect2 = [(corner_target[i,0], corner_target[i,2]) for i in range(3,-1,-1)] 

            poly1 = geometry.Polygon(rect1)
            poly2 = geometry.Polygon(rect2)

            inter_area = poly1.intersection(poly2).area

            ymin = max(corner_pred[0,1], corner_target[0,1])
            ymax = min(corner_pred[4,1], corner_target[4,1])

            inter_vol = inter_area * max(0.0, ymax-ymin)
            
            vol1 = box3d_vol(corner_pred)
            vol2 = box3d_vol(corner_target)
            iou = inter_vol / (vol1 + vol2 - inter_vol)


            iou_row.append(iou)

        IOU.append(iou_row)

    IOU = np.array(IOU)
    
    return IOU
            

    


def compute_recall(pred, target, threshold):
    '''
    Task 1
    input
        pred (N,7) proposed 3D bounding box labels
        target (M,7) ground truth 3D bounding box labels
        threshold (float) threshold for positive samples
    output
        recall (float) recall for the scene
    '''
    FN = 0
    TP = 0
    iou = get_iou(pred, target)

    iou = np.asarray(iou)
    iou = iou.T
    
    for i in range(len(iou)):
        iou_max = np.max(iou[i])
        if iou_max > threshold:
            TP+=1
        elif iou_max < threshold:
            FN+=1
    return TP/(FN+TP)
