import numpy as np

from utils.task1 import get_iou

def nms(pred, score, threshold):
    '''
    Task 5
    Implement NMS to reduce the number of predictions per frame with a threshold
    of 0.1. The IoU should be calculated only on the BEV.
    input
        pred (N,7) 3D bounding box with (x,y,z,h,w,l,ry)
        score (N,) confidence scores
        threshold (float) upper bound threshold for NMS
    output
        s_f (M,7) 3D bounding boxes after NMS
        c_f (M,1) corresopnding confidence scores
    '''
    s_f = []
    c_f = []

    while pred.shape[0] != 0:
        i = np.argmax(score)
        Di = pred[i]

        s_f.append(Di)
        c_f.append(score[i])
        pred = np.delete(pred, i, axis=0)
        score = np.delete(score, i)

        co_iou = get_iou_2d(pred, Di).reshape(-1)
        pred = pred[co_iou<threshold]
        score = score[co_iou<threshold]

    s_f = np.vstack(s_f)
    c_f = np.array(c_f).reshape(-1,1)

    return s_f, c_f

def get_iou_2d(pred, target):
    pred_2d = pred.copy()
    target_2d = target.copy().reshape(-1,7)
    pred_2d[:,1] = 0
    pred_2d[:,3] = 1
    target_2d[:,1] = 0
    target_2d[:,3] = 1
    return get_iou(pred_2d, target_2d)