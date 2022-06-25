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
    if len(pred) == 0:
        print("task5 pred shape", pred.shape)
        s_f = np.array([]).reshape(-1,7)
        c_f = np.array([]).reshape(-1,1)
        return s_f, c_f

    s_f = []
    c_f = []
    pred_2d = pred.copy()
    pred_2d[:,1] = 0
    pred_2d[:,3] = 1

    while pred.shape[0] != 0:
        N = len(score)
        i = np.argmax(score)
        Di = pred_2d[i].reshape(-1,7)

        s_f.append(pred[i])
        c_f.append(score[i])
        
        mask = np.arange(N) != i
        pred = pred[mask].reshape(-1,7)
        score = score[mask]
        pred_2d = pred_2d[mask].reshape(-1,7)

        co_iou = get_iou(pred_2d, Di).reshape(-1)
        mask = co_iou<threshold
        pred = pred[mask].reshape(-1,7)
        score = score[mask]
        pred_2d = pred_2d[mask].reshape(-1,7)

    s_f = np.vstack(s_f)
    c_f = np.array(c_f).reshape(-1,1)

    return s_f, c_f
