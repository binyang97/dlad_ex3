import numpy as np

def label2corners(label):
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
        corner_3d = [[l/2, h,  w/2],
                    [-l/2, h,  w/2],
                    [-l/2, h, -w/2],
                    [ l/2, h, -w/2],
                    [ l/2,  0,  w/2],
                    [-l/2,  0,  w/2],
                    [-l/2,  0, -w/2],
                    [ l/2,  0, -w/2]]
        corner_3d = np.array(corner_3d)
        corner_3d = np.hstack([corner_3d, np.ones((8,1))])
        corner_3d = corner_3d.T

        cos_ry = np.cos(ry)
        sin_ry = np.sin(ry)
        T = [[ cos_ry, 0, sin_ry, -x],
            [0,       1, 0,      -y],
            [-sin_ry, 0, cos_ry, -z],
            [0,       0, 0,      1]]
        T = np.array(T)

        corner_3d = np.dot(T, corner_3d)
        print(corner_3d)

    pass



def get_iou(pred, target):
    '''
    Task 1
    input
        pred (N,7) 3D bounding box corners
        target (N,7) 3D bounding box corners
    output
        iou (N,M) pairwise 3D intersection-over-union
    '''
    pass

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
    pass