# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

#from sklearn.metrics import explained_variance_score
import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
#from load_data import load_data
import yaml
from dataset import DatasetLoader
from utils.task1 import label2corners
from utils.task2 import roi_pool, enlarge_box
from utils.task3 import sample_proposals
from utils.ccs import global2canonical, canonical2global

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])

    def update(self, points):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        self.sem_vis.set_data(points, size=10)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

def test(label):
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
        corner_3d = [[l/2, -h,  w/2],
                    [-l/2, -h,  w/2],
                    [-l/2, -h, -w/2],
                    [ l/2, -h, -w/2],
                    [ l/2,  0,  w/2],
                    [-l/2,  0,  w/2],
                    [-l/2,  0, -w/2],
                    [ l/2,  0, -w/2]]

        corner_3d = np.array(corner_3d)
        corners.append(corner_3d)

    corners = np.array(corners)

    return corners

if __name__ == '__main__':
    #data = load_data('data/demo.p') # Change to data.p for your final submission 

    visualizer = Visualizer()
    config_path = 'config.yaml'
    config = yaml.safe_load(open(config_path, 'r'))
    ds = DatasetLoader(config['data'], 'minival')
    
    frame_id = 0

    valid_pred, pooled_xyz, pooled_feat = roi_pool(pred=ds.get_data(frame_id, 'detections'),
									xyz=ds.get_data(frame_id, 'xyz'),
									feat=ds.get_data(frame_id, 'features'),
									config=config['data'])

    # pooled_xyz = global2canonical(pooled_xyz, valid_pred)

    box_ind =np.arange(0,5) 

    # Visulization for task 2
    # points = pooled_xyz[box_ind].reshape(-1, pooled_xyz.shape[-1])
    # visualizer.update(points)

    # valid_corners_org = test(valid_pred)
    # valid_corners = test(enlarge_box(valid_pred,config['data']['delta']))

    # visualizer.update_boxes(valid_corners[box_ind])
    # visualizer.update_boxes(valid_corners_org[box_ind])
    # visualizer.update_boxes(np.vstack([valid_corners[box_ind],valid_corners_org[box_ind]]))
    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    # Visulization for task 3
    targets, xyz, feat, iou, pred = sample_proposals(pred=valid_pred,
                                                target=ds.get_data(frame_id, 'target'),
                                                xyz=pooled_xyz,
                                                feat=pooled_feat,
                                                config=config['data'],
                                                train=True,
                                                )

    points = xyz[box_ind].reshape(-1, xyz.shape[-1])
    visualizer.update(points)

    # targets = global2canonical(targets.reshape(-1, 1, targets.shape[1]), pred).squeeze(1)
    # targets[:, 6] -= pred[:, 6]

    # targets[:, 6] += pred[:, 6]
    # targets = canonical2global(torch.from_numpy(targets), pred).numpy()

    valid_corners = label2corners(targets[box_ind].reshape(-1,7))
    visualizer.update_boxes(valid_corners)

    print(valid_corners.shape)
    print(xyz.shape)
    print(targets[box_ind])


    vispy.app.run()

    # check

    # for (i, expand_box) in enumerate(valid_corners):
    #     xyz = valid_xyz[i]
    #     # Create a binary mask for each bounding box regarding to the location of each point
    #     min_x, max_x= np.min(expand_box[:,0]), np.max(expand_box[:,0])
    #     min_y, max_y = np.min(expand_box[:,1]), np.max(expand_box[:,1])
    #     min_z, max_z = np.min(expand_box[:,2]), np.max(expand_box[:,2])

    #     mask_x = np.logical_and((xyz[:, 0] >= min_x) , (xyz[:, 0] <= max_x))
    #     mask_y = np.logical_and((xyz[:, 1] >= min_y) , (xyz[:, 1] <= max_y))
    #     mask_z = np.logical_and((xyz[:, 2] >= min_z) , (xyz[:, 2] <= max_z))

    #     mask = mask_x & mask_y & mask_z

    #     print(np.all(mask))

