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
from utils.task2 import roi_pool
from utils.task1 import label2corners
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

if __name__ == '__main__':
    #data = load_data('data/demo.p') # Change to data.p for your final submission 

    # Visulization for task 2
    visualizer = Visualizer()
    config_path = 'config.yaml'
    config = yaml.safe_load(open(config_path, 'r'))
    ds = DatasetLoader(config['data'], 'minival')
    


    valid_pred, valid_xyz, valid_features = roi_pool(pred=ds.get_data(0, 'detections'),
									xyz=ds.get_data(0, 'xyz'),
									feat=ds.get_data(0, 'features'),
									config=config['data'])

    #print(isinstance(valid_xyz, np.ndarray))
    #print(valid_pred.shape)
    #print(valid_xyz.shape)

    points = valid_xyz.reshape(-1, valid_xyz.shape[-1])
    visualizer.update(points)
    valid_corners = label2corners(valid_pred, expand = True, delta = config['data']['delta'])
    visualizer.update_boxes(valid_corners)


    valid_corners_org = label2corners(valid_pred)

    #i = 97
    #visualizer.update(valid_xyz[i])
    #visualizer.update_boxes(valid_corners[i])
    #visualizer.update_boxes(np.vstack([valid_corners[0],valid_corners_org[0]]))
    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
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



