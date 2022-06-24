import os, sys, argparse
import pathlib
import psutil
import yaml
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timeit import default_timer as timer
import numpy as np
import torch

from dataset import DatasetLoader
from utils.task2 import roi_pool
from utils.task3 import sample_proposals

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_path', default='config.yaml')
parser.add_argument('--recordings_dir', default='tests/recordings')
parser.add_argument('--task', type=int)
args = parser.parse_args()

if __name__=='__main__':
    config = yaml.safe_load(open(args.config_path, 'r'))
    ds = DatasetLoader(config['data'], 'minival')


    valid_pred, valid_xyz, valid_features = roi_pool(pred=ds.get_data(0, 'detections'),
									xyz=ds.get_data(0, 'xyz'),
									feat=ds.get_data(0, 'features'),
									config=config['data'])
    
    target = ds.get_data(0, 'target')


    assigned_targets, xyz_ret, feat_ret, iou_ret = sample_proposals(valid_pred, target, valid_xyz, valid_features, config = config['data'], train = True)

    #print(assigned_targets.shape, xyz_ret.shape, feat_ret.shape, iou_ret.shape)

