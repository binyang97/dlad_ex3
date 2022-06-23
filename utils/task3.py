from dataclasses import replace
from secrets import choice
import numpy as np

from .task1 import get_iou

def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (string) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices 
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    '''
    co_iou = get_iou(pred, target)
    iou_target = np.max(co_iou, axis=1) #(N,)
    iou_target_index = np.argmax(co_iou, axis=1) #(N,)

    if not train:
        assigned_targets = target[iou_target_index] #(N,7)
        return assigned_targets, xyz, feat, iou_target
    else:
        fg_cr1_index = np.where(iou_target >= config['t_fg_lb'])[0]
        fg_cr1_num = fg_cr1_index.size

        fg_cr2_num = target.shape[0] #M
        iou_pred = np.max(co_iou, axis=0) #(fg_cr2_num,) w.r.t. pred
        iou_pred_index = np.argmax(co_iou, axis=0) #(fg_cr2_num,) pred

        fg_num = fg_cr1_num + fg_cr2_num


        bg_mask = iou_target < config['t_bg_up']
        bg_easy_mask = iou_target < config['t_bg_hard_lb']
        bg_hard_mask = bg_mask & ~bg_easy_mask
        bg_easy_index = np.where(bg_easy_mask)[0]
        bg_hard_index = np.where(bg_hard_mask)[0]
        bg_easy_num = bg_easy_index.size
        bg_hard_num = bg_hard_index.size
        bg_num = bg_easy_num + bg_hard_num

        fg_cr1_choice = np.array([])
        fg_cr2_choice = np.array([])
        bg_easy_choice = np.array([])
        bg_hard_choice = np.array([])

        if bg_num == 0:
            fg_cr1_choice, fg_cr2_choice = sample_fg(64, fg_cr1_num, fg_cr2_num)
        elif fg_num == 0:
            bg_easy_choice, bg_hard_choice = sample_bg(64, bg_easy_num, bg_hard_num, config['bg_hard_ratio'])
        else:
            num_fg_sample = np.min(fg_num,config['num_fg_sample'])
            fg_cr1_choice, fg_cr2_choice = sample_fg(num_fg_sample, fg_cr1_num, fg_cr2_num)
            bg_easy_choice, bg_hard_choice = sample_bg(64-num_fg_sample, bg_easy_num, bg_hard_num, config['bg_hard_ratio'])
        
        assigned_targets = np.vstack([target[iou_target_index[fg_cr1_index][fg_cr1_choice]],
                                      target[fg_cr2_choice],
                                      target[iou_target_index[bg_easy_index][bg_easy_choice]],
                                      target[iou_target_index[bg_hard_index][bg_hard_choice]]])

        xyz_ret = np.vstack([xyz[fg_cr1_index[fg_cr1_choice]],
                             xyz[iou_pred_index[fg_cr2_choice]],
                             xyz[bg_easy_index[bg_easy_choice]],
                             xyz[bg_hard_index[bg_hard_choice]]])

        feat_ret = np.vstack([feat[fg_cr1_index[fg_cr1_choice]],
                              feat[iou_pred_index[fg_cr2_choice]],
                              feat[bg_easy_index[bg_easy_choice]],
                              feat[bg_hard_index[bg_hard_choice]]])

        iou_ret = np.vstack([iou_target[fg_cr1_index[fg_cr1_choice]],
                             iou_pred[fg_cr2_choice],
                             iou_target[bg_easy_index[bg_easy_choice]],
                             iou_target[bg_hard_index[bg_hard_choice]]])

        return assigned_targets, xyz_ret, feat_ret, iou_ret

def sample_fg(num_needed, fg_cr1_num, fg_cr2_num):
    choice = sample_w_padding(fg_cr1_num+fg_cr2_num, num_needed)
    fg_cr1_choice = choice[choice<fg_cr1_num]
    fg_cr2_choice = choice[choice>=fg_cr1_num] - fg_cr1_num
    return np.sort(fg_cr1_choice), np.sort(fg_cr2_choice)


def sample_bg(num_needed, bg_easy_num, bg_hard_num, bg_hard_ratio):
    bg_easy_choice = np.array([])
    bg_hard_choice = np.array([])

    if bg_easy_num==0:
        bg_hard_choice = sample_w_padding(bg_hard_num, num_needed)
    elif bg_hard_num==0:
        bg_easy_choice = sample_w_padding(bg_easy_num, num_needed)
    else:
        bg_hard_sample_num = np.floor(bg_hard_ratio * num_needed).astype(int)
        bg_hard_choice = sample_w_padding(bg_hard_num, bg_hard_sample_num)
        bg_easy_choice = sample_w_padding(bg_easy_num, num_needed-bg_hard_sample_num)

    return np.sort(bg_easy_choice), np.sort(bg_hard_choice)


def sample_w_padding(num_elements, num_needed):
    if num_elements >= num_needed:
        return np.random.choice(num_elements, size=num_needed, replace=False)
    else:
        choice = np.arange(num_elements)
        repeat = np.random.choice(num_elements, size=num_needed-num_elements, replace=True)
        return np.concatenate((choice, repeat))