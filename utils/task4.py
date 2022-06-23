import torch
import torch.nn as nn

class RegressionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        mask = iou >= self.config['positive_reg_lb']
        pred = pred[mask]
        target = target[mask]
        loss_loc = self.loss(pred[:,:3], target[:,:3])
        loss_size = self.loss(pred[:,3:6], target[:,3:6])
        loss_rot = self.loss(pred[:,6], target[:,6])
        return loss_loc + 3*loss_size + loss_rot

class ClassificationLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,1) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        N = pred.size(0)
        target = torch.zeros(N,1)
        
        mask_pos = iou >= self.config['positive_cls_lb']
        mask_neg = iou <= self.config['negative_cls_ub']
        mask = mask_pos | mask_neg

        target[mask_pos] = 1
        target = target[mask]
        pred = pred[mask]

        loss = self.loss(pred, target)
        return loss