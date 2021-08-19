import numpy as np
import torch
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn as nn
import torch.nn.functional as F

from mmdet3d.core import (PseudoSampler, anchor, box3d_multiclass_nms, limit_period,
                          xywhr2xyxyr)
from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models import HEADS
from ..builder import build_loss


@HEADS.register_module()
class PVGAuxHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 bbox_coder=dict(type='PVGNetBBoxCoder'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0)):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fp16_enabled = False

        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = 2

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self._init_layers()
        self._init_assigner_sampler()

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        self.bbox_sampler = PseudoSampler()

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.conv_sahred = None
        # self.cls_out_channels = self.num_classes
        # self.conv_shared = nn.Sequential(
        #     nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(self.in_channels),
        #     nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(self.in_channels),
        #     nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(self.in_channels)
        # )
        # self.conv_shared = nn.Sequential(
        #     nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1),
        #     nn.BatchNorm2d(self.in_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1),
        #     nn.BatchNorm2d(self.in_channels),
        #     nn.ReLU(inplace=True),
        # )
        self.conv_cls = nn.Conv2d(self.in_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.box_code_size, 1)

    def init_weights(self):
        """Initialize the weights of head."""
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        if self.conv_shared is not None:
            x = self.conv_shared(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):

        return multi_apply(self.forward_single, feats)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             input_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        
        device = cls_scores[0].device
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(featmap_sizes,
                                           gt_bboxes,
                                           gt_labels,
                                           input_metas,
                                           label_channels,
                                           device)

        if cls_reg_targets is None:
            return None
        labels_list, bbox_targets_list = cls_reg_targets
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            bbox_targets_list)
        return dict(
            loss_2d_center=losses_cls, loss_2d_bbox=losses_bbox)

    def get_targets(self, featmap_sizes, gt_bboxes, gt_labels,
                    input_metas, label_channels, device):
        batch_size = len(gt_bboxes)
        valid_idx = [label != -1 for label in gt_labels]
        gt_bboxes = [res[idx] for res, idx in zip(gt_bboxes, valid_idx)]
        gt_labels = [res[idx] for res, idx in zip(gt_labels, valid_idx)]
        
        sizes = [((box[:, 2] - box[:, 0]) / featmap_sizes[0][1],
                  (box[:, 3] - box[:, 1]) / featmap_sizes[0][0]) for box in gt_bboxes]
        sizes = [torch.stack(res).T for res in sizes]
        centers = [(box[:, :2] + size / 2.).to(torch.long)\
            for box, size in zip(gt_bboxes, sizes)]

        cls_targets = torch.ones((batch_size, self.num_classes, *featmap_sizes[0]),
                                  dtype=torch.long, device=device)
        reg_targets = torch.zeros((batch_size, self.box_code_size, *featmap_sizes[0]),
                                   dtype=torch.float32, device=device)
        
        for i in range(batch_size):
            u = centers[i][:, 0]
            v = centers[i][:, 1]
            cls_targets[i, :, v, u] = gt_labels[i]
            reg_targets[i, :, v, u] = sizes[i].T

        return [cls_targets], [reg_targets]

    def loss_single(self, cls_score, bbox_pred, labels, bbox_targets):

        # classification loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes).contiguous()
        labels = labels.permute(0, 2, 3, 1).reshape(-1).contiguous()

        loss_cls = self.loss_cls(
            cls_score, labels, avg_factor=cls_score.shape[0])

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.box_code_size).contiguous()
        bbox_targets = bbox_targets.permute(0, 2, 3, 1).reshape(-1, self.box_code_size).contiguous()

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]

        if num_pos > 0:
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                avg_factor=pos_inds.shape[0])

        return loss_cls, loss_bbox
