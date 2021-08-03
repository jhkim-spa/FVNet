import numpy as np
import torch
from mmcv.runner import force_fp32
from torch import nn as nn

from mmdet3d.core import (PseudoSampler, box3d_multiclass_nms,
                          limit_period, xywhr2xyxyr)
from mmdet.core import build_assigner, build_bbox_coder, build_sampler, multi_apply
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
from mmdet.models import HEADS
from ..builder import build_loss
from .anchor3d_head import Anchor3DHead


@HEADS.register_module()
class PVGNetHead(nn.Module):

    def __init__(self,
                 num_classes=1,
                 feat_channels=384,
                 fg_weight=15,
                 bbox_coder=dict(type='PVGNetBBoxCoder'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.fg_weight = fg_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_encode_size = self.bbox_coder.encode_size
        self.box_decode_size = self.bbox_coder.decode_size

        # build loss function
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self._init_layers()

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feat_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        # Classification layer
        self.cls_fc = nn.Sequential(
            nn.Linear(64, self.num_classes)
        )
        # Regression layer
        self.reg_fc = nn.Sequential(
            nn.Linear(64, self.box_encode_size)
        )

    def init_weights(self):
        """Initialize the weights of head."""
        pass

    def forward_single(self, x):
        x = self.shared_fc(x)
        cls_score = self.cls_fc(x)
        bbox_pred = self.reg_fc(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             points,
             gt_bboxes,
             gt_labels,
             input_metas,
             gt_bboxes_ignore):

        cls_reg_targets = self.get_targets(
            points,
            gt_bboxes,
            gt_labels)

        (cls_targets, bbox_targets, pos_idx,
         num_total_list, num_pos_list) = cls_reg_targets

        # targets are the same for all scales
        num_scales = len(cls_scores)
        cls_targets = [cls_targets] * num_scales
        bbox_targets = [bbox_targets] * num_scales
        pos_idx = [pos_idx] * num_scales
        num_total_list = [num_total_list] * num_scales
        num_pos_list = [num_pos_list] * num_scales

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            cls_targets,
            bbox_targets,
            pos_idx,
            num_total_list,
            num_pos_list)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def loss_single(self, cls_score, bbox_pred,
                    cls_target, bbox_target, pos_idx,
                    num_total_list, num_pos_list):
        pos_inds = torch.where(
            (cls_target < self.num_classes) &\
            (cls_target >= 0)
        )[0]
        num_pos = pos_inds.shape[0]
        # classification loss
        cls_weight = torch.ones(cls_target.shape, device=cls_score.device)
        cls_weight[pos_inds] *= self.fg_weight
        loss_cls = self.loss_cls(cls_score,
                                 cls_target,
                                 cls_weight,
                                 avg_factor=cls_score.shape[0])
        # regression loss
        if num_pos > 0:
            bbox_pred = bbox_pred.split(num_total_list)
            pos_idx = pos_idx.split(num_pos_list)
            pos_bbox_pred = [pred[idx] for pred, idx in zip(bbox_pred, pos_idx)]
            pos_bbox_pred = torch.cat(pos_bbox_pred)
            loss_bbox = self.loss_bbox(pos_bbox_pred,
                                       bbox_target,
                                       avg_factor=num_pos)
        else:
            loss_bbox = bbox_pred.sum() * 0

        return loss_cls, loss_bbox

    def get_targets(self, points, gt_bboxes, gt_labels):
        batch_size = len(points)
        device = points[0].device

        cls_targets_list = []
        bbox_targets_list = []
        pos_idx_list = []
        num_total_list = []
        num_pos_list = []
        total_pos = 0
        for i in range(batch_size):
            res_points = points[i][:, :3]
            labels = gt_labels[i]
            boxes = gt_bboxes[i].tensor
            boxes = boxes.to(device)

            assigned_idx = points_in_boxes_gpu(res_points.unsqueeze(0),
                boxes.unsqueeze(0))[0].to(torch.long)
            pos_idx = torch.where(assigned_idx != -1)[0]
            neg_idx = torch.where(assigned_idx == -1)[0]
            num_total = res_points.shape[0]
            num_pos = pos_idx.shape[0]

            cls_targets = labels[assigned_idx]
            cls_targets[neg_idx] = self.num_classes

            if num_pos != 0:
                pos_idx = torch.where((assigned_idx != -1) & (labels[assigned_idx] >= 0))[0]
                boxes = boxes[assigned_idx][pos_idx]
                bbox_targets = self.bbox_coder.encode(res_points[pos_idx], boxes,
                    self.bbox_coder.prior_size)
                bbox_targets_list.append(bbox_targets)
                pos_idx_list.append(pos_idx)
                num_pos = pos_idx.shape[0]
            cls_targets_list.append(cls_targets)
            num_total_list.append(num_total)
            num_pos_list.append(num_pos)
            total_pos += num_pos
        
        cls_targets = torch.cat(cls_targets_list, dim=0)
        if total_pos != 0:
            pos_idx = torch.cat(pos_idx_list, dim=0)
            bbox_targets = torch.cat(bbox_targets_list, dim=0)
        else:
            pos_idx = None
            bbox_targets = None

        return (cls_targets, bbox_targets, pos_idx,
                num_total_list, num_pos_list)

    def levels_to_batches(self, mlvl_tensor):

        return None

    def get_bboxes(self,
                   points,
                   cls_scores,
                   bbox_preds,
                   input_metas,
                   rescale=False):
        """
        Implemented only for bath size of one.
        """
        cfg = self.test_cfg
        num_levels = len(cls_scores)
        input_meta = input_metas[0]

        points = torch.cat(points, dim=0)[:, :3]

        proposals = []
        for i in range(num_levels):
            cls_score = cls_scores[i]
            bbox_pred = bbox_preds[i]
            scores =cls_score.sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                res_points = points[topk_inds, :]
            
            bboxes = self.bbox_coder.decode(res_points, bbox_pred, self.bbox_coder.prior_size)
            bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
                bboxes, box_dim=self.box_decode_size).bev)

            padding = scores.new_zeros(scores.shape[0], 1)
            scores = torch.cat([scores, padding], dim=1)
    
            score_thr = cfg.get('score_thr', 0)
            results = box3d_multiclass_nms(bboxes, bboxes_for_nms, scores,
                                        score_thr, cfg.max_num, cfg)
            bboxes, scores, labels, _ = results
            bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_decode_size)
            proposals.append((bboxes, scores, labels))

        return proposals
