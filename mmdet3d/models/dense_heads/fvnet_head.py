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
from .train_mixins import AnchorTrainMixin


@HEADS.register_module()
class FVNetHead(nn.Module, AnchorTrainMixin):

    def __init__(self,
                 anchor_cfg,
                 num_classes,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 shared_conv=False,
                 feat_channels=256,
                 use_direction_classifier=True,
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0,
                 dir_limit_offset=1,
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2)):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.shared_conv = shared_conv
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.fp16_enabled = False
        self.num_anchors = 2
        self.num_levels = 1

        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)
        self.fp16_enabled = False

        self.anchor_cfg = anchor_cfg

        self._init_layers()
        self._init_assigner_sampler()

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_classes * self.num_anchors
        self.conv_shared = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.feat_channels),
            nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.feat_channels),
            nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.feat_channels))
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                        self.num_anchors * 2, 1)
    def init_weights(self):
        """Initialize the weights of head."""
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        """Forward function on a single-scale feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        x = self.conv_shared(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        return multi_apply(self.forward_single, feats)

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th \
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             gt_bboxes,
             gt_labels,
             input_metas,
             valid_coords,
             gt_bboxes_ignore=None):
        """Calculate losses.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Gt bboxes
                of each sample.
            gt_labels (list[torch.Tensor]): Gt labels of each sample.
            input_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and \
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification \
                    losses.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        batch_size = cls_scores[0].shape[0]
        
        device = cls_scores[0].device
        anchor_list = self.get_anchors(batch_size, valid_coords,
            self.anchor_cfg, featmap_sizes, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            gt_bboxes,
            input_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # num_total_samples = None
        losses_cls, losses_bbox, losses_dir = multi_apply(
            self.loss_single,
            valid_coords,
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            dir_targets_list,
            dir_weights_list,
            num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   input_metas,
                   valid_coords,
                   cfg=None,
                   rescale=False):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): Whether th rescale bbox.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]

        mlvl_anchors = self.get_anchors(batch_size, valid_coords,
            self.anchor_cfg, featmap_sizes, device=device)[0]
        mlvl_anchors = [
            anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
        ]

        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id].detach() for i in range(num_levels)
            ]

            input_meta = input_metas[img_id]
            proposals = self.get_bboxes_single(valid_coords,
                                               cls_score_list, bbox_pred_list,
                                               dir_cls_pred_list, mlvl_anchors,
                                               input_meta, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          mlvl_valid_coords,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          cfg=None,
                          rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (None | :obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors, valid_coords in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors,
                mlvl_valid_coords):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            if valid_coords['2d'].sum() == 0:
                continue
            valid_coords_2d = valid_coords['2d'][:, 1:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0)
            dir_cls_pred = dir_cls_pred[valid_coords_2d[:, 0],
                                        valid_coords_2d[:, 1], :].reshape(-1, 2)

            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2, 0)
            cls_score = cls_score[valid_coords_2d[:, 0],
                                  valid_coords_2d[:, 1], :].reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = bbox_pred[valid_coords_2d[:, 0],
                                  valid_coords_2d[:, 1], :].reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred,
                                            normalize=self.bbox_coder.normalize)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, cfg.max_num,
                                       cfg, mlvl_dir_scores)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        return bboxes, scores, labels

    def get_anchors(self, batch_size, mlvl_valid_coords, anchor_cfg,
        featmap_sizes, device):
        # multi scale 구현하기
        size = anchor_cfg['size']
        rotation = anchor_cfg['rotation']

        anchors_batch = []
        for i in range(batch_size):
            anchors_mlvl = []
            for j in range(len(featmap_sizes)):
                mask = mlvl_valid_coords[j]['3d'][:, 0] == i
                centers = mlvl_valid_coords[j]['3d'][mask][:, 1:]
                centers[:, 2] -= size[2] / 2

                num_points = centers.shape[0]
                res_centers = torch.repeat_interleave(centers, 2, dim=0)
                res_size = torch.tensor(size, device=device).repeat((num_points*2, 1))
                res_rotation = torch.tensor(rotation, device=device).reshape(-1, 1)
                res_rotation = res_rotation.repeat((num_points, 1))
                
                res_anchors_mlvl = torch.cat([res_centers, res_size, res_rotation], dim=1)
                anchors_mlvl.append(res_anchors_mlvl)
            anchors_batch.append(anchors_mlvl)

        return anchors_batch

    def loss_single(self, valid_coords, cls_score, bbox_pred, dir_cls_preds, labels,
                    label_weights, bbox_targets, bbox_weights, dir_targets,
                    dir_weights, num_total_samples):
        """Calculate loss of Single-level results.

        Args:
            cls_score (torch.Tensor): Class score in single-level.
            bbox_pred (torch.Tensor): Bbox prediction in single-level.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single-level.
            labels (torch.Tensor): Labels of class.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_targets (torch.Tensor): Targets of bbox predictions.
            bbox_weights (torch.Tensor): Weights of bbox loss.
            dir_targets (torch.Tensor): Targets of direction predictions.
            dir_weights (torch.Tensor): Weights of direction loss.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[torch.Tensor]: Losses of class, bbox \
                and direction, respectively.
        """
        if valid_coords['2d'].sum() == 0:
            loss_cls = 0 * bbox_pred.sum()
            loss_bbox = 0 * bbox_pred.sum()
            loss_dir = 0 * bbox_pred.sum()
            return loss_cls, loss_bbox, loss_dir

        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = torch.cat(labels)
        label_weights = torch.cat(label_weights)
        cls_score = cls_score.permute(0, 2, 3, 1)
        batch_size = cls_score.shape[0]
        cls_score_list = []
        for i in range(batch_size):
            batch_idx = valid_coords['2d'][:, 0] == i
            cls_score_list.append(cls_score[i, valid_coords['2d'][batch_idx][:, 1],
                valid_coords['2d'][batch_idx][:, 2], :].reshape(-1, self.num_classes))
        cls_score = torch.cat(cls_score_list).contiguous()
        # cls_score = cls_score.reshape(-1, self.num_classes)
        # assert labels.max().item() <= self.num_classes

        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)
        bbox_pred_list = []
        for i in range(batch_size):
            batch_idx = valid_coords['2d'][:, 0] == i
            bbox_pred_list.append(bbox_pred[i, valid_coords['2d'][batch_idx][:, 1],
                valid_coords['2d'][batch_idx][:, 2], :].reshape(-1, self.box_code_size))
        bbox_pred = torch.cat(bbox_pred_list)
        bbox_targets = torch.cat(bbox_targets)
        bbox_weights = torch.cat(bbox_weights)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # dir loss
        if self.use_direction_classifier:
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1)
            dir_cls_preds_list = []
            for i in range(batch_size):
                batch_idx = valid_coords['2d'][:, 0] == i
                dir_cls_preds_list.append(dir_cls_preds[i, valid_coords['2d'][batch_idx][:, 1],
                    valid_coords['2d'][batch_idx][:, 2], :].reshape(-1, 2))
            dir_cls_preds = torch.cat(dir_cls_preds_list)
            dir_targets = torch.cat(dir_targets)
            dir_weights = torch.cat(dir_weights)
            pos_dir_cls_preds = dir_cls_preds[pos_inds]
            pos_dir_targets = dir_targets[pos_inds]
            pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=num_total_samples)
        else:
            loss_bbox = pos_bbox_pred.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()

        return loss_cls, loss_bbox, loss_dir

    def anchor_target_3d(self,
                         anchor_list,
                         gt_bboxes_list,
                         input_metas,
                         gt_bboxes_ignore_list=None,
                         gt_labels_list=None,
                         label_channels=1,
                         num_classes=1,
                         sampling=True):

        num_imgs = len(input_metas)
        assert len(anchor_list) == num_imgs

        if isinstance(anchor_list[0][0], list):
            # sizes of anchors are different
            # anchor number of a single level
            num_level_anchors = [
                sum([anchor.size(0) for anchor in anchors])
                for anchors in anchor_list[0]
            ]
            for i in range(num_imgs):
                anchor_list[i] = anchor_list[i][0]
        else:
            # anchor number of multi levels
            num_level_anchors = [[
                res.view(-1, self.box_code_size).size(0)
                for res in anchors
            ] for anchors in anchor_list]
            # concat all level anchors and flags to a single tensor
            for i in range(num_imgs):
                anchor_list[i] = torch.cat(anchor_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         all_dir_targets, all_dir_weights, pos_inds_list,
         neg_inds_list) = multi_apply(
             self.anchor_target_3d_single,
             anchor_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             input_metas,
             label_channels=label_channels,
             num_classes=num_classes,
             sampling=sampling)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = self.images_to_levels(all_labels, num_level_anchors)
        label_weights_list = self.images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = self.images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = self.images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        dir_targets_list = self.images_to_levels(all_dir_targets, num_level_anchors)
        dir_weights_list = self.images_to_levels(all_dir_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, dir_targets_list, dir_weights_list,
                num_total_pos, num_total_neg)
    
    def images_to_levels(self, target, num_levels):
        batch_size = len(num_levels)
        num_featmap_size = len(num_levels[0])
        batch_targets = []
        for i in range(batch_size):
            start = 0
            level_targets = []
            for j in range(num_featmap_size):
                res_target = target[i]
                end = start + num_levels[i][j]
                level_targets.append(res_target[start: end])
                start = end
            batch_targets.append(level_targets)

        level_targets = []
        for i in range(num_featmap_size):
            res_targets = []
            for j in range(batch_size):
                res_targets.append(batch_targets[j][i])
            level_targets.append(res_targets)

        return level_targets
