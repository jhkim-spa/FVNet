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
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu


@HEADS.register_module()
class PVGAuxHead(nn.Module):

    def __init__(self,
                 in_channels,
                 loss_seg=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.1),
                 loss_depth=dict(
                     type='smoothL1Loss', beta=1.0 / 9.0, loss_weight=0.1)):
        super().__init__()
        self.in_channels = in_channels
        self.fp16_enabled = False

        # build loss function
        self.use_sigmoid_cls = loss_seg.get('use_sigmoid', False)
        self.sampling = loss_seg['type'] not in ['FocalLoss', 'GHMC']
        self.loss_seg = build_loss(loss_seg)
        self.loss_depth = build_loss(loss_depth)

        self._init_layers()
        self._init_assigner_sampler()

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        self.bbox_sampler = PseudoSampler()

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv_seg = nn.Conv2d(32, 1, 1)
        self.conv_depth = nn.Conv2d(32, 1, 1)

    def init_weights(self):
        """Initialize the weights of head."""
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_seg, std=0.01, bias=bias_cls)
        normal_init(self.conv_depth, std=0.01)

    def forward_single(self, x):
        x = self.upsample_layer(x)
        seg_score = self.conv_seg(x)
        depth_pred = self.conv_depth(x)
        return seg_score, depth_pred

    def forward(self, feats):

        return multi_apply(self.forward_single, feats)

    @force_fp32(apply_to=('seg_scores', 'depth_preds'))
    def loss(self,
             seg_scores,
             depth_preds,
             points,
             gt_bboxes,
             gt_labels,
             input_metas,
             gt_bboxes_ignore=None):

        device = seg_scores[0].device
        cls_reg_targets = self.get_targets(points,
                                           gt_bboxes,
                                           gt_labels,
                                           input_metas,
                                           device)
        if cls_reg_targets is None:
            return None
        seg_targets_list, depth_targets_list = cls_reg_targets
        losses_seg, losses_depth = multi_apply(
            self.loss_single,
            seg_scores,
            depth_preds,
            seg_targets_list,
            depth_targets_list)
        return dict(
            loss_aux_seg=losses_seg, loss_aux_depth=losses_depth)

    def get_targets(self, points, gt_bboxes, gt_labels,
                    input_metas, device):
        batch_size = len(gt_bboxes)
        seg_targets_list = []
        depth_targets_list = []
        for i in range(batch_size):
            res_points = points[i][:, :3]
            res_lidar2img = input_metas[i]['lidar2img'][:3]
            res_lidar2img = torch.from_numpy(res_lidar2img)

            # Back transform
            res_points = res_points / input_metas[i]['pcd_scale_factor']
            if input_metas[i].get('pcd_rotation') is not None:
                rotation = input_metas[i]['pcd_rotation'].to(device)
                res_points = res_points @ torch.inverse(rotation)
            if input_metas[i]['pcd_horizontal_flip']:
                res_points[:, 1] *= -1

            uv = self.project_to_img(res_points, res_lidar2img)
            width = input_metas[i]['img_shape'][1]
            height = input_metas[i]['img_shape'][0]

            w_scale, h_scale = input_metas[i]['scale_factor'][:2]
            uv[:, 0] *= h_scale
            uv[:, 1] *= w_scale
            uv = uv.to(torch.long)
        
            if input_metas[i]['flip']:
                uv[:, 1] = width - uv[:, 1] - 1
            valid_inds = torch.where(
                (uv[:, 0] < height) & (uv[:, 0] >= 0) &
                (uv[:, 1] < width)  & (uv[:, 1] >= 0)
            )[0]

            # filtering invalid points
            uv = uv[valid_inds]
            res_points = res_points[valid_inds]

            # segmentation targets
            seg_targets = torch.ones((height, width, 1),
                                      dtype=torch.long,
                                      device=device) * -1

            assigned_inds = points_in_boxes_gpu(points[i][:, :3].unsqueeze(dim=0),
                                                gt_bboxes[i].tensor.unsqueeze(dim=0)
                                                .to(device))
            assigned_inds = assigned_inds.squeeze(dim=0).to(torch.long)
            assigned_inds = assigned_inds[valid_inds]
            
            res_labels = torch.cat([gt_labels[i], torch.tensor([1]).to(device)])
            seg_targets[uv[:, 0], uv[:, 1], 0] = res_labels[assigned_inds]

            # depth targets
            depth_targets = torch.zeros((height, width, 1),
                                         dtype=torch.float32,
                                         device=device)
            depth_targets[uv[:, 0], uv[:, 1], 0] = res_points[:, 0] / 69.12

            seg_targets_list.append(seg_targets)
            depth_targets_list.append(depth_targets)

        seg_targets = torch.stack(seg_targets_list, dim=0)
        depth_targets = torch.stack(depth_targets_list, dim=0)
        return [seg_targets], [depth_targets]

    @torch.no_grad()
    def project_to_img(self, xyz, lidar2img):

        device = xyz.device
        lidar2img = lidar2img.to(device)
        num_points = xyz.shape[0]
        xyz_hom = torch.cat((xyz.T, torch.ones((1, num_points)).to(device)))
        uv_hom = lidar2img @ xyz_hom
        uv_hom[:2, :] /= uv_hom[2, :]
        uv = uv_hom[:2, :].T
        uv = uv[:, [1, 0]]

        return uv

    def loss_single(self, seg_score, depth_pred, seg_targets, depth_targets):
        # fg/bg segmentation loss
        num_valid_points = (seg_targets != -1).sum()
        seg_score = seg_score.permute(0, 2, 3, 1).contiguous().reshape(-1, 1)
        seg_targets = seg_targets.reshape(-1)
        loss_seg = self.loss_seg(seg_score, seg_targets, avg_factor=num_valid_points)

        # depth regression loss
        depth_pred = depth_pred.permute(0, 2, 3, 1).contiguous().reshape(-1, 1)
        depth_targets = depth_targets.reshape(-1, 1)
        valid_inds = depth_targets.nonzero()[:, 0]
        loss_depth = self.loss_depth(depth_pred[valid_inds],
                                     depth_targets[valid_inds],
                                     avg_factor=valid_inds.shape[0])

        return loss_seg, loss_depth
