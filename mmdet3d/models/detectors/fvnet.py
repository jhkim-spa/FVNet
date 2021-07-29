import copy
from os import path as osp

import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
import torch
from torch import nn as nn
from .. import builder

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS, build_backbone, build_neck, build_head
from .single_stage import SingleStage3DDetector
from mmdet3d.core import Box3DMode, show_result
from mmdet3d.ops import Voxelization
from torch.nn import functional as F


@DETECTORS.register_module()
class FVNet(SingleStage3DDetector):

    def __init__(self,
                 point_cloud_range=None,
                 feats_to_use=None,
                 backbone=None,
                 backbone_img=None,
                 backbone_bev=None,
                 neck=None,
                 neck_img=None,
                 neck_bev=None,
                 bbox_head=None,
                 voxel_layer=None,
                 voxel_encoder=None,
                 middle_encoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FVNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.point_cloud_range = point_cloud_range
        self.feats_to_use = feats_to_use
        if backbone_img is not None:
            self.backbone_img = build_backbone(backbone_img)
        if voxel_layer is not None:
            self.voxel_layer = Voxelization(**voxel_layer)
            self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
            self.middle_encoder = builder.build_middle_encoder(middle_encoder)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def get_valid_coords(self, fv):
        valid_coords = dict()
        valid_coords_2d = torch.nonzero(fv[:, -1, :, :])
        valid_coords_3d = fv[valid_coords_2d[:, 0],
                            :3,
                            valid_coords_2d[:, 1],
                            valid_coords_2d[:, 2]]
        valid_coords_reflectance = fv[valid_coords_2d[:, 0],
                                      3,
                                      valid_coords_2d[:, 1],
                                      valid_coords_2d[:, 2]].reshape(-1, 1)
        batch_idx = valid_coords_2d[:, 0].reshape(-1, 1)
        valid_coords_3d = torch.cat((batch_idx, valid_coords_3d), dim=1)
        valid_coords_reflectance = torch.cat((batch_idx, valid_coords_reflectance), dim=1)
        valid_coords['2d'] = valid_coords_2d
        valid_coords['3d'] = valid_coords_3d
        valid_coords['reflectance'] = valid_coords_reflectance
        return valid_coords

    def fusion(self, feats_fv, feats_bev, feats_img, mode='concat'):
        if mode == 'concat':
            feats = [feats_fv, feats_bev, feats_img]
            feats_fused = []
            for feat in feats:
                if feat is not None:
                    feats_fused.append(feat[0])
            feats_fused = torch.cat(feats_fused, dim=1)
            return [feats_fused]


    def extract_feat(self, fv, img=None):
        feats_fv = None
        feats_bev = None
        feats_img = None

        valid_coords = self.get_valid_coords(fv)
        if 'fv' in self.feats_to_use:
            feats_fv = self.backbone(fv)
            if self.with_neck:
                feats_fv = self.neck(feats_fv)

        if 'bev' in self.feats_to_use:
            batch_size = fv.shape[0]
            points = []
            for i in range(batch_size):
                batch_idx = valid_coords['3d'][:, 0] == i
                res_points = valid_coords['reflectance'][batch_idx][:, 1:].contiguous()
                points.append(res_points)

            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0].item() + 1
            feats_bev = self.middle_encoder(voxel_features, coors, batch_size)
            feats_bev = self.backbone_bev(feats_bev)
            if self.with_neck:
                feats_bev = self.neck_bev(feats_bev)
            
        if 'img' in self.feats_to_use:
            feats_img = self.backbone_img(img)

        feats = self.fusion(feats_fv, feats_bev, feats_img, mode='concat')

        return feats, [valid_coords]


    def forward_train(self,
                      fv,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      points=None,
                      img=None,
                      gt_bboxes_ignore=None):
        fv = torch.stack(fv)
        feats, valid_coords = self.extract_feat(fv, img)
        outs = self.bbox_head(feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        if valid_coords is not None:
            losses = self.bbox_head.loss(
                *loss_inputs, valid_coords, gt_bboxes_ignore=gt_bboxes_ignore)
        else:
            losses = self.bbox_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def simple_test(self, fv, img_metas, img=None, points= None, rescale=False):
        """Test function without augmentaiton."""
        fv = torch.stack(fv)
        feats, valid_coords = self.extract_feat(fv, img)
        outs = self.bbox_head(feats)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, valid_coords, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
    
    def forward_test(self, fv, img_metas, points=None, img=None, **kwargs):

        for var, name in [(fv, 'points'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(fv)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(fv), len(img_metas)))

        if num_augs == 1:
            img = [img] if img is None else img
            return self.simple_test(fv=fv[0], img_metas=img_metas[0], img=img[0], **kwargs)
        else:
            return self.aug_test(fv, img_metas, img, **kwargs)
    
    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (list[dict]): Input points and the information of the sample.
            result (list[dict]): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['fv'][0], DC):
                points = data['fv'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['fv'][0], torch.Tensor):
                points = data['fv'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['fv'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            points = self._load_points(pts_filename).reshape(-1, 4)[:, :3]

            pred_bboxes = copy.deepcopy(
                result[batch_id]['boxes_3d'].tensor.numpy())
            # for now we convert points into depth mode

            show_result(points, None, pred_bboxes, out_dir, file_name)

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        self.file_client_args = dict(backend='disk')
        self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points