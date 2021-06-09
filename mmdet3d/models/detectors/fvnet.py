import os
import copy
from os import path as osp

import numpy as np
import mmcv
from mmdet3d.models.detectors.base import Base3DDetector
from mmcv.parallel import DataContainer as DC
import torch
from torch import nn as nn

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS, build_backbone, build_neck, build_head
from .single_stage import SingleStage3DDetector
from mmdet3d.core import Box3DMode, show_result


@DETECTORS.register_module()
class FVNet(SingleStage3DDetector):

    def __init__(self,
                 depth_range,
                 backbone,
                 neck=None,
                 bbox_head=None,
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
        self.depth_range = depth_range

    def extract_feat(self, fv):

        fv = torch.stack(fv)
        feats = self.backbone(fv)
        if self.with_neck:
            feats = self.neck(feats)

        valid_coords = dict()
        valid_coords_2d = torch.nonzero(fv[:, -1, :, :])
        valid_coords_3d = fv[valid_coords_2d[:, 0],
                             :3,
                             valid_coords_2d[:, 1],
                             valid_coords_2d[:, 2]]
        batch_idx = valid_coords_2d[:, 0].reshape(-1, 1)
        valid_coords_3d = torch.cat((batch_idx, valid_coords_3d), dim=1)
        valid_coords['2d'] = valid_coords_2d
        valid_coords['3d'] = valid_coords_3d

        batch_size = len(fv)
        device = fv[0].device
        mlvl_valid_coords = []
        featmap_sizes = [featmap.size()[-2:] for featmap in feats]
        depth_range = self.depth_range
        fv_src = fv

        for i in range(len(feats)):
            featmap_size = featmap_sizes[i]
            h_des = featmap_size[0]
            w_des = featmap_size[1]
            h_scale = h_des / fv.shape[2]
            w_scale = w_des / fv.shape[3]

            fv_des = torch.zeros((batch_size, fv_src.shape[1],
                                    h_des, w_des)).to(device)
            idx_src = torch.nonzero(fv_src[:, -1, :, :], as_tuple=True)
            depth = fv_src[idx_src[0], 0, idx_src[1], idx_src[2]]

            mask_sort = torch.argsort(depth, descending=True)
            mask_range = torch.where((depth_range[i] < depth[mask_sort]) &\
                                        (depth_range[i+1] > depth[mask_sort]))[0]
            idx_src = [idx[mask_sort][mask_range] for idx in idx_src]
            idx_des = list()
            idx_des.append(idx_src[0])
            idx_des.append((h_scale * idx_src[1]).to(torch.long))
            idx_des.append((w_scale * idx_src[2]).to(torch.long))
            fv_des[idx_des[0], :, idx_des[1], idx_des[2]] = \
                fv_src[idx_src[0], :, idx_src[1], idx_src[2]]
            
            res_valid_coords = dict()
            res_valid_coords_2d = torch.nonzero(fv_des[:, -1, :, :])
            res_valid_coords_3d = fv_des[res_valid_coords_2d[:, 0],
                                            :3,
                                            res_valid_coords_2d[:, 1],
                                            res_valid_coords_2d[:, 2]]
            batch_idx = res_valid_coords_2d[:, 0].reshape(-1, 1)
            res_valid_coords_3d = torch.cat((batch_idx, res_valid_coords_3d),
                                                dim=1)
            res_valid_coords['2d'] = res_valid_coords_2d
            res_valid_coords['3d'] = res_valid_coords_3d
            mlvl_valid_coords.append(res_valid_coords)

        return feats, mlvl_valid_coords

    def forward_train(self,
                      fv,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):

        feats, valid_coords = self.extract_feat(fv)
        outs = self.bbox_head(feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, valid_coords, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def simple_test(self, fv, img_metas, img=None, rescale=False,
        gt_bboxes_3d=None):
        """Test function without augmentaiton."""
        feats, valid_coords = self.extract_feat(fv)
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
    
    def forward_test(self, fv, img_metas, img=None, **kwargs):

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
            return self.simple_test(fv[0], img_metas[0], img[0], **kwargs)
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