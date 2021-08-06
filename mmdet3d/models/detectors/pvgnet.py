import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector


@DETECTORS.register_module()
class PVGNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 interpolation,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PVGNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.interpolation = interpolation
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

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

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas)
        x = self.bev2pts(points, x, interpolation=self.interpolation) # (B, C, H, W) -> (B, N, C)
        outs = self.bbox_head(x)
        loss_inputs = outs + (points, gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def bev2pts(self, points, bev_feats, interpolation=False):
        """This function converts bev features to point-wise features.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            bev_feats (list[torch.Tensor]): BEV features of each scale
                with shape (B, C, H, W).
            interpolation (bool):
                If True, neighboring bev features are weighted sumed.
        
        Returns:
            pts_feats (list[torch.Tensor]): Point-wise features of each sample
                with shape (S, N, C) (S: number of scales)
        """
        point_cloud_range = self.voxel_layer.point_cloud_range
        x_range = point_cloud_range[3] - point_cloud_range[0]
        y_range = point_cloud_range[4] - point_cloud_range[1]
        num_levels = len(bev_feats)
        batch_size = len(points)

        num_pts = [len(pts) for pts in points]
        batch_idx = []
        for i in range(batch_size):
            batch_idx.append(torch.ones(num_pts[i]) * i)
        batch_idx = torch.cat(batch_idx, dim=0).T.to(torch.long)
        points = torch.cat(points, dim=0)[:, :3]

        pts_feats = []

        for i in range(num_levels):
            if interpolation:
                bev_feat = bev_feats[i]
                bev_shape = bev_feat.shape[2:]
                x = points[:, 0] * bev_shape[1] / x_range
                y = (points[:, 1] - point_cloud_range[1]) * bev_shape[0] / y_range
                x_ = torch.ceil(x).to(torch.long) - 1 # diffrent from trunc
                y_ = torch.ceil(y).to(torch.long) - 1

                pts_feat = None
                # zero padding
                import torch.nn.functional as F
                bev_feat = F.pad(input=bev_feat, pad=(1, 1, 1, 1), mode='constant', value=0)
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        # distance = torch.sqrt((x - torch.trunc(x+j))**2 + (y - torch.trunc(y+i))**2)
                        distance = torch.sqrt((x - (x_ + j))**2 + (y - (y_ + i))**2)
                        feat = bev_feat[batch_idx, :, y_ + i + 1, x_ + j + 1]
                        feat = (1 / (2*distance + 1)).unsqueeze(-1) * feat
                        if pts_feat is None:
                            pts_feat = feat
                        else:
                            pts_feat += feat
                pts_feat = torch.cat([points, pts_feat], dim=1)
                pts_feats.append(pts_feat)
            else:
                bev_feat = bev_feats[i]
                bev_shape = bev_feat.shape[2:]
                x = torch.ceil(points[:, 0] * bev_shape[1] / x_range) - 1
                y = torch.ceil((points[:, 1] - point_cloud_range[1]) * bev_shape[0] / y_range) - 1
                x = x.to(torch.long)
                y = y.to(torch.long)
                pts_feat = bev_feat[batch_idx, :, y, x]
                pts_feat = torch.cat([points,
                                      bev_feat[batch_idx, :, y, x]],
                                      dim=1)
                pts_feats.append(pts_feat)
        
        return pts_feats

    def simple_test(self, points, img_metas, imgs=None, rescale=False,
        gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        x = self.bev2pts(points, x, interpolation=self.interpolation)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            points, *outs, img_metas, rescale=rescale,
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
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
