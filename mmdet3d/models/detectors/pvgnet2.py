import torch
from mmcv.runner import force_fp32
from torch._C import device
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_voxel_encoder, build_middle_encoder, build_backbone
from .single_stage import SingleStage3DDetector
from torch.nn import UpsamplingNearest2d
import torch.nn as nn


@DETECTORS.register_module()
class PVGNet2(SingleStage3DDetector):

    def __init__(self,
                 bev_interp,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 voxel_layer2=None,
                 voxel_encoder2=None,
                 voxel_layer3=None,
                 voxel_encoder3=None,
                 img_backbone=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PVGNet2, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.point_cloud_range = voxel_layer.point_cloud_range
        self.bev_interp = bev_interp
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = build_voxel_encoder(voxel_encoder)
        self.middle_encoder = build_middle_encoder(middle_encoder)

        self.voxel_layer2 = Voxelization(**voxel_layer2)
        self.voxel_encoder2 = build_voxel_encoder(voxel_encoder2)
        self.voxel_layer3 = Voxelization(**voxel_layer3)
        self.voxel_encoder3 = build_voxel_encoder(voxel_encoder3)

        self.img_backbone = build_backbone(img_backbone)

        self.channel_reduct = nn.Linear(448, 256)
        self.alpha = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))

        # self.weight_conv = nn.Conv2d(1, 1, 1)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, voxel_layer):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = voxel_layer(res)
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

    def extract_feat(self, points, img, plidar, img_metas):
        """Extract features from points."""
        ## Extract LiDAR Features
        voxels, num_points, coors = self.voxelize(points, self.voxel_layer)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)

        bev_shape = x[0].shape[2:]
        voxels, num_points_pts, coors_pts = self.voxelize(points, self.voxel_layer2)
        voxel_features = self.voxel_encoder2(voxels, num_points_pts, coors_pts)
        coors_pts = coors_pts.to(torch.long)
        bev_feats = x[0][coors_pts[:, 0], :, coors_pts[:, 2], coors_pts[:, 3]]

        pts_feats = torch.cat([voxel_features, bev_feats], dim=1)
        pts_feats = self.channel_reduct(pts_feats)

        ## Extract RGB Features
        img_feats = self.img_backbone(img)

        ## Image features to bev
        # resize plidar same as img feats
        m = UpsamplingNearest2d(size=img_feats[0].shape[2:])
        plidar = m(plidar)

        img_feats_with_xyz = torch.cat([plidar, img_feats[0]], dim=1)
        img_feats_with_xyz = img_feats_with_xyz.reshape(batch_size,
            img_feats_with_xyz.shape[1], -1).permute([0, 2, 1]).contiguous()
        img_feats = [feats.to(torch.float32) for feats in img_feats_with_xyz]
        
        voxels, num_points_img, coors_img = self.voxelize(img_feats, self.voxel_layer3)
        voxel_features = self.voxel_encoder3(voxels, num_points_img, coors_img)
        coors_img = coors_img.to(torch.long)

        ## fusion
        device = plidar.device
        bev_feats_pts = torch.zeros((batch_size, 256, *bev_shape), dtype=torch.float32, device=device)
        bev_feats_img = torch.zeros((batch_size, 256, *bev_shape), dtype=torch.float32, device=device)
        bev_feats_pts[coors_pts[:, 0], :, coors_pts[:, 2], coors_pts[:, 3]] = pts_feats
        bev_feats_img[coors_img[:, 0], :, coors_img[:, 2], coors_img[:, 3]] = voxel_features

        num_pts = torch.zeros((batch_size, 1, *bev_shape), dtype=torch.float32, device=device)
        num_pts[coors_pts[:, 0], 0, coors_pts[:, 2], coors_pts[:, 3]] = num_points_pts.to(torch.float32)
        num_img = torch.zeros((batch_size, 1, *bev_shape), dtype=torch.float32, device=device)
        num_img[coors_img[:, 0], 0, coors_img[:, 2], coors_img[:, 3]] = num_points_img.to(torch.float32)
        ## learnable weights
        ## method1
        # weights_pts = (self.weight_conv(num_pts)).sigmoid()
        # weights_img = 1 - weights_pts
        ## method2
        weights_pts = (self.alpha * num_pts / (num_pts + num_img + self.beta)).sigmoid()
        weights_img = 1 - weights_pts

        fused_feats = bev_feats_pts * weights_pts + bev_feats_img * weights_img

        # find valid coords
        valid_idx = (num_pts + num_img).nonzero()[:, [0, 2, 3]]
        fused_feats = fused_feats[valid_idx[:, 0], :, valid_idx[:, 1], valid_idx[:, 2]]

        anchor_points = torch.zeros((batch_size, 3, *bev_shape), dtype=torch.float32, device=device)
        anchor_points[:, 0, :, :] = torch.repeat_interleave(torch.tensor(range(bev_shape[1]),
            dtype=torch.float32, device=device).reshape(1, -1), bev_shape[0], dim=0) + 0.5
        anchor_points[:, 1, :, :] = torch.repeat_interleave(torch.tensor(range(bev_shape[0]),
            dtype=torch.float32, device=device).reshape(-1, 1), bev_shape[1], dim=1) + 0.5

        x_range = self.point_cloud_range[3] - self.point_cloud_range[0]
        y_range = self.point_cloud_range[4] - self.point_cloud_range[1]
        anchor_points[:, 0, :, :] = (anchor_points[:, 0, :, :]) * x_range / bev_shape[1]
        anchor_points[:, 1, :, :] = (anchor_points[:, 1, :, :]) * y_range / bev_shape[0] + self.point_cloud_range[1]
        anchor_points[:, 2, :, :] = -1.7 + 1.56 / 2
        anchor_points = anchor_points[valid_idx[:, 0], :, valid_idx[:, 1], valid_idx[:, 2]]
        anchor_points = torch.cat([valid_idx[:, 0].reshape(-1, 1), anchor_points], dim=1)
        fused_feats = torch.cat([anchor_points[:, 1:], fused_feats], dim=1)

        return [fused_feats], anchor_points
    
    def forward_train(self,
                      points,
                      img,
                      plidar,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):

        x, anchor_points = self.extract_feat(points, img, plidar, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, anchor_points, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, plidar=None,rescale=False,
        gt_bboxes_3d=None, gt_labels_3d=None):

        x, anchor_points = self.extract_feat(points, imgs, plidar[0], img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            anchor_points, *outs, img_metas, rescale=rescale,
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        assert True, "Not implemented"
