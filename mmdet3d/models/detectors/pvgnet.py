import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from mmdet3d.models.builder import (build_neck, build_voxel_encoder,
                                    build_middle_encoder, build_backbone,
                                    build_head)
from .single_stage import SingleStage3DDetector
import torch.nn as nn


@DETECTORS.register_module()
class PVGNet(SingleStage3DDetector):

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 voxel_layer2=None,
                 voxel_encoder2=None,
                 img_backbone=None,
                 img_neck=None,
                 bbox_head=None,
                 aux_head=None,
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
        self.point_cloud_range = voxel_layer.point_cloud_range
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = build_voxel_encoder(voxel_encoder)
        self.middle_encoder = build_middle_encoder(middle_encoder)

        if voxel_layer2 is not None:
            self.voxel_layer2 = Voxelization(**voxel_layer2)
            self.voxel_encoder2 = build_voxel_encoder(voxel_encoder2)

        if img_backbone is not None:
            self.img_backbone = build_backbone(img_backbone)
        self.img_neck = None
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        
        self.aux_head = None
        if aux_head is not None:
            self.aux_head = build_head(aux_head)

        # self.lidar_channel_reduct_layer = nn.Sequential(
        #     nn.Linear(515, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True)
        # )
        self.img_channel_reduct_layer = nn.Sequential(
            nn.Linear(256*5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

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

    def fusion_mlvl(self, lidar_feats, img_feats, img_metas):

        device = lidar_feats.device
        lidar2img = torch.from_numpy(img_metas['lidar2img'])[:3]
        width = img_metas['img_shape'][1]
        height = img_metas['img_shape'][0]
        xyz = lidar_feats[:, :3].clone()

        ## Back transformation for projection
        ## scale -> rot -> flip
        # scale
        xyz = xyz / img_metas['pcd_scale_factor']
        # rot
        if img_metas.get('pcd_rotation') is not None:
            rotation = img_metas['pcd_rotation'].to(device)
            xyz = xyz @ torch.inverse(rotation)
        # flip
        if img_metas['pcd_horizontal_flip']:
            xyz[:, 1] *= -1
        uv = self.project_to_img(xyz, lidar2img)

        ## scale uv with img resize scale factor
        w_scale, h_scale = img_metas['scale_factor'][:2]
        uv[:, 0] *= h_scale
        uv[:, 1] *= w_scale

        # flip uv if image flip is used
        if img_metas['flip']:
            uv[:, 1] = width - uv[:, 1] - 1

        ## fov filter
        valid_inds = torch.where(
            (uv[:, 0] < height) & (uv[:, 0] >= 0) &
            (uv[:, 1] < width)  & (uv[:, 0] >= 0)
        )[0]
        uv = uv[valid_inds]
        lidar_feats = lidar_feats[valid_inds]

        ## scale uv with img feature scale factor
        num_scales = len(img_feats)
        matched_img_feats = []
        for i in range(num_scales):
            scale_factor = img_metas['img_shape'][0] / img_feats[i].shape[1]
            res_uv = uv / scale_factor
            res_uv = res_uv.to(torch.long)

            res_matched_img_feats = img_feats[i][:, res_uv[:, 0], res_uv[:, 1]].T
            matched_img_feats.append(res_matched_img_feats)
        matched_img_feats = torch.cat(matched_img_feats, dim=1)
        img_feats = self.img_channel_reduct_layer(matched_img_feats)
        # lidar_feats = self.lidar_channel_reduct_layer(lidar_feats)

        fused_feats = torch.cat([lidar_feats, img_feats], dim=1)

        return fused_feats

    def extract_img_feats(self, img):

        img_feats = self.img_backbone(img)
        if self.img_neck is not None:
            img_feats = self.img_neck(img_feats)

        return img_feats
    
    def extract_lidar_feats(self, points):
        # BEV feats
        voxels, num_points, coors = self.voxelize(points, self.voxel_layer)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        bev_feats = x[0]

        # VFE
        points_only_xyz = [res[:, :3].contiguous() for res in points]
        voxels, num_points, coors = self.voxelize(points_only_xyz, self.voxel_layer2)
        voxel_feats = self.voxel_encoder2(voxels, num_points, coors)

        anchor_centers = voxels.sum(dim=1) / num_points.reshape(-1, 1)

        # Concat features
        coors = coors.to(torch.long)
        bev_feats = bev_feats[coors[:, 0], :, coors[:, 2], coors[:, 3]]
        lidar_feats = torch.cat([anchor_centers, bev_feats, voxel_feats], dim=1)

        batch_size = len(points)
        num_samples = [(coors[:, 0] == i).sum().item() for i in range(batch_size)]
        lidar_feats = lidar_feats.split(num_samples)

        return lidar_feats, coors

    def extract_feat(self, points, img_metas, img=None):
        device = points[0].device

        lidar_feats, coors = self.extract_lidar_feats(points)

        if img is None:
            img_feats = None
            lidar_feats = torch.cat(lidar_feats, dim=0)
            anchor_centers = lidar_feats[:, :3].clone()
            anchor_centers = torch.cat([coors[:, :1], anchor_centers], dim=1)
            return [lidar_feats], anchor_centers, img_feats

        img_feats = self.extract_img_feats(img)
        batch_size = len(points)
        fused_feats_list = []
        num_samples = []
        for i in range(batch_size):
            img_feats_batch = [feats[i] for feats in img_feats]
            feats = self.fusion_mlvl(lidar_feats[i], img_feats_batch, img_metas[i])
            fused_feats_list.append(feats)
            num_samples.append(feats.shape[0])

        fused_feats = torch.cat(fused_feats_list)
        batch_idx = [[i] * num for i, num in enumerate(num_samples)]
        batch_idx = torch.cat([torch.tensor(l) for l in batch_idx]).reshape(-1, 1)
        batch_idx = batch_idx.to(device)
        anchor_centers = fused_feats[:, :3].clone()
        anchor_centers = torch.cat([batch_idx, anchor_centers], dim=1)
        return [fused_feats], anchor_centers, img_feats
    
    def forward_train(self,
                      points,
                      img_metas,
                      img=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_bboxes_ignore=None):

        x, anchor_centers, img_feats = self.extract_feat(points, img_metas, img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, anchor_centers, gt_bboxes_ignore=gt_bboxes_ignore)
        
        if self.aux_head is not None:
            outs_aux = self.aux_head([img_feats[0]])
            loss_inputs = outs_aux + (points, gt_bboxes_3d, gt_labels_3d, img_metas)
            losses_aux = self.aux_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(losses_aux)

        return losses

    def simple_test(self,
                    points,
                    img_metas,
                    imgs=None,
                    rescale=False,
                    gt_bboxes_3d=None,
                    gt_labels_3d=None):

        x, anchor_centers, img_feats = self.extract_feat(points, img_metas, imgs)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            anchor_centers, *outs, img_metas, rescale=rescale,
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results

    def aug_test(self,
                 points,
                 img_metas,
                 imgs=None,
                 rescale=False):

        assert True, "Not implemented"
