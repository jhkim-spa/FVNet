import torch
from mmcv.runner import force_fp32
from torch._C import device
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
class PVGNetD(SingleStage3DDetector):

    def __init__(self,
                 bev_interp,
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
        super(PVGNetD, self).__init__(
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

        if img_backbone is not None:
            self.img_backbone = build_backbone(img_backbone)
        self.img_neck = None
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        
        if aux_head is not None:
            self.aux_head = build_head(aux_head)
        self.weight_layer = nn.Sequential(
            nn.Linear(256*5 + 1, 5),
            nn.Sigmoid()
        )
        # self.weight_layer = nn.Sequential(
        #     nn.Linear(257, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid()
        # )
        self.channel_reduct_layer = nn.Sequential(
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

    # def fusion(self, lidar_feats, img_feats, img_metas):

    #     device = lidar_feats.device
    #     lidar2img = torch.from_numpy(img_metas['lidar2img'])[:3]
    #     width = img_metas['img_shape'][1]
    #     height = img_metas['img_shape'][0]
    #     xyz = lidar_feats[:, :3].clone()

    #     ## Back transformation for projection
    #     ## scale -> rot -> flip
    #     # scale
    #     xyz = xyz / img_metas['pcd_scale_factor']
    #     # rot
    #     if img_metas.get('pcd_rotation') is not None:
    #         rotation = img_metas['pcd_rotation'].to(device)
    #         xyz = xyz @ torch.inverse(rotation)
    #     # flip
    #     if img_metas['pcd_horizontal_flip']:
    #         xyz[:, 1] *= -1
    #     uv = self.project_to_img(xyz, lidar2img)

    #     ## scale uv with img resize scale factor
    #     w_scale, h_scale = img_metas['scale_factor'][:2]
    #     uv[:, 0] *= h_scale
    #     uv[:, 1] *= w_scale

    #     # flip uv if image flip is used
    #     if img_metas['flip']:
    #         uv[:, 1] = width - uv[:, 1] - 1

    #     ## fov filter
    #     valid_inds = torch.where(
    #         (uv[:, 0] < height) & (uv[:, 0] >= 0) &
    #         (uv[:, 1] < width)  & (uv[:, 1] >= 0)
    #     )[0]

    #     ## scale uv with img feature scale factor
    #     scale_factor = img_metas['img_shape'][0] / img_feats.shape[1]
    #     uv /= scale_factor

    #     uv = uv.to(torch.long)
    #     uv = uv[valid_inds] 
    #     lidar_feats = lidar_feats[valid_inds]

    #     matched_img_feats = img_feats[:, uv[:, 0], uv[:, 1]].T
    #     fused_feats = torch.cat([lidar_feats, matched_img_feats], dim=1)

    #     return fused_feats

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

        ### Visualize anchor centers
        # import matplotlib.pyplot as plt
        # img = torch.flip(img.cpu().permute([1, 2, 0]), [0])
        # plt.imshow(img.cpu().permute([1, 2, 0]))
        # plt.scatter(uv[:, 1].cpu().detach(),
        #             height-uv[:, 0].cpu().detach(),
        #             s=0.1,
        #             color='red')
        # plt.xlim(0, width)
        # plt.ylim(0, height)
        # plt.savefig('test.png', dpi=300)

        ## fov filter
        valid_inds = torch.where(
            (uv[:, 0] < height) & (uv[:, 0] >= 0) &
            (uv[:, 1] < width)  & (uv[:, 0] >= 0)
        )[0]
        uv = uv[valid_inds]
        lidar_feats = lidar_feats[valid_inds]

        ## scale uv with img feature scale factor
        num_scales = len(img_feats)
        matched_img_feats_list = []
        for i in range(num_scales):
            scale_factor = img_metas['img_shape'][0] / img_feats[i].shape[1]
            res_uv = uv / scale_factor
            res_uv = res_uv.to(torch.long)

            res_matched_img_feats = img_feats[i][:, res_uv[:, 0], res_uv[:, 1]].T
            ########### weighting with depth ##################
            # depth = xyz[:, :1][valid_inds]
            # res_img_feats_with_depth = torch.cat([depth, res_matched_img_feats], dim=1)
            # mlp
            # weights = self.weight_layer(res_img_feats_with_depth)
            matched_img_feats_list.append(res_matched_img_feats)
        
        matched_img_feats = torch.cat(matched_img_feats_list, dim=1)
        depth = xyz[:, :1][valid_inds]
        matched_img_feats = torch.cat([depth, matched_img_feats], dim=1)
        weights = self.weight_layer(matched_img_feats)

        matched_img_feats_list = [feats * weight.unsqueeze(dim=-1) for feats, weight\
            in zip(matched_img_feats_list, weights.T)]

        matched_img_feats = torch.cat(matched_img_feats_list, dim=1)
        matched_img_feats = self.channel_reduct_layer(matched_img_feats)
        fused_feats = torch.cat([lidar_feats, matched_img_feats], dim=1)

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

        ## Get anchor centers
        # method1. grid center
        # x_range = self.point_cloud_range[3] - self.point_cloud_range[0]
        # y_range = self.point_cloud_range[4] - self.point_cloud_range[1]
        # bev_shape = bev_feats.shape[2:]
        # anchor_size = self.bbox_head.anchor_cfg.size
        # anchor_grid = coors[:, [2, 3]].to(torch.float32) + 0.5
        # anchor_centers = torch.zeros((anchor_grid.shape[0], 3),
        #                               dtype=torch.float32,
        #                               device=device)
        # anchor_centers[:, 0] = anchor_grid[:, 1] * x_range / bev_shape[1]
        # anchor_centers[:, 1] = anchor_grid[:, 0] * y_range / bev_shape[0]\
        #                         + self.point_cloud_range[1]
        # lidar_height = 1.7
        # anchor_centers[:, 2] = -lidar_height + anchor_size[2] / 2
        # method2. voxel mean
        anchor_centers = voxels.sum(dim=1) / num_points.reshape(-1, 1)

        # Concat features
        coors = coors.to(torch.long)
        bev_feats = bev_feats[coors[:, 0], :, coors[:, 2], coors[:, 3]]
        lidar_feats = torch.cat([anchor_centers, bev_feats, voxel_feats], dim=1)

        batch_size = len(points)
        num_samples = [(coors[:, 0] == i).sum().item() for i in range(batch_size)]
        lidar_feats = lidar_feats.split(num_samples)

        return lidar_feats

    def extract_feat(self, points, img_metas, img=None):
        # TODO: Multi scale image features
        use_mlvl = True

        device = points[0].device

        lidar_feats = self.extract_lidar_feats(points)

        if img is None:
            img_feats = None
            """
            """
            return [lidar_feats], anchor_centers, img_feats

        if use_mlvl:
            img_feats = self.extract_img_feats(img)
            batch_size = len(points)
            fused_feats_list = []
            num_samples = []
            for i in range(batch_size):
                img_feats_batch = [feats[i] for feats in img_feats]
                feats = self.fusion_mlvl(lidar_feats[i], img_feats_batch, img_metas[i])
                fused_feats_list.append(feats)
                num_samples.append(feats.shape[0])
        else:
            img_feats = self.extract_img_feats(img)[0]
            batch_size = len(points)
            fused_feats_list = []
            num_samples = []
            for i in range(batch_size):
                feats = self.fusion(lidar_feats[i], img_feats[i], img_metas[i])
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
            # seg_result = outs_aux[0][0][0]
            # depth_result = outs_aux[1][0][0]

            # import matplotlib.pyplot as plt
            # seg_result = seg_result.sigmoid()
            # seg_result = seg_result.permute(1, 2, 0).reshape(384, 1248, 1)
            # seg_result = seg_result.detach().cpu()
            # plt.imshow(seg_result)
            # plt.savefig('seg.png', dpi=300)

            # plt.cla()
            # depth_result = depth_result
            # depth_result = depth_result.permute(1, 2, 0).reshape(384, 1248, 1)
            # depth_result = depth_result.detach().cpu()
            # plt.imshow(depth_result)
            # plt.savefig('depth.png', dpi=300)
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

        # seg, depth visualize test
        # if self.aux_head is not None:
        #     outs_aux = self.aux_head([img_feats[0]])
        #     seg_result = outs_aux[0][0]
        #     depth_result = outs_aux[1][0]

        #     import matplotlib.pyplot as plt
        #     seg_result = seg_result.sigmoid()
        #     seg_result = seg_result.permute(2, 3, 0, 1).reshape(384, 1248, 1)
        #     seg_result = seg_result.cpu()
        #     plt.imshow(seg_result)
        #     plt.savefig('seg.png', dpi=300)

        #     plt.cla()
        #     depth_result = depth_result
        #     depth_result = depth_result.permute(2, 3, 0, 1).reshape(384, 1248, 1)
        #     depth_result = depth_result.cpu()
        #     plt.imshow(depth_result)
        #     plt.savefig('depth.png', dpi=300)


        #     print('test')

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
