import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from mmdet3d.models.builder import build_voxel_encoder, build_middle_encoder, build_backbone
from .single_stage import SingleStage3DDetector


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

        self.img_backbone = build_backbone(img_backbone)

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

    def extract_feat(self, points, img, img_metas):
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
        voxels, num_points, coors = self.voxelize(points, self.voxel_layer2)
        voxel_features = self.voxel_encoder2(voxels, num_points, coors)
        coors = coors.to(torch.long)
        bev_feats = x[0][coors[:, 0], :, coors[:, 2], coors[:, 3]]

        anchor_points = coors[:, [0, 2, 3]].to(torch.float32)
        x_range = self.point_cloud_range[3] - self.point_cloud_range[0]
        y_range = self.point_cloud_range[4] - self.point_cloud_range[1]
        anchor_size = self.bbox_head.anchor_cfg.size
        anchor_points[:, 1] = (anchor_points[:, 1] + 0.5)\
                              * y_range / bev_shape[0] + self.point_cloud_range[1]
        anchor_points[:, 2] = (anchor_points[:, 2] + 0.5) * x_range / bev_shape[1]

        z = torch.ones((anchor_points.shape[0], 1),
                        dtype=torch.float32,
                        device=coors.device) * (-1.7 + anchor_size[2] / 2)
        anchor_points = torch.cat([anchor_points[:, [2]], anchor_points[:, [1]], z], dim=1)
        pts_feats = [torch.cat([anchor_points, voxel_features, bev_feats], dim=1)]

        ## Extract RGB Features
        img_feats = self.img_backbone(img)

        ## Fusion
        # project anchor_points to image (x, y, z) -> (u, v)
        pts_2d = []
        for i in range(batch_size):
            res_lidar2img = img_metas[i]['lidar2img']
            img_shape = img_metas[i]['ori_shape'][:2]
            res_pts2d = self.project_to_img(anchor_points, img_shape, res_lidar2img)
            pts_2d.append(res_pts2d)
        print('test')


        return fused_feats, anchor_points
    
    def project_to_img(self, points, img_shape, lidar2img):
        device = points.device
        proj_mat = torch.from_numpy(lidar2img[:3]).to(device)
        points = points.transpose(1, 0)
        num_pts = points.shape[1]
        points = torch.cat((points, torch.ones((1, num_pts)).to(device)))
        points = proj_mat @ points
        points[:2, :] /= points[2, :]
        pts_2d = points[:2, :]
        pts_2d = pts_2d.permute(1, 0)
        pts_2d = torch.floor(pts_2d).to(torch.long)

        #filter points out of image
        valid_idx = torch.where(
            (pts_2d[:, 0] >= 0)&
            (pts_2d[:, 0] < img_shape[1])&
            (pts_2d[:, 1] >= 0)&
            (pts_2d[:, 1] < img_shape[0])
        )[0]
        pts_2d[valid_idx]
        return pts_2d

    def forward_train(self,
                      points,
                      img,
                      pseudo_lidar,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        #test
        import matplotlib.pyplot as plt
        import open3d as o3d
        from open3d import geometry
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.set_full_screen(True)
        vis.get_render_option().point_size = 0.01
        plidar = pseudo_lidar[0].cpu().reshape(3, -1).T
        valid_idx = torch.where(plidar[:, 0] != -1)[0]
        plidar = plidar[valid_idx]




        x, anchor_points = self.extract_feat(points, img, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, anchor_points, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False,
        gt_bboxes_3d=None, gt_labels_3d=None):

        x, anchor_points = self.extract_feat(points, imgs, img_metas)
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
