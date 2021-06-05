from mmdet3d.models.detectors.base import Base3DDetector
import torch
from torch import nn as nn

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS, build_backbone, build_neck, build_head
from .base import Base3DDetector


@DETECTORS.register_module()
class FVNet(Base3DDetector):

    def __init__(self,
                 backbone_fv,
                 backbone_img=None,
                 neck_fv=None,
                 neck_img=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FVNet, self).__init__()
        self.backbone_fv = build_backbone(backbone_fv)
        self.backbone_img = build_backbone(backbone_img)
        if neck_fv is not None:
            self.neck_fv = build_neck(neck_fv)
        if neck_img is not None:
            self.neck_img = build_neck(neck_img)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize weights of detector."""
        super(FVNet, self).init_weights(pretrained)
        self.backbone_img.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck_img, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck_img.init_weights()
        self.bbox_head.init_weights()

    def extract_feat_fv(self, fv):
        x = self.backbone_fv(fv)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def extract_feat_img(self, img):
        x = self.backbone_img(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat(self, fv, img, img_metas):
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

        feats_fv = self.extract_feat_fv(fv)
        if img is not None:
            feats_img = self.extract_feat_img(img)
        else:
            feats_img = None
        return feats_fv, feats_img, valid_coords

    def forward_train(self,
                      fv,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img=None,
                      gt_bboxes_ignore=None):

        feats_fv, feats_img, valid_coords = self.extract_feat(fv, img, img_metas)
        if img is not None:
            # concat
            x = [torch.cat((feat_fv, feat_img), dim=1)for feat_fv, feat_img\
                in zip(feats_fv, feats_img)]
        else:
            x = feats_fv
        outs = self.bbox_head(x, valid_coords['2d'])
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, valid_coords, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
    
    def simple_test(self, fv, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        feats_fv, feats_img, valid_coords = self.extract_feat(fv, img, img_metas)
        if img is not None:
            # concat
            x = [torch.cat((feat_fv, feat_img), dim=1)for feat_fv, feat_img\
                in zip(feats_fv, feats_img)]
        else:
            x = feats_fv
        outs = self.bbox_head(x, valid_coords['2d'])
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
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(fv, 'fv'), (img_metas, 'img_metas')]:
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