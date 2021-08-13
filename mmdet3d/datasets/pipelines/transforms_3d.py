import numpy as np
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg
import torch
from copy import deepcopy

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import box_np_ops
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip
from ..registry import OBJECTSAMPLERS
from .data_augment_utils import noise_per_object_v3_
import numba


@PIPELINES.register_module()
class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        super(RandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            input_dict['points'] = input_dict[key].flip(
                direction, points=input_dict['points'])

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # filp 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(sync_2d={},'.format(self.sync_2d)
        repr_str += 'flip_ratio_bev_vertical={})'.format(
            self.flip_ratio_bev_vertical)
        return repr_str


@PIPELINES.register_module()
class ObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module()
class ObjectNoise(object):
    """Apply noise to each GT objects in the scene.

    Args:
        translation_std (list[float], optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list[float], optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list[float], optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(self,
                 translation_std=[0.25, 0.25, 0.25],
                 global_rot_range=[0.0, 0.0],
                 rot_range=[-0.15707963267, 0.15707963267],
                 num_try=100):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def __call__(self, input_dict):
        """Call function to apply noise to each ground truth in the scene.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after adding noise to each object, \
                'points', 'gt_bboxes_3d' keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        # TODO: check this inplace function
        numpy_box = gt_bboxes_3d.tensor.numpy()
        numpy_points = points.tensor.numpy()

        noise_per_object_v3_(
            numpy_box,
            numpy_points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(numpy_box)
        input_dict['points'] = points.new_point(numpy_points)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(num_try={},'.format(self.num_try)
        repr_str += ' translation_std={},'.format(self.translation_std)
        repr_str += ' global_rot_range={},'.format(self.global_rot_range)
        repr_str += ' rot_range={})'.format(self.rot_range)
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of ranslation
            noise. This apply random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        if not isinstance(self.translation_std, (list, tuple, np.ndarray)):
            translation_std = [
                self.translation_std, self.translation_std,
                self.translation_std
            ]
        else:
            translation_std = self.translation_std
        translation_std = np.array(translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        rotation = self.rot_range
        if not isinstance(rotation, list):
            rotation = [-rotation, rotation]
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T
        # input_dict['points_instance'].rotate(noise_rotation)

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys()
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(rot_range={},'.format(self.rot_range)
        repr_str += ' scale_ratio_range={},'.format(self.scale_ratio_range)
        repr_str += ' translation_std={})'.format(self.translation_std)
        repr_str += ' shift_height={})'.format(self.shift_height)
        return repr_str


@PIPELINES.register_module()
class PointShuffle(object):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        # input_dict['points'].shuffle()
        shuffle_idx = torch.randperm(input_dict['points'].__len__())
        input_dict['points'] = input_dict['points'][shuffle_idx]
        input_dict['shuffle_inds'] = shuffle_idx
        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.bev_range = self.pcd_range[[0, 1, 3, 4]]

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(self.bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        input_dict['points_range_mask'] = points_mask
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(point_cloud_range={})'.format(self.pcd_range.tolist())
        return repr_str


@PIPELINES.register_module()
class ObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@PIPELINES.register_module()
class IndoorPointSample(object):
    """Indoor point sample.

    Sampling data to a certain number.

    Args:
        name (str): Name of the dataset.
        num_points (int): Number of points to be sampled.
    """

    def __init__(self, num_points):
        self.num_points = num_points

    def points_random_sampling(self,
                               points,
                               num_samples,
                               replace=None,
                               return_choices=False):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (np.ndarray): 3D Points.
            num_samples (int): Number of samples to be sampled.
            replace (bool): Whether the sample is with or without replacement.
            Defaults to None.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[np.ndarray] | np.ndarray:

                - points (np.ndarray): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if replace is None:
            replace = (points.shape[0] < num_samples)
        choices = np.random.choice(
            points.shape[0], num_samples, replace=replace)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        points, choices = self.points_random_sampling(
            points, self.num_points, return_choices=True)

        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)
        results['points'] = points

        if pts_instance_mask is not None and pts_semantic_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            pts_semantic_mask = pts_semantic_mask[choices]
            results['pts_instance_mask'] = pts_instance_mask
            results['pts_semantic_mask'] = pts_semantic_mask

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(num_points={})'.format(self.num_points)
        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilter(object):
    """Filter background points near the bounding box.

    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (is_tuple_of(bbox_enlarge_range, float)
                and len(bbox_enlarge_range) == 3) \
            or isinstance(bbox_enlarge_range, float), \
            f'Invalid arguments bbox_enlarge_range {bbox_enlarge_range}'

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(
            bbox_enlarge_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        points = input_dict['points']
        gt_bboxes_3d = input_dict['gt_bboxes_3d']

        gt_bboxes_3d_np = gt_bboxes_3d.tensor.numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.numpy()
        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        points_numpy = points.tensor.numpy()
        foreground_masks = box_np_ops.points_in_rbbox(points_numpy,
                                                      gt_bboxes_3d_np)
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d)
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks,
                                      enlarge_foreground_masks)

        input_dict['points'] = points[valid_masks]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[valid_masks]

        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[valid_masks]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(bbox_enlarge_range={})'.format(
            self.bbox_enlarge_range.tolist())
        return repr_str


@PIPELINES.register_module()
class VoxelBasedPointSampler(object):
    """Voxel based point sampler.

    Apply voxel sampling to multiple sweep points.

    Args:
        cur_sweep_cfg (dict): Config for sampling current points.
        prev_sweep_cfg (dict): Config for sampling previous points.
        time_dim (int): Index that indicate the time dimention
            for input points.
    """

    def __init__(self, cur_sweep_cfg, prev_sweep_cfg=None, time_dim=3):
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg['max_num_points'] == \
                cur_sweep_cfg['max_num_points']
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.

        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimention of each points

        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros([
                sampler._max_voxels - voxels.shape[0], sampler._max_num_points,
                point_dim
            ],
                                      dtype=points.dtype)
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def __call__(self, results):
        """Call function to sample points from multiple sweeps.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        original_dim = points.shape[1]

        # TODO: process instance and semantic mask while _max_num_points
        # is larger than 1
        # Extend points with seg and mask fields
        map_fields2dim = []
        start_dim = original_dim
        points_numpy = points.tensor.numpy()
        extra_channel = [points_numpy]
        for idx, key in enumerate(results['pts_mask_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        start_dim += len(results['pts_mask_fields'])
        for idx, key in enumerate(results['pts_seg_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        points_numpy = np.concatenate(extra_channel, axis=-1)

        # Split points into two part, current sweep points and
        # previous sweeps points.
        # TODO: support different sampling methods for next sweeps points
        # and previous sweeps points.
        cur_points_flag = (points_numpy[:, self.time_dim] == 0)
        cur_sweep_points = points_numpy[cur_points_flag]
        prev_sweeps_points = points_numpy[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(cur_sweep_points,
                                               self.cur_voxel_generator,
                                               points_numpy.shape[1])
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(prev_sweeps_points,
                                                     self.prev_voxel_generator,
                                                     points_numpy.shape[1])

            points_numpy = np.concatenate(
                [cur_sweep_points, prev_sweeps_points], 0)
        else:
            points_numpy = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points_numpy = points_numpy.squeeze(1)
        results['points'] = points.new_point(points_numpy[..., :original_dim])

        # Restore the correspoinding seg and mask fields
        for key, dim_index in map_fields2dim:
            results[key] = points_numpy[..., dim_index]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""

        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split('\n')
            repr_str = [' ' * indent + t + '\n' for t in repr_str]
            repr_str = ''.join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += '(\n'
        repr_str += ' ' * indent + f'num_cur_sweep={self.cur_voxel_num},\n'
        repr_str += ' ' * indent + f'num_prev_sweep={self.prev_voxel_num},\n'
        repr_str += ' ' * indent + f'time_dim={self.time_dim},\n'
        repr_str += ' ' * indent + 'cur_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.cur_voxel_generator), 8)},\n'
        repr_str += ' ' * indent + 'prev_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.prev_voxel_generator), 8)})'
        return repr_str


@PIPELINES.register_module()
class ProjectToImage(object):

    def project_to_image(self, points, proj_mat):
        num_pts = points.shape[1]

        points = np.concatenate((points, np.ones((1, num_pts))))
        points = proj_mat @ points
        points[:2, :] /= points[2, :]
        return points[:2, :]

    def render_lidar_on_image(self, points, width, height, lidar2img):
        reflectance = points[:, -1]
        points = points[:, :3]
        proj_velo2cam2 = lidar2img[:3]

        pts_2d = self.project_to_image(points.transpose(1, 0),
                                       proj_velo2cam2)

        inds = np.where((pts_2d[0, :] < width) & (pts_2d[0, :] >= 0) &
                        (pts_2d[1, :] < height) & (pts_2d[1, :] >= 0) &
                        (points[:, 0] > 0)
                        )[0]

        imgfov_pc_pixel = pts_2d[:, inds]
        imgfov_pc_velo = points[inds, :]
        reflectance = reflectance[inds]

        pc_projected = np.zeros((height, width, 5),
            dtype=np.float32)
        x_coords = np.trunc(imgfov_pc_pixel[0]).astype(np.int32)
        y_coords = np.trunc(imgfov_pc_pixel[1]).astype(np.int32)
        pc_projected[y_coords, x_coords, :3] = imgfov_pc_velo
        pc_projected[y_coords, x_coords, 3] = reflectance
        flag_channel = (pc_projected[:, :, 0] != 0)
        pc_projected[:, :, -1] = flag_channel
        return pc_projected

    def __call__(self, input_dict):
        from mmdet3d.core.bbox.box_np_ops import points_in_rbbox
        width = input_dict['img_info']['width']
        height = input_dict['img_info']['height']
        lidar2img = input_dict['lidar2img']
        points = deepcopy(input_dict['points'])

        points = points.tensor.numpy()
        fv = self.render_lidar_on_image(points, width, height, lidar2img)

        # if True:
        if False:
            gt_bboxes_3d = input_dict['ann_info']['gt_bboxes_3d']
            objectness_idx = points_in_rbbox(points, gt_bboxes_3d.tensor.numpy())
            objectness_idx = objectness_idx.sum(axis=1).astype(np.int32)
            objectness_idx = np.where(objectness_idx == 1)[0]
            objectness_points = points[objectness_idx]
            objectness = self.get_objectness(objectness_points, width, height, lidar2img)
            fv = np.concatenate((fv, objectness), axis=2)
            fv = fv[:, :, [0, 1, 2, 3, 5, 4]]

        input_dict['fv'] = fv
        return input_dict


@PIPELINES.register_module()
class ResizeFV(object):

    def __init__(self, size):
        self.size = size

    def _resize_fv(self, fv, size):
        w_des, h_des = size
        w_scale = w_des / fv.shape[1]
        h_scale = h_des / fv.shape[0]

        if (w_scale == 1. and h_scale == 1.):
            return fv
        else:
            fv_src = fv
            fv_des = np.zeros((h_des, w_des, fv_src.shape[-1]),
                               dtype=np.float32)
            idx_src = np.nonzero(fv_src)
            idx_des = list()
            idx_des.append((h_scale * idx_src[0]).astype(np.int32))
            idx_des.append((w_scale * idx_src[1]).astype(np.int32))
            fv_des[idx_des[0], idx_des[1], :] = \
                fv_src[idx_src[0], idx_src[1], :]
            return fv_des

    def __call__(self, results):

        size = self.size
        fv = results['fv']
        fv = self._resize_fv(fv, size)
        results['fv'] = fv
        return results


@PIPELINES.register_module()
class Fusion(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        inputs = []
        for key in self.keys:
            res = results[key]
            if key == 'img':
                res = torch.from_numpy(res)
            inputs.append(res)
        results['points'] = torch.cat(inputs, dim=2)
        return results


@PIPELINES.register_module()
class PadFV(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_fv(self, results):
        """Pad images according to ``self.size``."""
        for key in ['fv']:
            if self.size is not None:
                if self.size != results[key].shape[:2]:
                    results[key] = self._pad(
                        results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                divisor = self.size_divisor
                pad_h = int(np.ceil(results[key].shape[0] / divisor)) * divisor
                pad_w = int(np.ceil(results[key].shape[1] / divisor)) * divisor
                size = (pad_h, pad_w)
                if size != results[key].shape[:2]:
                    results[key] = self._pad(
                        results[key], shape=size, pad_val=self.pad_val)

    def _pad(self, fv, shape, pad_val=0.):
        ori_shape = fv.shape
        padded = np.ones((shape[0], shape[1], ori_shape[-1]),
                          dtype=np.float32) * pad_val
        padded[:ori_shape[0], :ori_shape[1], :] = fv
        return padded

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_fv(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class ScalePoints(object):

    def __init__(self,
                 scale_ratio_range=[0.95, 1.05]):
        self.scale_ratio_range = scale_ratio_range

    def _scale_bbox_points(self, input_dict):

        scale = input_dict['pcd_scale_factor']
        fv = input_dict['fv']
        fv[:, :, :3] *= scale
        input_dict['fv'] = fv

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):

        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):

        self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        return input_dict


@PIPELINES.register_module()
class RotPoints(object):

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816]):
        self.rot_range = rot_range

    def _rot_bbox_points(self, input_dict):

        rotation = self.rot_range
        if not isinstance(rotation, list):
            rotation = [-rotation, rotation]
        noise_rotation = np.random.uniform(rotation[0], rotation[1])
        fv = input_dict['fv']

        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                input_dict[key].rotate(noise_rotation)
                fv, rot_mat_T = self._rotate(fv, noise_rotation)
                input_dict['fv'] = fv
                input_dict['pcd_rotation'] = rot_mat_T

    def _rotate(self, fv, angle):
        angle = np.array(angle, dtype=np.float32)
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        rot_mat_T = np.array([[rot_cos, -rot_sin, 0],
                              [rot_sin, rot_cos, 0], [0, 0, 1]],
                              dtype=np.float32)
        fv = fv
        idx = fv[:, :, 0].nonzero()
        points = fv[:, :, :3][idx[0], idx[1], :]
        points = points @ rot_mat_T
        fv[:, :, :3][idx[0], idx[1], :] = points


        return fv, rot_mat_T

    def __call__(self, input_dict):

        self._rot_bbox_points(input_dict)

        return input_dict


@PIPELINES.register_module()
class RandomFlipFV(object):

    def __init__(self,
                 flip_ratio=0.5,
                 sync_2d=False):
        self.flip_ratio = flip_ratio
        self.sync_2d = sync_2d

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        input_dict['fv'][:, :, 1] *= -1
        input_dict['fv'] = np.flip(input_dict['fv'], 1)
        input_dict['gt_bboxes_3d'].flip(direction)

    def __call__(self, input_dict):

        if 'pcd_horizontal_flip' not in input_dict:
            flip_horizontal = True if np.random.rand(
            ) < self.flip_ratio else False
            input_dict['pcd_horizontal_flip'] = flip_horizontal

        if self.sync_2d:
            if input_dict['flip']:
                self.random_flip_data_3d(input_dict, 'horizontal')
        else:
            if input_dict['pcd_horizontal_flip']:
                self.random_flip_data_3d(input_dict, 'horizontal')
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    

@PIPELINES.register_module()
class ImagePointsMatching(object):

    def __init__(self, phase):
        self.phase = phase
    
    def project_to_image(self, points, proj_mat):
        num_pts = points.shape[1]

        points = torch.cat((points, torch.ones((1, num_pts))))
        points = proj_mat @ points
        points[:2, :] /= points[2, :]
        return points[:2, :]

    def render_lidar_on_image(self, points, lidar2img):
        points = points[:, :3]
        proj_velo2cam2 = torch.from_numpy(lidar2img[:3])
        pts_2d = self.project_to_image(points.transpose(1, 0),
                                        proj_velo2cam2)
        pts_2d = pts_2d.permute(1, 0)
        pts_2d = torch.floor(pts_2d).to(torch.long)

        return pts_2d

    def __call__(self, input_dict):
        if self.phase == 'initial':
            lidar2img = input_dict['lidar2img']
            points = input_dict['points'].tensor
            pts_2d = self.render_lidar_on_image(points, lidar2img)
            input_dict['pts_2d'] = pts_2d
            return input_dict

        elif self.phase == 'resize':
            scale_factor = input_dict['scale_factor']
            pts_2d = input_dict['pts_2d']
            pts_2d[:, 0] = (pts_2d[:, 0] * scale_factor[0]).to(torch.long)
            pts_2d[:, 1] = (pts_2d[:, 1] * scale_factor[1]).to(torch.long)
            input_dict['pts_2d'] = pts_2d
            return input_dict

        elif self.phase == 'flip':
            if input_dict['flip']:
                w = input_dict['img'].shape[1]
                pts_2d = input_dict['pts_2d']
                pts_2d[:, 0] = (pts_2d[:, 0] - w + 1) * -1
                input_dict['pts_2d'] = pts_2d
                return input_dict
            else:
                return input_dict

        elif self.phase == 'points_range':
            points_mask = input_dict['points_range_mask']
            input_dict['pts_2d'] = input_dict['pts_2d'][points_mask]
            return input_dict

        elif self.phase == 'points_shuffle':
            shuffle_inds = input_dict['shuffle_inds']
            input_dict['pts_2d'] = input_dict['pts_2d'][shuffle_inds]
            return input_dict


@PIPELINES.register_module()
class DepthToLidarPoints(object):
    def __call__(self, input_dict):
        depth = input_dict['depth']
        P2 = input_dict['calib_info']['P2']
        r_rect = input_dict['calib_info']['rect']
        velo2cam = input_dict['calib_info']['velo2cam']
        depth = np.array(depth)
        lidar_points = self.depth_to_lidar_points(
            depth[0], 0, P2, r_rect, velo2cam)
        input_dict['pseudo_lidar'] = lidar_points
        return input_dict

    def depth_to_points(self, depth, trunc_pixel):
        points = np.ones((3, depth.shape[0], depth.shape[1]), dtype=depth.dtype)
        points[0, :, :] = np.tile(np.array(range(depth.shape[1])).reshape(1, -1),
                                 (depth.shape[0], 1))
        points[1, :, :] = np.tile(np.array(range(depth.shape[0])).reshape(-1, 1),
                                 (1, depth.shape[1]))
        lidar_points = points * depth
        return lidar_points

    def camera_to_lidar(self, points, r_rect, velo2cam):
        points_shape = list(points.shape[0:-1])
        if points.shape[-1] == 3:
            points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
        lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
        return lidar_points[..., :3]

    def depth_to_lidar_points(self, depth, trunc_pixel, P2, r_rect, velo2cam):
        pts = self.depth_to_points(depth, trunc_pixel)
        ori_shape = pts.shape
        pts = pts.reshape(3, -1).T
        points_shape = list(pts.shape[0:-1])
        points = np.concatenate([pts, np.ones(points_shape + [1])], axis=-1)
        points = points @ np.linalg.inv(P2.T)
        lidar_points = self.camera_to_lidar(points, r_rect, velo2cam)
        # lidar_points = lidar_points.T.reshape(ori_shape)
        return lidar_points


@PIPELINES.register_module()
class PsuedoPointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        pseudo_lidar = input_dict['pseudo_lidar']
        ignore_mask = np.where(
            (pseudo_lidar[:, 0] < self.pcd_range[0]) |
            (pseudo_lidar[:, 1] < self.pcd_range[1]) |
            (pseudo_lidar[:, 2] < self.pcd_range[2]) |
            (pseudo_lidar[:, 0] > self.pcd_range[3]) |
            (pseudo_lidar[:, 1] > self.pcd_range[4]) |
            (pseudo_lidar[:, 2] > self.pcd_range[5])
        )[0]
        pseudo_lidar[ignore_mask, :] = -1
        return input_dict


@PIPELINES.register_module()
class PseudoPointsFlip(object):
    def __call__(self, input_dict):
        if input_dict['pcd_horizontal_flip']:
            input_dict['pseudo_lidar'][:, 1] *= -1
        return input_dict


@PIPELINES.register_module()
class PseudoPointsRotScale(object):
    def __call__(self, input_dict):
        rotation = np.array(input_dict['pcd_rotation'])
        scale = input_dict['pcd_scale_factor']
        pseudo_lidar = input_dict['pseudo_lidar']
        pseudo_lidar = pseudo_lidar @ rotation
        pseudo_lidar *= scale
        input_dict['pseudo_lidar'] = pseudo_lidar
        return input_dict


@PIPELINES.register_module()
class PseudoPointsToImage(object):
    def __call__(self, input_dict):
        img_shape = input_dict['img_shape']
        pseudo_lidar = input_dict['pseudo_lidar']
        pseudo_lidar = pseudo_lidar.T.reshape(img_shape[2],
                                              img_shape[0],
                                              img_shape[1])
        input_dict['pseudo_lidar'] = pseudo_lidar
        return input_dict