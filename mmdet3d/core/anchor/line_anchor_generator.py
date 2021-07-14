import mmcv
import torch
import numpy as np
from mmdet3d.core.bbox import Box3DMode, CameraInstance3DBoxes, get_box_type

from mmdet.core.anchor import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register_module()
class LineAnchorGenerator(object):

    def __init__(self,
                 dist_list,
                 num_bins,
                 ranges,
                 sizes=[[1.6, 3.9, 1.56]],
                 scales=[1],
                 rotations=[0, 1.5707963],
                 custom_values=(),
                 reshape_out=True,
                 size_per_range=True):
        assert mmcv.is_list_of(ranges, list)
        if size_per_range:
            if len(sizes) != len(ranges):
                assert len(ranges) == 1
                ranges = ranges * len(sizes)
            assert len(ranges) == len(sizes)
        else:
            assert len(ranges) == 1
        assert mmcv.is_list_of(sizes, list)
        assert isinstance(scales, list)

        self.dist_list = dist_list
        self.num_bins = num_bins
        self.sizes = sizes
        self.scales = scales
        self.ranges = ranges
        self.rotations = rotations
        self.custom_values = custom_values
        self.cached_anchors = None
        self.reshape_out = reshape_out
        self.size_per_range = size_per_range
        self.multi_level_anchors = None

        self.box_type_3d, self.box_mode_3d = get_box_type('LiDAR')
        # self.cam2lidar = torch.tensor([[2.3477350e-04,  1.0449406e-02,  9.9994540e-01,  2.7290344e-01],
        #     [-9.9994421e-01,  1.0565354e-02,  1.2436594e-04, -1.9692658e-03],
        #     [-1.0563478e-02, -9.9988955e-01,  1.0451305e-02, -7.2285898e-02],
        #     [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
        #     dtype=torch.float32)
        self.cam2lidar = torch.tensor([[ 7.5337449e-03,  1.4802488e-02,  9.9986202e-01,  2.7290344e-01],
            [-9.9997145e-01,  7.2807324e-04,  7.5237905e-03, -1.9692658e-03],
            [-6.1660202e-04, -9.9989015e-01,  1.4807552e-02, -7.2285898e-02],
            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
            dtype=torch.float32)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'anchor_range={self.ranges},\n'
        s += f'scales={self.scales},\n'
        s += f'sizes={self.sizes},\n'
        s += f'rotations={self.rotations},\n'
        s += f'reshape_out={self.reshape_out},\n'
        s += f'size_per_range={self.size_per_range})'
        return s

    @property
    def num_base_anchors(self):
        """list[int]: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    @property
    def num_levels(self):
        """int: Number of feature levels that the generator is applied to."""
        return len(self.scales)

    def grid_anchors(self, featmap_sizes, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.

        Returns:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature lavel, \
                num_base_anchors is the number of anchors for that level.
        """
        if self.multi_level_anchors is not None:
            return self.multi_level_anchors, self.multi_level_val_masks
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        multi_level_val_masks = []
        dist_list = []
        for i in range(self.num_levels):
            dist_list.append(self.dist_list[i*self.num_bins:(i+1)*self.num_bins]) 
        dist_list.reverse()
        for i in range(self.num_levels):
            anchors, val_masks = self.single_level_grid_anchors(
                featmap_sizes[i], dist_list[i], self.scales[i], device=device)
            if self.reshape_out:
                anchors = anchors.reshape(-1, anchors.size(-1))
            multi_level_anchors.append(anchors)
            multi_level_val_masks.append(val_masks)
        self.multi_level_anchors = multi_level_anchors
        self.multi_level_val_masks = multi_level_val_masks
        return multi_level_anchors, multi_level_val_masks

    def single_level_grid_anchors(self, featmap_size, dist_list,
                                  scale, device='cuda'):
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            featmap_size (tuple[int]): Size of the feature map.
            scale (float): Scale factor of the anchors in the current level.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        # We reimplement the anchor generator using torch in cuda
        # torch: 0.6975 s for 1000 times
        # numpy: 4.3345 s for 1000 times
        # which is ~5 times faster than the numpy implementation
        if not self.size_per_range:
            return self.anchors_single_range(
                featmap_size,
                self.ranges[0],
                scale,
                self.sizes,
                self.rotations,
                device=device)

        mr_anchors = []
        mr_val_masks = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchor, mr_val_mask = self.anchors_single_range(
                                        featmap_size,
                                        dist_list,
                                        anchor_range,
                                        scale,
                                        anchor_size,
                                        self.rotations,
                                        device=device)
            mr_anchors.append(mr_anchor)
            mr_val_masks.append(mr_val_mask)
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        mr_val_masks = torch.cat(mr_val_masks, dim=-2)
        return mr_anchors, mr_val_masks

    def anchors_single_range(self,
                             feature_size,
                             dist_list,
                             anchor_range,
                             scale=1,
                             sizes=[[1.6, 3.9, 1.56]],
                             rotations=[0, 1.5707963],
                             device='cuda'):
        # TODO
        # 1. Convert camera coord to lidar coord
        # 2. Multi-class
        # 3. filtering with anchor_range
        anchors = []
        val_masks = []
        sizes = torch.tensor(sizes).reshape(-1, 3)
        rotations = torch.tensor(rotations).reshape(-1, 1)
        for dist in dist_list:
            ref_centers = self.get_ref_points(*feature_size).T
            k = dist / torch.sqrt((1. + ref_centers[:, 0]**2 + ref_centers[:, 1]**2))
            center = k.reshape(-1, 1) * ref_centers
            # idx 0: y, 1: z, 2: x
            num_centers = center.shape[0]
            center = torch.repeat_interleave(center, 2, dim=0)
            size = sizes.repeat((num_centers * 2, 1))
            rotation = rotations.repeat(num_centers, 1)
            anchor = torch.cat((center, size, rotation), dim=1)
            anchor = anchor.reshape(-1, 2, 7)

            # import matplotlib.pyplot as plt
            # anchor = anchor.cpu()
            # plt.scatter(anchor[..., 2], -anchor[..., 0], s=0.1)
            # plt.savefig('test_.png')

            anchor = self.convert_cam2lidar(anchor)
            anchor[..., 2] = anchor[..., 2] - anchor[..., 5] / 2
            val_mask = (anchor[..., 0] >= anchor_range[0]) &\
                       (anchor[..., 0] <= anchor_range[3]) &\
                       (anchor[..., 1] >= anchor_range[1]) &\
                       (anchor[..., 1] <= anchor_range[4]) &\
                       (anchor[..., 2] >= anchor_range[2]) &\
                       (anchor[..., 2] <= anchor_range[5])
            # anchor[..., 2] = -1.78
            # val_mask = anchor[..., 0] > -100000
            anchors.append(anchor)
            val_masks.append(val_mask)
        anchors = torch.stack(anchors)
        anchors = anchors.permute(1, 0, 2, 3)
        anchors = anchors.reshape(1, *feature_size, 1, -1, 7)
        val_masks = torch.stack(val_masks)
        val_masks = val_masks.permute(1, 0, 2)
        val_masks = val_masks.reshape(1, *feature_size, 1, -1)

        return anchors, val_masks

    def convert_cam2lidar(self, boxes):
        # boxes = boxes.reshape(-1, 7)
        # boxes[:, [3, 4, 5]] = boxes[:, [4, 5, 3]]
        # boxes = boxes.contiguous()
        # boxes = CameraInstance3DBoxes(boxes).convert_to(self.box_mode_3d,
        #     self.cam2lidar)
        
        # boxes = boxes.tensor.reshape(-1, 2, 7)

        boxes = boxes.reshape(-1, 7)
        boxes[:, [0, 1, 2]] = boxes[:, [2, 0, 1]]
        boxes[:, [1, 2]] = -boxes[:, [1, 2]]

        # cam2 to cam0
        boxes[:, 1] += 0.06
        # cam0 to velo
        boxes[:, 2] -= 0.08
        boxes[:, 0] += 0.27

        # boxes = boxes.tensor.reshape(-1, 2, 7)
        boxes = boxes.reshape(-1, 2, 7)
        return boxes

    def get_ref_points(self, height, width):
        K = self.intrinsic_from_fov(height, width, 90)
        K_inv = torch.tensor(np.linalg.inv(K), dtype=torch.float32)
        pixel_coords = torch.tensor(self.pixel_coord_np(width, height),
                                    dtype=torch.float32)
        cam_coords = K_inv[:3, :3] @ pixel_coords * 1.0

        return cam_coords

    def pixel_coord_np(self, width, height):
        """
        Pixel in homogenous coordinate
        Returns:
            Pixel coordinate:       [3, width * height]
        """
        x = np.linspace(0, width - 1, width).astype(np.int)
        y = np.linspace(0, height - 1, height).astype(np.int)
        [x, y] = np.meshgrid(x, y)
        return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


    def intrinsic_from_fov(self, height, width, fov=90):
        """
        Basic Pinhole Camera Model
        intrinsic params from fov and sensor width and height in pixels
        Returns:
            K:      [4, 4]
        """
        px, py = (width / 2, height / 2)
        hfov = fov / 360. * 2. * np.pi
        fx = width / (2. * np.tan(hfov / 2.))

        vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
        fy = height / (2. * np.tan(vfov / 2.))

        return np.array([[fx, 0, px, 0.],
                        [0, fy, py, 0.],
                        [0, 0, 1., 0.],
                        [0., 0., 0., 1.]])
