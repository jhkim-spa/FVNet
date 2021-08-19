import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

from ..builder import BACKBONES


class PyramidFeatures(nn.Module):
    '''
    FPN pyramid layer
    '''
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5 = self.P5_1(C5)
        P5_up = self.P5_upsampled(P5)
        P5 = self.P5_2(P5)

        P4 = self.P4_1(C4)
        P4 = P4 + P5_up
        P4_up = self.P4_upsampled(P4)
        P4 = self.P4_2(P4)

        P3 = self.P3_1(C3)
        P3 = P3 + P4_up
        P3 = self.P3_2(P3)

        P6 = self.P6(C5)
        P7 = self.P7_1(P6)
        P7 = self.P7_2(P7)

        return [P3, P4, P5, P6, P7]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, \
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes))
        self.stride = stride

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


@BACKBONES.register_module()
class FPN18(nn.Module):
    def __init__(self):
        super(FPN18, self).__init__()
        num_blocks = [2,2,2,2]
        bb_block = BasicBlock

        self.f_in_planes_det = 64

        # For RGB Feature Network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer_det(bb_block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer_det(bb_block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer_det(bb_block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_det(bb_block, 512, num_blocks[3], stride=2)
        fpn_sizes = [
                self.layer2[1].conv2.out_channels,
                self.layer3[1].conv2.out_channels,
                self.layer4[1].conv2.out_channels]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

    def _make_layer_det(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.f_in_planes_det, planes, stride))
            self.f_in_planes_det = planes * block.expansion
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        pth_path = 'pretrained/FPN18_retinanet_968.pth'
        pre_weights = torch.load(pth_path)
        new_res_state_dict = OrderedDict()
        model_dict = self.state_dict()
        for k,v in pre_weights['state_dict'].items():
            if ('regressionModel' not in k) and ('classificationModel' not in k):
                # name = k.replace('module', 'rpn')
                name = '.'.join(k.split('.')[1:])
                new_res_state_dict[name] = v
        model_dict.update(new_res_state_dict) 
        self.load_state_dict(model_dict)

    def forward(self, x):
        """Forward function."""
        f1 = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        x = self.fpn([f3, f4, f5])

        return x
