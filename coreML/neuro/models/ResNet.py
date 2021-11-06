import torch 
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor


__all__ = ["binary18", "binary34", "binary50", "binary101", "binary152", "binary50_32x4d"]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int =1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1
        
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: Tensor) -> Tensor:
        indentity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            indentity = self.downsample(x)
        
        out += indentity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    expansion: int = 4
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        indentity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            indentity = self.downsample(x)
        
        out += indentity
        out = self.relu(out)

        return out


class Binary(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Binary, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                            'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=5, stride=(1, 3), bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])
        #self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(512 * block.expansion, 80),
                                 nn.ReLU(),
                                 nn.Dropout())
        self.fc2 = nn.Linear(80, num_classes)
    
        
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                        base_width=self.base_width, dilation=self.dilation,
                        norm_layer=norm_layer))
        
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        # N x   1 x 60 x 601
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print(x.shape)
        # N x  64 x 56 x 66
        x = self.maxpool(x)
        #print(x.shape)
        # N x  64 x 27 x 32

        x = self.layer1(x)
        #print(x.shape)
        # N x 256 x 27 x 32
        x = self.layer2(x)
        #print(x.shape)
        # N x 512 x 14 x 16
        x = self.layer3(x)
        #print(x.shape)
        # N x 1024 x 7 x 8
        x = self.layer4(x)
        #print(x.shape)
        # N x 2048 x 4 x 4

        #x = self.maxpool(x)
        x = self.avgpool(x)
        #print(x.shape)
        #N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _binary(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    num_classes: int = 1,
    **kwargs: Any
) -> Binary:
    return Binary(block, layers, num_classes=num_classes, **kwargs)


def binary18() -> Binary:
    return _binary(BasicBlock, [2, 2, 2, 2])


def binary34() -> Binary:
    return _binary(BasicBlock, [3, 4, 6, 3])


def binary50() -> Binary:
    return _binary(Bottleneck, [3, 4, 6, 3])


def binary101() -> Binary:
    return _binary(Bottleneck, [3, 4, 23, 3])


def binary152() -> Binary:
    return _binary(Bottleneck, [3, 8, 36, 3])


def binary50_32x4d(**kwargs: Any) -> Binary:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _binary(Bottleneck, [3, 4, 6, 3], **kwargs)