# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.structures import BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from .point_features import point_sample

POINT_HEAD_REGISTRY = Registry("POINT_HEAD")
POINT_HEAD_REGISTRY.__doc__ = """
Registry for point heads, which makes prediction for a given set of per-point features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

@POINT_HEAD_REGISTRY.register()
class BoxFeatureMerger(nn.Module):
    """
    To merge box_features and ref_box_features from last frame
    """

    def __init__(self, cfg, input_channels: int):
        """
        The following attributes are parsed from config:
            num_fc: the number of FC layers
        """
        super(BoxFeatureMerger, self).__init__()
        # fmt: off
        num_conv                      = cfg.MODEL.FEATRURE_MERGE.NUM_CONV
        # fmt: on
        conv_dim_in = input_channels * 2
        conv_dim = input_channels
        self.conv_layers = []
        for k in range(num_conv):
            conv = nn.Conv2d(conv_dim_in, conv_dim, kernel_size=3, stride=1, padding=1, bias=True)
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_layers.append(conv)
            conv_dim_in = conv_dim

        for layer in self.conv_layers:
            weight_init.c2_msra_fill(layer)

    def forward(self, box_feature, ref_box_feature):
        x = torch.cat((box_feature, ref_box_feature), dim=1)
        for layer in self.conv_layers:
            x = F.relu(layer(x))
        x = x + box_feature
        return x

