# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat, Conv2d
from detectron2.structures import BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from .point_features import point_sample

TRACK_HEAD_REGISTRY = Registry("TRACK_HEAD")
TRACK_HEAD_REGISTRY.__doc__ = """
Registry for track heads, which tracks bbox in frames of videos.

The registered object will be called with `obj(cfg, input_shape)`.
"""

@POINT_HEAD_REGISTRY.register()
class StandardTrackHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super(StandardTrackHead, self).__init__()
        # fmt: off
        num_conv                    = cfg.MODEL.TRACK_HEAD.NUM_CONV
        input_channels              = input_shape.channels
        # fmt: on

        fc_dim_in = input_channels
        self.fc_layers = []
        for k in range(num_conv):
            fc = Conv2d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.fc_layers:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, fine_grained_features, coarse_features):
        x = torch.cat((fine_grained_features, coarse_features), dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = cat((x, coarse_features), dim=1)
        return self.predictor(x)


def build_point_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.POINT_HEAD.NAME
    return POINT_HEAD_REGISTRY.get(head_name)(cfg, input_channels)
