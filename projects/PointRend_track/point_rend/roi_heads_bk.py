# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import cv2

from detectron2.layers import ShapeSpec, cat, interpolate
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.mask_head import (
    build_mask_head,
    mask_rcnn_inference,
    mask_rcnn_loss,
)
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals

from .point_features import (
    generate_regular_grid_point_coords,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_fine_grained_features,
)
from .point_head import build_point_head, roi_mask_point_loss


def calculate_uncertainty(logits, classes):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.

    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.

    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -(torch.abs(gt_class_logits))


@ROI_HEADS_REGISTRY.register()
class PointRendROIHeads(StandardROIHeads):
    """
    The RoI heads class for PointRend instance segmentation models.

    In this class we redefine the mask head of `StandardROIHeads` leaving all other heads intact.
    To avoid namespace conflict with other heads we use names starting from `mask_` for all
    variables that correspond to the mask head in the class's namespace.
    """

    def __init__(self, cfg, input_shape):
        # TODO use explicit args style
        super().__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels+1 for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on                 = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        self.mask_coarse_in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES
        self.mask_coarse_side_size   = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self._feature_scales         = {k: 1.0 / v.stride for k, v in input_shape.items()}
        # fmt: on

        in_channels = np.sum([input_shape[f].channels+1 for f in self.mask_coarse_in_features])
        self.mask_coarse_head = build_mask_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                width=self.mask_coarse_side_size,
                height=self.mask_coarse_side_size,
            ),
        )
        self._init_point_head(cfg, input_shape)

    def _init_point_head(self, cfg, input_shape):
        # fmt: off
        self.mask_point_on                      = cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON
        if not self.mask_point_on:
            return
        assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        self.mask_point_in_features             = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.mask_point_train_num_points        = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        self.mask_point_oversample_ratio        = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
        self.mask_point_importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO
        # next two parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_steps       = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.mask_point_subdivision_num_points  = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        # fmt: on

        in_channels = np.sum([input_shape[f].channels for f in self.mask_point_in_features])
        self.mask_point_head = build_point_head(
            cfg, ShapeSpec(channels=in_channels, width=1, height=1)
        )
    '''
    def _init_track_head(self, cfg, input_shape):
        # fmt: off
        self.mask_track_on                      = cfg.MODEL.ROI_MASK_HEAD.TRACK_HEAD_ON
        if not self.mask_track_on:
            return
        self.mask_track_in_features             = cfg.MODEL.TRACK_HEAD.IN_FEATURES
        # fmt: on

        in_channels = np.sum([input_shape[f].channels for f in self.mask_track_in_features])
        self.mask_track_head = build_track_head(
            cfg, ShapeSpec(channels=in_channels, width=1, height=1)
        )
    '''

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        if not mask:
            mask = np.zeros((10, 10), dtype=np.float32)

        add_premask_on_features(features, mask)

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_coarse_logits = self._forward_mask_coarse(features, proposal_boxes)

            losses = {"loss_mask": mask_rcnn_loss(mask_coarse_logits, proposals)}
            losses.update(self._forward_mask_point(features, mask_coarse_logits, proposals))
            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_coarse_logits = self._forward_mask_coarse(features, pred_boxes)

            mask_logits = self._forward_mask_point(features, mask_coarse_logits, instances)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_mask_coarse(self, features, boxes):
        """
        Forward logic of the coarse mask head.
        """
        point_coords = generate_regular_grid_point_coords(
            np.sum(len(x) for x in boxes), self.mask_coarse_side_size, boxes[0].device
        )
        mask_coarse_features_list = [features[k] for k in self.mask_coarse_in_features]
        features_scales = [self._feature_scales[k] for k in self.mask_coarse_in_features]
        # For regular grids of points, this function is equivalent to `len(features_list)' calls
        # of `ROIAlign` (with `SAMPLING_RATIO=2`), and concat the results.
        mask_features, _ = point_sample_fine_grained_features(
            mask_coarse_features_list, features_scales, boxes, point_coords
        )
        return self.mask_coarse_head(mask_features)

    def _forward_mask_point(self, features, mask_coarse_logits, instances):
        """
        Forward logic of the mask point head.
        """
        if not self.mask_point_on:
            return {} if self.training else mask_coarse_logits

        mask_features_list = [features[k] for k in self.mask_point_in_features]
        features_scales = [self._feature_scales[k] for k in self.mask_point_in_features]

        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            gt_classes = cat([x.gt_classes for x in instances])
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    mask_coarse_logits,
                    lambda logits: calculate_uncertainty(logits, gt_classes),
                    self.mask_point_train_num_points,
                    self.mask_point_oversample_ratio,
                    self.mask_point_importance_sample_ratio,
                )

            fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
                mask_features_list, features_scales, proposal_boxes, point_coords
            )
            coarse_features = point_sample(mask_coarse_logits, point_coords, align_corners=False)
            point_logits = self.mask_point_head(fine_grained_features, coarse_features)
            return {
                "loss_mask_point": roi_mask_point_loss(
                    point_logits, instances, point_coords_wrt_image
                )
            }
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_classes = cat([x.pred_classes for x in instances])
            # The subdivision code will fail with the empty list of boxes
            if len(pred_classes) == 0:
                return mask_coarse_logits

            mask_logits = mask_coarse_logits.clone()
            for subdivions_step in range(self.mask_point_subdivision_steps):
                mask_logits = interpolate(
                    mask_logits, scale_factor=2, mode="bilinear", align_corners=False
                )
                # If `mask_point_subdivision_num_points` is larger or equal to the
                # resolution of the next step, then we can skip this step
                H, W = mask_logits.shape[-2:]
                if (
                    self.mask_point_subdivision_num_points >= 4 * H * W
                    and subdivions_step < self.mask_point_subdivision_steps - 1
                ):
                    continue
                uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.mask_point_subdivision_num_points
                )
                fine_grained_features, _ = point_sample_fine_grained_features(
                    mask_features_list, features_scales, pred_boxes, point_coords
                )
                coarse_features = point_sample(
                    mask_coarse_logits, point_coords, align_corners=False
                )
                point_logits = self.mask_point_head(fine_grained_features, coarse_features)

                # put mask point predictions to the right places on the upsampled grid.
                R, C, H, W = mask_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                    mask_logits.reshape(R, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(R, C, H, W)
                )
            return mask_logits

def add_premask_on_features(features, mask):
    for key, value in features.items():
        r, c, h, w = value.shape()
        print(r, c, h, w)
        mask = torch.from_numpy(cv2.resize(mask, (h, w))).repeat(r, 1, h, w)
        features[key] = torch.cat([value, mask], 1)
