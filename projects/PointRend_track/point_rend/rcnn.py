# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import cv2

from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN



@META_ARCH_REGISTRY.register()
class MaskGuildRCNN(GeneralizedRCNN):
    """
    The MaskGuildRCNN. A stantard RCNN with MaskGuild module for instance segmentation
    """

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images, ref_images = self.preprocess_image(batched_inputs)
        mask = [binput['mask'] for binput in batched_inputs]
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        with torch.no_grad():
            features = self.backbone(images.tensor)
            ref_features = self.backbone(ref_images.tensor)
            ref_features = [ref_features[f] for f in self.box_in_features]

            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, mask)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        masks = [binput['mask'] for binput in batched_inputs]
        if batched_inputs[0]['frame_id']==0:
            ref_features = None
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None, masks)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        ref_features = [features[f] for f in self.box_in_features]

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        ref_images = [x["ref_image"].to(self.device) for x in batched_inputs]
        ref_images = ImageList.from_tensors(ref_images, self.backbone.size_divisibility)
        return images, ref_images
