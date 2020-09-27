#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
PointRend Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch
import logging
import numpy as np
import itertools
from fvcore.common.file_io import PathManager
import pickle
from collections import OrderedDict

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
import detectron2.utils.comm as comm
from detectron2.modeling import build_model
from detectron2.data import build_batch_data_loader, MapDataset
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results, DatasetEvaluator, DatasetEvaluators
from point_rend import (
    ColorAugSSDTransform, 
    add_pointrend_config, 
    YTVOSDataset,
    COCODataset, 
    YTVOSEvaluator, 
    COCODataset,
    inference_on_dataset_timestep,
    DatasetMapper, 
)


def build_detection_train_loader(dataset, cfg, mapper=None):
    """
    A data loader:
    Args:
        cfg (CfgNode): the config
    Returns:
        an infinite iterator of training data
    """
    if mapper is None:
        mapper = DatasetMapper(cfg, True, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


def build_detection_test_loader(dataset, cfg, mapper=None):
    """
    A data loader:
    Args:
        cfg (CfgNode): the config
    Returns:
        an infinite iterator of training data
    """
    if mapper is None:
        mapper = DatasetMapper(cfg, False, False)
    dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn = lambda x:x, 
    )
    return data_loader
    

def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    augs.append(T.RandomFlip())
    return augs

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        """
        model = build_model(cfg)
        print(model)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        for param in model.parameters():
            param.requires_grad = False
        #for param in model.roi_heads.box_merger.parameters():
        #    param.requires_grad = True
        #for param in model.roi_heads.mask_coarse_head.parameters():
        #    param.requires_grad = True
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder='./output'):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        data_root = '/data/youtube-VIS/'
        ann_file=data_root + 'annotation/valid_e.json'
        return YTVOSEvaluator(dataset_name, ann_file, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        data_root = '/data/youtube-VIS/'
        train=dict(
            ann_file=data_root + 'annotation/train_e.json',
            img_prefix=data_root + 'train/JPEGImages',
            )
        dataset = YTVOSDataset(**train)
        return build_detection_train_loader(dataset, cfg)

    @classmethod
    def build_test_loader_e(cls, cfg, dateset_name):
        dataset_type = 'YTVOSDataset'
        data_root = '/data/youtube-VIS/'
        img_norm_cfg = dict(
                mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)

        val=dict(
                ann_file=data_root + 'annotation/valid.json',
                img_prefix=data_root + 'valid/JPEGImages',
                img_scale=(1280, 720),
                img_norm_cfg=img_norm_cfg,
                size_divisor=32,
                flip_ratio=0,
                with_mask=True,
                with_crowd=True,
                with_label=True,
                test_mode=True)
        dataset = YTVOSDataset(**val)
        return build_detection_test_loader(dataset, cfg)
    
    @classmethod
    def build_test_loader(cls, cfg, dateset_name):
        data_root = '/data/youtube-VIS/'

        val=dict(
                ann_file=data_root + 'annotation/valid_e.json',
                img_prefix=data_root + 'train/JPEGImages',
                test_mode=True, 
                )
        dataset = COCODataset(**val)
        return build_detection_test_loader(dataset, cfg)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
