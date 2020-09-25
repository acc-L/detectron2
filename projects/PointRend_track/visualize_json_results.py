#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import pickle
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from pycocotools2.coco import COCO
from point_rend import COCODataset

def _get_vis_instances_meta():
    thing_ids = list(range(1, 41))
    metadata = MetadataCatalog.get('coco_2017_val')
    thing_colors = metadata.thing_colors[:40]
    print(metadata.thing_classes.index('person'))
    print(metadata.thing_classes.index('cat'))
    print(metadata.thing_classes.index('dog'))
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_classes=['', 'person','giant_panda','lizard','parrot','skateboard','sedan',
        'ape','dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
        'train','horse','turtle','bear','motorbike','giraffe','leopard',
        'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
        'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
        'tennis_racket']
    ret = {
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def register_vis_instances(name, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda x:x)
    metadata = _get_vis_instances_meta()

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="youtube-VIS", **metadata
    )


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([predictions[i]["category_id"] for i in chosen])
    #labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="ytvos")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--ann-file", default='/data/youtube-VIS/annotation/valid_e.json')
    parser.add_argument("--img-prefix", default='/data/youtube-VIS/train/JPEGImages')
    args = parser.parse_args()
    register_vis_instances(args.dataset, args.ann_file, args.img_prefix)

    def load_annotations(ann_file):
        ytvos = COCO(ann_file)
        cat_ids = ytvos.getCatIds()
        cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(cat_ids)
        }
        vid_ids = ytvos.getImgIds()
        vid_infos = []
        for i in vid_ids:
            info = ytvos.loadImgs([i])[0]
            vid_infos.append(info)
        return vid_infos

    vid_infos = load_annotations(args.ann_file)

    logger = setup_logger()

    with open(args.input, "rb") as f:
        predictions = pickle.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[(p['instances'][0]['image_id'])].extend(p['instances'])

    #dicts = list(DatasetCatalog.get(args.dataset))
    img_norm_cfg = dict(
            mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
    val = dict(
                ann_file=args.ann_file,
                img_prefix=args.img_prefix,
                img_scale=(1280, 720),
                img_norm_cfg=img_norm_cfg,
                size_divisor=32,
                flip_ratio=0,
                with_mask=True,
                with_crowd=True,
                with_label=True,
                ) 
    dicts = list(COCODataset(**val))
    metadata = MetadataCatalog.get(args.dataset)
    print(metadata)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    else:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id-1


    os.makedirs(args.output, exist_ok=True)

    for p in predictions:
        for ins in p['instances']:
            print(ins['category_id'])

    for dic in tqdm.tqdm(dicts):
        image_id = dic['image_id'] - 1
        vid_info = vid_infos[image_id]
        assert vid_info['file_name'] == dic['img_file'], vid_info['file_name']
        img = cv2.imread(os.path.join(args.img_prefix, vid_info['file_name']))[:, :, ::-1]
        #img = cv2.resize(img, (1280, 720))
        dirname = os.path.dirname(vid_info['file_name'])
        if not os.path.exists(os.path.join(args.output, dirname)):
            os.mkdir(os.path.join(args.output, dirname))

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, vid_info['file_name']), concat[:, :, ::-1])
