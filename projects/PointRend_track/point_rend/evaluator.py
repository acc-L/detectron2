import pickle
import numpy as np
import torch
import os
import itertools
from collections import OrderedDict
import logging
import datetime
import time

from detectron2.evaluation import COCOEvaluator, DatasetEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou
import detectron2.utils.comm as comm
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds

import pycocotools.mask as mask_util
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
from fvcore.common.file_io import PathManager
from contextlib import contextmanager

def instances_to_ytvis_json(instances, img_id, vid):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "video_id": vid,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results

def convert_for_eval(prediction):
    seg = [res["segmentation"] for res in prediction['instances']]
    bbox = [res["bbox"] for res in prediction['instances']]
    category = [res["category_id"] for res in prediction['instances']]
    score = [res["score"] for res in prediction['instances']]
    prediction.pop('instances')
    prediction["segmentation"] = seg
    prediction["bboxes"] = bbox
    prediction["category_ids"] = category
    prediction["scores"] = score
    return prediction


class YTVOSEvaluator(COCOEvaluator):
    """
    Evaluator for youtube-VIS
    based on COCOAPI
    """
    def __init__(self, dataset_name, annfile, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self.annfile = annfile

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"frame_id": input["frame_id"], 
                    "video_id":input["video_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_ytvis_json(instances, 
                        prediction["frame_id"], prediction["video_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pkl")
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(predictions, f)

        res = OrderedDict()
        result = [convert_for_eval(p) for p in predictions]
        self._eval_ytvos_res(self.annfile, result)
        res['res'] = {'vis':[0,0,0,0]}
        return res

    def _eval_ytvos_res(self, ann_file, predictions):
        vos_gt = YTVOS(ann_file)
        vos_dt = vos_gt.loadRes(predictions)
        vos_eval = YTVOSeval(vos_gt, vos_dt)
        vos_eval.evaluate()
        vos_eval.accumulate()
        vos_eval.summarize()


def inference_on_dataset_timestep(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.
    Run model 

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        return {}
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    tmask = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            print(idx)
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            if inputs[0]['frame_id']>0:
                inputs[0]['mask'] = tmask
            
            start_compute_time = time.perf_counter()
            outputs = model(inputs)

            outputs_instances = outputs[0]['instances'].to('cpu')
            output_masks = outputs_instances.pred_masks.detach().numpy()
            output_scores = outputs_instances.scores.detach().numpy()
            final_choose_masks = np.array(np.zeros((1,)+inputs[0]['img_shape']), dtype=np.uint8)
            final_choose_idx = []
            for other_id in range(len(output_scores)):
                if output_scores[other_id] > 0.5:
                    final_choose_idx.append(other_id)
            if len(final_choose_idx) > 0:
                final_choose_masks = output_masks[final_choose_idx]
            tmask = np.sum(final_choose_masks, axis=0).astype(np.float32)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
