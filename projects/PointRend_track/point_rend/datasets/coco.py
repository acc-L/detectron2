import numpy as np
import os.path as osp
import random
import cv2
from .custom import CustomDataset
from .extra_aug import ExtraAugmentation
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from pycocotools2.coco import COCO
#from mmcv.parallel import DataContainer as DC
from .utils import to_tensor, random_scale
from detectron2.structures import Boxes,BoxMode

class COCODataset(CustomDataset):
    CLASSES=('person','giant_panda','lizard','parrot','skateboard','sedan',
        'ape','dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
        'train','horse','turtle','bear','motorbike','giraffe','leopard',
        'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
        'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
        'tennis_racket')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 proposal_file=None,
                 num_max_proposals=1000,
                 with_track=False,
                 test_mode=False
                 ):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(ann_file)

        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training

        self.with_track = with_track
        
        # in test mode or not
        self.test_mode = test_mode


    def __len__(self):
        return len(self.vid_infos)
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        data = self.prepare_train_img(idx)
        return data
    
    def load_annotations(self, ann_file):
        self.ytvos = COCO(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getImgIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadImgs([i])[0]
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, idx):
        vid_id = self.vid_infos[idx]['id']
        #print(vid_id)
        ann_ids = self.ytvos.getAnnIds(imgIds=[vid_id])
        #print(ann_ids)
        ann_info = self.ytvos.loadAnns(ann_ids)
        #print([info['image_id'] for info in ann_info])
        #return self._parse_ann_info(ann_info, frame_id)
        return ann_info


    def prepare_train_img(self, idx):
        # prepare a pair of image in a sequence
        vid_info = self.vid_infos[idx]
        # load image
        file_name = osp.join(self.img_prefix, vid_info['file_name'])
        #img = cv2.resize(img, self.img_scales[0])
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        anns = self.get_ann_info(idx)
        [ann.update({'bbox_mode': BoxMode.XYWH_ABS}) for ann in anns]
        #gt_masks = [cv2.resize(self.ytvos.annToMask(ann), self.img_scales[0]) for ann in anns]
        #gt_masks = [self.ytvos.annToMask(ann) for ann in anns]

        #gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
        #                               False)

        #gt_masks = self.mask_transform(gt_masks, pad_shape, scale_factor, False)
        
        # compute matching of reference frame with current frame
        # 0 denote there is no matching
        # skip the image if there is no valid gt bbox
        if len(anns) == 0:
            return None

        ori_shape = (vid_info['height'], vid_info['width'], 3)

        data = dict(
            file_name=file_name,
            image_id=vid_info['id']
        )
        ann = []
        '''
        for i, bbox in enumerate(gt_bboxes):
            instance = {'bbox':bbox, 'bbox_mode':0}
            if self.proposals is not None:
                instance['proposals'] = to_tensor(proposals)
            instance['category_id']=gt_labels[i]
            instance['segmentation']=gt_masks[i]
            ann.append(instance)
        '''
        data['annotations'] = anns
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        vid_info = self.vid_infos[idx]
        file_name = osp.join(self.img_prefix, vid_info['file_name'])
        is_first = (vid_info['id'] == vid_info['start_id'])
        data = dict(file_name=file_name, is_first=is_first, 
                image_id=vid_info['id'], id=vid_info['id'])
        #data['is_first'] = True
        return data
