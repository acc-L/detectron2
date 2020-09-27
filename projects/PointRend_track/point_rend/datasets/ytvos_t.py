import numpy as np
import os.path as osp
import random
from .custom import CustomDataset
from pycocotools.ytvos import YTVOS
from .utils import to_tensor, random_scale

class YTVOSDataset(CustomDataset):
    CLASSES=('person','giant_panda','lizard','parrot','skateboard','sedan',
        'ape','dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
        'train','horse','turtle','bear','motorbike','giraffe','leopard',
        'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
        'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
        'tennis_racket')

    TO_COCO = {1:0, 4:14, 5:36, 6:2, 8:16, 14:15, 15:19, 17:6, 18:17, 20:21, 21:3,
    22:23, 27:37, 28:4, 29:7, 30:22, 32:20, 33:31, 34:8, 38:14, 40:38}

    def __init__(self,
                 ann_file,
                 img_prefix,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 coco_format = True, 
                 test_mode=False):
        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations (and proposals)
        self.vid_infos = self.load_annotations(ann_file)
        img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
          for frame_id in range(len(vid_info['file_names'])):
            img_ids.append((idx, frame_id))
        self.img_ids = img_ids
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None
        # filter images with no annotation during training
        if not test_mode:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                if len(self.get_ann_info(v, f))]
            self.img_ids = [self.img_ids[i] for i in valid_inds]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        self.coco_format = coco_format

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        return self.prepare_test_img(self.img_ids[idx])
    
    def load_annotations(self, ann_file):
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_id, _ = self.img_ids[i]
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1

    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_samples = []
        for i in sample_range:
          # check if the frame id is valid
          ref_idx = (vid, i)
          if i != frame_id and ref_idx in self.img_ids:
              valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)

    def prepare_train_img(self, idx):
        # prepare a pair of image in a sequence
        vid,  frame_id = idx
        vid_info = self.vid_infos[vid]
        # load data
        file_name = vid_info['file_names'][frame_id]
        ann = self.get_ann_info(vid, frame_id)

        if not self.test_mode:
            _, ref_frame_id = self.sample_ref(idx)
            ref_file_name = vid_info['file_names'][ref_frame_id]
            ref_ann = self.get_ann_info(vid, ref_frame_id)
        

        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
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
        
        # skip the image if there is no valid gt bbox
        if len(ann) == 0:
            return None

        data = dict(
            file_name = file_name,
            ref_file_name = ref_file_name,
            annotations = ann,
            ref_annotations = ref_ann,
        )
        if ref_ann is not None:
            data['ref_annotations'] = ref_ann
        if ref_file_name is not None:
            data['ref_file_name'] = ref_file_name

        return data

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        ann_frame = []
        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area
            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            anno = dict(bbox=bbox, bbox_mode=0)
            if ann['iscrowd']:
                anno = ann['iscrowd']
            else:
                cat_id = self.cat2label[ann['category_id']]
                if self.coco_format:
                    cat_id = self.TO_COCO.get(cat_id, 80)
                anno['category_id'] = cat_id
                anno['obj_id'] = ann['id']
            if with_mask:
                #gt_masks.append(self.ytvos.annToMask(ann, frame_id))
                mask_polys = [
                    p for p in segm if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                anno['segmentation'] = mask_polys
            ann_frame.append(anno)
        return ann_frame
