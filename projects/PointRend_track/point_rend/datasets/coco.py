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
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_track=False,
                 extra_aug=None,
                 aug_ref_bbox_param=None,
                 resize_keep_ratio=True,
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

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # max proposals per image
        self.num_max_proposals = num_max_proposals
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor
        self.img_scales = img_scale if isinstance(img_scale,list) else [img_scale]

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track
        # params for augmenting bbox in the reference frame
        self.aug_ref_bbox_param = aug_ref_bbox_param
        # in test mode or not
        self.test_mode = test_mode

        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_transform = BboxTransform()
        self.mask_transform = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()

        # if use extra augmentation
        if extra_aug is not None:
            self.extra_aug = ExtraAugmentation(**extra_aug)
        else:
            self.extra_aug = None

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

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

    def bbox_aug(self, bbox, img_size):
        assert self.aug_ref_bbox_param is not None
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]
        
        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2*np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2*np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:,:2]+ bbox[:,2:])/2.
        sizes = bbox[:,2:] - bbox[:,:2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes/2.
        new_x2y2 = new_centers + new_sizes/2.
        c_min = [0,0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1,new_x2y2)).astype(np.float32)
        return bbox

    def prepare_train_img(self, idx):
        # prepare a pair of image in a sequence
        vid_info = self.vid_infos[idx]
        # load image
        img = cv2.imread(osp.join(self.img_prefix, vid_info['file_name']))
        img_shape = img.shape[:2]
        #img = cv2.resize(img, self.img_scales[0])
        basename = osp.basename(vid_info['file_name'])

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
        #print(idx)
        #print(vid_info)
        #print(anns)
        gt_labels = np.array([ann['category_id'] for ann in anns])
        gt_bboxes = np.array([ann['bbox'] for ann in anns])
        gt_bboxes[:,2] += (gt_bboxes[:,0] - 1)
        gt_bboxes[:,3] += (gt_bboxes[:,1] - 1)
        #gt_masks = [cv2.resize(self.ytvos.annToMask(ann), self.img_scales[0]) for ann in anns]
        gt_masks = [self.ytvos.annToMask(ann) for ann in anns]

        img, _, pad_shape, scale_factor = self.img_transform(
            img, self.img_scales[0], False, keep_ratio=self.resize_keep_ratio)
        #img = img.copy()
        #gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
        #                               False)

        #gt_masks = self.mask_transform(gt_masks, pad_shape, scale_factor, False)
        
        # compute matching of reference frame with current frame
        # 0 denote there is no matching
        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        ori_shape = (vid_info['height'], vid_info['width'], 3)
        '''
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            #pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)
        '''

        data = dict(
            image=to_tensor(img),
            #img_meta=img_meta,
            height=img_shape[0],
            width=img_shape[1], 
            img_shape=img_shape, 
            image_id = vid_info['id'], 
        )
        ann = []
        for i, bbox in enumerate(gt_bboxes):
            instance = {'bbox':bbox, 'bbox_mode':0}
            if self.proposals is not None:
                instance['proposals'] = to_tensor(proposals)
            if self.with_label:
                instance['category_id']=gt_labels[i]
            if self.with_mask:
                instance['segmentation']=gt_masks[i]
            ann.append(instance)

        data['annotations'] = ann
        data['img_file'] = vid_info['file_name']
        return data

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        vid_info = self.vid_infos[idx]
        file_name = vid_info['file_name']
        is_first = (vid_info['id'] == vid_info['start_id'])
        data = dict(file_name=file_name, is_first=is_first, 
                id=vid_info['id'])
        #data['is_first'] = True
        return data
