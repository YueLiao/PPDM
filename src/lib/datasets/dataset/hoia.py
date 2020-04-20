import numpy as np
import cv2
import os, random
import torch
from torch.utils.data import Dataset
import json

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
import math

def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

class HOIA(Dataset):
    num_classes = 11
    num_classes_verb = 10
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    def __init__(self,  opt, split = 'train', resize_keep_ratio=False, multiscale_mode='value'):
        self.opt = opt
        self.root = os.path.join(self.opt.root_path, 'hoia')
        self.image_dir = self.opt.image_dir
        if split == 'train':
            self.hoi_annotations = json.load(open(os.path.join(self.root, 'annotations', 'train_hoia.json'), 'r'))
            self.resize_keep_ratio = resize_keep_ratio
            self.multiscale_mode = multiscale_mode
            self.flip = True
            self.ids = []
            for i, hoia in enumerate(self.hoi_annotations):
                flag_bad = 0
                for hoi in hoia['hoi_annotation']:
                    if hoi['subject_id'] >= len(hoia['annotations']) or hoi['object_id'] >= len(hoia['annotations']):
                        flag_bad = 1
                        break
                if flag_bad == 0:
                    self.ids.append(i)
            if split == 'train':
                self.shuffle()
            self.num_classes = opt.num_classes
            self.max_objs = 128
            self.max_rels = 64
            self._valid_ids = list(range(1,12))
            self._valid_ids_verb = list(range(1,11))

            self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

            self.cat_ids_verb = {v: i for i, v in enumerate(self._valid_ids_verb)}
            self._data_rng = np.random.RandomState(123)
            self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                     dtype=np.float32)
            self._eig_vec = np.array([
                [-0.58752847, -0.69563484, 0.41340352],
                [-0.5832747, 0.00994535, -0.81221408],
                [-0.56089297, 0.71832671, 0.41158938]
            ], dtype=np.float32)

            self.default_resolution = [512, 512]
            self.mean = np.array([0.40789654, 0.44719302, 0.47026115],
                                 dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array([0.28863828, 0.27408164, 0.27809835],
                                dtype=np.float32).reshape(1, 1, 3)

            self.split = split
            self.num_classes_verb = len(self._valid_ids_verb)
        else:
            self.hoi_annotations = json.load(open(os.path.join(self.root, 'annotations', 'test_hoia.json'), 'r'))
            self.ids = list(range(len(self.hoi_annotations)))


    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def __getitem__(self, index):
        img_id = self.ids[index]

        file_name = self.hoi_annotations[img_id]['file_name']
        img_path = os.path.join(self.root, self.image_dir, file_name)
        anns = self.hoi_annotations[img_id]['annotations']
        hoi_anns = self.hoi_annotations[img_id]['hoi_annotation']
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.7, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        hm_rel = np.zeros((self.num_classes_verb, output_h, output_w), dtype = np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        sub_offset = np.zeros((self.max_rels, 2), dtype=np.float32)
        obj_offset = np.zeros((self.max_rels, 2), dtype=np.float32)


        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []

        bbox_ct = []
        num_rels = min(len(hoi_anns), self.max_rels)
        for k in range(num_objs):
            ann = anns[k]
            bbox = np.asarray(ann['bbox'])
            if isinstance(ann['category_id'], str):
                ann['category_id'] =  int(ann['category_id'].replace('\n', ''))
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]


            ct = np.array(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

            ct_int = ct.astype(np.int32)
            bbox_ct.append(ct_int.tolist())
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                draw_gaussian(hm[cls_id], ct_int, radius)


                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])



        offset_mask = np.zeros((self.max_rels), dtype=np.uint8)
        rel_ind = np.zeros((self.max_rels), dtype=np.int64)
        for k in range(num_rels):
            hoi = hoi_anns[k]
            if isinstance(hoi['category_id'], str):
                hoi['category_id'] = int(hoi['category_id'].replace('\n', ''))
            if hoi['category_id'] == 0:
                continue
            hoi_cate = int(self.cat_ids_verb[hoi['category_id']])
            sub_ct = bbox_ct[hoi['subject_id']]
            obj_ct = bbox_ct[hoi['object_id']]
            offset_mask[k] = 1
            rel_ct = np.array([(sub_ct[0] + obj_ct[0]) / 2,
                               (sub_ct[1] + obj_ct[1]) / 2], dtype=np.float32)
            radius = gaussian_radius((math.ceil(abs(sub_ct[0] - obj_ct[0])), math.ceil(abs(sub_ct[1] - obj_ct[1]))))
            radius = max(0, int(radius))
            radius = self.opt.hm_gauss if self.opt.mse_loss else radius
            rel_ct_int = rel_ct.astype(np.int32)
            draw_gaussian(hm_rel[hoi_cate], rel_ct_int, radius)
            rel_sub_offset = np.array([rel_ct_int[0] - sub_ct[0], rel_ct_int[1] - sub_ct[1]], dtype=np.float32)
            rel_obj_offset = np.array([rel_ct_int[0] - obj_ct[0], rel_ct_int[1] - obj_ct[1]], dtype=np.float32)
            sub_offset[k] = 1.* rel_sub_offset[0], 1.*rel_sub_offset[1]
            obj_offset[k] = 1.* rel_obj_offset[0], 1.*rel_obj_offset[1]
            rel_ind[k] = rel_ct_int[1] * output_w + rel_ct_int[0]


        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hm_rel': hm_rel, 'sub_offset': sub_offset, 'obj_offset': obj_offset, 'offset_mask': offset_mask, 'rel_ind': rel_ind}
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        return ret

    def __len__(self):
        return len(self.ids)

    def shuffle(self):
        random.shuffle(self.ids)
