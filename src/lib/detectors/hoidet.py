from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch

from models.decode import hoidet_decode
from utils.post_process import ctdet_post_process

from .base_detector import BaseDetector
import os

class HoidetDetector(BaseDetector):
    def __init__(self, opt):
        super(HoidetDetector, self).__init__(opt)
        self.opt = opt
        if 'hico' in self.opt.dataset:
            self.corre_mat = np.load(os.path.join(self.opt.root_path,'hico_det/annotations/corre_hico.npy'))
        elif 'vcoco' in opt.dataset:
            self.corre_mat = np.load(os.path.join(self.opt.root_path, 'verbcoco/annotations/corre_vcoco.npy'))
        elif 'hoia' in opt.dataset:
            self.corre_mat = np.load(os.path.join(self.opt.root_path, 'hoia/annotations/corre_hoia.npy'))
        self.triplet_labels = np.nonzero(self.corre_mat)
        self.triplet_labels = list(zip(self.triplet_labels[0], self.triplet_labels[1]))
        self.corre_mat = torch.tensor(self.corre_mat).float().cuda()
    def process(self, images, return_time=False):

        with torch.no_grad():
            output = self.model(images)[-1]
            hm_obj = output['hm'].sigmoid_()
            hm_rel = output['hm_rel'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            sub_offset = output['sub_offset']
            obj_offset = output['obj_offset']
            torch.cuda.synchronize()
            forward_time = time.time()

            dets_obj, dets_sub, rel = hoidet_decode(hm_obj, wh, hm_rel, sub_offset, obj_offset, reg=reg,
                                                    corremat=self.corre_mat, is_sub_verb=self.opt.use_verb_sub)

        if return_time:
            return output, dets_obj, dets_sub, rel, forward_time, images.size()[2], images.size()[3]
        else:
            return output, dets_obj, dets_sub, rel, images.size()[2], images.size()[3]

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        return dets

    def bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_hoi_output(self, det_sub, det_obj, rel, c):
        output = {'predictions': [], 'hoi_prediction': []}
        obj_match_dict = {}
        sub_match_dict = {}
        count = 0
        h = c[0] * 2
        w = c[1] * 2
        if 'hico' in self.opt.dataset or 'vcoco' in self.opt.dataset:
            obj_cate_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                         24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                         37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                         48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                         58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                         72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                         82, 84, 85, 86, 87, 88, 89, 90]
        if 'hico' in self.opt.dataset:
            verb_cate_ids = list(range(118))
            verb_cate_ids.remove(0)
        elif 'vcoco' in self.opt.dataset:
            verb_cate_ids = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 28]
        if 'hoia' in self.opt.dataset:
            obj_cate_ids= list(range(1, 12))
            verb_cate_ids = list(range(1, 11))
        for i in range(rel.shape[0]):
            rel_i = rel[i, :]
            sub_id = int(rel_i[0])
            obj_id = int(rel_i[1])
            if (int(rel_i[2]),int(det_obj[0,obj_id,-1])) not in self.triplet_labels:
                continue
            if sub_id not in sub_match_dict.keys():
                sub_match_dict[sub_id] = count
                count = count + 1
                bbox_i = [det_sub[0, sub_id, 0], det_sub[0, sub_id, 1], det_sub[0, sub_id, 2], det_sub[0, sub_id, 3]]
                bbox_i_refine = self.bbox_clip(bbox_i[0], bbox_i[1], bbox_i[2] - bbox_i[0], bbox_i[3] - bbox_i[1],
                                               (w, h))
                output['predictions'].append({'bbox': [bbox_i_refine[0], bbox_i_refine[1],
                                                       bbox_i_refine[0] + bbox_i_refine[2],
                                                       bbox_i_refine[1] + bbox_i_refine[3]],
                                              'category_id': obj_cate_ids[int(det_sub[0, sub_id, -1])]})
            if obj_id not in obj_match_dict.keys():
                obj_match_dict[obj_id] = count
                count = count + 1
                bbox_i = [det_obj[0, obj_id, 0], det_obj[0, obj_id, 1], det_obj[0, obj_id, 2], det_obj[0, obj_id, 3]]
                bbox_i_refine = self.bbox_clip(bbox_i[0], bbox_i[1], bbox_i[2] - bbox_i[0], bbox_i[3] - bbox_i[1],
                                               (w, h))
                output['predictions'].append({'bbox': [bbox_i_refine[0], bbox_i_refine[1],
                                                       bbox_i_refine[0] + bbox_i_refine[2],
                                                       bbox_i_refine[1] + bbox_i_refine[3]],
                                              'category_id': obj_cate_ids[int(det_obj[0, obj_id, -1])]})

            output['hoi_prediction'].append({'subject_id': sub_match_dict[sub_id], 'object_id': obj_match_dict[obj_id],
                                             'category_id': verb_cate_ids[int(rel_i[2])], 'score': rel_i[3]})
        return output

