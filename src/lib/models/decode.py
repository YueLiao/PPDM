from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from utils import topk, nms_maxpool, tranpose_and_gather_feat
import numpy as np


def get_bboxes(c, wh, inds, scores, clses, reg=None):
    x, y = c
    if reg is not None:
        reg = tranpose_and_gather_feat(reg, inds)
        reg = reg.view(-1, 2)
        x = x + reg[:, 0:1]
        y = y + reg[:, 1:2]

    wh = tranpose_and_gather_feat(wh, inds)
    wh = wh.view(-1, 2)
    bboxes = torch.cat([
        x - wh[:, 0:1] / 2, y - wh[:, 1:2] / 2, x + wh[:, 0:1] / 2,
        y + wh[:, 1:2] / 2
    ],
        dim=1)

    detections = torch.cat([bboxes, scores, clses], dim=1)
    return detections


def get_matching_scores(point_a, point_b, offset):
    x_a, y_a = point_a
    x_b, y_b = point_b

    x_coarse_b = x_a - offset[:, 0:1]
    y_coarse_b = y_a - offset[:, 1:2]
    dis_x = abs(
        x_coarse_b.view(-1, 1).repeat(1,
                                      x_b.size()[0]) -
        x_b.view(1, -1).repeat(x_coarse_b.size()[0], 1))
    dis_y = abs(
        y_coarse_b.view(-1, 1).repeat(1,
                                      y_b.size()[0]) -
        y_b.view(1, -1).repeat(y_coarse_b.size()[0], 1))
 
    return (1 / (dis_x + 1)) * (1 / (dis_y + 1))


def hoidet_decode(heat_obj, wh, heat_rel, sub_offset, obj_offset, reg=None, hoi_cate_mask=None, K_obj=100, K_sub=100,
                  K_rel=100):
    batch, cat_obj, height, width = heat_obj.size()
    heat_sub = heat_obj[:, 0, :, :].view(batch, 1, height, width)

    heat_sub = nms_maxpool(heat_sub)
    heat_obj = nms_maxpool(heat_obj)

    scores_sub, inds_sub, clses_sub, ys_sub, xs_sub = \
        topk(heat_sub, K=K_sub)
    scores_sub = scores_sub.view(K_sub, 1)
    clses_sub = clses_sub.view(K_sub, 1).float()
    sub_points = (xs_sub.view(K_sub, 1), ys_sub.view(K_sub, 1))

    scores_obj, inds_obj, clses_obj, ys_obj, xs_obj = \
        topk(heat_obj, K=K_obj)
    scores_obj = scores_obj.view(K_obj, 1)
    clses_obj = clses_obj.view(K_obj, 1).float()
    obj_points = (xs_obj.view(K_obj, 1), ys_obj.view(K_obj, 1))

    heat_rel = nms_maxpool(heat_rel)
    scores_rel, inds_rel, clses_rel, ys_rel, xs_rel = \
        topk(heat_rel, K=K_rel)
    scores_rel = scores_rel.view(K_rel, 1)
    clses_rel = clses_rel.view(K_rel, 1).float()
    rel_point = (xs_rel.view(K_rel, 1), ys_rel.view(K_rel, 1))

    sub_offset = tranpose_and_gather_feat(sub_offset, inds_rel)
    sub_offset = sub_offset.view(K_rel, 2)

    obj_offset = tranpose_and_gather_feat(obj_offset, inds_rel)
    obj_offset = obj_offset.view(K_rel, 2)

    sub_matching_score = \
        get_matching_scores(rel_point, sub_points, sub_offset) * \
        (scores_sub.view(1, K_sub).repeat(K_rel, 1))

    obj_matching_score = \
        get_matching_scores(rel_point, obj_points, obj_offset) * \
        (scores_obj.view(1, K_obj).repeat(K_rel, 1))

    if hoi_cate_mask is not None:
        selected_hoi_cate_mask = \
            hoi_cate_mask[clses_rel.view(K_rel).long(), :]
        selected_hoi_cate_mask = \
            selected_hoi_cate_mask[:, clses_obj.view(K_obj).long()]
        obj_matching_score = \
            obj_matching_score * selected_hoi_cate_mask

    interacted_sub_ids = torch.argmax(
        sub_matching_score, dim=-1)
    interacted_obj_ids = torch.argmax(
        obj_matching_score, dim=-1)

    interacted_sub_score = (scores_sub.view(K_sub))[interacted_sub_ids.long()]
    interacted_obj_score = (scores_obj.view(K_sub))[interacted_obj_ids.long()]
    hoi_score = interacted_sub_score.view(K_rel) * \
                interacted_obj_score.view(K_rel) * scores_rel.view(K_rel)

    hoi_triplet = \
        torch.cat((interacted_sub_ids.view(K_rel, 1).float(),
                   interacted_obj_ids.view(K_rel, 1).float(),
                   clses_rel.view(K_rel, 1),
                   hoi_score.view(K_rel, 1)), -1)

    hoi_triplet = hoi_triplet.cpu().numpy()
    hoi_triplet = hoi_triplet[np.argsort(-hoi_triplet[:, -1])]
    _, selected_hoi_id = \
        np.unique(hoi_triplet[:, [0, 1, 2]], axis=0, return_index=True)

    rel_triplet = hoi_triplet[selected_hoi_id]
    detections_sub = \
        get_bboxes(sub_points, wh, inds_sub, scores_sub,
                   clses_sub, reg)
    detections_obj = \
        get_bboxes(obj_points, wh, inds_obj, scores_obj,
                   clses_obj, reg)

    return detections_sub.unsqueeze(0), detections_obj.unsqueeze(0), rel_triplet

