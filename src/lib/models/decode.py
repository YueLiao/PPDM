from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
import numpy as np
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def match_rel_box(rel, bbox, K_rel, K_bbox):
    rel  = rel.view(K_rel,1)
    rel = rel.repeat(1, K_bbox)
    bbox = bbox.view(1, K_bbox)
    bbox = bbox.repeat(K_rel, 1)
    dis_mat = abs(rel - bbox)
    return dis_mat

def hoidet_decode( heat_obj, wh, heat_rel, offset_sub, offset_obj, reg=None, corremat = None, K_obj=100, K_human = 100, K_rel = 100, is_sub_verb = 0):
    batch, cat_obj, height, width = heat_obj.size()
    heat_obj = _nms(heat_obj)
    heat_rel = _nms(heat_rel)
    heat_human = heat_obj[:,0,:,:].view(batch,1, height, width)

    scores_obj, inds_obj, clses_obj, ys_obj, xs_obj = _topk(heat_obj, K=K_obj)
    scores_obj = scores_obj.view(batch, K_obj, 1)
    clses_obj = clses_obj.view(batch, K_obj, 1).float()

    scores_human, inds_human, clses_human, ys_human, xs_human = _topk(heat_human, K=K_human)
    scores_human = scores_human.view(batch, K_human, 1)
    clses_human = clses_human.view(batch, K_human, 1).float()

    scores_rel, inds_rel, clses_rel, ys_rel, xs_rel = _topk(heat_rel, K=K_rel)
    scores_rel = scores_rel.view(batch, K_rel, 1)
    clses_rel = clses_rel.view(batch, K_rel, 1).float()

    offset_sub = _tranpose_and_gather_feat(offset_sub, inds_rel)
    offset_sub = offset_sub.view(batch, K_rel, 2)
    dist_sub_xs = xs_rel.view(batch, K_rel, 1) - offset_sub[:, :, 0:1]
    dist_sub_ys = ys_rel.view(batch, K_rel, 1) - offset_sub[:, :, 1:2]
    match_sub_xs = match_rel_box(dist_sub_xs, xs_human.view(batch, K_human, 1), K_rel, K_human)
    match_sub_ys = match_rel_box(dist_sub_ys, ys_human.view(batch, K_human, 1), K_rel, K_human)

    offset_obj = _tranpose_and_gather_feat(offset_obj, inds_rel)
    offset_obj = offset_obj.view(batch, K_rel, 2)
    dist_obj_xs = xs_rel.view(batch, K_rel, 1) - offset_obj[:,:,0:1]
    dist_obj_ys = ys_rel.view(batch, K_rel, 1) - offset_obj[:,:,1:2]
    match_obj_xs = match_rel_box(dist_obj_xs, xs_obj.view(batch, K_obj, 1), K_rel, K_obj)
    match_obj_ys = match_rel_box(dist_obj_ys, ys_obj.view(batch, K_obj, 1), K_rel, K_obj)


    if corremat is not None:
        this_corremat = corremat[clses_rel.view(K_rel).long(),:]
        this_corremat = this_corremat[:, clses_obj.view(K_obj).long()]
    else:
        this_corremat = np.ones((K_rel, K_obj))


    dis_sub_score = (1/(match_sub_xs + 1)) * (1/(match_sub_ys + 1)) * ((scores_human.view(1, K_human).repeat(K_rel,1)))
    sub_rel_ids = torch.argmax(dis_sub_score, dim=1)
    sub_scores_rel = (scores_human.view(K_human))[sub_rel_ids.long()]
    dis_obj_score = (1 / (match_obj_xs + 1)) * (1 / (match_obj_ys + 1)) * ((scores_obj.view(1, K_obj).repeat(K_rel, 1)))
    dis_obj_score = dis_obj_score * this_corremat
    obj_rel_ids = torch.argmax(dis_obj_score, dim=1)
    obj_scores_rel = (scores_obj.view(K_obj))[obj_rel_ids.long()]
    score_hoi = sub_scores_rel * obj_scores_rel * (scores_rel.view(K_rel))
    hoi_triplet = (torch.cat((sub_rel_ids.view(K_rel,1).float(), obj_rel_ids.view(K_rel,1).float(), clses_rel.view(K_rel,1).float(), score_hoi.view(K_rel,1)), 1)).cpu().numpy()
    hoi_triplet = hoi_triplet[np.argsort(-hoi_triplet[:,-1])]
    _, u_hoi_id = np.unique(hoi_triplet[:,[0,1,2]], axis=0, return_index=True)
    rel_triplet = hoi_triplet[u_hoi_id]
    if reg is not None:
        reg_obj = _tranpose_and_gather_feat(reg, inds_obj)
        reg_obj = reg_obj.view(batch, K_obj, 2)

        reg_human = _tranpose_and_gather_feat(reg, inds_human)
        reg_human = reg_human.view(batch, K_human, 2)

        xs_human = xs_human.view(batch, K_human, 1) + reg_human[:, :, 0:1]
        ys_human = ys_human.view(batch, K_human, 1) + reg_human[:, :, 1:2]
        xs_obj = xs_obj.view(batch, K_obj, 1) + reg_obj[:, :, 0:1]
        ys_obj = ys_obj.view(batch, K_obj, 1) + reg_obj[:, :, 1:2]
    wh_human = _tranpose_and_gather_feat(wh, inds_human)
    wh_obj = _tranpose_and_gather_feat(wh, inds_obj)
    wh_obj = wh_obj.view(batch, K_obj, 2)
    wh_human = wh_human.view(batch, K_human, 2)


    obj_bboxes = torch.cat([xs_obj - wh_obj[..., 0:1] / 2,
                        ys_obj - wh_obj[..., 1:2] / 2,
                        xs_obj + wh_obj[..., 0:1] / 2,
                        ys_obj + wh_obj[..., 1:2] / 2], dim=2)

    obj_detections = torch.cat([obj_bboxes, scores_obj, clses_obj], dim=2)


    human_bboxes = torch.cat([xs_human - wh_human[..., 0:1] / 2,
                            ys_human - wh_human[..., 1:2] / 2,
                            xs_human + wh_human[..., 0:1] / 2,
                            ys_human + wh_human[..., 1:2] / 2], dim=2)
    if is_sub_verb > 0:
        clses_human[:] = 0.0
    human_detections = torch.cat([human_bboxes, scores_human, clses_human], dim=2)

    return obj_detections, human_detections, rel_triplet
