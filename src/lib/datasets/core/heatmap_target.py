import math
import numpy as np
import torch

from utils.image import gaussian_radius, draw_umich_gaussian


def get_heatmap_bbox_target(bbox_list, output_w, output_h, max_objs,
                            num_classes, num_objs):
    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((max_objs, 2), dtype=np.float32)
    reg = np.zeros((max_objs, 2), dtype=np.float32)
    ind = np.zeros(max_objs, dtype=np.int64)
    reg_mask = np.zeros(max_objs, dtype=np.uint8)

    for k in range(num_objs):
        bbox = bbox_list[k]
        ct = np.array(bbox[:2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        w, h = bbox[2:4]
        cls_id = bbox[-1]
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            draw_umich_gaussian(hm[cls_id], ct_int, radius)
    return hm, wh, reg, ind, reg_mask


def get_heatmap_hoi_target(hoi_list, bbox_list, output_w, output_h, max_rels,
                           num_classes, num_rels):
    hm_rel = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    sub_offset = np.zeros((max_rels, 2), dtype=np.float32)
    obj_offset = np.zeros((max_rels, 2), dtype=np.float32)
    offset_mask = np.zeros(max_rels, dtype=np.uint8)
    ind = np.zeros(max_rels, dtype=np.int64)
    for k in range(num_rels):
        hoi = hoi_list[k]
        sub_box_ct = np.array(bbox_list[hoi[0]][:2],
                              dtype=np.float32).astype(np.int32)
        obj_box_ct = np.array(bbox_list[hoi[1]][:2],
                              dtype=np.float32).astype(np.int32)
        rel_ct = np.array([(sub_box_ct[0] + obj_box_ct[0]) / 2,
                           (sub_box_ct[1] + obj_box_ct[1]) / 2],
                          dtype=np.float32)
        rel_ct_int = rel_ct.astype(np.int32)
        rel_sub_offset = np.array(
            [rel_ct_int[0] - sub_box_ct[0], rel_ct_int[1] - sub_box_ct[1]],
            dtype=np.float32)
        rel_obj_offset = np.array(
            [rel_ct_int[0] - obj_box_ct[0], rel_ct_int[1] - obj_box_ct[1]],
            dtype=np.float32)
        radius = gaussian_radius(
            (math.ceil(abs(sub_box_ct[0] - obj_box_ct[0])),
             math.ceil(abs(sub_box_ct[1] - obj_box_ct[1]))))
        radius = max(0, int(radius))

        draw_umich_gaussian(hm_rel[hoi[-1]], rel_ct_int, radius)
        sub_offset[k] = 1. * rel_sub_offset[0], 1. * rel_sub_offset[1]
        obj_offset[k] = 1. * rel_obj_offset[0], 1. * rel_obj_offset[1]
        ind[k] = rel_ct_int[1] * output_w + rel_ct_int[0]
        offset_mask[k] = 1

    return hm_rel, sub_offset, obj_offset, ind, offset_mask


# def get_heatmap_hoi_target(hoi_list, bbox_list, output_w, output_h, max_rels,
#                        num_classes, num_rels):
#     bbox_list = torch.tensor(bbox_list)
#     hm_rel = torch.zeros((4, num_classes, output_h, output_w),
#                          dtype=torch.float)
#     offset_mask = torch.zeros(max_rels, dtype=torch.uint8)
#     ind = torch.zeros(max_rels, dtype=torch.uint8)
#     sub_ct = bbox_list.index_select(0, hoi_list[:, 0])[:, :2]
#     obj_ct = bbox_list.index_select(0, hoi_list[:, 1])[:, :2]
#     rel_ct = torch.stack(torch.arange(output_w))
#     rel_ct = torch.stack([(sub_ct[:, 0] + obj_ct[:, 0]) / 2,
#                           (sub_ct[:, 1] + obj_ct[:, 1]) / 2],
#                          1, dtype=torch.int8)
#     sub_offset = rel_ct - sub_ct
#     obj_offset = rel_ct - obj_ct
#     for k in range(num_rels):
#         offset_mask[k] = 1
#     return hm_rel, sub_offset, obj_offset, ind, offset_mask


def get_set_hoi_target(hoi_list, bbox_list, output_w, output_h, max_rels, num_classes, num_rels):
    hm_rel = torch.zeros(max_rels, dtype=torch.long)
    hoi_list = torch.tensor(hoi_list)[:num_rels, :]
    sub_obj_bbox_id, hm_rel[:num_rels] = hoi_list[:, :2], hoi_list[:, -1]
    bbox_list = torch.tensor(bbox_list).int()
    matrix = torch.stack([torch.arange(0, output_w).repeat(output_h, 1).T,
                          torch.arange(0, output_h).repeat(output_w, 1)], dim=0).repeat(max_rels, 2, 1, 1)
    sub_obj_ct = torch.zeros(max_rels, 4, 1, 1)
    sub_obj_ct[:num_rels, :] = bbox_list.index_select(0, sub_obj_bbox_id.flatten())[:, :2].reshape(-1, 4, 1, 1)
    offset = matrix - sub_obj_ct
    offset_mask = torch.arange(max_rels) < len(hoi_list)
    return hm_rel, offset, offset_mask


def get_heatmap_target(bbox_list, hoi_list, output_w, output_h, max_objs,
                       max_rels, num_obj_classes, num_verb_classes, num_objs,
                       num_rels):
    hm, wh, reg, ind, reg_mask = get_heatmap_bbox_target(
        bbox_list, output_w, output_h, max_objs, num_obj_classes, num_objs)
    hm_rel, offset, offset_mask = \
        get_set_hoi_target(hoi_list, bbox_list, output_w, output_h, max_rels, num_verb_classes, num_rels)

    ret = {
        'hm': hm,
        'reg_mask': reg_mask,
        'ind': ind,
        'wh': wh,
        'hm_rel': hm_rel,
        'offset': offset,
        'offset_mask': offset_mask,
        'reg': reg
    }
    return ret
