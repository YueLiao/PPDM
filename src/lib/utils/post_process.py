from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def get_pred_depth(depth):
    """
    Get the depth of a node.

    Args:
        depth: (int): write your description
    """
  return depth

def get_alpha(rot):
    """
    Get alpha matrix.

    Args:
        rot: (str): write your description
    """
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  



def ctdet_post_process(dets, c, s, h, w):
    """
    Ctdet_post_post_process_process_process_process_process

    Args:
        dets: (todo): write your description
        c: (todo): write your description
        s: (todo): write your description
        h: (todo): write your description
        w: (todo): write your description
    """
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    # print("det size {}".format(dets.shape))
    # aa  = input()
    # classes = dets[i, :, -1]
    # for j in range(num_classes):
    #   inds = (classes == j)
    #   top_preds[j + 1] = np.concatenate([
    #     dets[i, inds, :4].astype(np.float32),
    #     dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    # ret.append(top_preds)
  return dets

