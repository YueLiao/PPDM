import numpy as np
import cv2

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform

_data_rng = np.random.RandomState(123)
_eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                    dtype=np.float32)
_eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)


def data_augmentation(img, w, h, keep_res=False, rand_crop=False,
                      flip=False, with_color_aug=False, pad=31, scale=1, shift=0.1, down_ratio=4):
    flipped = False
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if keep_res:
        input_h = (height | pad) + 1
        input_w = (width | pad) + 1
        s = np.array([input_w, input_h], dtype=np.float32)
    else:
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = h, w
        if rand_crop:
            s = s * np.random.choice(np.arange(0.7, 1.4, 0.1))
            w_border = _get_border(128, img.shape[1])
            h_border = _get_border(128, img.shape[0])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        else:
            sf = scale
            cf = shift
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < flip:
            flipped = True
            img = img[:, ::-1, :]
            c[0] = width - c[0] - 1

    trans_input = get_affine_transform(
        c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if with_color_aug:
        color_aug(_data_rng, inp, _eig_val, _eig_vec)

    output_h = input_h // down_ratio
    output_w = input_w // down_ratio
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
    return inp, trans_output, flipped, output_w, output_h


def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
