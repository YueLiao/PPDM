import numpy as np
import cv2
import os, random
import json

from torch.utils.data import Dataset

from utils.image import get_affine_transform, affine_transform
from datasets.core import get_heatmap_target
from datasets.core import data_augmentation

MC_AVAILABLE = False

try:
    import sys

    sys.path.append(r'/mnt/lustre/share/pymc/py3')
    import mc

    MC_AVAILABLE = True
except ModuleNotFoundError:
    print('Memcached is not available')


class HOIDataset(Dataset):
    num_classes = 0
    num_classes_verb = 0
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
    dataset_tag = None
    ann_tag = None
    _valid_ids = []
    _valid_ids_verb = []

    def __init__(self, opt, split='train'):
        self.opt = opt
        self.root = os.path.join(self.opt.root_path, self.dataset_tag)
        self.image_dir = self.opt.image_dir
        if split == 'train':
            self.hoi_annotations = json.load(open(os.path.join(self.root, 'annotations', self.ann_tag[split]), 'r'))
            self.flip = self.opt.flip
            self.ids = []
            self.filter_bad_anns()
            self.shuffle()
            self.max_objs = 128
            self.max_rels = 64
            self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
            self.cat_ids_verb = {v: i for i, v in enumerate(self._valid_ids_verb)}
            self.split = split
            self.num_classes_verb = self.num_classes_verb
        else:
            self.hoi_annotations = json.load(open(os.path.join(self.root, 'annotations', self.ann_tag[split]), 'r'))
            self.ids = list(range(len(self.hoi_annotations)))

        if MC_AVAILABLE:
            self.mclient = None

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = self.load_img(img_id)
        anns, hoi_anns = self.load_anns(img_id)
        num_objs = min(len(anns), self.max_objs)
        num_rels = min(len(hoi_anns), self.max_rels)

        height, width = img.shape[0], img.shape[1]

        # img augmentation
        inp, trans_output, flipped, output_w, output_h = \
            data_augmentation(img, self.opt.input_w, self.opt.input_h, keep_res=self.opt.keep_res,
                              rand_crop=not self.opt.not_rand_crop, flip=self.opt.flip,
                              with_color_aug=not self.opt.no_color_aug, pad=self.opt.pad,
                              down_ratio=self.opt.down_ratio)
        # normalize img
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        bbox_list = self.load_bbox(anns, trans_output, num_objs, width, output_w, output_w, flipped)
        hoi_list = self.load_hoi(hoi_anns, num_rels)

        num_rels_update = min(len(hoi_list), self.max_rels)
        ret = get_heatmap_target(bbox_list, hoi_list, output_w, output_h, self.max_objs, self.max_rels,
                                 self.num_classes, self.num_classes_verb, num_objs, num_rels_update)
        # import pdb; pdb.set_trace()
        ret.update({'input': inp})
        return ret

    def __len__(self):
        return len(self.ids)

    def filter_bad_anns(self):
        for i, hoi_anns in enumerate(self.hoi_annotations):
            flag_bad = 0
            for hoi in hoi_anns['hoi_annotation']:
                if hoi['subject_id'] >= len(hoi_anns['annotations']) or hoi['object_id'] >= len(
                        hoi_anns['annotations']):
                    flag_bad = 1
                    break
            if flag_bad == 0:
                self.ids.append(i)

    # load image with memcached
    def _ensure_memcached(self):
        if self.mclient is None:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
        return

    def read_img_with_mc(self, img_path):
        self._ensure_memcached()
        value = mc.pyvector()
        self.mclient.Get(img_path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    def shuffle(self):
        random.shuffle(self.ids)

    def load_img(self, img_id):
        file_name = self.hoi_annotations[img_id]['file_name']
        img_path = os.path.join(self.root, self.image_dir, file_name)
        if MC_AVAILABLE:
            img = self.read_img_with_mc(img_path)
        else:
            img = cv2.imread(img_path)
        return img

    def load_anns(self, img_id):
        anns = self.hoi_annotations[img_id]['annotations']
        hoi_anns = self.hoi_annotations[img_id]['hoi_annotation']
        return anns, hoi_anns

    def load_bbox(self, anns, trans_output, num_objs, width, output_w, output_h, flipped):
        bbox_list = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = np.asarray(ann['bbox'])
            if isinstance(ann['category_id'], str):
                ann['category_id'] = int(ann['category_id'].replace('\n', ''))
            cls_id = int(self.cat_ids[ann['category_id']])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            bbox_list.append([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1. * w, 1. * h, cls_id])
        return bbox_list

    def load_hoi(self, hoi_anns, num_rels):
        hoi_list = []
        for k in range(num_rels):
            hoi = hoi_anns[k]
            if isinstance(hoi['category_id'], str):
                hoi['category_id'] = int(hoi['category_id'].replace('\n', ''))
            if isinstance(hoi['category_id'], int):
                hoi_cate = int(self.cat_ids_verb[hoi['category_id']])
                hoi_list.append([int(hoi['subject_id']), int(hoi['object_id']), hoi_cate])
            if isinstance(hoi['category_id'], list):
                for hoi_cate in hoi['category_id']:
                    hoi_list.append([int(hoi['subject_id']), int(hoi['object_id']), int(self.cat_ids_verb[hoi_cate])])
        return hoi_list
