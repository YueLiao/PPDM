from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import _init_paths
from progress.bar import Bar
import torch
from eval.hico_eval import hico
from eval.vcoco_eval import vcoco
from eval.hoia_eval import hoia
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import get_dataset
from eval.save_json import save_json

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.pre_process_func = pre_process_func
    self.opt = opt
    self.root = dataset.root
    self.image_dir = dataset.image_dir
    self.hoi_annotations = dataset.hoi_annotations
    self.ids = dataset.ids

  def __getitem__(self, index):
    img_id = self.ids[index]
    img_name = self.hoi_annotations[img_id]['file_name']
    img_path = os.path.join(self.root, self.image_dir, img_name)
    image = cv2.imread(img_path)
    images, meta = {}, {}
    for scale in opt.test_scales:
        images[scale], meta[scale] = self.pre_process_func(image, scale)
    return img_id, {'images': images, 'image': image, 'meta': meta}

  def __len__(self):
    return len(self.ids)

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  Dataset = get_dataset(opt.dataset)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  Logger(opt)
  Detector = detector_factory[opt.task]
  dataset = Dataset(opt, 'test')
  model_begin = 100
  model_end = 140
  if opt.load_model != '':
    model_begin = 0
    model_end = 0
  if opt.test_with_eval:
    map_dcit = {'best_id': 0, 'best_map': 0}
    best_output = []
  for model_id in range(model_begin, model_end+1):
    model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') else opt.save_dir
    if opt.load_model == '' or model_id > model_begin: 
      opt.load_model = os.path.join(model_path, 'model_' + str(model_id) + '.pth')
    detector = Detector(opt)

    data_loader = torch.utils.data.DataLoader(
      PrefetchDataset(opt, dataset, detector.pre_process),
      batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    num_iters = len(dataset)
    print("----epoch :{} -----".format(model_id))
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    output_hoi = []

    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
      ret = detector.run(pre_processed_images)
      output_i = ret['results_rel'].copy()
      output_i['file_name'] = dataset.hoi_annotations[int(img_id)]['file_name']
      output_hoi.append(output_i)
      Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
        ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
      for t in avg_time_stats:
        avg_time_stats[t].update(ret[t])
        Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
          t, tm=avg_time_stats[t])
      bar.next()
    bar.finish()


    if opt.test_with_eval:
      if 'hico' in opt.dataset:
        hoi_eval = hico(os.path.join(opt.root_path, 'hico_det/annotations/test_hico.json'))
      elif 'vcoco' in opt.dataset:
        hoi_eval = vcoco(os.path.join(opt.root_path,'verbcoco/annotations/test_vcoco.json'))
      elif 'hoia' in opt.dataset:
        hoi_eval = hoia(os.path.join(opt.root_path,'hoia/annotations/test_hoia.json'))
      map = hoi_eval.evalution(output_hoi)
      if map>map_dcit['best_map']:
        map_dcit['best_map'] = map
        map_dcit['best_id'] = model_id
        best_output = output_hoi

    if opt.save_predictions:
      save_json(output_hoi, model_path, 'predictions_model_' + str(model_id) + '.json')
  if opt.test_with_eval:
    print('best model id: {}, best map: {}'.format(map_dcit['best_id'], map_dcit['best_map']))
    save_json(best_output, model_path, 'best_predictions.json')

if __name__ == '__main__':
  opt = opts().parse()
  prefetch_test(opt)
