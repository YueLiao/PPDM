#!/usr/bin/env bash
set -x
cd src

#train
python -u  main.py hoidet --exp_id hoidet_hico_res18 --arch resdcn_18 --batch_size 128 --master_batch 9  --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --val_intervals 100000 --image_dir images/train2015 --load_model ../models/ctdet_coco_resdcn18.pth --dataset hico --root_path '/mnt/lustre/liaoyue/datasets'

#test
python -u test_hoi.py hoidet --exp_id hoidet_hico_res18 --arch resdcn_18  --gpus 0 --image_dir images/test2015 --dataset hico --test_with_eval
