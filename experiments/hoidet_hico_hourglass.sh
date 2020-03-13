#!/usr/bin/env bash
set -x

#!/usr/bin/env bash
set -x
cd src

#train
python -u  main.py hoidet --exp_id hoidet_hico_hourglass --arch hourglass  --batch_size 31 --master_batch 3 --lr 3e-4  --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --val_intervals 100000 --load_model ../models/ctdet_coco_hg.pth --image_dir images/train2015 --dataset hico --root_path '/mnt/lustre/liaoyue/datasets'

#test
python -u test_hoi.py hoidet --arch hourglass --exp_id hoidet_hico_hourglass  --gpus 0 --image_dir images/test2015 --dataset hico --test_with_eval

