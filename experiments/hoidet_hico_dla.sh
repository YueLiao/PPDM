#!/usr/bin/env bash
set -x

PARTITION=$1
JOB_NAME=$2
GPUS=${4:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-"--validate"}
cd src
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=8 \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u  main.py Hoidet --exp_id hoidet_hico_dla_140epoch_v1_5e-4_v2 --batch_size 128  --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 8 --val_intervals 100000 --image_dir images/train2015 --load_model ../models/ctdet_coco_dla_2x.pth --dataset hico --slurm --dist --apex --sync_bn --print_iter 100

