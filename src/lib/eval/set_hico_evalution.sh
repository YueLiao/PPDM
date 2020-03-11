#!/bin/bash

# ---------------HICO-DET Dataset------------------

echo "Downloading HICO-DET Evaluation Code"

git clone https://github.com/ywchao/ho-rcnn.git
cd ../
cp Generate_detection.m ho-rcnn/
cp save_mat.m ho-rcnn/
cp load_mat.m ho-rcnn/

mkdir ho-rcnn/data/hico_20160224_det/
cp ../../../Dataset/hico/annotations/anno_bbox.mat ho-rcnn/data/hico_20160224_det/
cp ../../../Dataset/hico/annotations/anno.mat ho-rcnn/data/hico_20160224_det/
