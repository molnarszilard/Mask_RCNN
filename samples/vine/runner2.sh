#!bin/bash

CUDA_VISIBLE_DEVICES=3 python vine.py train --weights=coco --dataset=/mnt/ssd1/datasets/vineyards/vineye_leaves/leaves_dataset/lab_infield/images_hls/ --cs=hls

nvidia-smi

CUDA_VISIBLE_DEVICES=3 python vine.py train --weights=coco --dataset=/mnt/ssd1/datasets/vineyards/vineye_leaves/leaves_dataset/lab_infield/images_ycrcb/ --cs=ycrcb
