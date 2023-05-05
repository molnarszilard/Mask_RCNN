#!bin/bash

CUDA_VISIBLE_DEVICES=0 python vine.py train --weights=coco --dataset=/mnt/ssd1/datasets/vineyards/vineye_leaves/leaves_dataset/lab_infield/images/ --cs=rgb

nvidia-smi

CUDA_VISIBLE_DEVICES=0 python vine.py train --weights=coco --dataset=/mnt/ssd1/datasets/vineyards/vineye_leaves/leaves_dataset/lab_infield/images_hsv/ --cs=hsv
