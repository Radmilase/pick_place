#!/bin/bash
python3 /home/rad/yolov5/train.py \
  --img 640 \
  --batch 16 \
  --epochs 5 \
  --data /home/rad/master/p/data.yaml\
  --weights yolov5s.pt \
  --name custom_train \
  --cache disk \
  --device cpu \
  --workers 2 \
