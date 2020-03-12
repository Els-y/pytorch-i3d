#!/usr/bin/env bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

OUTPUT_DIR="output/sthsth_`date +%m-%d_%H:%M:%S`"
BACKUP_DIR="$OUTPUT_DIR/codes"
LOG_FILE="$OUTPUT_DIR/log.txt"

mkdir -p $BACKUP_DIR
cp $0 $OUTPUT_DIR
cp *.py $BACKUP_DIR

python train_i3d.py --net resnet50 \
                    --epoch 80 \
                    --batch 8 \
                    --lr 0.01 \
                    --pretrained true \
                    --worker 8 \
                    --output $OUTPUT_DIR

