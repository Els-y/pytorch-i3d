#!/usr/bin/env bash

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

OUTPUT_DIR="output/sthsth_`date +%m-%d_%H:%M:%S`"
LOG_FILE="$OUTPUT_DIR/log.txt"

mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR

python train_i3d.py --epoch 70 \
                    --batch 8 \
                    --lr 0.001 \
                    --pretrained true \
                    --worker 8 \
                    --output $OUTPUT_DIR > $LOG_FILE && tail $LOG_FILE -f

