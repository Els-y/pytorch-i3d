#!/usr/bin/env bash

# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
python test_i3d.py --net resnet50 \
                   --phase val \
                   --weight output/1.3_sthsth_03-11_10:52:33/weights/sthsth_80.pt \
                   --batch 1 \
                   --worker 1

