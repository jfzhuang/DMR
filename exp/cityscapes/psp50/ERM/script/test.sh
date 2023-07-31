#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/code/local/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:/code/DRM
export PYTHONPATH=$PYTHONPATH:/code/DRM/mmsegmentation

cd /code/DRM && \
python exp/cityscapes/psp50/ERM/python/test.py \
            --data_path /data/bitahub/Cityscapes \
            --im_path /data/bitahub/Cityscapes \
            --model_weight /data/jfzhuang/VSPW_480p/work_dirs_v2/DRM/exp/cityscapes/psp50/ERM/best.pth \
            --num_workers 8
            