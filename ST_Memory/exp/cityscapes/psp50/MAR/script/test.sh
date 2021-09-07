#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/code/local/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:/code/ST_Memory
export PYTHONPATH=$PYTHONPATH:/code/mmsegmentation

cd /code/ST_Memory && \
python exp_new/cityscapes/psp50/MAR/python/test.py \
            --data_path /data/bitahub/Cityscapes \
            --im_path /data/bitahub/Cityscapes \
            --model_weight /data/jfzhuang/VSPW_480p/work_dirs_v2/ST_Memory/exp_new/cityscapes/psp50/MAR/best.pth \
            --num_workers 8
            