#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/code/local/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:/code/DRM
export PYTHONPATH=$PYTHONPATH:/code/DRM/mmsegmentation

cd /code/DRM && \
python3 exp/cityscapes/psp50/generate_memory/python/test.py \
            --data_path /data/bitahub/Cityscapes \
            --im_path /data/bitahub/Cityscapes \
            --num_workers 8
            