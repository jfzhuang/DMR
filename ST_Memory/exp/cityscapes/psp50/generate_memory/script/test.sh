#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/code/local/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:/code/ST_Memory
export PYTHONPATH=$PYTHONPATH:/code/mmsegmentation

cd /code/ST_Memory && \
python3 exp/cityscapes/psp50/generate_memory/python/test.py \
            --data_path /data/bitahub/Cityscapes \
            --im_path /data/bitahub/Cityscapes \
            --num_workers 8
            