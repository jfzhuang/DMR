#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/code/local/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:/code/DRM
export PYTHONPATH=$PYTHONPATH:/code/DRM/mmsegmentation

cd /code/DRM && \
python ./exp/cityscapes/psp50/CRM/python/train.py \
        --exp_name CRM \
        --root_data_path /data/bitahub/Cityscapes \
        --root_im_path /data/bitahub/Cityscapes \
        --random_seed 666 \
        --lr 1e-5 \
        --momentum 0.9 \
        --weight_decay 1e-4 \
        --train_power 0.9 \
        --train_batch_size 4 \
        --train_num_workers 4 \
        --val_batch_size 4 \
        --val_num_workers 4 \
        --num_epoch 50 \
        --snap_shot 2 \
        --model_save_path /output/saved_model \
        --tblog_dir /output/tblog