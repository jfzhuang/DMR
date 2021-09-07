import cv2
import numpy as np

from mmcv.utils import Config
from mmseg.datasets import build_dataset, build_dataloader


def transform(im):
    mean = [123.675, 116.28, 103.53]
    mean = np.array(mean).reshape((1, 1, 3))
    std = [58.395, 57.12, 57.375]
    std = np.array(std).reshape((1, 1, 3))

    im = im.permute(1, 2, 0).numpy()
    im = im * std + mean
    im = im.astype(np.uint8)

    return im


def main_temporal():
    config = '/ghome/zhuangjf/ST-Fusion/configs/_base_/datasets/multi_frame.py'
    cfg = Config.fromfile(config)

    # cfg = cfg.data.train
    # cfg['data_root'] = '/ghome/zhuangjf/ST-Fusion/data/cityscapes/'
    # cfg['split'] = '/gdata/zhuangjf/cityscapes/original/list/train_3_frames.txt'
    # cfg['img_dir'] = '/gdata/zhuangjf/cityscapes/original/leftImg8bit_sequence/train'
    # dataset = build_dataset(cfg)
    # data_loader = build_dataloader(dataset, 1, 1, drop_last=True)
    # for i, data in enumerate(data_loader):
    #     clip = data['clip']
    #     gt = data['gt_semantic_seg'].data[0]
    #     print('{}/{}'.format(i, len(data_loader)), len(clip), clip[0].data[0].shape, gt.shape)

    cfg = cfg.data.val
    cfg['data_root'] = '/ghome/zhuangjf/ST-Fusion/data/cityscapes/'
    cfg['split'] = '/gdata/zhuangjf/cityscapes/original/list/val_3_frames.txt'
    cfg['img_dir'] = '/gdata/zhuangjf/cityscapes/original/leftImg8bit_sequence/val'
    dataset = build_dataset(cfg)
    data_loader = build_dataloader(dataset, 1, 1, drop_last=True)
    for i, data in enumerate(data_loader):
        clip = data['clip']
        print('{}/{}'.format(i, len(data_loader)), len(clip[0]), clip[0][0].data[0].shape)


if __name__ == '__main__':
    main_temporal()
