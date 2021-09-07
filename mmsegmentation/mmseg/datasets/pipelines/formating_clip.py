from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class DefaultFormatBundle_Clip(object):
    def __call__(self, results):
        if 'clip' in results:
            clip = results['clip']
            for i in range(len(clip)):
                if len(clip[i].shape) < 3:
                    clip[i] = np.expand_dims(clip[i], -1)
                clip[i] = np.ascontiguousarray(clip[i].transpose(2, 0, 1))
                clip[i] = DC(to_tensor(clip[i]), stack=True)
            results['clip'] = clip
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...].astype(np.int64)), stack=True
            )
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ImageToTensor_Clip(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            clip = results[key]
            for i in range(len(clip)):
                if len(clip[i].shape) < 3:
                    clip[i] = np.expand_dims(clip[i], -1)
                clip[i] = to_tensor(clip[i].transpose(2, 0, 1))
            results[key] = clip
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
