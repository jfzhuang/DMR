import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile_Clip(object):
    def __init__(
        self, to_float32=False, color_type='color', file_client_args=dict(backend='disk'), imdecode_backend='cv2'
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        clip = []
        for filename in results['img_info']['filename']:
            if results.get('img_prefix') is not None:
                filename = osp.join(results['img_prefix'], filename)
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                img = img.astype(np.float32)
            clip.append(img)

        results['filename'] = results['img_info']['filename']
        results['ori_filename'] = results['img_info']['filename']
        results['clip'] = clip
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32), std=np.ones(num_channels, dtype=np.float32), to_rgb=False
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
