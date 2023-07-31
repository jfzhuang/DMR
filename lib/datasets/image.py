import numpy as np
import os
import cv2
import random
from PIL import Image
import torch

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                 (250, 170, 29), (219, 219, 0), (106, 142, 35), (152, 250, 152), (69, 129, 180), (219, 19, 60),
                 (255, 0, 0), (0, 0, 142), (0, 0, 69), (0, 60, 100), (0, 79, 100), (0, 0, 230), (119, 10, 32),
                 (0, 0, 0)]
label_colours_camvid = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (0, 0, 192), (128, 128, 0),
                        (192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0), (0, 128, 192), (0, 0, 0)]


def load_img(dir_path, img_path):
    img = cv2.imread(os.path.join(dir_path, img_path))
    return img


def load_gt(dir_path, img_path):
    gt = cv2.imread(os.path.join(dir_path, img_path), 0)
    return gt


def transform_f(img_f):
    img_f = np.transpose(img_f, (2, 0, 1))
    img_f = img_f.astype(np.float32) / 255.0
    return img_f


def transform_s(img_s):
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    img_s = img_s.transpose((2, 0, 1))
    img_s = img_s.astype(np.float32) / 255.0
    return img_s


def transform_inference_s(img_s, overlap=64, input_size=[1024, 2048]):

    height = input_size[0] // 2
    width = input_size[1] // 2

    image = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    image = image[np.newaxis, :, :, :]
    image = np.concatenate(
        (image[:, :height + overlap, :width + overlap, :], image[:, :height + overlap, width - overlap:, :],
         image[:, height - overlap:, :width + overlap, :], image[:, height - overlap:, width - overlap:, :]), 0)
    image = image.astype(np.float32) / 255.0
    image_s = np.transpose(image, [0, 3, 1, 2])
    return image_s


def transform_inference_f(img_f, overlap=64, input_size=[1024, 2048]):
    height = input_size[0] // 2
    width = input_size[1] // 2
    image = img_f[np.newaxis, :, :, :]
    image = np.concatenate(
        (image[:, :height + overlap, :width + overlap, :], image[:, :height + overlap, width - overlap:, :],
         image[:, height - overlap:, :width + overlap, :], image[:, height - overlap:, width - overlap:, :]), 0)
    image = image.astype(np.float32) / 255.0
    image_f = np.transpose(image, [0, 3, 1, 2])

    return image_f


def transform_inference_s(img_s, overlap=64, input_size=[1024, 2048]):

    height = input_size[0] // 2
    width = input_size[1] // 2

    image = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    image = image[np.newaxis, :, :, :]
    image = np.concatenate(
        (image[:, :height + overlap, :width + overlap, :], image[:, :height + overlap, width - overlap:, :],
         image[:, height - overlap:, :width + overlap, :], image[:, height - overlap:, width - overlap:, :]), 0)
    image = image.astype(np.float32) / 255.0
    image_s = np.transpose(image, [0, 3, 1, 2])
    return image_s


def transform_inference_f(img_f, overlap=64, input_size=[1024, 2048]):
    height = input_size[0] // 2
    width = input_size[1] // 2
    image = img_f[np.newaxis, :, :, :]
    image = np.concatenate(
        (image[:, :height + overlap, :width + overlap, :], image[:, :height + overlap, width - overlap:, :],
         image[:, height - overlap:, :width + overlap, :], image[:, height - overlap:, width - overlap:, :]), 0)
    image = image.astype(np.float32) / 255.0
    image_f = np.transpose(image, [0, 3, 1, 2])

    return image_f


def transform_crop(img_list, gt, outpit_size, random=True):
    crop = Crop(outpit_size, random)
    img_list, gt = crop(img_list, gt)
    return img_list, gt


def transform_crop_v2(img_list, outpit_size, random=True):
    crop = Crop_v2(outpit_size, random)
    img_list = crop(img_list)
    return img_list


def transform_crop_list(img_s_list, img_f_list, gt, outpit_size, random=True):
    crop = Crop_list(outpit_size, random)
    img_s_list, img_f_list, gt = crop(img_s_list, img_f_list, gt)
    return img_s_list, img_f_list, gt


def transform_scale_crop(img_list, gt, outpit_size, random=True):
    crop = Scale_Crop(outpit_size, random)
    img_list, gt = crop(img_list, gt)
    return img_list, gt


def decode_labels_pytorch(mask, num_classes):
    n, h, w, c = mask.shape
    num_classes += 1
    mask[mask == 255] = 19
    color_table = np.array(label_colours)
    color_table = torch.from_numpy(color_table).long()

    onehot_output = torch.zeros((n * h * w * c, num_classes))
    if mask.is_cuda:
        color_table = color_table.cuda()
        onehot_output = onehot_output.cuda()
    onehot_output = onehot_output.scatter(1, mask.view(-1, 1), 1)
    outs = torch.mm(onehot_output.float(), color_table.float())
    outs = outs.view(n, h, w, 3)
    return outs


def decode_labels(mask):
    h, w = mask.shape
    mask[mask == 255] = 19
    color_table = np.array(label_colours, dtype=np.float32)
    out = np.take(color_table, mask, axis=0)
    out = out.astype(np.uint8)
    out = out[:, :, ::-1]
    return out


def decode_labels_camvid(mask):
    h, w = mask.shape
    mask[mask == 255] = 11
    color_table = np.array(label_colours_camvid, dtype=np.float32)
    out = np.take(color_table, mask, axis=0)
    out = out.astype(np.uint8)
    out = out[:, :, ::-1]
    return out


def new_overlap(pred, max_value, input_size=[512, 1024], overlap=64):
    height, width = input_size

    for i in range(4):
        pred[i] = pred[i].cpu().numpy()
        max_value[i] = max_value[i].cpu().numpy()

    pred_list = np.zeros([4, height * 2, width * 2], dtype=np.int64)
    pred_list[0, :height + overlap, :width + overlap] = pred[0]
    pred_list[1, :height + overlap, width - overlap:] = pred[1]
    pred_list[2, height - overlap:, :width + overlap] = pred[2]
    pred_list[3, height - overlap:, width - overlap:] = pred[3]

    value_list = np.zeros([4, height * 2, width * 2])
    value_list[0, :height + overlap, :width + overlap] = max_value[0]
    value_list[1, :height + overlap, width - overlap:] = max_value[1]
    value_list[2, height - overlap:, :width + overlap] = max_value[2]
    value_list[3, height - overlap:, width - overlap:] = max_value[3]
    select_list = np.argmax(value_list, axis=0)

    final_pred = np.choose(select_list, pred_list)
    return final_pred


class Crop(object):
    def __init__(self, output_size, random=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.random = random

    def __call__(self, img_list, gt):

        h, w = gt.shape
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        if self.random:
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
        else:
            top_list = [0, h - new_h]
            left_list = [0, w - new_w]
            top = random.sample(top_list, 1)
            left = random.sample(left_list, 1)
            top = top[0]
            left = left[0]

        for i in range(len(img_list)):
            img_list[i] = img_list[i][:, top:top + new_h, left:left + new_w]
        gt = gt[top:top + new_h, left:left + new_w]

        return img_list, gt


class Crop_v2(object):
    def __init__(self, output_size, random=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.random = random

    def __call__(self, img_list):

        _, h, w = img_list[0].shape
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        if self.random:
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
        else:
            top_list = [0, h - new_h]
            left_list = [0, w - new_w]
            top = random.sample(top_list, 1)
            left = random.sample(left_list, 1)
            top = top[0]
            left = left[0]

        for i in range(len(img_list)):
            if len(img_list[i].shape) == 3:
                img_list[i] = img_list[i][:, top:top + new_h, left:left + new_w]
            else:
                img_list[i] = img_list[i][top:top + new_h, left:left + new_w]

        return img_list


class Crop_list(object):
    def __init__(self, output_size, random=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.random = random

    def __call__(self, img_list1, img_list2, gt):

        h, w = gt.shape
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        if self.random:
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)
        else:
            top_list = [0, h - new_h]
            left_list = [0, w - new_w]
            top = random.sample(top_list, 1)
            left = random.sample(left_list, 1)
            top = top[0]
            left = left[0]

        for i in range(len(img_list1)):
            img_list1[i] = img_list1[i][:, top:top + new_h, left:left + new_w]
            img_list2[i] = img_list2[i][:, top:top + new_h, left:left + new_w]
        gt = gt[top:top + new_h, left:left + new_w]

        return img_list1, img_list2, gt


class Scale_Crop(object):
    def __init__(self, output_size, random=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.random = random
        self.ratio = [0.5, 2]

    def __call__(self, img_list, gt):

        h, w = gt.shape
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        ratio = random.uniform(self.ratio[0], self.ratio[1])
        scale_h, scale_w = int(round(h * ratio)), int(round(w * ratio))
        scale_img_list = []
        for img in img_list:
            img = img.transpose([1, 2, 0])
            scale_img = cv2.resize(img, dsize=(scale_w, scale_h), interpolation=cv2.INTER_LINEAR)
            scale_img = scale_img.transpose([2, 0, 1])
            scale_img_list.append(scale_img)
        scale_gt = cv2.resize(gt, dsize=(scale_w, scale_h), interpolation=cv2.INTER_NEAREST)

        if self.random:
            top = np.random.randint(0, scale_h - new_h + 1)
            left = np.random.randint(0, scale_w - new_w + 1)
        else:
            top_list = [0, scale_h - new_h]
            left_list = [0, scale_w - new_w]
            top = random.sample(top_list, 1)
            left = random.sample(left_list, 1)
            top = top[0]
            left = left[0]

        for i in range(len(scale_img_list)):
            scale_img_list[i] = scale_img_list[i][:, top:top + new_h, left:left + new_w]
        scale_gt = scale_gt[top:top + new_h, left:left + new_w]

        return scale_img_list, scale_gt


def pred2im(pred):
    pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    pred = decode_labels(pred)
    return pred


def tensor2array(tensor, BGR=False, normalize=True):
    tensor = tensor.squeeze().cpu().numpy()
    tensor = tensor.transpose((1, 2, 0))
    if BGR:
        tensor = tensor[:, :, ::-1]
    if normalize:
        tensor *= 255
    tensor = tensor.astype(np.uint8)
    return tensor


def obtain_mask(im_size, parts=10, maxVertex=20, maxLength=100, maxBrushWidth=25, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.transpose(mask, [2, 0, 1])
    mask = np.expand_dims(mask, 0)
    mask = torch.from_numpy(mask)
    mask = mask.detach().cuda()
    return mask


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)

    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)

        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)

        cv2.line(mask, (startX, startY), (nextX, nextY), 1, brushWidth)
        cv2.circle(mask, (startX, startY), brushWidth // 2, 2)

        startY, startX = nextY, nextX
    cv2.circle(mask, (startX, startY), brushWidth // 2, 2)
    return mask