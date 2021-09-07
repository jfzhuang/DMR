import math
import torch


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    print('position:', position)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    print('div_term:', div_term)
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def positionalencoding3d(d_model, length, height, width):
    """
    :param d_model: dimension of the model
    :param length: length of the positions
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*length*height*width position matrix
    """
    if d_model % 8 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, length, height, width)
    # Each dimension use half of d_model
    d_m = int(d_model / 4)
    div_term = torch.exp(torch.arange(0., d_m, 2, dtype=torch.float) * -(math.log(10000.0) / d_m))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pos_l = torch.arange(0., length).unsqueeze(1)

    pe[0:d_m:2, :, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).view(d_m // 2, 1, 1,
                                                                            width).repeat(1, length, height, 1)
    pe[0:d_m:2, :, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).view(d_m // 2, 1, 1,
                                                                            width).repeat(1, length, height, 1)
    pe[d_m:d_m * 2:2, :, :, :] = torch.sin(pos_l * div_term).transpose(0, 1).view(d_m // 2, length, 1,
                                                                                  1).repeat(1, 1, height, width)
    pe[d_m + 1:d_m * 2:2, :, :, :] = torch.cos(pos_l * div_term).transpose(0, 1).view(d_m // 2, length, 1,
                                                                                      1).repeat(1, 1, height, width)
    pe[d_m * 2:d_m * 3:2, :, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).view(d_m // 2, 1, height,
                                                                                      1).repeat(1, length, 1, width)
    pe[d_m*2 + 1:d_m * 3:2, :, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).view(d_m // 2, 1, height,
                                                                                        1).repeat(1, length, 1, width)
    pe[d_m * 3::2, :, :, :] = torch.sin(pos_l * div_term).transpose(0, 1).view(d_m // 2, length, 1,
                                                                               1).repeat(1, 1, height, width)
    pe[d_m*3 + 1::2, :, :, :] = torch.cos(pos_l * div_term).transpose(0, 1).view(d_m // 2, length, 1,
                                                                                 1).repeat(1, 1, height, width)

    return pe


if __name__ == '__main__':
    import cv2
    import numpy as np

    # pe = positionalencoding1d(d_model=8, length=6)
    # print(pe)

    # pe = positionalencoding2d(d_model=4, height=6, width=6)
    # for i in range(4):
    #     print(pe[i, ...])

    pe = positionalencoding3d(d_model=256, length=3, height=256, width=256)
    print(pe.shape)

    # for i in range(48):
    #     for j in range(4):
    #         im = pe[i, j, :, :]
    #         im = (im+1) / 2 * 255
    #         im = im.numpy().astype(np.uint8)
    #         cv2.imwrite('/ghome/zhuangjf/DAVSS/exp/transformer/original_v65/model/tmp/{}_{}.png'.format(i, j), im)
