import os
import ast
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from lib.metrics import runningScore
from lib.datasets.CityscapesDataset import CityscapesDataset
from exp_new.cityscapes.psp50.STF.model.refine import SegRefine


def get_arguments():
    parser = argparse.ArgumentParser(description="Train Deeplabv3+")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--root_data_path", type=str, help="root path to the dataset")
    parser.add_argument("--root_im_path", type=str, help="root path to the image")
    parser.add_argument("--crop_size", type=int, default=768, help="crop_size")

    ###### training setting ######
    parser.add_argument("--resume", type=ast.literal_eval, default=False, help="resume or not")
    parser.add_argument("--resume_epoch", type=int, help="from which epoch for resume")
    parser.add_argument("--resume_load_path", type=str, help="resume model load path")
    parser.add_argument("--train_load_path", type=str, help="train model load path")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--random_seed", type=int, help="random seed")
    parser.add_argument("--train_power", type=float, help="power value for linear learning rate schedule")
    parser.add_argument("--momentum", type=float, help="momentum")
    parser.add_argument("--weight_decay", type=float, help="weight_decay")
    parser.add_argument("--train_batch_size", type=int, help="train batch size")
    parser.add_argument("--train_num_workers", type=int, default=8, help="num cpu use")
    parser.add_argument("--num_epoch", type=int, default=100, help="num of epoch in training")
    parser.add_argument("--snap_shot", type=int, default=1, help="save model every per snap_shot")
    parser.add_argument("--model_save_path", type=str, help="model save path")

    ###### testing setting ######
    parser.add_argument("--val_batch_size", type=int, default=1, help="batch_size for validation")
    parser.add_argument("--val_num_workers", type=int, default=4, help="num of used cpus in validation")

    ###### tensorboard setting ######
    parser.add_argument("--use_tensorboard", type=ast.literal_eval, default=True, help="use tensorboard or not")
    parser.add_argument("--tblog_dir", type=str, help="log save path")
    parser.add_argument("--tblog_interval", type=int, default=50, help="interval for tensorboard logging")

    return parser.parse_args()


def make_dirs(args):
    if args.use_tensorboard and not os.path.exists(args.tblog_dir):
        os.makedirs(args.tblog_dir)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)


def train():
    args = get_arguments()
    print(args)
    make_dirs(args)

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('random seed:{}'.format(random_seed))
    
    tblogger = SummaryWriter(args.tblog_dir)

    train_dataset = CityscapesDataset(args.root_data_path, args.root_im_path, 'train', resize_size=(768, 1536))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=args.train_num_workers,
        drop_last=True,
        persistent_workers=True,
    )
    val_dataset = CityscapesDataset(args.root_data_path, args.root_im_path, 'val')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=args.val_num_workers,
        drop_last=False,
        persistent_workers=True,
    )

    net = SegRefine()
        
    if args.resume:
        old_weight = torch.load(args.resume_load_path, map_location='cpu')
        start_epoch = args.resume_epoch
        new_weight = {}
        for k, v in old_weight.items():
            k = k.replace('module.', '')
            new_weight[k] = v
        net.load_state_dict(new_weight, strict=True)
    else:
        start_epoch = 0

    print('Successful loading model!')
    
    params = []
    for p in net.STF.parameters():
        params.append(p)
    for p in net.pspnet.decode_head.conv_seg.parameters():
        params.append(p)
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    net.cuda()
    net = torch.nn.DataParallel(net)    

    running_loss = 0.0
    miou_cal = runningScore(n_classes=19)
    current_miou = 0
    itr = start_epoch * len(train_dataloader)
    max_itr = args.num_epoch * len(train_dataloader)

    criterion_semantic = nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(start_epoch, args.num_epoch):
        net.train()
        net.module.pspnet.eval()
        for step, data in enumerate(train_dataloader):
            im_list, gt = data

            for i in range(len(im_list)):
                im_list[i] = im_list[i].cuda()
            gt = gt.cuda()

            optimizer.zero_grad()
#             pred = net(im_list)
#             loss_semantic = criterion_semantic(pred, gt)
            
            loss_semantic = net(im_list, gt)
            loss_semantic = loss_semantic.mean()
            
            loss_semantic.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.1)
            optimizer.step()

            print('epoch:{}/{} batch:{}/{} itr:{} loss:{:02f}'.format(epoch, args.num_epoch, step,
                                                                          len(train_dataloader), itr,
                                                                          loss_semantic.item()))
            if args.use_tensorboard and itr % args.tblog_interval == 0:
                tblogger.add_scalar('data/loss', loss_semantic.item(), itr)

            itr += 1

#             if step == 20:
#                 break

        if (epoch + 1) % args.snap_shot == 0:
            net.eval()
            for step, data in enumerate(val_dataloader):
                print('{}/{}'.format(step, len(val_dataloader)))
                im_list, gt = data
                gt = gt.numpy()
                for i in range(len(im_list)):
                    im_list[i] = im_list[i].cuda()
                with torch.no_grad():
                    pred = net(im_list)
                    out = torch.argmax(pred, dim=1)
                out = out.cpu().numpy()
                miou_cal.update(gt, out)

#                 if step == 10:
#                     break

            miou = miou_cal.get_scores()
            miou_cal.reset()
        
            save_path = os.path.join(args.model_save_path, 'epoch_{}.pth'.format(epoch))
            torch.save(net.module.state_dict(), save_path)

            print('miou:{}'.format(miou))
            tblogger.add_scalar('data/mIoU', miou, epoch)
            if miou > current_miou:
                save_path = os.path.join(args.model_save_path, 'best.pth')
                torch.save(net.module.state_dict(), save_path)
                current_miou = miou
            
            torch.cuda.empty_cache()


    save_path = os.path.join(args.model_save_path, 'last.pth')
    torch.save(net.module.state_dict(), save_path)
    if args.use_tensorboard:
        tblogger.close()
    print('%s has been saved' % save_path)


if __name__ == '__main__':
    train()
