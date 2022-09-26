#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from models.PairedSimSiam import PairedSimSiam
import json
from util import utils
import logging
import math
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from os.path import join
from datasets.PairedImagesDataset import TestImagesDataset
import os

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Self supervised pre-training for RPRs Testing')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='test_data/sanity1/')
parser.add_argument('--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 32)')
parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size ')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default="0", type=str, help='GPU id to use.')
# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int, help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int, help='hidden dimension of the predictor (default: 512)')
#added params
parser.add_argument('--img_size', default=448, type=int,help='image size after crop (square)')
parser.add_argument('--test_mode', default=0, type=int, help='0:pos+neg, 1:pos, 2: neg')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
resume_save_dir = os.path.join(os.path.dirname(args.resume),"test")
writer, args.save_dir = utils.init_log(args, save_dir=resume_save_dir)

def main():
    with torch.no_grad():

        if args.gpu is not None:
            print("Use GPU: {} for training".format(args.gpu))

        if args.resume and not os.path.isfile(args.resume):
            print('Failed! missing checkpoint: ' + args.resume)
            return

        # Set the seeds and the device
        use_cuda = torch.cuda.is_available()
        device_id = 'cpu'
        torch_seed = 0
        torch.manual_seed(torch_seed)
        if use_cuda:
             torch.backends.cudnn.deterministic = True
             torch.backends.cudnn.benchmark = False
             device_id = 'cuda:0'
        device = torch.device(device_id)

        # Create model
        print("Creating model with a '{}' encoder".format(args.arch))
        model = PairedSimSiam(
            models.__dict__[args.arch],
            args.dim, args.pred_dim, 'test')
        model = model.to(device)

        # define loss function (criterion) and optimizer
        criterion_cosine = nn.CosineSimilarity(dim=1).cuda()
        criterion_MSE = nn.MSELoss().cuda()

        # optionally resume from a checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:0'
                checkpoint = torch.load(args.resume, map_location=loc)
            #model.load_state_dict(checkpoint['state_dict'])
            #torch.save(model.state_dict(), os.path.join(args.save_dir,os.path.basename(args.resume).split(".tar")[0]), _use_new_zipfile_serialization=False)
            #print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        cudnn.benchmark = True

        # Data loading code
        testdir = os.path.join(args.data)

        augmentation = [
            transforms.Resize((args.img_size,args.img_size)),
            transforms.ToTensor()
        ]
        test_dataset = TestImagesDataset(
            testdir,
            transforms.Compose(augmentation), args.test_mode, 'anchor', 'positive', 'negative')


        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=True)

        test(test_loader, model, criterion_cosine, args, writer)


def test(test_loader, model, criterion_cosine, args, writer):
    with torch.no_grad():
        batch_time = utils.AverageMeter('Time', ':6.3f')
        data_time = utils.AverageMeter('Data', ':6.3f')
        # mse = AverageMeter('mse_negative', ':.4f')
        cosine_similarity_positive = utils.AverageMeter('cosine_similarity_positive', ':.4f')
        cosine_similarity_negative = utils.AverageMeter('cosine_similarity_negative', ':.4f')
        progress = utils.ProgressMeter(
            len(test_loader),
            [batch_time, data_time, cosine_similarity_positive, cosine_similarity_negative],
            prefix="Test:")

        # switch to train mode
        model.eval()
        fails=0
        n_imgs=len(test_loader)
        end = time.time()
        for i, images in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            anchor_x = images[0].cuda() #anchor_x, anchor_y
            anchor_y = images[1].cuda() #anchor_x, anchor_y
            positive_x = images[2].cuda()
            positive_y = images[3].cuda()
            negative_x = images[4].cuda()
            negative_y = images[5].cuda()
            img_name = images[6]

            # compute output and loss
            z_a = model(x1=anchor_x, y1=anchor_y, x2=None, y2=None)
            if args.test_mode==0 or args.test_mode==1:
                z_p = model(x1=positive_x, y1=positive_y, x2=None, y2=None)
            if args.test_mode==0 or args.test_mode==2:
                z_n = model(x1=negative_x, y1=negative_y, x2=None, y2=None)

            cos_sim_pos,cos_sim_neg = 0,0

            # for j in range(len(all_a)):
            #     cos_sim_pos += criterion_cosine(all_a[j], all_p[j]).mean()/len(all_a)
            #     cos_sim_neg += criterion_cosine(all_a[j], all_n[j]).mean()/len(all_a)
            if args.test_mode==0 or args.test_mode==1:
                cos_sim_pos += criterion_cosine(z_a, z_p).mean()
                cosine_similarity_positive.update(cos_sim_pos.item(), images[0].size(0))
            if args.test_mode == 0 or args.test_mode == 2:
                cos_sim_neg += criterion_cosine(z_a, z_n).mean()
                cosine_similarity_negative.update(cos_sim_neg.item(), images[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            progress.display(i)

            utils.log_img_to_tensorboard_triplet(writer, images, step=i)
            if args.test_mode==0 and cosine_similarity_negative.val>cosine_similarity_positive.val:
                fails+=1
                print("Failed -- Img:{}".format(img_name))
                progress.display(i)
                utils.log_img_to_tensorboard_triplet(writer, images, step=i, miss=True)
        print("")
        print("fails = {}/{}".format(fails,n_imgs))
        print("COSINE_SIM positive = {:.4f}, COSINE_SIM negative = {:.4f}".format(
            cosine_similarity_positive.avg, cosine_similarity_negative.avg
        ))

if __name__ == '__main__':
    main()
