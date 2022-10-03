#!/usr/bin/env python
# Extension of SimSiam to RPR

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
from datasets.PairedImagesDataset import PairedImagesDataset
import os


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Self supervised pre-training for RPRs')
parser.add_argument('--dataset_path',  help='path to dataset', default='/nfstemp/Datasets/7Scenes/')
parser.add_argument('--pairs_file', help='file with pairs', default='7scenes_training_pairs.csv')
parser.add_argument('--config_file', help='config file', default='pre_rpr_config.json')
parser.add_argument('--lr', help='learning rate', type=float, default=0.05)
parser.add_argument('--batch_size', help='batch size', type=int, default=12)
parser.add_argument('--momentum', default=0.9, type=float,help='momentum of SGD solver')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--n_freq_print', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--n_freq_checkpoint', default=10, type=int, help='checkpoint frequency (default: 10)')
parser.add_argument('--n_workers', default=4, type=int, help='number of data loading workers (default: 32)')
parser.add_argument('--start_epoch', help='start epoch', type=int, default=0)
parser.add_argument('--epochs', help='epochs', type=int, default=512)
parser.add_argument('--gpu_id', help='gpu id', default='3')
parser.add_argument('--arch', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('--exp_name', default='default', type=str, help='experiment name for logging.')
# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int, help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int, help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true', help='Fix learning rate for the predictor')
parser.add_argument('--weight_latest_loss', help='weight_latest_loss', type=float, default=0.01)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    utils.set_proxy()
    writer, args.save_dir = utils.init_log(args, saved_path=args.exp_name)
    utils.init_logger(args.save_dir)

    # Record execution details
    logging.info("Start {} experiment for RPR".format(args.arch))
    logging.info("Using dataset: {}".format(args.dataset_path))
    #logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    #with open(args.config_file, "r") as read_file:
    #    config = json.load(read_file)
    #print("Running with configuration:\n{}".format(
    #    '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

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
       args.dim, args.pred_dim, 'train')
    model = model.to(device)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).to(device)
    l1_loss = torch.nn.L1Loss().to(device)

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    train_dataset = PairedImagesDataset(
        args.dataset_path, args.pairs_file,
        transform=utils.PairedTransform(transforms.Compose(utils.pre_augmentation)))

    loader_params = {'batch_size': args.batch_size,
                     'shuffle': True,
                     'num_workers': args.n_workers,
                     'drop_last':True}
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_params)

    n_freq_print = args.n_freq_print
    n_freq_checkpoint = args.n_freq_checkpoint
    checkpoint_prefix = join(args.save_dir, utils.get_stamp_from_log())

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        batch_time = utils.AverageMeter('Time', ':6.3f')
        data_time = utils.AverageMeter('Data', ':6.3f')
        losses = utils.AverageMeter('Loss', ':.4f')
        progress = utils.ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        for i, sample in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            x1 = sample["x1"].to(device)
            y1 = sample["y1"].to(device)
            x2 = sample["x2"].to(device)
            y2 = sample["y2"].to(device)
            x1_ = sample["x1_"].to(device)
            y1_ = sample["y1_"].to(device)
            x2_ = sample["x2_"].to(device)
            y2_ = sample["y2_"].to(device)

            batch_size = x1.shape[0]
            # compute output and loss
            p1, z2, p1_, z2_ = model(x1, y1, x2, y2, x1_, y1_, x2_, y2_)
            #p1_, z2_ = model(x1_, y1_, x2_, y2_)
            #latent_loss = l1_loss(z2, z2_)
            loss = -(criterion(p1, z2).mean() + criterion(p1_, z2_).mean())*0.5
            #loss += latent_loss * args.weight_latest_loss

            losses.update(loss.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % n_freq_print == 0:
                progress.display(i)
                utils.log_to_tensorboard(writer, progress, step=epoch * len(train_loader) + i)
                #if epoch == 0:
                #    utils.log_img_to_tensorboard(writer, sample, step=epoch * len(train_loader) + i)


        if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
            torch.save(model.state_dict(), checkpoint_prefix + '_rpr_pretraining_checkpoint-{}.pth'.format(epoch))
