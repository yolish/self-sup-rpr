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
parser.add_argument('--lr', help='learning rate', type=float, default=0.005)
parser.add_argument('--batch_size', help='batch size', type=int, default=24)
parser.add_argument('--start_epoch', help='start epoch', type=int, default=0)
parser.add_argument('--epochs', help='epochs', type=int, default=512)
parser.add_argument('--gpu_id', help='gpu id', default='3')
parser.add_argument('--arch', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    utils.init_logger()
    utils.set_proxy()

    # Record execution details
    logging.info("Start {} experiment for RPR".format(args.arch))
    logging.info("Using dataset: {}".format(args.dataset_path))
    #logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    print("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    device = torch.device(device_id)

    # Create model
    print("Creating model with a '{}' encoder".format(args.arch))
    model = PairedSimSiam(
        models.__dict__[args.arch],
        config.get("dim"), config.get("pred_dim"))
    #TODO load the backbone weights
    model = model.to(device)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).to(device)

    if config.get("fix_pred_lr"):
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=config.get("momentum"),
                                weight_decay=config.get("weight_decay"))

    train_dataset = PairedImagesDataset(
        args.dataset_path, args.pairs_file,
        transform=utils.PairedTransform(transforms.Compose(utils.pre_augmentation)))

    loader_params = {'batch_size': args.batch_size,
                     'shuffle': True,
                     'num_workers': config.get('n_workers'),
                     'drop_last':True}
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_params)

    n_freq_print = config.get("n_freq_print")
    n_freq_checkpoint = config.get("n_freq_checkpoint")
    n_epochs = config.get("n_epochs")
    checkpoint_prefix = join(utils.create_output_dir('out'), utils.get_stamp_from_log())

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

            batch_size = x1.shape[0]
            # compute output and loss
            p1, z2 = model(x1, y1, x2, y2)
            loss = -(criterion(p1, z2).mean())

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


        if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
            torch.save(model.state_dict(), checkpoint_prefix + '_rpr_pretraining_checkpoint-{}.pth'.format(epoch))






