import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
import time
from os import mkdir, getcwd
import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageFilter
import ssl
import os
import datetime
from tensorboardX import SummaryWriter
import sys
from torchvision.utils import make_grid

def set_proxy():
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["http_proxy"] = "http://127.0.0.1:3128"
    os.environ["https_proxy"] = "http://127.0.0.1:3128"
    os.environ["ftp_proxy"] = "http://127.0.0.1:3128"
    os.environ["socks_proxy"] = "http://127.0.0.1:3128"

# Logging and output utils
##########################
def get_stamp_from_log():
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log","")


def create_output_dir(name):
    """
    Create a new directory for outputs, if it does not already exist
    :param name: (str) the name of the directory
    :return: the path to the outpur directory
    """
    out_dir = join(getcwd(), name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_logger(save_dir):
    """
    Initialize the logger and create a time stamp for the file
    """
    path = split(realpath(__file__))[0]

    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M", time.localtime()), ".log"])

        # Creating logs' folder is needed
        log_path = create_output_dir(save_dir)

        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, filename)
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)



# Evaluation utils
##########################
def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
    orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / np.pi
    return posit_err, orient_err


# pre-training augmentations
########################################################
# Code from SimSiam repo
class PairedTransform:
    """Apply two augmentations for a pair of images"""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x, y):
        x1 = self.base_transform(x)
        x2 = self.base_transform(x)
        y1 = self.base_transform(y)
        y2 = self.base_transform(y)
        return x1, x2, y1, y2


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709 WITHOUT horizontal flip and gray scale
pre_augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.ToTensor(),
    normalize
]
#########################################

# RPR Augmentations
train_transforms = {
    'baseline': transforms.Compose([transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                    transforms.ToTensor(),
                                    normalize])

}
test_transforms = {
    'baseline': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize
        ])
}
#########################################
# utils for reporting progress from SimSiam repo:
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def init_log(args, saved_path='default', save_dir=None):
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    if save_dir is None:
        save_dir = os.path.join('out', "{}_{}".format(saved_path, current_time))
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))
    # Saving run line
    with open(os.path.join(save_dir, 'run_line.txt'), 'w') as f:
        f.write("python {}".format(' '.join(sys.argv[:])))
        print("python {}".format(' '.join(sys.argv[:])))
    # Saving arguments to json
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print("Saving arguments to: {}".format(os.path.join(save_dir, 'args.txt')))
    return writer, save_dir

def log_to_tensorboard(writer, progress, step):
    with torch.no_grad():
        writer.add_scalar(progress.meters[-1].name, progress.meters[-1].val, step)

def log_img_to_tensorboard( writer, images, step):
    with torch.no_grad():
        batch_size = images["x1"].shape[0]
        images_0_grid = make_grid(images["x1"][:min(5, batch_size)], nrow=min(5, batch_size), normalize=False)
        images_1_grid = make_grid(images["x2"][:min(5, batch_size)], nrow=min(5, batch_size), normalize=False)
        images_2_grid = make_grid(images["y1"][:min(5, batch_size)], nrow=min(5, batch_size), normalize=False)
        images_3_grid = make_grid(images["y2"][:min(5, batch_size)], nrow=min(5, batch_size), normalize=False)
        image_grid = torch.cat((images_0_grid, images_1_grid,images_2_grid, images_3_grid), 1)
        writer.add_image('train/x1_x2_y1_y2', image_grid, step)

def log_img_to_tensorboard_triplet( writer, images, step, miss=False):
    with torch.no_grad():
        batch_size = images[0].shape[0]
        images_0_grid = make_grid(images[0][:min(5, batch_size)], nrow=min(5, batch_size), normalize=False)
        images_1_grid = make_grid(images[1][:min(5, batch_size)], nrow=min(5, batch_size), normalize=False)
        images_2_grid = make_grid(images[2][:min(5, batch_size)], nrow=min(5, batch_size), normalize=False)
        image_grid = torch.cat((images_0_grid, images_1_grid, images_2_grid), 2)
        if miss:
            writer.add_image('test{}/anchor_positive_negative'.format("_MISS"), image_grid, step)
        else:
            writer.add_image('test/anchor_positive_negative', image_grid, step)
