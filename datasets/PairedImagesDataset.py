import pandas as pd
from torch.utils.data import Dataset
import torch
from skimage.io import imread
import os
from PIL import Image
import glob

class PairedImagesDataset(Dataset):
    """
        A class used to represent a dataset with match pair dataset
    """
    def __init__(self, dataset_path, pairs_file, transform):
        super(PairedImagesDataset, self).__init__()
        self.dataset_path = dataset_path
        df = pd.read_csv(pairs_file)
        self.img0_paths, self.img1_paths = df["img_path0"].values, df["img_path1"].values
        self.transform = transform

    def __len__(self):
        return len(self.img0_paths)

    def __getitem__(self, idx):
        x = Image.open(os.path.join(self.dataset_path, self.img0_paths[idx]))
        y = Image.open(os.path.join(self.dataset_path,self.img1_paths[idx]))
        x1, x2, y1, y2 = self.transform(x, y)

        return {"x1":x1, "y1":y1, "x2":x2, "y2":y2}


class TestImagesDataset(Dataset):
    def __init__(self, root, transform, test_mode, anchor, positive, negative):
        self.test_mode  = test_mode
        self.anchor_files_x = sorted(glob.glob(os.path.join(root, "{}".format(anchor)) + "/x/*.*"))
        self.anchor_files_y = sorted(glob.glob(os.path.join(root, "{}".format(anchor)) + "/y/*.*"))
        if self.test_mode==0 or self.test_mode==1:
            self.positive_files_x = sorted(glob.glob(os.path.join(root, "{}".format(positive)) + "/x/*.*"))
            self.positive_files_y = sorted(glob.glob(os.path.join(root, "{}".format(positive)) + "/y/*.*"))
        if self.test_mode == 0 or self.test_mode == 2:
            self.negative_files_x = sorted(glob.glob(os.path.join(root, "{}".format(negative)) + "/x/*.*"))
            self.negative_files_y = sorted(glob.glob(os.path.join(root, "{}".format(negative)) + "/y/*.*"))
        #assert (len(self.anchor_files)==len(self.positive_files)==len(self.negative_files))
        self.transform = transform

    def __getitem__(self, index):
        file_name = self.anchor_files_x[index % len(self.anchor_files_x)]
        anchor_x = self.transform(Image.open(self.anchor_files_x[index % len(self.anchor_files_x)]))
        anchor_y = self.transform(Image.open(self.anchor_files_y[index % len(self.anchor_files_y)]))
        positive_x  = positive_y = negative_x = negative_y = torch.zeros_like(anchor_x)
        if self.test_mode == 0 or self.test_mode == 1:
            positive_x = self.transform(Image.open(self.positive_files_x[index % len(self.anchor_files_x)]))
            positive_y = self.transform(Image.open(self.positive_files_y[index % len(self.anchor_files_y)]))
        if self.test_mode == 0 or self.test_mode == 2:
            negative_x = self.transform(Image.open(self.negative_files_x[index % len(self.anchor_files_x)]))
            negative_y = self.transform(Image.open(self.negative_files_y[index % len(self.anchor_files_y)]))

        return [anchor_x, anchor_y, positive_x, positive_y, negative_x, negative_y, os.path.basename(file_name)]

    def __len__(self):
        return len(self.anchor_files_x)