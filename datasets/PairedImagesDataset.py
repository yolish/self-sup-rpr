import pandas as pd
from torch.utils.data import Dataset
from skimage.io import imread

class PairedImagesDataset(Dataset):
    """
        A class used to represent a dataset with match pair dataset
    """
    def __init__(self, dataset_path, pairs_file, transform):
        super(PairedImagesDataset, self).__init__()
        self.dataset_path = dataset_path
        df = pd.read_csv(pairs_file)
        self.img0_paths, self.img1_paths = df["img_path0"].values, df["img_path0"].values
        self.transform = transform

    def __len__(self):
        return len(self.img0_paths)

    def __getitem__(self, idx):
        x = imread(self.img0_path[idx])
        y = imread(self.img1_path[idx])
        x1, x2, y1, y2 = self.transform(x, y)

        return {"x1":x1, "y1":y1, "x2":x2, "y2":y2}
