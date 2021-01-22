# This script defines the chexpert dataset on MSI hard drive

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np


def padAndResize(img):
    w, h = img.size
    if w > h:
        pad = (w - h) // 2
    elif w < h:
        pad = (h - w) // 2
    else:
        pass
    return img.resize((224, 224))


class CheXpert(Dataset):
    def __init__(self, df, train=True):
        self.path = list(df['Path'])
        self.label = np.asarray(df.iloc[:, 1:])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        p = self.path[index]
        img = Image.open(p).convert("RGB")
        img = padAndResize(img)
        label = torch.FloatTensor(list(map(float, self.label[index])))
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.path)


def preprocess(df, rootPath):
    # df.loc[:, 'Path'] = rootPath + df.loc[:, 'Path']
    df['Path'] = rootPath + df['Path']
    kept_col = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    df = df.iloc[:, kept_col].fillna(0)
    df = df.replace(-1, 1)
    return df


if __name__ == "__main__":
    import pandas as pd

    RootPath = "/home/jusun/shared/Stanford_Dataset/"
    df = pd.read_csv(RootPath + "CheXpert-v1.0-small/train.csv")
    df = preprocess(df, RootPath)
    dataset = CheXpert(df)

    for img, label in dataset:
        print(img.shape, label)