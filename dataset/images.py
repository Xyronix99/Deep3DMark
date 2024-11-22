import os
import numpy as np
import mlconfig
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from tqdm import tqdm
from six.moves import urllib
from torch.utils.data import Dataset, DataLoader

from util import V
from model import *

def pc_normalize(pc):
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
    pc = pc / m
    return pc, centroid, m

def build_img_from_graph(xyz, idx, centroid, m):
    xyz = xyz * m + centroid
    H = int(np.sqrt(xyz.shape[0]))
    img = torch.zeros((3, H, H))
    for i, v in enumerate(xyz):
        img[:, i//H, i%H] = v
    return transforms.ToPILImage()(img)

class CIFAR10(Dataset):
    def __init__(self, root, split, **kwargs):
        super().__init__()
        self.save_path = os.path.join(root, f"cifar10_{split}.dat")
        if not os.path.exists(self.save_path):
            transform = transforms.Compose(
                [transforms.ToTensor()])
            
            dataset = torchvision.datasets.CIFAR10(root=root, train=(split=='train'),
                                            download=True, transform=transform)
            self.list_of_points, self.list_of_idx, self.list_of_centroid, self.list_of_m = [], [], [], []
            for img, label in dataset:
                pc, idx, c, m = _build_graph_from_img(img)
                self.list_of_points.append(pc)
                self.list_of_idx.append(idx)
                self.list_of_centroid.append(c)
                self.list_of_m.append(m)

            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_faces, self.list_of_centroid, self.list_of_m], f)
        else:
            V().info('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_idx, self.list_of_centroid, self.list_of_m = pickle.load(f)
            
    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, index):
        return self.list_of_points[index], self.list_of_idx[index], self.list_of_centroid[index], self.list_of_m[index]

@mlconfig.register
class CIFAR10DataLoader(DataLoader):
    def __init__(self, batch_size, shuffle, **dataset_kwargs):
        dataset = CIFAR10(**dataset_kwargs)
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10)


class COCODataset(Dataset):
    def __init__(self, root, split):
        super().__init__()

        self.save_path = os.path.join(root, f"cifar10_{split}.dat")

        if not os.path.exists(self.save_path):
            data_path_list = glob(os.path.join(root, f'{split}2017', "*.jpg"))
            transform = {
                'train': transforms.Compose([
                    transforms.RandomCrop((64, 64), pad_if_needed=True),
                    transforms.ToTensor(),
                ]),
                'val': transforms.Compose([
                    transforms.CenterCrop((64, 64)),
                    transforms.ToTensor(),
                ])
            }[split]

            self.list_of_points, self.list_of_idx, self.list_of_centroid, self.list_of_m = [], [], [], []
            
            i = 0
            batch_size = 128
            while i < len(data_path_list):
                list_of_imgs = []
                for cnt in range(batch_size):
                    path = data_path_list[i]
                    with open(path, "rb") as f:
                        img = Image.open(f).convert('RGB')
                    img = transform(img)
                    list_of_imgs.append(img)
                    i += 1
                    if i >= len(data_path_list): break
                imgs = torch.stack(list_of_imgs).cuda()
                xyzs, idxs = build_graph_from_img(imgs)
                for xyz, idx in zip(xyzs, idxs):
                    xyz, c, m = pc_normalize(xyz)
                    self.list_of_points.append(xyz.detach().cpu().numpy())
                    self.list_of_idx.append(idx.detach().cpu().numpy())
                    self.list_of_centroid.append(c.detach().cpu().numpy())
                    self.list_of_m.append(m.detach().cpu().numpy())

            with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_idx, self.list_of_centroid, self.list_of_m], f)
        else:
            V().info('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_idx, self.list_of_centroid, self.list_of_m = pickle.load(f)
    
    def __len__(self):
        return len(self.list_of_points)
    
    def __getitem__(self, index):
        return self.list_of_points[index], self.list_of_idx[index], self.list_of_centroid[index], self.list_of_m[index]

@mlconfig.register
def COCO(root, split, batch_size, **kwargs):
    
    dataset = COCODataset(root, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=10)
    return loader