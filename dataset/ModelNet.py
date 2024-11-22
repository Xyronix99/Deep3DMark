
import os
import numpy as np
import warnings
import pickle
import mlconfig
import glob
from tqdm import tqdm
import pyvista
import open3d as o3d
import torch
import torch.utils.data.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from util import *
from model.backend.backend_utils import k_neighbor_query

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataset(Dataset):
    def __init__(self, root, split, npoints, **kwargs):
        self.root = root

        self.save_path = os.path.join(root, 'modelnet40_%s_%dpts_fps.dat' % (split, npoints))
        
        if not os.path.exists(self.save_path):
            obj_path_list = glob.glob(os.path.join(root, split, "*", f"*.off"))
            V().info('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = []
            self.list_of_faces = []
            self.list_of_centroid = []
            self.list_of_m = []

            for index in tqdm(range(len(obj_path_list)), total=len(obj_path_list)):
                obj_path = obj_path_list[index]

                mesh = o3d.io.read_triangle_mesh(obj_path)
                point_set = np.asarray(mesh.vertices).astype(np.float32)
                face_set = np.asarray(mesh.triangles).astype(np.int32)

                if npoints != point_set.shape[0]:
                    continue
                
                point_set[:, 0:3], centroid, m = pc_normalize(point_set[:, 0:3])

                self.list_of_points.append(point_set)
                self.list_of_centroid.append(centroid)
                self.list_of_m.append(m)
                self.list_of_faces.append(
                    face_set
                )

            with open(self.save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_faces, self.list_of_centroid, self.list_of_m], f)
        else:
            V().info('Load processed data from %s...' % self.save_path)
            with open(self.save_path, 'rb') as f:
                self.list_of_points, self.list_of_faces, self.list_of_centroid, self.list_of_m = pickle.load(f)
    
    @staticmethod
    def collect_fn(data):
        pc, faces, centroid, m = zip(*data)
        pc = torch.tensor(pc, dtype=torch.float32)
        centroid = torch.tensor(centroid, dtype=torch.float32)
        m = torch.tensor(m, dtype=torch.float32)
        new_faces = pad_sequence([torch.tensor(face, dtype=torch.int32) for face in faces], batch_first=True, padding_value=-1)
        return pc, new_faces, centroid, m



    def __len__(self):
        return len(self.list_of_points)

    def _get_item(self, index):
        point_set, faces, centroid, m = self.list_of_points[index], self.list_of_faces[index], self.list_of_centroid[index], self.list_of_m[index]

        return point_set, faces, centroid, m

    def __getitem__(self, index):
        return self._get_item(index)

@mlconfig.register
class ModelNetDataLoader(DataLoader):
    def __init__(self, batch_size, shuffle, **dataset_kwargs):
        dataset = ModelNetDataset(**dataset_kwargs)
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10, collate_fn=ModelNetDataset.collect_fn)