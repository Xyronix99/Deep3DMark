
import os
import numpy as np
import warnings
import pickle
import mlconfig
import glob
from tqdm import tqdm
import pyvista
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

class HandsDataset(Dataset):
    def __init__(self, root, split, npoints, **kwargs):
        self.root = root

        obj_path_list = glob.glob(os.path.join(root, split, f"*.obj"))
        obj_path_list.extend(glob.glob(os.path.join(root, split, f"*.off")))

        self.save_path = os.path.join(root, 'hands_%s_%dpts_fps.dat' % (split, npoints))
        
        if not os.path.exists(self.save_path):
            V().info('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = []
            self.list_of_faces = []
            self.list_of_centroid = []
            self.list_of_m = []

            for index in tqdm(range(len(obj_path_list)), total=len(obj_path_list)):
                obj_path = obj_path_list[index]
                # import open3d as o3d
                # mesh = o3d.io.read_triangle_mesh(obj_path)
                # point_set = np.asarray(mesh.vertices).astype(np.float32)
                # face_set = np.asarray(mesh.triangles).astype(np.int32)
                
                mesh = pyvista.PolyData(obj_path)
                point_set = np.asarray(mesh.points).astype(np.float32)
                face_set = np.asarray(mesh.faces).astype(np.int32)
                face_set = face_set.reshape(-1, 4)[:, 1:]

                if point_set.shape[0]!=npoints:
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
class HandsDataLoader(DataLoader):
    def __init__(self, batch_size, shuffle, **dataset_kwargs):
        dataset = HandsDataset(**dataset_kwargs)
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, collate_fn=HandsDataset.collect_fn)
    
if __name__ == "__main__":
    root = '/data_HDD/zhuxingyu/.dataset/hands'
    obj_path_list = glob.glob(os.path.join(root, 'train', "*.ply"))
    for index in tqdm(range(len(obj_path_list)), total=len(obj_path_list)):
        obj_path = obj_path_list[index]

        mesh = pyvista.PolyData(obj_path)
        print(mesh.points.shape)
        pass