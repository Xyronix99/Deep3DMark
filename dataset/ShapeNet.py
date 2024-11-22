
import os
import numpy as np
import warnings
import pickle
import mlconfig
import glob
from tqdm import tqdm
import pyvista
import trimesh
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

class ShapeNetDataset(Dataset):
    def __init__(self, root, split, npoints, **kwargs):
        self.root = root

        self.save_path = os.path.join(root, 'shapenet40_%s_%dpts_fps.dat' % (split, npoints))
        
        if not os.path.exists(self.save_path):
            matcher = {"train":"[012]","test":"[4]"}
            obj_path_list = glob.glob(os.path.join(root, f"0{matcher[split]}*", "*", f"*.obj"))
            V().info('Processing data %s (only running in the first time)...' % self.save_path)
            self.list_of_points = []
            self.list_of_faces = []
            self.list_of_centroid = []
            self.list_of_m = []

            for index in tqdm(range(len(obj_path_list)), total=len(obj_path_list)):
                obj_path = obj_path_list[index]

                mesh = pyvista.PolyData(obj_path)
                point_set = np.asarray(mesh.points).astype(np.float32)
                face_set = np.asarray(mesh.faces).astype(np.int32)
                face_set = face_set.reshape(-1, 4)[:, 1:]

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
class ShapeNetDataLoader(DataLoader):
    def __init__(self, batch_size, shuffle, **dataset_kwargs):
        dataset = ShapeNetDataset(**dataset_kwargs)
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10, collate_fn=ShapeNetDataset.collect_fn)


class ShapeNetPartial(Dataset):
    def __init__(self):
        self.selected_path = [
            "/data/xingyu/shapenetcore_watertight/02691156/2_watertight/122963149f6a04272620819ddac40644.off",
            "/data/xingyu/shapenetcore_watertight/02691156/2_watertight/b837c3b8eec02a4967c54439d6177032.off",
            "/data/xingyu/shapenetcore_watertight/02691156/2_watertight/db567b8afbaaa95060a762246a709d46.off",
            "/data/xingyu/shapenetcore_watertight/02691156/2_watertight/2495267afeb60584c3a35cee92bb95b.off",
            "/data/xingyu/shapenetcore_watertight/02691156/2_watertight/a57802c776ba1b69e44b1dd0f956e84.off",
            "/data/xingyu/shapenetcore_watertight/02691156/2_watertight/ec2ceb5d65007bacfbb51ecfb25331aa.off",
        ]
        self.list_of_points = []
        self.list_of_faces = []
        self.list_of_centroid = []
        self.list_of_m = []
        for index in tqdm(range(len(self.selected_path))):
            off_file_path = self.selected_path[index]
            mesh = trimesh.load_mesh(off_file_path)

            point_set = np.asarray(mesh.vertices).astype(np.float32)
            face_set = np.asarray(mesh.faces).astype(np.int32)

            point_set[:, 0:3], centroid, m = pc_normalize(point_set[:, 0:3])

            self.list_of_points.append(point_set)
            self.list_of_centroid.append(centroid)
            self.list_of_m.append(m)
            self.list_of_faces.append(
                face_set
            )


    def __len__(self):
        return len(self.selected_path)
    
    def __getitem__(self, index):
        point_set, faces, centroid, m = self.list_of_points[index], self.list_of_faces[index], self.list_of_centroid[index], self.list_of_m[index]

        return point_set, faces, centroid, m

@mlconfig.register
class ShapeNetPartialDataLoader(DataLoader):
    def __init__(self, batch_size, shuffle, **dataset_kwargs):
        dataset = ShapeNetPartial()
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10)