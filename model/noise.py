import numpy as np
import torch.nn as nn

from model.backend.backend_utils import *

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, xyz):
        return xyz

class Cropping(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio
    def forward(self, xyz):
        """
            xyz: (B, N, 3)
            cropped_xyz: (B, R, 3)
        """
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        cropped_xyz, cropped_faces = random_crop(xyz, self.ratio)
        return cropped_xyz

class Gauss(nn.Module):
    def __init__(self, var=0.001):
        super().__init__()
        self.mean = 0
        self.var = var
    def forward(self, xyz):
        gauss = torch.normal(mean=self.mean, std=self.var, size=xyz.shape).cuda()
        restricted_gauss = gauss
        return xyz + restricted_gauss

def rotation(xyz, theta, dim=0):
    """
        xyz: (B, N, 3)
        dim = 0->x, 1->y, 2->z
    """
    one = torch.ones(1)
    zero = torch.zeros(1)
    rotation_M = [
        torch.stack([ # x
            torch.stack([one, zero, zero]),
            torch.stack([zero, torch.cos(theta), -torch.sin(theta)]),
            torch.stack([zero, torch.sin(theta), torch.cos(theta)])
        ]).cuda().reshape(3,3),
        torch.stack([ # y
            torch.stack([torch.cos(theta), zero, torch.sin(theta)]),
            torch.stack([zero, one, zero]),
            torch.stack([-torch.sin(theta), zero, torch.cos(theta)]),
        ]).cuda().reshape(3,3),
        torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta), zero]),
            torch.stack([torch.sin(theta), torch.cos(theta), zero]),
            torch.stack([zero, zero, one])
        ]).cuda().reshape(3,3)
    ]
    return torch.matmul(rotation_M[dim], xyz.permute(0, 2, 1)).permute(0, 2, 1)

class Rotation(nn.Module):
    def __init__(self, theta=None):
        super().__init__()
        self.theta=theta
    def forward(self, xyz):
        if self.theta is not None:
            theta = torch.tensor([self.theta])
        else:
            theta = torch.rand(1)
        dim = torch.randint(low=0, high=3, size=(1,)).item()
        return rotation(xyz, theta, dim)

class Translation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, xyz):
        B, N, D = xyz.shape
        delta = torch.randn((B, 1, D)).cuda()
        return xyz + delta

class Scaling(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
    def forward(self, xyz):
        return xyz * self.s
    
class Quantization(nn.Module):
    def __init__(self, Nq=6):
        super().__init__()
        self.Nq = Nq
    def forward(self, xyz):
        max_v = torch.max(xyz)
        v0 = xyz / max_v
        v0 = v0 * (1 << self.Nq)
        v0 = torch.floor(v0)
        return v0 / (1 << self.Nq) * max_v
