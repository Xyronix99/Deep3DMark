import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from model.base import *
from model.backend import *


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(ConvBNRelu2D(last_channel, out_channel, kernel_size = 1))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, feature):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, N, C = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, C, N]
        if self.group_all:
            new_xyz = torch.zeros(B, 1, C).cuda()
            new_feature = feature.view(B, -1, 1, N)
        else:
            new_xyz = gather_operation(
                xyz_trans,
                furthest_point_sample(xyz, self.npoint)
            ).permute(0, 2, 1).contiguous() # [B, npoint, C]
            new_feature = grouping_operation(
                feature, 
                k_neighbor_query(new_xyz, xyz, self.nsample)
            ) # [B, D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            new_feature = conv(new_feature)

        new_feature = torch.max(new_feature, 3)[0] # [i, j, k] = max_l{[i, j, k, l]} => [B, hidden, npoint]
        return new_xyz, new_feature


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, without_relu=False):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp[:-1]:
            self.mlp_convs.append(ConvBNRelu1D(last_channel, out_channel, kernel_size=1))
            last_channel = out_channel
        
        self.mlp_convs.append(
            ConvBN1D(last_channel, mlp[-1], kernel_size=1) if without_relu else ConvBNRelu1D(last_channel, mlp[-1], kernel_size=1)
        )

    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, npoint, C]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, npoint]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        B, N, C = xyz1.shape
        _, npoint, _ = xyz2.shape

        if npoint == 1:
            interpolated_feat = feat2.repeat(1, 1, N) # [B, D, N]
        else:
            dist, idx = three_nn(xyz1, xyz2)
            reverse_density = 1.0 / (dist + 1e-8)
            norm = torch.sum(reverse_density, dim=2, keepdim=True)
            reverse_density = reverse_density / norm
            interpolated_feat = three_interpolate(feat2, idx, reverse_density) # [B, D, N]

        # interploated_points [B, N, D]
        concated_feat = torch.cat([interpolated_feat, feat1], dim=1) # [B, D+D, N]
        for i, conv in enumerate(self.mlp_convs):
            concated_feat = conv(concated_feat)
        return concated_feat
