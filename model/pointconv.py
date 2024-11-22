import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.backend.backend_utils import *
from model.base import *
from util import V

class PointConv(nn.Module):
    def __init__(self, in_channel, out_channel=64, weight_channel=64,
                npoints=256, with_relu=True):
        super().__init__()
        self.npoints = npoints

        self.grouping_strategy = V().cfg.grouping_strategy()
        
        weight_mlp_channels = [weight_channel] * 3
        self.weight_net = WeightNet(mlp_channels=weight_mlp_channels)
        self.final = ConvBNRelu1D(in_channel * weight_mlp_channels[-1], out_channel, kernel_size=1) \
                        if with_relu else \
                    nn.Conv1d(in_channel * weight_mlp_channels[-1], out_channel, kernel_size=1)


    def forward(self, xyz, feature):
        '''
            xyz: (B, N, 3)
            feature: (B, D, N)
            convolution:
                    centered at new_xyz
        '''
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous()           # [B, 3, N]

        new_xyz_trans = gather_operation(
            xyz_trans, furthest_point_sample(xyz, self.npoints)
        )                                                       # [B, 3, npoints]
        new_xyz = new_xyz_trans.permute(0, 2, 1).contiguous()   # [B, npoints, 3]
        idx = self.grouping_strategy(new_xyz, xyz, None)  # [B, npoints, nsample]
        
        grouped_xyz_trans = grouping_operation(
            xyz_trans, idx
        ) - new_xyz_trans.unsqueeze(-1) # [B, 3, npoints, nsample] - [B, 3, npoints, 1] = [B, 3, npoints, nsample]

        grouped_feature = grouping_operation(
            feature, idx
        ) # [B, D, N, nsample]

        weights = self.weight_net(grouped_xyz_trans)  # [B, D', npoints, nsample]
        new_feature = torch.matmul(
            grouped_feature.permute(0, 2, 1, 3),      # [B, npoints, D, nsample]
            weights.permute(0, 2, 3, 1)           # [B, npoints, nsample, D']
        ).view(B, self.npoints, -1).permute(0, 2, 1) # [B, D*D', npoints]
        return new_xyz, self.final(new_feature)

class PointDeconv(nn.Module):
    def __init__(self, in_channel, out_channel, weight_channel=64, with_relu=True):
        super().__init__()
        self.grouping_strategy = V().cfg.grouping_strategy()
        weight_mlp_channels = [weight_channel] * 3
        self.weight_net = WeightNet(mlp_channels=weight_mlp_channels)
        self.final = ConvBNRelu1D(in_channel[1]*weight_mlp_channels[-1]+in_channel[0], out_channel, kernel_size=1) \
                        if with_relu else \
                    nn.Conv1d(in_channel[1]*weight_mlp_channels[-1]+in_channel[0], out_channel, kernel_size=1)
    def forward(self, xyz, new_xyz, feature, new_feature):
        '''
            xyz: (B, N, 3)
            new_xyz: (B, npoints, 3)
            feature: (B, D, N)
            new_feature: (B, D', npoints)
        '''
        B, N, _ = xyz.shape

        idx = self.grouping_strategy(xyz, new_xyz, None) # [B, N, nsample]
        grouped_new_xyz_trans = grouping_operation(
            new_xyz.permute(0, 2, 1).contiguous(), idx
        ) - xyz.permute(0, 2, 1).unsqueeze(-1) # [B, 3, N, nsample] - [B, 3, N, 1]
        
        grouped_new_feature = grouping_operation(
            new_feature, idx
        ) # [B, D', N, nsample]

        weights = self.weight_net(grouped_new_xyz_trans) # [B, weight_channel, N, nsample]
        new_feature = torch.matmul(
            grouped_new_feature.permute(0, 2, 1, 3), # [B, N, D', nsample]
            weights.permute(0, 2, 3, 1) # [B, N, nsample, weight_channel]
        ).view(B, N, -1).permute(0, 2, 1) # [B, weight_channel*D', N]
        concated_feature = torch.cat([new_feature, feature], dim=1) # [B, weight_channel*D'+D, N]
        return self.final(concated_feature)


class CNC(nn.Module):
    def __init__(self, in_channel, out_channel, with_relu=True):
        super().__init__()
        # self.nsample = V().cfg.grouping_strategy.nsample
        self.nsample = 1
        weight_mlp = [64] * 3
        self.weight_net = WeightNet(weight_mlp)
        self.final = ConvBNRelu1D(in_channel * weight_mlp[-1], out_channels=out_channel, kernel_size=1)\
                    if with_relu else \
                    nn.Conv1d(in_channel * weight_mlp[-1], out_channels=out_channel, kernel_size=1)
        self.scale_size = V().cfg.scale_size
    def forward(self, xyz, feature, idx):
        """
            xyz: (B, N, 6)
            feature: (B, D, N)
        """
        B, _, N = feature.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        
        grouped_xyz_trans = grouping_operation(
            xyz_trans, idx
        ) - xyz_trans.view(B, -1, N, 1)

        feature = grouping_operation(
            feature, idx
        ) / self.nsample # [B, D, N, nsample]

        weights = self.weight_net(grouped_xyz_trans/self.scale_size) # [B, D1, N, sample]
        feature = torch.matmul(
            feature.permute(0, 2, 1, 3), # [B, N, D, nsample]
            weights.permute(0, 2, 3, 1) # [B, N, nsample, D1]
        ).view(B, N, -1) # [B, N, D*D1]
        return self.final(feature.permute(0, 2, 1))

class CNCAgg(nn.Module):
    def __init__(self, in_channel, out_channel, with_relu=True):
        super().__init__()
        weight_mlp = [64] * 3
        self.weight_net = WeightNet1D(weight_mlp)
        self.final = ConvBNRelu1D(in_channel * weight_mlp[-1], out_channels=out_channel, kernel_size=1)\
                    if with_relu else \
                    nn.Conv1d(in_channel * weight_mlp[-1], out_channels=out_channel, kernel_size=1)
        self.scale_size = V().cfg.scale_size
    def forward(self, xyz, feature, idx):
        B, _, N = feature.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        weights = self.weight_net(xyz_trans/self.scale_size) # [B, D1, N]
        feature = feature / N
        feature = torch.matmul(
            feature, # [B, D, N]
            weights.permute(0, 2, 1) # [B, N, D1]
        ).view(B, -1, 1) # [B, D', 1]
        return self.final(feature)


class CNCBlock(nn.Module):
    def __init__(self, in_channel, out_channel, with_relu=True):
        super().__init__()
        self.with_relu = with_relu
        last_channel = in_channel
        mlp_channels = [64] * 1
        self.mlps = nn.ModuleList()
        for channel in mlp_channels:
            self.mlps.append(CNC(in_channel=last_channel, out_channel=channel))
            last_channel = channel
        
        self.mlps.append(CNC(in_channel=last_channel, out_channel=out_channel, with_relu=False))

        self.shortcut = lambda x: x
        if in_channel != out_channel:
            self.shortcut = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)

    def forward(self, xyz, feature, idx):
        feat = feature
        for mlp in self.mlps:
            feat = mlp(xyz, feat, idx)
        feat = feat + self.shortcut(feature)
        if self.with_relu:
            feat = F.leaky_relu(feat, inplace=True)
        return feat