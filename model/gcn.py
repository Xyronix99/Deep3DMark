import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backend import *
from model.base import *
from util import V

class WangGCN(nn.Module):
    def __init__(self, in_channel, out_channel, with_relu=True):
        super().__init__()
        self.with_relu = with_relu
        self.w0 = nn.parameter.Parameter(torch.rand((in_channel, out_channel)))
        self.w1 = nn.parameter.Parameter(torch.rand((in_channel, out_channel)))
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, xyz, feature, idx):
        """
            xyz: (B, N, 3)
            feature: (B, D, N)
            idx: (B, N, N)
        """
        B, N, _ = xyz.shape
        grouped_feat = torch.matmul(feature, idx.float()).permute(0, 2, 1) # (B, N, D)
        num = idx.sum(dim=-1, keepdim=True)
        grouped_feat = grouped_feat / num.float().clamp_min(1e-6) # (B, N, D)
        new_feat = torch.matmul(
            grouped_feat, # (B, N, D_in)
            self.w1 # (D_in, D_out) ==> (B, N, D_out)
        ) # (B, N, D_out)
        
        new_feat = (
            new_feat + torch.matmul(
                feature.permute(0, 2, 1), # (B, N, D_in)
                self.w0 # (D_in, D_out)
            )
        ).permute(0, 2, 1) # (B, D_out, N)
        new_feat = self.bn(new_feat)
        if self.with_relu:
            new_feat = self.relu(new_feat)
        return new_feat
        # B, N, _ = xyz.shape
        # idx, num = idx[:,:,:-1], idx[:,:,-1] # (B, N, S), (B, N)
        # num = num.view(B, 1, N, 1) # (B, 1, N, 1)

        # num -= 1
        # idx = idx[:, :, 1:].contiguous()

        # grouped_feat = grouping_operation(
        #     feature, idx
        # ) / num.float().clamp_min(1e-6) # [B, D, N, S]

        # grouped_feat = grouped_feat.sum(dim=-1).permute(0, 2, 1)

        # new_feat = torch.matmul(
        #     grouped_feat, # (B, N, D_in)
        #     self.w1 # (D_in, D_out) ==> (B, N, D_out)
        # ) # (B, N, D_out)
        
        # new_feat = (
        #     new_feat + torch.matmul(
        #         feature.permute(0, 2, 1), # (B, N, D_in)
        #         self.w0 # (D_in, D_out)
        #     )
        # ).permute(0, 2, 1) # (B, D_out, N)
        # new_feat = self.bn(new_feat)
        # if self.with_relu:
        #     new_feat = self.relu(new_feat)
        # return new_feat

class WangBlock(nn.Module):
    def __init__(self, in_channel, out_channel, with_relu=True):
        super().__init__()
        self.with_relu = with_relu
        last_channel = in_channel
        mlp_channels = [64] * 1
        self.mlps = nn.ModuleList()
        for channel in mlp_channels:
            self.mlps.append(WangGCN(in_channel=last_channel, out_channel=channel))
            last_channel = channel
        
        self.mlps.append(WangGCN(in_channel=last_channel, out_channel=out_channel, with_relu=False))

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

class GAT(nn.Module):
    def __init__(self, in_channel, out_channel, with_relu=True):
        super().__init__()
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
            idx: (B, N, S+1)
        """
        B, _, N = feature.shape
        idx, num = idx[:,:,:-1].contiguous(), idx[:,:,-1] # (B, N, S), (B, N)
        num = num.view(B, 1, N, 1) # (B)
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        
        grouped_xyz_trans = grouping_operation(
            xyz_trans, idx
        ) - xyz_trans.view(B, -1, N, 1)

        feature = grouping_operation(
            feature, idx
        ) / num # [B, D, N, nsample]

        weights = self.weight_net(grouped_xyz_trans/self.scale_size) # [B, D1, N, sample]
        feature = torch.matmul(
            feature.permute(0, 2, 1, 3), # [B, N, D, nsample]
            weights.permute(0, 2, 3, 1) # [B, N, nsample, D1]
        ).view(B, N, -1) # [B, N, D*D1]
        return self.final(feature.permute(0, 2, 1))
