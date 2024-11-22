import torch
import torch.nn as nn
from torch.nn import AdaptiveAvgPool2d

from model.base import *
from model.pointnet2 import PointNetSetAbstraction
from model.pointnet import STN3d, PointNet
from model.pointconv import PointConv, CNC, CNCBlock
from model.gcn import WangBlock, GAT
from model.backend import *
from util import *

class MsgDecoder(nn.Module):
    def __init__(self, msg_length, ydim):
        super().__init__()
        last_channel = ydim
        mlp_channels = [500, 500]
        self.mlps = nn.ModuleList()
        for out_channel in mlp_channels:
            self.mlps.append(LinearRelu(last_channel, out_channel, bias=False))
            last_channel = out_channel
        self.get_x_hat = nn.Linear(last_channel, msg_length, bias=False)
    
    def forward(self, y):
        """
            x: (B, L)
        """
        for mlp in self.mlps:
            y = mlp(y)
        return self.get_x_hat(y)

class PointNet2Decoder(nn.Module):
    def __init__(self, ydim):
        super().__init__()
        last_channel = 3
        self.stn = STN3d(3)
        self.sa1 = PointNetSetAbstraction(npoint=512, nsample=32, in_channel=3, mlp=[128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, nsample=64, in_channel=128, mlp=[256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, nsample=None, in_channel=256, mlp=[1024], group_all=True)
        self.final = nn.Linear(1024, ydim)


    def forward(self, xyz):
        B, N, _ = xyz.shape
        trans = self.stn(xyz.permute(0, 2, 1))                  # trans [B, 3, 3]
        xyz = torch.bmm(xyz, trans)                     # [B, N, D]
        l1_xyz, l1_feat = self.sa1(xyz, xyz.permute(0, 2, 1).contiguous())
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)
        l3_feat = l3_feat.view(B, -1)
        msg = self.final(l3_feat)
        return msg

class PointNetDecoder(nn.Module):
    def __init__(self, ydim):
        super().__init__()
        mlp_channels = [64, 64, 64, 128, 1024]
        self.pointnet = PointNet(mlp_channels=mlp_channels, global_feat=True)
        
        self.final = nn.Linear(mlp_channels[-1], ydim)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        glob_feat, _, _ = self.pointnet(xyz) # [B, D]
        return self.final(glob_feat)

class PointConvDecoder(nn.Module):
    def __init__(self, ydim):
        super().__init__()
        self.mlps = nn.ModuleList([
            PointConv(in_channel=3, out_channel=64, npoints=256),
            PointConv(in_channel=64, out_channel=64, npoints=128),
            PointConv(in_channel=64, out_channel=ydim, npoints=64),
        ])

        self.pooling = nn.AdaptiveAvgPool1d(output_size=(1))
        self.final = nn.Linear(ydim, ydim)
        
    def forward(self, xyz, idx=None):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1) # [B, 3, N]
        xyzs = [xyz]
        feats = [xyz_trans.contiguous()]
        for id, mlp in enumerate(self.mlps):
            new_xyz, new_feat = mlp(xyzs[id], feats[id])
            xyzs.append(new_xyz)
            feats.append(new_feat)
        # xyz_feat: len=4
        feat = self.pooling(feats[-1]).squeeze(-1)
        return self.final(feat)

class CNCDecoder(nn.Module):
    def __init__(self, ydim, channel_size):
        super().__init__()
        input_channel = 3
        last_channel = input_channel
        self.mlps = nn.ModuleList()
        mlp_channels = [channel_size] * (V().cfg.decoder_block - 1)
        for out_channel in mlp_channels:
            self.mlps.append(CNC(in_channel=last_channel, out_channel=out_channel))
            last_channel = out_channel

        self.mlps.append(CNC(in_channel=last_channel, out_channel=ydim, with_relu=True))
        self.pooling = nn.AdaptiveAvgPool1d(output_size=(1))
        self.final = nn.Linear(ydim, ydim)
    
    def forward(self, xyz, idx):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, 3, N]
        feat = xyz_trans
        for conv in self.mlps:
            feat = conv(xyz, feat, idx)
        # [B, ydim, N]
        feat = self.pooling(feat).squeeze(-1)
        return self.final(feat)

class WangDecoder(nn.Module):
    def __init__(self, ydim):
        super().__init__()
        input_channel = 3
        last_channel = input_channel
        self.mlps = nn.ModuleList()
        mlp_channels = [64] * (V().cfg.decoder_block - 1)
        for out_channel in mlp_channels:
            self.mlps.append(WangBlock(in_channel=last_channel, out_channel=out_channel))
            last_channel = out_channel

        self.mlps.append(WangBlock(in_channel=last_channel, out_channel=ydim, with_relu=False))
        self.pooling = nn.AdaptiveAvgPool1d(output_size=(1))
        self.final = nn.Linear(ydim, ydim)
    
    def forward(self, xyz, idx):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, 3, N]
        feat = xyz_trans
        for conv in self.mlps:
            feat = conv(xyz, feat, idx)
        # [B, ydim, N]
        feat = self.pooling(feat).squeeze(-1)
        return self.final(feat)

class GATDecoder(nn.Module):
    def __init__(self, ydim, channel_size):
        super().__init__()
        input_channel = 3
        last_channel = input_channel
        self.mlps = nn.ModuleList()
        mlp_channels = [channel_size] * (V().cfg.decoder_block - 1)
        for out_channel in mlp_channels:
            self.mlps.append(GAT(in_channel=last_channel, out_channel=out_channel))
            last_channel = out_channel

        self.mlps.append(GAT(in_channel=last_channel, out_channel=ydim, with_relu=True))
        self.pooling = nn.AdaptiveAvgPool1d(output_size=(1))
        self.final = nn.Linear(ydim, ydim)
    
    def forward(self, xyz, idx):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, 3, N]
        feat = xyz_trans
        # print("before: ", feat)
        for conv in self.mlps:
            feat = conv(xyz, feat, idx)
        # [B, ydim, N]
        # print('feat: ', feat)
        feat = self.pooling(feat).squeeze(-1)
        return self.final(feat)