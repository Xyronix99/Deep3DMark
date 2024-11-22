import torch
import torch.nn as nn
import mlconfig

from model.base import *
from model.pointnet2 import PointNetSetAbstraction
from model.backend import *
from model.pointnet import STN3d, PointNet
from model.pointconv import CNCBlock, PointConv, PointDeconv, CNC
from model.gcn import WangBlock, GAT
from util import V

@mlconfig.register
class PointNet2Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = STN3d(3)
        self.sa1 = PointNetSetAbstraction(npoint=512, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, nsample=64, in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, nsample=None, in_channel=256, mlp=[256, 512, 1024], group_all=True)
        self.final = nn.Linear(1024, 1)


    def forward(self, xyz, faces, idx):
        B, N, _ = xyz.shape
        trans = self.stn(xyz.permute(0, 2, 1))                  # trans [B, 3, 3]
        xyz = torch.bmm(xyz, trans)                     # [B, N, D]
        l1_xyz, l1_feat = self.sa1(xyz, xyz.permute(0, 2, 1).contiguous())
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)
        l3_feat = l3_feat.view(B, -1)
        return self.final(l3_feat)


@mlconfig.register
class PointConvDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlps = nn.ModuleList([
            PointConv(in_channel=3, out_channel=64, npoints=256),
            PointConv(in_channel=64, out_channel=64, npoints=128),
            PointConv(in_channel=64, out_channel=64, npoints=64),
        ])

        self.pooling = nn.AdaptiveAvgPool1d(output_size=(1))

        self.final = nn.Linear(64, 1)
        
    def forward(self, xyz, faces, idx=None):
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

@mlconfig.register
class PointNetDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        mlp_channels = [64, 64, 64, 128, 1024]
        self.pointnet = PointNet(mlp_channels=mlp_channels, global_feat=True)
        
        self.final = nn.Linear(mlp_channels[-1], 1)

    def forward(self, xyz, faces, idx):
        B, N, _ = xyz.shape
        glob_feat, _, _ = self.pointnet(xyz) # [B, D]
        return self.final(glob_feat)

@mlconfig.register
class CNCDiscriminator(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        last_channel = 3
        self.mlps = nn.ModuleList()
        mlp_channels = [channel_size] * (V().cfg.discriminator_block - 1)
        for out_channel in mlp_channels:
            self.mlps.append(CNC(in_channel=last_channel, out_channel=out_channel))
            last_channel = out_channel
        self.mlps.append(CNC(in_channel=last_channel, out_channel=out_channel, with_relu=True))
        
        self.pooling = nn.AdaptiveAvgPool1d(output_size=(1))
        self.final = nn.Linear(mlp_channels[-1], 1)
    
    def forward(self, xyz, faces, idx):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, 6, N]
        feat = xyz_trans
        for conv in self.mlps:
            feat = conv(xyz, feat, idx)
        # [B, msg_length, N]
        feat = self.pooling(feat).squeeze(-1)
        return self.final(feat)

@mlconfig.register
class WangDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        last_channel = 3
        self.mlps = nn.ModuleList()
        mlp_channels = [64] * (V().cfg.discriminator_block)
        for out_channel in mlp_channels:
            self.mlps.append(WangBlock(in_channel=last_channel, out_channel=out_channel))
            last_channel = out_channel
        
        self.pooling = nn.AdaptiveAvgPool1d(output_size=(1))
        self.final = nn.Linear(mlp_channels[-1], 1)
    
    def forward(self, xyz, faces, idx):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, 6, N]
        feat = xyz_trans
        for conv in self.mlps:
            feat = conv(xyz, feat, idx)
        # [B, msg_length, N]
        feat = self.pooling(feat).squeeze(-1)
        return self.final(feat)

@mlconfig.register
class GATDiscriminator(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        last_channel = 3
        self.mlps = nn.ModuleList()
        mlp_channels = [channel_size] * (V().cfg.discriminator_block - 1)
        for out_channel in mlp_channels:
            self.mlps.append(GAT(in_channel=last_channel, out_channel=out_channel))
            last_channel = out_channel
        self.mlps.append(GAT(in_channel=last_channel, out_channel=out_channel, with_relu=True))
        
        self.pooling = nn.AdaptiveAvgPool1d(output_size=(1))
        self.final = nn.Linear(mlp_channels[-1], 1)
    
    def forward(self, xyz, faces, idx):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, 6, N]
        feat = xyz_trans
        for conv in self.mlps:
            feat = conv(xyz, feat, idx)
        # [B, msg_length, N]
        feat = self.pooling(feat).squeeze(-1)
        return self.final(feat)