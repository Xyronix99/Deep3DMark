import torch
import torch.nn as nn
import torch.distributions as D

from model.base import ConvBNRelu1D, LinearRelu
from model.pointconv import CNCBlock, PointConv, PointDeconv, CNC
from model.pointnet import STN3d, STNkd, PointNet
from model.pointnet2 import PointNetSetAbstraction, PointNetFeaturePropagation
from model.gcn import WangBlock, GAT
from model.backend import *
from util import *

def merge_msg(mat, msg):
    B, _, N = mat.shape
    batched_msg = msg.view(B, -1, 1).repeat(1, 1, N) # [B, L, N]
    return torch.cat([mat, batched_msg], dim=1)

class MsgEncoder(nn.Module):
    def __init__(self, msg_length, ydim, noise, vimco_samples):
        super().__init__()
        self.noise = noise
        self.vimco_samples = vimco_samples
        last_channel = msg_length
        mlp_channels = [500]
        self.mlps = nn.ModuleList()
        for out_channel in mlp_channels:
            self.mlps.append(LinearRelu(last_channel, out_channel))
            last_channel = out_channel
        self.get_y_hat_prob = nn.Linear(last_channel, ydim, bias=False)
    
    def forward(self, x):
        """
            x: (B, L)
        """
        for mlp in self.mlps:
            x = mlp(x)
        z = self.get_y_hat_prob(x)
        q = D.Bernoulli(logits=z)
        y_hat = q.sample()

        y_hat_prob = torch.sigmoid(z)
        combined_prob = y_hat_prob - (2 * y_hat_prob * self.noise) + self.noise    # (B, ydim)

        q = D.Bernoulli(probs=combined_prob)
        y = q.sample((self.vimco_samples,)) # (nsample, B, ydim)
        return combined_prob, y_hat, y, q


class PointNet2Encoder(nn.Module):
    def __init__(self, ydim):
        super().__init__()
        self.stn = STN3d(3)

        self.sa1 = PointNetSetAbstraction(npoint=512, nsample=32, in_channel=3, mlp=[128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, nsample=64, in_channel=128 + ydim, mlp=[256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, nsample=None, in_channel=256 + ydim, mlp=[1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280 + 2 * ydim, mlp=[256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384 + ydim, mlp=[128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+3, mlp=[128])

        self.conv1 = ConvBNRelu1D(in_channels=128, out_channels=128, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1)

    
    def forward(self, xyz, msg):
        B, N, D = xyz.shape
        '''input transform'''
        trans = self.stn(xyz.permute(0, 2, 1))                  # trans [B, 3, 3]
        xyz = torch.bmm(xyz, trans)                     # [B, N, D]
        '''TODO: add more feature'''
        l0_feat = xyz.permute(0, 2, 1).contiguous()
        l0_xyz = xyz

        l1_xyz, l1_feat = self.sa1(l0_xyz, l0_feat)
        l1_feat = merge_msg(l1_feat, msg)

        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l2_feat = merge_msg(l2_feat, msg)

        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)
        l3_feat = merge_msg(l3_feat, msg)

        l2_feat = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)
        l1_feat = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_feat)
        l0_feat = self.fp1(l0_xyz, l1_xyz, l0_feat, l1_feat)    # [B, D, N]
        
        out = self.conv1(l0_feat)
        out = self.conv2(out)
        return out.permute(0, 2, 1).contiguous()

class PointNetEncoder(nn.Module):
    def __init__(self, ydim):
        super().__init__()
        mlp_channels = [64, 64, 64, 128, 1024]
        self.pointnet = PointNet(mlp_channels=mlp_channels, global_feat=False)
        
        mlp_channel_after_concate = [512, 512, 64]
        last_channel = mlp_channels[-1]+mlp_channels[1]+ydim
        self.after_concate = nn.Sequential()
        for channel in mlp_channel_after_concate:
            self.after_concate.append(ConvBNRelu1D(in_channels=last_channel, out_channels=channel, kernel_size=1))
            last_channel = channel
        self.final = nn.Conv1d(in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, xyz, msg):
        B, N, _ = xyz.shape
        expanded_msg = msg.view(B, 1, -1).repeat(1, N, 1) # [B, N, L]
        xyz = torch.cat([xyz, expanded_msg], dim=-1) # [B, N, L+3]
        feat, _, _ = self.pointnet(xyz) # [B, D, N]
        feat = self.after_concate(feat) # [B, D, N]
        return self.final(feat).permute(0, 2, 1).contiguous() # [B, N, 3]


class PointConvEncoder(nn.Module):
    def __init__(self, ydim):
        super().__init__()
        self.mlps = nn.ModuleList([
            PointConv(in_channel=3+ydim, out_channel=64, npoints=256),
            PointConv(in_channel=64, out_channel=64, npoints=128),
            PointConv(in_channel=64, out_channel=64, npoints=64),
        ])
        
        self.demlps = nn.ModuleList([
            PointDeconv(in_channel=[64, 64], out_channel=64),
            PointDeconv(in_channel=[64, 64], out_channel=64),
            PointDeconv(in_channel=[3+ydim, 64], out_channel=3)
        ])
        

    def forward(self, xyz, msg, idx=None):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1) # [B, 3, N]
        expanded_msg = msg.view(B, -1, 1).repeat(1, 1, N) # [B, L, N]
        xyzs = [xyz]
        feats = [torch.cat([xyz_trans, expanded_msg], dim=1).contiguous()]
        for id, mlp in enumerate(self.mlps):
            new_xyz, new_feat = mlp(xyzs[id], feats[id])
            xyzs.append(new_xyz)
            feats.append(new_feat)
        # xyz_feat: len=4
        dep = len(xyzs) - 1
        for id, mlp in enumerate(self.demlps):
            new_xyz = xyzs[dep - id]
            new_feat = feats[dep - id]
            xyz = xyzs[dep - id - 1]
            feat = feats[dep - id - 1]
            f = mlp(xyz, new_xyz, feat, new_feat)
            feats[dep - id -1] = f
        return feats[0].permute(0, 2, 1).contiguous()

class CNCEncoder(nn.Module):
    def __init__(self, ydim, channel_size):
        super().__init__()
        last_channel = 3
        self.mlps = nn.ModuleList()
        mlp_channels = [channel_size] * (V().cfg.encoder_block - 2)
        for out_channel in mlp_channels:
            self.mlps.append(CNC(in_channel=last_channel, out_channel=out_channel))
            last_channel = out_channel

        self.after_concat = CNC(in_channel=3+last_channel+ydim, out_channel=64)
        self.final = CNC(in_channel=64, out_channel=3, with_relu=False)
    def forward(self, xyz, msg, idx):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, 3, N]
        feat = xyz_trans
        for conv in self.mlps:
            feat = conv(xyz, feat, idx)
        expanded_msg = msg.view(B, -1, 1).repeat(1, 1, N) # [B, L, N]
        feat = torch.cat([feat, xyz_trans, expanded_msg], dim=1)
        feat = self.after_concat(xyz, feat, idx)
        feat = self.final(xyz, feat, idx)
        return feat.permute(0, 2, 1).contiguous()

class WangEncoder(nn.Module):
    def __init__(self, ydim):
        super().__init__()
        last_channel = 3
        self.mlps = nn.ModuleList()
        mlp_channels = [64] * (V().cfg.encoder_block - 2)
        for out_channel in mlp_channels:
            self.mlps.append(WangBlock(in_channel=last_channel, out_channel=out_channel))
            last_channel = out_channel

        self.after_concat = WangBlock(in_channel=3+last_channel+ydim, out_channel=64)
        self.final = WangBlock(in_channel=64, out_channel=3, with_relu=False)
    def forward(self, xyz, msg, idx):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, 3, N]
        feat = xyz_trans
        for conv in self.mlps:
            feat = conv(xyz, feat, idx)
        expanded_msg = msg.view(B, -1, 1).repeat(1, 1, N) # [B, L, N]
        feat = torch.cat([feat, xyz_trans, expanded_msg], dim=1)
        feat = self.after_concat(xyz, feat, idx)
        feat = self.final(xyz, feat, idx)
        return feat.permute(0, 2, 1).contiguous()

class GATEncoder(nn.Module):
    def __init__(self, ydim, channel_size):
        super().__init__()
        last_channel = 3
        self.mlps = nn.ModuleList()
        mlp_channels = [channel_size] * (V().cfg.encoder_block - 2)
        for out_channel in mlp_channels:
            self.mlps.append(GAT(in_channel=last_channel, out_channel=out_channel))
            last_channel = out_channel

        self.after_concat = GAT(in_channel=3+last_channel+ydim, out_channel=channel_size)
        self.final = GAT(in_channel=channel_size, out_channel=3, with_relu=False)
    def forward(self, xyz, msg, idx):
        B, N, _ = xyz.shape
        xyz_trans = xyz.permute(0, 2, 1).contiguous() # [B, 3, N]
        feat = xyz_trans
        for conv in self.mlps:
            feat = conv(xyz, feat, idx)
        expanded_msg = msg.view(B, -1, 1).repeat(1, 1, N) # [B, L, N]
        feat = torch.cat([feat, xyz_trans, expanded_msg], dim=1)
        feat = self.after_concat(xyz, feat, idx)
        f1 = feat
        feat = self.final(xyz, feat, idx)
        return feat.permute(0, 2, 1).contiguous()