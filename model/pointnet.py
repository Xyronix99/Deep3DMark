import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


from util import *
from model.base import ConvBNRelu1D

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0] # [B, D, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) # [B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0] # [B, 1024]
        x = x.view(-1, 1024) # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # [B, 9]

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3) # [B, 3, 3]
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.to(V().device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, mlp_channels, global_feat=False):
        super().__init__()
        last_channel = 3
        self.stn = STN3d(3)
        self.fstn = STNkd(k=64)
        
        self.mlp_before_trans = nn.ModuleList()
        for out_channel in mlp_channels[:2]:
            self.mlp_before_trans.append(ConvBNRelu1D(in_channels=last_channel, out_channels=out_channel, kernel_size=1))
            last_channel = out_channel
        
        self.mlp_after_trans = nn.ModuleList()
        for out_channel in mlp_channels[2:]:
            self.mlp_after_trans.append(ConvBNRelu1D(in_channels=last_channel, out_channels=out_channel, kernel_size=1))
            last_channel = out_channel
        
        self.global_feat = global_feat

    def forward(self, xyz):
        B, N, D = xyz.shape
        '''pc transform'''
        trans = self.stn(xyz.permute(0, 2, 1)) # [B, 3, 3]
        if D > 3:
            feature = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]     # [B, N, 3]
        xyz = torch.bmm(xyz, trans) # [B, N, 3]
        if D > 3:
            xyz = torch.cat([xyz, feature], dim=2) # [B, N, D]

        '''feat conv before feat trans'''
        xyz = xyz.transpose(2, 1) # [B, D, N]
        for conv in self.mlp_before_trans:
            xyz = conv(xyz)

        '''feat transform'''
        feat_trans = self.fstn(xyz) # [B, 64, 64]
        xyz = xyz.transpose(2, 1)   # [B, N, 64]
        xyz = torch.bmm(xyz, feat_trans) # [B, N, 64]
        xyz = xyz.transpose(2, 1) # [B, 64, N]

        '''feat conv after feat trans'''
        pointfeat = xyz # [B, 64, N]
        for conv in self.mlp_after_trans:
            xyz = conv(xyz)

        '''max pooling'''
        xyz = torch.max(xyz, 2, keepdim=True)[0] # [B, 1024, 1]
        xyz = xyz.view(-1, 1024) # [B, 1024]
        if self.global_feat:
            return xyz, trans, feat_trans
        else:
            xyz = xyz.view(-1, 1024, 1).repeat(1, 1, N) # [B, 1024, N]
            return torch.cat([xyz, pointfeat], 1), trans, feat_trans #[B, 1024, ]


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
