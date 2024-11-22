import mlconfig
import torch.nn as nn

from model.encoder import *
from model.decoder import *
from model.noise import *

def pc_normalize(pc):
    B, N, _ = pc.shape
    centroid = torch.mean(pc, dim=1).view(B, 1, 3)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1)), dim=-1).values.view(B, 1, 1)
    pc = pc / m
    return pc, centroid, m

@mlconfig.register
class MsgED(nn.Module):
    def __init__(self):
        super().__init__()
        noise = V().cfg.noise
        msg_length = V().cfg.msg_length
        ydim = V().cfg.ydim
        vimco_samples = V().cfg.vimco_samples
        assert noise < 0.5
        self.encoder = MsgEncoder(msg_length=msg_length, ydim=ydim, noise=noise, vimco_samples=vimco_samples)
        self.decoder = MsgDecoder(msg_length=msg_length, ydim=ydim)
    def forward(self, x):
        raise Exception


@mlconfig.register
class PointNet2ED(nn.Module):
    def __init__(self):
        ydim = V().cfg.ydim
        super().__init__()
        self.encoder = PointNet2Encoder(ydim=ydim)
        self.decoder = PointNet2Decoder(ydim=ydim)
    
    def forward(self, xyz, faces, msg, idx):
        encoded_xyz = self.encoder(xyz, msg)
        decoded_msg = self.decoder(encoded_xyz)
        return encoded_xyz, decoded_msg

@mlconfig.register
class PointNetED(nn.Module):
    def __init__(self):
        ydim = V().cfg.ydim
        super().__init__()
        self.noise_layer = [
            Identity(),
            # Gauss(), Scaling(s=10)
            # Cropping(ratio=0.7), Gauss(), Scaling(s=10)
        ]
        self.encoder = PointNetEncoder(ydim=ydim)
        self.decoder = PointNetDecoder(ydim=ydim)

    def forward(self, xyz, faces, msg, idx):
        noise_layer = np.random.choice(self.noise_layer, 1)[0]
        encoded_xyz = self.encoder(xyz, msg)
        noised_xyz = noise_layer(encoded_xyz)
        decoded_msg = self.decoder(noised_xyz)
        return encoded_xyz, decoded_msg
        

@mlconfig.register
class PointConvED(nn.Module):
    def __init__(self) -> None:
        ydim = V().cfg.ydim
        super().__init__()
        self.noise_layer = [
            Identity(),
        ]
        self.encoder = PointConvEncoder(ydim=ydim)
        self.decoder = PointConvDecoder(ydim=ydim)
    def forward(self, xyz, faces, msg, idx):
        encoded_xyz = self.encoder(xyz, msg)
        decoded_msg = self.decoder(encoded_xyz)
        return encoded_xyz, decoded_msg

@mlconfig.register
class CNCED(nn.Module):
    def __init__(self, channel_size):
        ydim = V().cfg.ydim
        super().__init__()
        self.noise_layer = [
            Identity(),
            Gauss(), Rotation(),
            # Scaling(s=10), Translation()
        ]
        self.encoder = CNCEncoder(ydim=ydim, channel_size=channel_size)
        self.decoder = CNCDecoder(ydim=ydim, channel_size=channel_size)
    
    def forward(self, xyz, faces, msg, idx):

        encoded_xyz = self.encoder(xyz, msg, idx)
        # noise_layer = np.random.choice(self.noise_layer, 1)[0]
        # noised_xyz = noise_layer(encoded_xyz)
        noised_xyz = []
        idxs = []
        for noise_layer in self.noise_layer:
            noised_xyz.append(noise_layer(encoded_xyz))
            idxs.append(idx)
        noised_xyz = torch.cat(noised_xyz)
        idxs = torch.cat(idxs)
        noised_xyz, _, _ = pc_normalize(noised_xyz)

        decoded_msg = self.decoder(noised_xyz, idxs)
        return encoded_xyz, decoded_msg

@mlconfig.register
class GATED(nn.Module):
    def __init__(self, channel_size):
        ydim = V().cfg.ydim
        super().__init__()
        self.noise_layer = [
            Identity(),
            Gauss(),
            Rotation(),
            # Scaling(s=10), Translation()
        ]
        self.encoder = GATEncoder(ydim=ydim, channel_size=channel_size)
        self.decoder = GATDecoder(ydim=ydim, channel_size=channel_size)
    
    def forward(self, xyz, faces, msg, idx):

        encoded_xyz = self.encoder(xyz, msg, idx)
        # noise_layer = np.random.choice(self.noise_layer, 1)[0]
        # noised_xyz = noise_layer(encoded_xyz)
        noised_xyz = []
        idxs = []
        for noise_layer in self.noise_layer:
            noised_xyz.append(noise_layer(encoded_xyz))
            idxs.append(idx)
        noised_xyz = torch.cat(noised_xyz)
        idxs = torch.cat(idxs)
        noised_xyz, _, _ = pc_normalize(noised_xyz)

        decoded_msg = self.decoder(noised_xyz, idxs)
        return encoded_xyz, decoded_msg

@mlconfig.register
class WangED(nn.Module):
    def __init__(self):
        ydim = V().cfg.ydim
        super().__init__()
        self.noise_layer = [
            Identity(),
            Gauss(), Rotation(),
            # Scaling(s=10), Translation()
        ]
        self.encoder = WangEncoder(ydim=ydim)
        self.decoder = WangDecoder(ydim=ydim)
    
    def forward(self, xyz, faces, msg, idx):

        encoded_xyz = self.encoder(xyz, msg, idx)
        # noise_layer = np.random.choice(self.noise_layer, 1)[0]
        # noised_xyz = noise_layer(encoded_xyz)
        noised_xyz = []
        idxs = []
        for noise_layer in self.noise_layer:
            noised_xyz.append(noise_layer(encoded_xyz))
            idxs.append(idx)
        noised_xyz = torch.cat(noised_xyz)
        idxs = torch.cat(idxs)
        noised_xyz, _, _ = pc_normalize(noised_xyz)

        decoded_msg = self.decoder(noised_xyz, idxs)
        return encoded_xyz, decoded_msg