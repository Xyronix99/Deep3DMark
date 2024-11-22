import torch.nn as nn


# class Sequential(nn.Sequential):
#     def __init__(self, *args):
#         super().__init__(*args)
#     def forward(self, input):
#         result = []
#         for module in self:
#             input = module(input)
#             result.append(input)
#         return result

class LinearRelu(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channel, out_channel, bias=bias),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class ConvBNRelu1D(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class ConvBN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)

class ConvBNRelu2D(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class ConvBN2D(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)

class WeightNet(nn.Module):
    def __init__(self, mlp_channels=[64, 64, 64]):
        super().__init__()
        in_channel = 3
        self.mlps = nn.Sequential()
        for idx, channel in enumerate(mlp_channels):
            self.mlps.append(
                nn.Conv2d(in_channels=in_channel, out_channels=channel, kernel_size=1) \
                    if idx == len(mlp_channels) - 1 else \
                ConvBNRelu2D(in_channels=in_channel, out_channels=channel, kernel_size=1)
            )
            in_channel = channel

    def forward(self, offset):
        return self.mlps(offset)
    
class WeightNet1D(nn.Module):
    def __init__(self, mlp_channels=[64, 64, 64]):
        super().__init__()
        in_channel = 3
        self.mlps = nn.Sequential()
        for idx, channel in enumerate(mlp_channels):
            self.mlps.append(
                nn.Conv1d(in_channels=in_channel, out_channels=channel, kernel_size=1) \
                    if idx == len(mlp_channels) - 1 else \
                ConvBNRelu1D(in_channels=in_channel, out_channels=channel, kernel_size=1)
            )
            in_channel = channel

    def forward(self, offset):
        return self.mlps(offset)