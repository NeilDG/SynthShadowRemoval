import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, channel: int, ratio: int):
        super(ChannelAttention, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, padding=0, bias=False),
            # nn.ReLU(inplace=True),
            nn.Conv2d(channel // ratio, channel, 1, padding=0, bias=False)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_avg = self.shared_mlp(self.avg_pool(x))
        feat_max = self.shared_mlp(self.max_pool(x))

        return self.sigmoid(feat_avg + feat_max)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_avg = torch.mean(x, dim=1, keepdim=True)
        feat_max = torch.max(x, dim=1, keepdim=True)[0]

        feature = torch.cat((feat_avg, feat_max), dim=1)

        return self.sigmoid(self.conv(feature))

class CbamResblock(nn.Module):
    def __init__(self, channel: int, ratio=16):
        super(CbamResblock, self).__init__()

        self.conv_lower = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channel)
            # nn.BatchNorm2d(channel),
            # nn.ReLU(inplace=True)
        )

        self.conv_upper = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channel)
            # nn.BatchNorm2d(channel)
        )

        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention()

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path = self.conv_lower(x)
        path = self.conv_upper(path)

        path = self.ca(path) * path
        path = self.sa(path) * path

        # return self.relu(path + x)
        return path + x