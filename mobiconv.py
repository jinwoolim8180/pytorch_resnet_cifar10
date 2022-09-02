import torch
import torch.nn as nn
import torch.nn.functional as F


class MobiConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, padding=1, stride=1, bias=True,
                 n_pools=3, n_layers=16, n_pruned=0, ratio=[1, 0]):
        super(MobiConvBlock, self).__init__()
        # out_channels should be divisible by n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.n_pools = n_pools
        self.n_layers = n_layers
        self.n_pruned = n_pruned
        self.ratio = ratio
        assert out_channels >= n_pools * n_layers

        self.convs = nn.ModuleList()
        if groups != 1:
            for i in range(n_pools):
                self.convs.append(
                    nn.Conv2d(in_channels, n_layers, kernel_size=kernel_size, groups=n_layers,
                              padding=padding, stride=stride, bias=bias)
                )
            self.convs.append(
                nn.Conv2d(in_channels, out_channels - n_pools * n_layers, kernel_size=kernel_size,
                          groups=out_channels - n_pools * n_layers,
                          padding=padding, stride=stride, bias=bias)
            )
        else:
            for i in range(n_pools):
                self.convs.append(
                    nn.Conv2d(in_channels, n_layers, kernel_size=kernel_size,
                              padding=padding, stride=stride, bias=bias)
                )
            self.convs.append(
                nn.Conv2d(in_channels, out_channels - n_pools * n_layers, kernel_size=kernel_size,
                          padding=padding, stride=stride, bias=bias)
            )

    def forward(self, x):
        N, C, H, W = x.shape
        size = 2 ** self.n_pools
        out = []
        table = torch.ones(N, 1, H // self.stride, W // self.stride).cuda()
        for conv in self.convs:
            h = F.max_pool2d(x, kernel_size=size, stride=size)
            h = conv(h)
            h = F.upsample(h, scale_factor=size, mode='nearest')
            h *= table
            threshold = self.ratio[0] * torch.mean(h, dim=(-2, -1), keepdim=True)
            threshold += self.ratio[1] * torch.amax(h, dim=(-2, -1), keepdim=True)
            threshold /= self.ratio[0] + self.ratio[1]
            table = torch.sum(torch.ge(h, threshold).float(), dim=1, keepdim=True)
            table += self.n_pruned * torch.ones(N, 1, H // self.stride, W // self.stride).cuda()
            out.append(h)
            size //= 2
        out = torch.cat(out, dim=1)
        return out


class SmartPool2d(nn.Module):
    def __init__(self, scale, mode='avgpool', ratio=0.1):
        super(SmartPool2d, self).__init__()
        self.scale = scale
        self.mode = mode
        self.ratio = ratio

    def _crop(self, x):
        N, C, H, W = x.shape
        threshold = self.ratio * torch.amax(x, dim=(-2, -1))
        table = torch.ge(x, threshold.unsqueeze(2).unsqueeze(3))
        x_range = torch.tile(torch.arange(H), (N, C, W, 1)).permute(0, 1, 3, 2).cuda()
        y_range = torch.tile(torch.arange(W), (N, C, H, 1)).cuda()
        x_min = torch.amin(torch.logical_not(x_range) * 1e5 + table * x_range, dim=(-2, -1))
        x_max = torch.amax(torch.logical_not(x_range) * -1e5 + table * x_range, dim=(-2, -1))
        y_min = torch.amin(torch.logical_not(y_range) * 1e5 + table * y_range, dim=(-2, -1))
        y_max = torch.amax(torch.logical_not(y_range) * -1e5 + table * y_range, dim=(-2, -1))
        out = []
        for n in range(N):
            stack = []
            for c in range(C):
                feature = x[n, c, int(x_min[n, c].item()):int(x_max[n, c].item()) + 1,
                            int(y_min[n, c].item()):int(y_max[n, c].item()) + 1]
                print(feature.shape)
                stack.append(feature)
            out.append(torch.stack(stack, dim=1))
        return torch.stack(out, dim=0)

    def forward(self, x):
        N, C, H, W = x.shape
        threshold = self.ratio * torch.amax(x, dim=(-2, -1))
        table = torch.ge(x, threshold.unsqueeze(2).unsqueeze(3))
        x_range = torch.tile(torch.arange(H), (N, C, W, 1)).permute(0, 1, 3, 2).cuda()
        y_range = torch.tile(torch.arange(W), (N, C, H, 1)).cuda()
        x_min = torch.amin(torch.logical_not(x_range) * 1e5 + table * x_range, dim=(-2, -1))
        x_max = torch.amax(torch.logical_not(x_range) * -1e5 + table * x_range, dim=(-2, -1))
        y_min = torch.amin(torch.logical_not(y_range) * 1e5 + table * y_range, dim=(-2, -1))
        y_max = torch.amax(torch.logical_not(y_range) * -1e5 + table * y_range, dim=(-2, -1))
        out = []
        for n in range(N):
            stack = []
            for c in range(C):
                feature = x[n, c, int(x_min[n, c].item()):int(x_max[n, c].item()) + 1,
                          int(y_min[n, c].item()):int(y_max[n, c].item()) + 1]
                if self.mode == 'avgpool':
                    feature = F.interpolate(feature.unsqueeze(0).unsqueeze(1), size=(H // self.scale, W // self.scale),
                                            align_corners=False, antialias=True, mode='bilinear')
                elif self.mode == 'maxpool':
                    feature = F.interpolate(feature.unsqueeze(0).unsqueeze(1), size=(H, W),
                                            align_corners=False, antialias=True, mode='bilinear')
                    feature = F.max_pool2d(feature, kernel_size=self.scale, stride=self.scale)
                feature = feature.squeeze(1).squeeze(0)
                stack.append(feature)
            out.append(torch.stack(stack, dim=0))
        out = torch.stack(out, dim=0)
        return out