import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class DSConv(nn.Module):
    def __init__(self, channels, padding=1, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, 3, padding=padding, dilation=dilation,
            groups=channels, bias=False,
        )
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))


class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x_local, x_global):
        g = self.gate(x_local + x_global)
        return g * x_local + (1 - g) * x_global


class CAEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_local = DSConv(channels, padding=1, dilation=1)
        self.conv_global = DSConv(channels, padding=2, dilation=2)
        self.fusion = GatedFusion(channels)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        feat_local = self.relu(self.conv_local(x))
        # CCC final, section 3.2.1 / Fig. 1(b): both branches receive x.
        feat_global = self.conv_global(x)
        out = self.fusion(feat_local, feat_global)
        out = self.ca(out)
        out = self.sa(out)
        return out + residual


class LGAN(nn.Module):
    """CCC final LGAN; user-confirmed 3-block default matches Table 2's budget.

    The paper does not specify C, N, activations, reduction or clipping.
    These details follow the supplied model, except parallel branch wiring.
    """
    def __init__(self, features=32, num_blocks=3):
        super().__init__()
        if features < 8 or num_blocks < 1:
            raise ValueError("features must be >= 8 and num_blocks must be >= 1")
        self.conv_in = nn.Conv2d(3, features, 3, padding=1)
        self.body = nn.Sequential(*[CAEBlock(features) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(features, 3, 3, padding=1)

    def forward(self, x):
        feat = F.leaky_relu(self.conv_in(x), 0.2)
        out = self.body(feat)
        residual = self.conv_out(out)
        return torch.clamp(x + residual, 0, 1)


def build_model(**kwargs):
    return LGAN(**kwargs)


# Import aliases only: both now use the CCC parallel-branch computation.
ICRNetV2 = LGAN
MicroCARBlockV2 = CAEBlock
