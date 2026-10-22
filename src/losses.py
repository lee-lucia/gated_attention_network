
import torch
from torch import nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, ssim_weight=0.5):
        super().__init__()
        if ssim_weight < 0:
            raise ValueError("ssim_weight must be nonnegative")
        self.ssim_weight = ssim_weight
        coordinates = torch.arange(11, dtype=torch.float32) - 5
        gaussian = torch.exp(-coordinates.square() / (2 * 1.5 ** 2))
        gaussian /= gaussian.sum()
        window = gaussian[:, None] * gaussian[None, :]
        self.register_buffer("window", window.expand(3, 1, 11, 11).contiguous())

    def forward(self, prediction, target):
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("Loss expects equally sized B x 3 x H x W RGB tensors")
        mse = F.mse_loss(prediction, target)
        if self.ssim_weight == 0:
            return mse
        if min(prediction.shape[-2:]) < 1:
            raise ValueError("SSIM training loss requires positive height and width")
        x, y = prediction.clamp(0, 1), target.clamp(0, 1)
        window = self.window.to(dtype=x.dtype)

        def average(image):
            return F.conv2d(image, window, padding=5, groups=3)

        mean_x, mean_y = average(x), average(y)
        variance_x = average(x.square()) - mean_x.square()
        variance_y = average(y.square()) - mean_y.square()
        covariance = average(x * y) - mean_x * mean_y
        ssim = ((2 * mean_x * mean_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
                / ((mean_x.square() + mean_y.square() + 0.01 ** 2)
                   * (variance_x + variance_y + 0.03 ** 2)))
        return mse + self.ssim_weight * (1 - ssim.mean())
