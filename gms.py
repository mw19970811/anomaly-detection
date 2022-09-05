import numpy as np
import torch
import torch.nn.functional as F


class GMS(torch.nn.Module):
    """SSIM. Modified from:
    https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
    """

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.channel = 1
        self._create_prewitt_kernel(self.channel)


    def forward(self, img1, img2):
        assert len(img1.shape) == 4

        channel = img1.size()[1]

        if channel == self.channel and self.x_kernel.data.type() == img1.data.type():
            pass
        else:
            self._create_prewitt_kernel(channel)

            self.x_kernel = self.x_kernel.type_as(img1)
            self.y_kernel = self.y_kernel.type_as(img1)

            self.channel = channel

        return self._gms(img1, img2, self.size_average)

    def _create_prewitt_kernel(self, channel):
        x_kernel = torch.tensor([[-1.,0.,1.],[-1.,0.,1.],[-1.,0.,1.]]).expand(channel, 1, 3, 3).contiguous()
        y_kernel = torch.tensor([[-1.,-1.,-1.],[0.,0.,0.],[1.,1.,1.]]).expand(channel, 1, 3, 3).contiguous()
        self.register_buffer('x_kernel',x_kernel,persistent=False)
        self.register_buffer('y_kernel',y_kernel,persistent=False)
    
    def _create_gm_map(self, img):
        x_gms_map = F.conv2d(img, self.x_kernel, padding=1, groups=self.channel)
        y_gms_map = F.conv2d(img, self.y_kernel, padding=1, groups=self.channel)
        gms_map = torch.sqrt(x_gms_map ** 2 + y_gms_map ** 2)
        return gms_map

    def _gms(self, img1, img2, size_average=True):
        dx = self._create_gm_map(img1)
        dy = self._create_gm_map(img2)

        C = 0.01 ** 2

        gms_map = (2 * dx * dy + C) / ( dx ** 2 + dy ** 2 + C)

        if size_average:
            return gms_map.mean()

        else:
            return gms_map

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return
