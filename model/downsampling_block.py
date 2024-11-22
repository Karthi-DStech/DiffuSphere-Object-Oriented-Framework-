import torch
import torch.nn as nn


class DownSample(nn.Module):
    """
    This class implements the downsampling block for the UNet model.

    Parameters
    ----------
    C : int
        Number of channels in the input image
    """

    def __init__(self, C):
        super(DownSample, self).__init__()

        self._conv = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the downsampling block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Downsampled image of shape (B, C, H/2, W/2)
        """
        B, C, H, W = x.shape
        x = self._conv(x)

        expected_shape = (B, C, H // 2, W // 2)
        assert (
            x.shape == expected_shape
        ), f"Expected output shape {expected_shape}, but got {x.shape}"

        return x


# Test the function of the downsampling block

"""
t = (torch.rand (100) * 10).long()
get_timestep_embedding (t, 64)

downsample = DownSample(64)
img = torch.randn((10, 64, 400, 400))
downsample(img)

"""
