import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Module):
    """
    This class implements the upsampling block for the UNet model.

    Parameters
    ----------
    C : int
        Number of channels in the input image
    """

    def __init__(self, C):
        super(UpSample, self).__init__()

        self._conv = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the downsampling block.

        Parameters
        ----------
        x : torch.Tensor
            Input image

        Returns
        -------
        torch.Tensor
            Downsampled image
        """

        if x.ndim != 4:
            raise ValueError(
                f"Input tensor must have 4 dimensions (B, C, H, W), but got {x.ndim} dimensions"
            )

        try:

            B, C, H, W = x.shape

            x = F.interpolate(x, size=None, scale_factor=2, mode="nearest")
            x = self._conv(x)

            assert x.shape == (B, C, H * 2, W * 2)
            return x

        except Exception as e:
            raise ValueError(
                f"An error occured in the forward pass of the upsampling block: {e}"
            )
