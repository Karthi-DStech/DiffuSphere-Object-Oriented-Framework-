import torch
import torch.nn as nn
import torch.nn.functional as F
from model.nin_block import Nin


class ResNetBlock(nn.Module):
    """
    This class implements the Residual block for the UNet model.

    Parameters
    ----------
    in_ch : int
        Number of input channels

    out_ch : int
        Number of output channels

    dropout_rate : float
        Dropout rate for the block
    """

    def __init__(self, in_ch, out_ch, dropout_rate=0.1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.dense = nn.Linear(512, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)

        if not (in_ch == out_ch):
            self.nin = Nin(in_ch, out_ch)

        self.dropout_rate = dropout_rate
        self.nonlinearity = torch.nn.SiLU()

    def forward(self, x: torch.Tensor, temb) -> torch.Tensor:
        """
        The forward pass for the Residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input image

        temb : torch.Tensor
            Time embedding for the timesteps T

        Returns
        -------
        torch.Tensor
            Output image
        """

        h = self.nonlinearity(F.group_norm(x, num_groups=32))
        h = self.conv1(x)

        # add in timestep embedding
        h += self.dense(self.nonlinearity(temb))[:, :, None, None]
        h = self.nonlinearity(F.group_norm(h, num_groups=32))
        h = F.dropout(h, p=self.dropout_rate)
        h = self.conv2(h)

        if not (x.shape[1] == h.shape[1]):
            x = self.nin(x)

        assert x.shape == h.shape
        return x + h
