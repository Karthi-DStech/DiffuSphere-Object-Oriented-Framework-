import torch
import torch.nn as nn
import numpy as np


class Nin(nn.Module):
    """
    This class implements the Network in Network block for the UNet model.

    Parameters
    ----------
    in_dim : int
        Input dimension

    out_dim : int
        Output dimension

    scale : float
        Scaling factor for the weights
    """

    def __init__(self, in_dim, out_dim, scale=1e-10):
        super(Nin, self).__init__()

        n = (in_dim + out_dim) / 2
        limit = np.sqrt(3 * scale / n)

        # Initialize the weights
        self.W = torch.nn.Parameter(
            torch.zeros((in_dim, out_dim), dtype=torch.float32).uniform_(-limit, limit)
        )

        # Initialize the bias
        self.b = torch.nn.Parameter(
            torch.zeros((1, out_dim, 1, 1), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the Network in Network block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, O, H, W) where O is the output dimension.
        """
        return torch.einsum("bchw, co->bowh", x, self.W) + self.b
