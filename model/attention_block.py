import torch
import torch.nn as nn
import torch.nn.functional as F
from model.nin_block import Nin


class AttentionBlock(nn.Module):
    """
    This class implements the Attention block for the UNet model.

    Parameters
    ----------

    ch : int
        Number of channels in the input image
    """

    def __init__(self, ch):
        super(AttentionBlock, self).__init__()

        self.Q = Nin(ch, ch)
        self.K = Nin(ch, ch)
        self.V = Nin(ch, ch)

        self.ch = ch

        self.nin = Nin(ch, ch, scale=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the Attention block.

        Parameters
        ----------
        x : torch.Tensor
            Input image

        Returns
        -------
        torch.Tensor
            Output image
        """

        B, C, H, W = x.shape
        assert C == self.ch

        h = F.group_norm(x, num_groups=32)
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        w = torch.einsum("bchw, bcHW->bhwHW", q, k) * int((C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))

        h = torch.einsum("bhwHW, bcHW->bchw", w, v)
        h = self.nin(h)

        assert h.shape == x.shape
        return x + h


# Test the function of the attention block


"""
t = (torch.rand (10) * 10).long()
temb= get_timestep_embedding (t, 512)

downsample = DownSample(64)
img = torch.randn((10, 64, 16, 16))
hidden  = downsample(img)

upsample = UpSample(64)
img = upsample(hidden)
print(img.shape)

nin = Nin(64, 128)
print(nin(img).shape)
img = nin(img)

resnet = ResNetBlock(128, 128, 0.1)
img = resnet(img, temb)
print(img.shape)

resnet = ResNetBlock(128, 64, 0.1)
img = resnet(img, temb)
print(img.shape)

att = AttentionBlock(64)
img = att(img)
print(img.shape)

"""
