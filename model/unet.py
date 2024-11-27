import torch
import torch.nn as nn
import torch.nn.functional as F

from model.downsampling_block import DownSample
from model.upsampling_block import UpSample
from model.resnet_block import ResNetBlock
from model.attention_block import AttentionBlock
from model.timestep_embedding import get_timestep_embedding

from option.config import TrainOptionsConfig
from model.networks import BaseNetwork
# from option.train_options import TrainOptions

# ----- UNet Archiecture for Diffusion Model (DDPM) ----->>>>


class UNet(BaseNetwork):
    """
    This class implements the UNet model for the diffusion model.

    Parameters
    ----------
    opt: TrainOptionsConfig
        The options used to initialize the model.

    ch : int
        Number of channels in the input image

    in_ch : int
        Number of input channels
    """

    def __init__(self, opt: TrainOptionsConfig, ch: int, in_ch: int) -> None:
        super(UNet, self).__init__()

        self.ch = ch
        self.in_ch = in_ch
        self._opt = opt

        self.linear1 = nn.Linear(ch, 4 * ch)
        self.linear2 = nn.Linear(4 * ch, 4 * ch)

        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride=1, padding=1)

        self.down = nn.ModuleList(
            [
                ResNetBlock(ch, 1 * ch),
                ResNetBlock(1 * ch, 1 * ch),
                DownSample(1 * ch),
                ResNetBlock(1 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                DownSample(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                DownSample(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
            ]
        )

        self.middle = nn.ModuleList(
            [
                ResNetBlock(2 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
            ]
        )

        self.up = nn.ModuleList(
            [
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                UpSample(2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                UpSample(2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(3 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                UpSample(2 * ch),
                ResNetBlock(3 * ch, ch),
                ResNetBlock(2 * ch, ch),
                ResNetBlock(2 * ch, ch),
            ]
        )

        self.final_conv = nn.Conv2d(ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x, t):
        """
        This method implements the forward pass for the
        CFG UNet model.

        parameters
        ----------
        x : torch.Tensor
            The input image tensor

        t : torch.Tensor
            The timestep tensor
        """

        temb = get_timestep_embedding(t, self.ch)
        temb = torch.nn.functional.silu(self.linear1(temb))
        temb = self.linear2(temb)
        assert temb.shape == (t.shape[0], self.ch * 4)

        x1 = self.conv1(x)

        # Down
        x2 = self.down[0](x1, temb)
        x3 = self.down[1](x2, temb)
        x4 = self.down[2](x3)
        x5 = self.down[3](x4, temb)
        x6 = self.down[4](x5)
        x7 = self.down[5](x6, temb)
        x8 = self.down[6](x7)
        x9 = self.down[7](x8)
        x10 = self.down[8](x9, temb)
        x11 = self.down[9](x10, temb)
        x12 = self.down[10](x11)
        x13 = self.down[11](x12, temb)
        x14 = self.down[12](x13, temb)

        # Middle
        x = self.middle[0](x14, temb)
        x = self.middle[1](x)
        x = self.middle[2](x, temb)

        # Up
        x = self.up[0](torch.cat((x, x14), dim=1), temb)
        x = self.up[1](torch.cat((x, x13), dim=1), temb)
        x = self.up[2](torch.cat((x, x12), dim=1), temb)
        x = self.up[3](x)
        x = self.up[4](torch.cat((x, x11), dim=1), temb)
        x = self.up[5](torch.cat((x, x10), dim=1), temb)
        x = self.up[6](torch.cat((x, x9), dim=1), temb)
        x = self.up[7](x)
        x = self.up[8](torch.cat((x, x8), dim=1), temb)
        x = self.up[9](x)
        x = self.up[10](torch.cat((x, x6), dim=1), temb)
        x = self.up[11](x)
        x = self.up[12](torch.cat((x, x4), dim=1), temb)
        x = self.up[13](x)
        x = self.up[14](x)
        x = self.up[15](torch.cat((x, x3), dim=1), temb)
        x = self.up[16](torch.cat((x, x2), dim=1), temb)
        x = self.up[17](torch.cat((x, x1), dim=1), temb)

        x = F.silu(F.group_norm(x, num_groups=32))
        x = self.final_conv(x)

        return x


# ----- Condition based UNet Archiecture for Diffusion Model (DDPM) ----->>>>
# ----- Classifier Free Guidance ----->>>>


class UnetCFG(BaseNetwork):
    """
    This class implements the UNet model for the
    Classifier Free Guidance - Diffusion model.

    Parameters
    ----------
    opt: TrainOptionsConfig
        The options used to initialize the model.

    ch : int
        Number of channels in the input image

    in_ch : int
        Number of input channels

    num_classes : int
        Number of classes in the dataset for label embedding.
    """

    def __init__(
        self, opt: TrainOptionsConfig, ch: int, in_ch: int, num_classes: int
    ) -> None:
        super(UnetCFG, self).__init__()

        self.ch = ch
        self.in_ch = in_ch
        self._opt = opt
        self.num_classes = num_classes

        self.linear1 = nn.Linear(ch, 4 * ch)
        self.linear2 = nn.Linear(4 * ch, 4 * ch)

        # Label embedding layer
        self.label_embedding = nn.Embedding(num_classes, ch)

        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_ch, ch, 3, stride=1, padding=1)

        # Downsampling blocks
        self.down = nn.ModuleList(
            [
                ResNetBlock(ch, 1 * ch),
                ResNetBlock(1 * ch, 1 * ch),
                DownSample(1 * ch),
                ResNetBlock(1 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                DownSample(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                DownSample(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
            ]
        )

        # Middle blocks
        self.middle = nn.ModuleList(
            [
                ResNetBlock(2 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(2 * ch, 2 * ch),
            ]
        )

        # Upsampling blocks
        self.up = nn.ModuleList(
            [
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                UpSample(2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                UpSample(2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(4 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                ResNetBlock(3 * ch, 2 * ch),
                AttentionBlock(2 * ch),
                UpSample(2 * ch),
                ResNetBlock(3 * ch, ch),
                ResNetBlock(2 * ch, ch),
                ResNetBlock(2 * ch, ch),
            ]
        )

        # Final convolution layer
        self.final_conv = nn.Conv2d(ch, in_ch, 3, stride=1, padding=1)

        # GroupNorm instance for final normalization
        self.final_norm = nn.GroupNorm(32, ch)

    def forward(self, x, t, labels=None):
        """
        This method implements the forward pass for the
        CFG UNet model.

        parameters
        ----------
        x : torch.Tensor
            The input image tensor

        t : torch.Tensor
            The timestep tensor

        labels : torch.Tensor
            The label tensor
        """

        temb = get_timestep_embedding(t, self.ch)
        temb = torch.nn.functional.silu(self.linear1(temb))
        temb = self.linear2(temb)

        # Label embedding & Combine embeddings (timestep and label)
        if labels is not None:
            lemb = self.label_embedding(labels)
            # Reshaping lemb to match temb dynamically
            lemb = (
                lemb.unsqueeze(1)
                .expand(-1, temb.shape[1] // self.ch, -1)
                .reshape_as(temb)
            )
            temb = temb + lemb

        assert temb.shape == (t.shape[0], self.ch * 4)

        x1 = self.conv1(x)

        # Down
        x2 = self.down[0](x1, temb)
        x3 = self.down[1](x2, temb)
        x4 = self.down[2](x3)
        x5 = self.down[3](x4, temb)
        x6 = self.down[4](x5)
        x7 = self.down[5](x6, temb)
        x8 = self.down[6](x7)
        x9 = self.down[7](x8)
        x10 = self.down[8](x9, temb)
        x11 = self.down[9](x10, temb)
        x12 = self.down[10](x11)
        x13 = self.down[11](x12, temb)
        x14 = self.down[12](x13, temb)

        # Middle
        x = self.middle[0](x14, temb)
        x = self.middle[1](x)
        x = self.middle[2](x, temb)

        # Up
        x = self.up[0](torch.cat((x, x14), dim=1), temb)
        x = self.up[1](torch.cat((x, x13), dim=1), temb)
        x = self.up[2](torch.cat((x, x12), dim=1), temb)
        x = self.up[3](x)
        x = self.up[4](torch.cat((x, x11), dim=1), temb)
        x = self.up[5](torch.cat((x, x10), dim=1), temb)
        x = self.up[6](torch.cat((x, x9), dim=1), temb)
        x = self.up[7](x)
        x = self.up[8](torch.cat((x, x8), dim=1), temb)
        x = self.up[9](x)
        x = self.up[10](torch.cat((x, x6), dim=1), temb)
        x = self.up[11](x)
        x = self.up[12](torch.cat((x, x4), dim=1), temb)
        x = self.up[13](x)
        x = self.up[14](x)
        x = self.up[15](torch.cat((x, x3), dim=1), temb)
        x = self.up[16](torch.cat((x, x2), dim=1), temb)
        x = self.up[17](torch.cat((x, x1), dim=1), temb)

        x = F.silu(self.final_norm(x))
        x = self.final_conv(x)

        return x
