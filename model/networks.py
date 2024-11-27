import os
import sys

import torch
import torch.nn as nn

from typing import Tuple
from option.enums import NetworkNames

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseNetwork(nn.Module):
    """
    This class is an abstract class for networks
    """

    def __init__(self) -> None:
        """
        Initializes the BaseNetwork class
        """
        super().__init__()
        self._name = NetworkNames.BASE_NETWORK

    @property
    def name(self) -> str:
        """
        Returns the name of the network
        """
        return self._name

    def forward(self, x: torch, t) -> torch.Tensor:
        """
        Forward pass for the network

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Returns the string representation of the network
        """
        return self._name

    def get_num_params(self) -> Tuple[int, int]:
        """
        Returns the number of parameters in the network

        Returns
        -------
        all_params: int
            The total number of parameters in the network
        trainable_params: int
            The total number of trainable parameters in the network
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return all_params, trainable_params
