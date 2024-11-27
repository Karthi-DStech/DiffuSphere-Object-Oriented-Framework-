import torch
import torch.nn as nn

from model.models import BaseModel
from call_methods import make_network
from option.enums import ModelNames, NetworkNames
from option.config import TrainOptionsConfig

# ------ Model instance for the DPPM Model ----->>>


class DDPM(BaseModel):
    """
    The class for the DDPM model.

    Parameters
    ----------
    T : int
        The number of time steps.

    opt : TrainOptionsConfig
        The options used to initialize the model.
    """

    def __init__(self, T: int, opt: TrainOptionsConfig):
        """
        Initializes the DDPM model.

        Implemts
        --------
        _create_networks :
            Creates the UNet model as the network for the DDPM.

        _make_loss :
            Creates the MSE loss function.

        _make_optimizer :
            Creates the optimiser for the model.
        """

        super(DDPM, self).__init__(opt)

        self._name = ModelNames.DDPM
        self.T = T

        # Initialize the model (e.g., UNet) and move it to the device
        self._create_networks()
        self.model = self._unet.to(self._device)

        # Define the betas and alphas for DDPM
        self.beta = torch.linspace(1e-4, 0.02, T).to(self._device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        if self._is_train:
            self._make_loss()
            self._make_optimizer()

    def _create_networks(self):
        """
        Creates the UNet model as the network for the DDPM.

        Returns
        -------
        Initialised the UNet model.
        """
        self._unet = make_network(
            network_name=NetworkNames.DDPM_Unet,
            opt=self._opt,
            ch=self._opt.unet_ch,
            in_ch=self._opt.in_channels,
        )

        self._send_to_device(self._unet)

    def _make_loss(self):
        """
        Creates the MSE loss function.
        """
        self._loss = nn.MSELoss()
        self._send_to_device(self._loss)

    def _compute_loss(self, eps, eps_predicted):
        """
        Computes the loss between the predicted and actual noise.

        Parameters
        ----------
        eps : torch.Tensor
            The actual noise.

        eps_predicted : torch.Tensor
            The predicted noise.
        """
        return self._loss(eps_predicted, eps)

    def _get_current_performance(self, do_visualization: bool = False) -> None:
        """
        Prints the current training performance.

        Parameters
        ----------
        do_visualization : bool
            Whether to visualize the performance.

        Returns
        -------
            Current performance of the model.
        """
        self._current_performance = {
            "Training Loss (MSE)": self._loss.item(),
        }

        if do_visualization:
            self._visualize_performance()

        print(f"Current Performance: {self._current_performance}")

    def forward(self, x, t):
        """
        Forward pass for the DDPM model.

        Parameters
        ----------
        x : torch.Tensor
            The input image.

        t : int
            The time step.
        """
        return self.model(x, t)


# ----- Condition based model instance for Diffusion Model (DDPM) ----->>>>
# ----- Classifier Free Guidance and CFG ++ ----->>>>


class DDPMCFG(BaseModel):
    """
    The class for the DDPM model.

    Parameters
    ----------
    T : int
        The number of time steps.

    opt : TrainOptionsConfig
        The options used to initialize the model.
    """

    def __init__(self, T: int, opt: TrainOptionsConfig):
        """
        Initializes the DDPM model.

        Implemts
        --------
        _create_networks :
            Creates the UNet model as the network for the DDPM.

        _make_loss :
            Creates the MSE loss function.

        _make_optimizer :
            Creates the optimiser for the model.
        """

        super(DDPMCFG, self).__init__(opt)
        self._name = ModelNames.CFG_DDPM
        self.T = T

        # Initialize the model (e.g., UNet) and move it to the device
        self._create_networks()
        self.model = self._cfg_unet.to(self._device)

        # Define the betas and alphas for DDPM
        self.beta = torch.linspace(1e-4, 0.02, T).to(self._device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        if self._is_train:
            self._make_loss()
            self._make_optimizer()

    def _create_networks(self):
        """
        Creates the function approximator for the CFG DDPM.

        Returns
        -------
        Initialised the UNet model.
        """
        self._cfg_unet = make_network(
            network_name=NetworkNames.CFG_Unet,
            opt=self._opt,
            ch=self._opt.unet_ch,
            in_ch=self._opt.in_channels,
            num_classes=self._opt.num_classes,
        )

        self._send_to_device(self._cfg_unet)

    def _make_loss(self):
        """
        Creates the MSE loss function.
        """
        self._loss = nn.MSELoss()
        self._send_to_device(self._loss)

    def _compute_loss(self, eps, eps_predicted):
        """
        Computes the loss between the predicted and actual noise.

        Parameters
        ----------
        eps : torch.Tensor
            The actual noise.

        eps_predicted : torch.Tensor
            The predicted noise.
        """
        return self._loss(eps_predicted, eps)

    def _get_current_performance(self, do_visualization: bool = False) -> None:
        """
        Prints the current training performance.

        Parameters
        ----------
        do_visualization : bool
            Whether to visualize the performance.

        Returns
        -------
            Current performance of the model.
        """
        self._current_performance = {
            "Training Loss (MSE)": self._loss.item(),
        }

        if do_visualization:
            self._visualize_performance()

    def forward(self, x, t, labels):
        """
        Forward pass for the DDPM model.

        Parameters
        ----------
        x : torch.Tensor
            The input image.

        t : int
            The time step.
        """
        return self.model(x, t, labels)
