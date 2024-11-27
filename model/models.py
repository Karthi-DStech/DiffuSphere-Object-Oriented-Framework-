import os
import sys

import torch.nn as nn
import torch
from typing import Union
import copy
from datetime import datetime
import numpy as np
from option.enums import ModelNames, OptimiserNames
from option.config import TrainOptionsConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseModel(object):
    """
    The base class for all diffusion models.

    Parameters
    ----------
    opt : TrainOptionsConfig
        The options used to initialize the model.

    model : nn.Module
        The model to use for the diffusion model.

    """

    def __init__(self, opt: TrainOptionsConfig, model: nn.Module = None) -> None:
        """
        Initializes the BaseModel class.

        Implements
        ----------
        _get_device :
            Creates the networks.

        artifcats_dir : str
            The directory to save the model.
        """
        super().__init__()
        self._name = "BaseModel"
        self._opt = opt
        self._is_train = self._opt.is_train

        self._get_device()

        # Initialize the model if provided
        self.model = model.to(self._device) if model else None

        # Check if the artifacts folder exists
        artifacts_dir = self._opt.save_dir
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            print(f"Created directory: {artifacts_dir}")
        else:
            print(f"Directory already exists: {artifacts_dir}")

        # Create a sub-directory for this specific model and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(
            artifacts_dir, f"{self._opt.model_name}_{current_time}"
        )
        os.makedirs(self.save_dir, exist_ok=True)

    def _get_device(self) -> None:
        """
        Creates the networks

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def _create_networks(self) -> None:
        """
        Creates the networks

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def _make_loss(self) -> None:
        """
        Creates the loss functions

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def _forward_ddpm(self):
        """
        Forward pass for the model

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def _get_current_performance(self, do_visualization: bool = False) -> None:
        """
        Gets the current performance of the model

        Parameters
        ----------
        do_visualization: bool
            Whether to visualize the performance

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Returns the name of the model
        """
        return self._name

    def _apply_ema(self):
        """
        Applies Exponential Moving Average (EMA) updates to the
        model's parameters.

        If EMA has not been initialized, this method initializes
        the EMA model by creating a copy of the current model
        and setting it to evaluation mode.

        EMA updates are performed if `self._opt.ema_apply` is enabled.

        Raises
        ------
        ValueError
            If the base model (`self.model`) is not defined.
        """
        if self._opt.ema_apply:
            if not hasattr(self, "ema_model") or self.ema_model is None:

                # Check if the model exists
                if not hasattr(self, "model") or self.model is None:
                    raise ValueError("Model is not defined. Cannot apply EMA.")

                from model.ema import EMA

                # Initialize EMA model and EMA object
                self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
                self.ema = EMA(
                    beta=self._opt.ema_beta, ema_step_start=self._opt.ema_start_step
                )
                print(f"EMA is initialised for {self._opt.model_name}.")

            # Update EMA model's parameters
            self.ema.step_ema(self.ema_model, self.model)

    def _apply_power_law_ema(self):
        """
        Applies Power-Law Exponential Moving Average (Power-Law EMA)
        updates to the model's parameters.

        If Power-Law EMA has not been initialized, this method
        initializes the EMA model by creating a copy of the
        current model and setting it to evaluation mode.

        Power-Law EMA updates are performed if `self._opt.power_ema_apply`
        is enabled.

        Raises
        ------
        ValueError
            If the base model (`self.model`) is not defined.
        """
        if self._opt.power_ema_apply:
            if not hasattr(self, "ema_model") or self.ema_model is None:

                # Check if the model exists
                if not hasattr(self, "model") or self.model is None:
                    raise ValueError("Model is not defined. Cannot apply EMA.")

                from model.ema import PowerLawEMA

                # Initialize EMA model and Power-Law EMA object
                self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
                self.ema = PowerLawEMA(
                    gamma=self._opt.power_ema_gamma,
                )
                print(f"Power-Law EMA is initialized for {self._opt.model_name}.")

            # Update EMA model's parameters
            self.ema.step_ema(self.ema_model, self.model)

    def _apply_ema_logic(self, model_name: str) -> None:
        """
        Applies the appropriate EMA logic based on the model name and conditions.

        Parameters
        ----------
        model_name : str
            The name of the model to determine the type of EMA to apply.
        """
        model_name = model_name.lower()

        # Early exit if no EMA is applicable
        if not self._opt.ema_apply and not self._opt.power_ema_apply:
            return

        # Map model names to their EMA logic with conditions
        ema_mapping = {
            ModelNames.DDPMwithEMA: (self._opt.ema_apply, self._apply_ema),
            ModelNames.DDPMwithPowerLawEMA: (
                self._opt.power_ema_apply,
                self._apply_power_law_ema,
            ),
            ModelNames.CFG_DDPM_EMA: (self._opt.ema_apply, self._apply_ema),
            ModelNames.CFG_DDPM_PowerLawEMA: (
                self._opt.power_ema_apply,
                self._apply_power_law_ema,
            ),
            ModelNames.CFG_Plus_DDPM_EMA: (self._opt.ema_apply, self._apply_ema),
            ModelNames.CFG_Plus_DDPM_PowerLawEMA: (
                self._opt.power_ema_apply,
                self._apply_power_law_ema,
            ),
        }

        # Check and apply EMA logic if conditions are met
        if model_name in ema_mapping:
            condition, ema_function = ema_mapping[model_name]
            if condition:  # Ensure the condition is met
                ema_function()

    def train(self, batch_size: int, dataset=None) -> None:
        """
        This method trains the model.

        Parameters
        ----------
        batch_size : int
            The batch size to use for training.

        dataset : torch.Tensor
            The dataset to use for training.

        train_diffusion : bool
            Whether to train the diffusion model.

        Returns
        -------
        eps : torch.Tensor
            The actual noise.

        eps_predicted : torch.Tensor
            The predicted noise.
        """
        self.model.train()

        self.batch_size = self._opt.batch_size
        self.dataset = dataset

        x0 = dataset
        actual_batch_size = x0.size(0)
        t = torch.randint(
            1, self.T + 1, (actual_batch_size,), device=self._device, dtype=torch.long
        )
        eps = torch.randn_like(x0)

        # Calculate noisy input
        alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        eps_predicted = self.model(
            torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps, t - 1
        )

        # Train the model
        if self._is_train:
            self.model.optimizer.zero_grad()
            loss = self._compute_loss(eps, eps_predicted)
            loss.backward()
            self.model.optimizer.step()

            # Apply EMA logic
            self._apply_ema_logic(self._opt.model_name)

        return eps, eps_predicted

    def cfg_train(
        self,
        batch_size: int,
        cfg_scale: float,
        label_usage: float,
        dataset=None,
    ) -> None:
        """
        This method trains the model using the CFG method

        Parameters
        ----------
        batch_size : int
            The batch size to use for training.

        cfg_scale : float
            The scale of the Gaussian noise.

        label_usage : float
            The percentage of labels to use.

        dataset : torch.Tensor
            The dataset to use for training.
        """
        self.model.train()

        x0 = dataset
        actual_batch_size = x0.size(0)
        t = torch.randint(
            1, self.T + 1, (actual_batch_size,), device=self._device, dtype=torch.long
        )
        eps = torch.randn_like(x0)

        # Calculate noisy input
        alpha_bar_t = self.alpha_bar[t - 1].view(-1, 1, 1, 1)
        noisy_input = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

        # Labels
        labels = self._send_to_device(
            torch.randint(0, self._opt.num_classes, (actual_batch_size,))
        )

        # Randomly decide whether to use labels
        if np.random.random() >= label_usage:
            labels = None

        # Conditional and unconditional predictions for CFG
        eps_uncond = self.model(noisy_input, t - 1, labels=None)
        eps_cond = (
            self.model(noisy_input, t - 1, labels=labels)
            if labels is not None
            else eps_uncond
        )

        # CFG interpolation
        eps_predicted = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # Train the model
        if self._is_train:

            self.model.optimizer.zero_grad()
            loss = self._compute_loss(eps, eps_predicted)
            loss.backward()
            self.model.optimizer.step()

            # Apply EMA logic
            self._apply_ema_logic(self._opt.model_name)

        return eps, eps_predicted

    def cfg_plus_train(
        self,
        batch_size: int,
        cfg_scale: float,
        label_usage: float,
        dataset=None,
    ) -> None:
        """
        This method trains the model using an enhanced CFG method (CFG+).

        Parameters
        ----------
        batch_size : int
            The batch size to use for training.

        cfg_scale : float
            The scale of the Gaussian noise.

        label_usage : float
            The percentage of labels to use.

        dataset : torch.Tensor
            The dataset to use for training.

        augmentation_func : Callable, optional
            A function to apply additional augmentations or modifications to inputs.
        """

        self.model.train()

        if self._opt.model_name == ModelNames.CFG_Plus_DDPM:
            if self._opt.control_cfg_scale:
                if not (0 <= cfg_scale <= 1):
                    raise ValueError("cfg_scale must be in the range [0, 1] for CFG++")

        x0 = dataset
        actual_batch_size = x0.size(0)
        t = torch.randint(
            1, self.T + 1, (actual_batch_size,), device=self._device, dtype=torch.long
        )
        eps = torch.randn_like(x0)

        # Calculate noisy input
        alpha_bar_t = self.alpha_bar[t - 1].view(-1, 1, 1, 1)
        noisy_input = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

        # Labels
        labels = self._send_to_device(
            torch.randint(0, self._opt.num_classes, (actual_batch_size,))
        )

        # Randomly decide whether to use labels
        if np.random.random() >= label_usage:
            labels = None

        # Conditional and unconditional predictions
        eps_uncond = self.model(noisy_input, t - 1, labels=None)
        eps_cond = (
            self.model(noisy_input, t - 1, labels=labels)
            if labels is not None
            else eps_uncond
        )

        # CFG interpolation (only used for denoising step)
        eps_predicted = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # CFG ++ interpolation (used for renoising step)
        x_denoised = (
            noisy_input - torch.sqrt(1 - alpha_bar_t) * eps_predicted
        ) / torch.sqrt(alpha_bar_t)

        noisy_input = (
            torch.sqrt(self.alpha_bar[t - 2].view(-1, 1, 1, 1)) * x_denoised
            + torch.sqrt(1 - self.alpha_bar[t - 2].view(-1, 1, 1, 1)) * eps_uncond
        )

        if self._is_train:
            self.model.optimizer.zero_grad()
            loss = self._compute_loss(eps, eps_predicted)
            loss.backward()
            self.model.optimizer.step()

            # Apply EMA logic
            self._apply_ema_logic(self._opt.model_name)

        return eps, eps_predicted

    def _print_num_params(self) -> None:
        """
        Prints the number of parameters of the model

        Raises
        ------
        ValueError
            If the networks are not created yet
        """
        if self._networks is None:
            raise ValueError("Networks are not created yet")
        else:
            for network in self._networks:
                all_params, trainable_params = network.get_num_params()
                print(
                    f"{network.name} has {all_params/1e3:.1f}K parameters ({trainable_params/1e3:.1f}K trainable)"
                )

    def _make_optimizer(self) -> None:
        """
        This method creates the optimizer for the model.

        Parameters
        ----------
        None

        Raises
        ------
        NotImplementedError
            If the method is not implemented
        """
        if self._opt.optimizer == OptimiserNames.ADAM:
            self.model.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self._opt.lr
            )
        elif self._opt.optimizer == OptimiserNames.ADAM_W:
            self.model.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self._opt.lr
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self._opt.optimizer} is not implemented"
            )

    def _get_device(self) -> None:
        """
        Gets the device to train the model
        """
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Using device: {self._device}")

    def _send_to_device(
        self, data: Union[torch.Tensor, list]
    ) -> Union[torch.Tensor, list]:
        """
        Sends the data to the device

        Parameters
        ----------
        data: torch.Tensor
            The data to send to the device

        Returns
        -------
        torch.Tensor
            The data in the device
        """
        if isinstance(data, list):
            return [x.to(self._device) for x in data]
        else:
            return data.to(self._device)

    def save_networks(self, epoch: Union[int, str]) -> None:
        """
        Saves the model and optimizer state.

        Parameters
        ----------
        epoch : Union[int, str]
            The current epoch number or a string indicating "final".

        Returns
        -------
        None
        """
        # Define the path to save the model and optimizer separately
        model_save_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
        optimizer_save_path = os.path.join(
            self.save_dir, f"optimizer_epoch_{epoch}.pth"
        )

        # Save the model and optimizer
        torch.save(self.model.cpu(), model_save_path)
        torch.save(self.model.optimizer.state_dict(), optimizer_save_path)

        # Move the model back to the original device
        self.model.to(self._device)

        print(f"Saved model: {model_save_path}")
        print(f"Saved optimizer: {optimizer_save_path}")
