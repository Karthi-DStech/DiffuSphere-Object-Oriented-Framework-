import argparse
import os
import sys

import torch.nn as nn
import torch
from typing import Union
from tqdm import tqdm
from datetime import datetime
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseModel(object):
    """
    The base class for all diffusion models.

    Parameters
    ----------
    opt : argparse.Namespace
        The options used to initialize the model.

    model : nn.Module
        The model to use for the diffusion model.

    """

    def __init__(self, opt: argparse.Namespace, model: nn.Module = None) -> None:
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

        if model is not None:
            self.function_approximator = model.to(self._device)
        else:
            self.function_approximator = None

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

    def train(
        self, batch_size: int, dataset=None, train_diffusion: bool = True
    ) -> None:
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

        self.model.train()

        # Train the model
        if train_diffusion:
            self.model.zero_grad()
            loss = self._compute_loss(eps, eps_predicted)
            loss.backward()
            self.model.optimizer.step()

        return eps, eps_predicted

    def cfg_train(
        self,
        batch_size: int,
        cfg_scale: float,
        label_usage: float,
        dataset=None,
        train_diffusion: bool = True,
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

        train_diffusion : bool
            Whether to train the diffusion model.
        """
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
        eps_uncond = self.function_approximator(noisy_input, t - 1, labels=None)
        eps_cond = (
            self.function_approximator(noisy_input, t - 1, labels=labels)
            if labels is not None
            else eps_uncond
        )

        # CFG interpolation
        eps_predicted = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # Train the model
        if train_diffusion:
            self.function_approximator.train()
            optimizer = torch.optim.Adam(
                self.function_approximator.parameters(), lr=self._opt.lr
            )
            optimizer.zero_grad()
            loss = self._compute_loss(eps, eps_predicted)
            loss.backward()
            optimizer.step()

        return eps, eps_predicted

    def test(self) -> None:
        """
        This method tests the model.
        """

        self._unet.eval()
        self._unet.zero_grad()
        self._unet._get_current_performance()

    @torch.no_grad()
    def predict(
        self, num_samples: int, image_channels, img_size, use_tqdm: True
    ) -> None:
        """
        This method generates samples from the diffusion model using no grad.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.

        image_channels : int
            The number of channels in the image.

        img_size : Tuple[int, int]
            The size of the image.

        use_tqdm : bool
            Whether to use tqdm for progress bar.

        Returns
        -------
        x : torch.Tensor
            The generated samples.
        """

        num_samples = self._opt.num_images
        image_channels = self._opt.image_channels
        img_size = self._opt.img_size

        x = torch.randn(
            (num_samples, image_channels, img_size[0], img_size[1]), device=self.device
        )

        progress_bar = tqdm if use_tqdm else lambda x: x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)

            t = torch.ones(num_samples, dtype=torch.long, device=self.device) * t

            beta_t = self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = (
                self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )

            mean = (
                1
                / torch.sqrt(alpha_t)
                * (
                    x
                    - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t))
                    * self.function_approximator(x, t - 1)
                )
            )
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z

        return x

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
        if self._opt.optimizer == "adam":
            self.model.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self._opt.lr
            )
        elif self._opt.optimizer == "adamw":
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
