# ============== Exponential Moving Average (EMA) ==============


class EMA:
    """
    Implements Exponential Moving Average (EMA) for model parameters.

    EMA maintains a smoothed version of model parameters by applying a
    weighted average over updates during training.

    Parameters
    ----------
    beta : float
        The decay factor for the exponential moving average. A value
        closer to 1 results in slower updates.

    step_start_ema : int
        The training step at which EMA updates should start.

    Formula
    --------
        EMA = beta * old + (1 - beta) * new
    """

    def __init__(self, beta: float, ema_step_start: int):
        """
        Initialises the EMA class.
        """
        self.beta = beta
        self.ema_step_start = ema_step_start
        self.step = 0

    def update_moving_average(self, ema_model, model):
        """
        Updates the EMA model's parameters using the exponential moving average.

        Parameters
        ----------
        ema_model : nn.Module
            The model that stores the EMA parameters.

        model : nn.Module
            The current model being trained.
        """
        print(f"Step {self.step}: Updating moving averages.")
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weights, new_weights = ema_param.data, current_param.data
            ema_param.data = self.update_average(old_weights, new_weights)

    def update_average(self, old, new):
        """
        Computes the updated average using the EMA formula.

        Parameters
        ----------
        old : torch.Tensor
            The previous EMA value.

        new : torch.Tensor
            The current value of the model parameter.

        Returns
        -------
        torch.Tensor
            The updated EMA value.
        """
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model):
        """
        Performs a single EMA update step.

        If the current step is less than `step_start_ema`, it resets the EMA model
        parameters to match the current model. Otherwise, it applies EMA updates.

        Parameters
        ----------
        ema_model : nn.Module
            The model that stores the EMA parameters.

        model : nn.Module
            The current model being trained.
        """

        if self.step < self.ema_step_start:
            print(
                f"Step {self.step}: Resetting EMA model parameters (start step: {self.ema_step_start})."
            )
            self.reset_parameters(ema_model, model)
        else:
            print(f"Step {self.step}: Applying EMA updates.")
            self.update_moving_average(ema_model, model)

        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Resets the EMA model parameters to match the current model.

        Parameters
        ----------
        ema_model : nn.Module
            The model that stores the EMA parameters.
        model : nn.Module
            The current model being trained.
        """
        ema_model.load_state_dict(model.state_dict())


# ============ Power-Law Weight Decay for EMA ==============
class PowerLawEMA:
    """
    Implements Power-Law Weight Decay for Exponential Moving
    Average (EMA) of model parameters.

    This class applies a power-law decay technique to EMA updates,
    allowing the decay factor to evolve dynamically based on the
    training step.

    Power-law decay adjusts the influence of past parameters,
    providing a flexible and adaptive method for smoothing.

    parameters
    ----------
    gamma : float
        The power-law decay factor.

    formula
    --------
        EMA = (1 - 1 / step) ** (gamma + 1)
    """

    def __init__(self, gamma: float):
        """
        Initialises the PowerLawEMA class.
        """
        self.gamma = gamma
        self.step = 0

    def _compute_decay_factor(self):
        """
        Computes the decay factor for the current step based on
        the power-law formula.

        Returns
        -------
        float
            The decay factor for the current training step.

            Returns 0.0 for step 0 to avoid updates in
            the first iteration.
        """
        if self.step < 1:
            return 0.0  # No decay at step 0
        return (1 - 1 / self.step) ** (self.gamma + 1)

    def update_moving_average(self, ema_model, model):
        """
        Updates the EMA model's parameters using the computed
        power-law decay factor.

        Parameters
        ----------
        ema_model : nn.Module
            The model storing the EMA parameters.

        model : nn.Module
            The current model being trained, providing the
            updated parameters.
        """
        decay_factor = self._compute_decay_factor()
        print(f"Step {self.step}: Updating moving averages.")
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weights, new_weights = ema_param.data, current_param.data
            ema_param.data = (
                decay_factor * old_weights + (1 - decay_factor) * new_weights
            )

    def step_ema(self, ema_model, model):
        """
        Performs a single update step for Power-Law EMA.

        This method calculates the decay factor, applies it to the
        moving averages, and increments the training step.

        Parameters
        ----------
        ema_model : nn.Module
            The model storing the EMA parameters.

        model : nn.Module
            The current model being trained, providing the
            updated parameters.
        """

        self.update_moving_average(ema_model, model)
        print(f"Step {self.step}: Applying EMA updates")
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """
        Resets the EMA model parameters to match the current
        model's parameters.

        Parameters
        ----------
        ema_model : nn.Module
            The model storing the EMA parameters.

        model : nn.Module
            The current model being trained, providing the
            updated parameters.
        """
        ema_model.load_state_dict(model.state_dict())
