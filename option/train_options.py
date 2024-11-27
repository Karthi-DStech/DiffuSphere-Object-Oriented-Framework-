import os
import sys

from option.base_options import BaseOptions
from option.enums import (
    ModelNames,
    OptimiserNames,
)

from option.config import BaseOptionsConfig, TrainOptionsConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainOptions(BaseOptions):
    """
    Training options for the DDPM model
    """

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """
        Initializes the training options from the
        base options.
        """
        BaseOptions.initialize(self)

        self._parser.add_argument(
            "--model_name",
            type=str,
            default=ModelNames.DDPM,
            choices=[
                # -- DDPM Variant Names -- >
                ModelNames.DDPM,
                ModelNames.CFG_DDPM,
                ModelNames.DDPMwithPowerLawEMA,
                # -- DDPM CFGVariant Names -- >
                ModelNames.CFG_DDPM,
                ModelNames.CFG_DDPM_EMA,
                ModelNames.CFG_DDPM_PowerLawEMA,
                # -- DDPM CFG ++ Variant Names -- >
                ModelNames.CFG_Plus_DDPM,
                ModelNames.CFG_Plus_DDPM_EMA,
                ModelNames.CFG_Plus_DDPM_PowerLawEMA,
            ],
            help="Name of the model to use",
        )

        self._parser.add_argument(
            "--lr", type=float, default=TrainOptionsConfig.LR, help="Learning rate"
        )

        self._parser.add_argument(
            "--Time_steps_FD",
            type=int,
            default=TrainOptionsConfig.TIME_STEPS_FD,
            help="Number of time steps for the diffusion model",
        )

        self._parser.add_argument(
            "--nb_images",
            type=int,
            default=TrainOptionsConfig.NB_IMAGES,
            help="Number of images to generate",
        )

        self._parser.add_argument(
            "--unet_ch",
            type=int,
            default=TrainOptionsConfig.UNET_CH,
            help="Number of filters for the UNet",
        )

        self._parser.add_argument(
            "--optimizer",
            type=str,
            default=OptimiserNames.ADAM,
            choices=["adam", "adamw"],
            help="Optimizer to use",
        )

        self._parser.add_argument(
            "--mean",
            type=float,
            default=TrainOptionsConfig.MEAN,
            help="Mean of the dataset",
        )

        self._parser.add_argument(
            "--std",
            type=float,
            default=TrainOptionsConfig.STD,
            help="Standard deviation of the dataset",
        )

        self._parser.add_argument(
            "--continue_train",
            type=bool,
            default=TrainOptionsConfig.CONTINUE_TRAIN,
            help="Continue training",
        )

        self._parser.add_argument(
            "--print_freq",
            type=int,
            default=TrainOptionsConfig.PRINT_FREQ,
            help="Print frequency",
        )

        self._parser.add_argument(
            "--save_freq",
            type=int,
            default=TrainOptionsConfig.SAVE_FREQ,
            help="Checkpoint saving frequency of the model over epochs",
        )

        # ----- CFG parameters ------>>>>

        self._parser.add_argument(
            "--cfg_scale",
            type=float,
            default=TrainOptionsConfig.CFG_SCALE,
            help="Scale of the Gaussian noise",
        )

        self._parser.add_argument(
            "--label_usage",
            type=float,
            default=TrainOptionsConfig.LABEL_USAGE,
            help="Percentage of labels to use",
        )

        self._parser.add_argument(
            "--num_classes",
            type=int,
            default=TrainOptionsConfig.NUM_CLASSES,
            help="Number of classes",
        )

        # ----- CFG ++ parameters ------>>>>

        self._parser.add_argument(
            "--control_cfg_scale",
            type=bool,
            default=TrainOptionsConfig.CONTROL_CFG_SCALE,
            help="To control the scale weights for the CFG Plus DDPM",
        )

        # ----- EMA parameters ------>>>>

        # --ema_apply is set to false by default.
        # If the model_name is either "ddpm_ema" or "cfg_ema_ddpm" or "cfg_plus_ddpm_ema"
        # The flag will be set to true in the call_methods.py file.

        self._parser.add_argument(
            "--ema_apply",
            type=bool,
            default=TrainOptionsConfig.EMA_APPLY,
            help="Apply EMA for Diffusion Models",
        )

        self._parser.add_argument(
            "--ema_beta",
            type=float,
            default=TrainOptionsConfig.EMA_BETA,
            help="Beta value for the EMA",
        )

        self._parser.add_argument(
            "--ema_start_step",
            type=int,
            default=TrainOptionsConfig.EMA_START_STEP,  # 2000
            help="Step to start the EMA",
        )

        # ----- Power Law EMA parameters ------>>>>

        # --power_ema_apply is set to false by default.
        # If the model_name is either "ddpm_power_law_ema" or "cfg_power_law_ema_ddpm" or "cfg_plus_ddpm_power_law_ema"
        # The flag will be set to true in the call_methods.py file.

        self._parser.add_argument(
            "--power_ema_apply",
            type=bool,
            default=TrainOptionsConfig.POWER_EMA_APPLY,
            help="Apply Power Law EMA for Diffusion Models",
        )

        self._parser.add_argument(
            "--power_ema_gamma",
            type=float,
            default=TrainOptionsConfig.POWER_EMA_GAMMA,
            choices=[
                6.94,
                16.97,
            ],  # Add more values if needed
            help="Gamma value for the Power Law EMA. Use 6.94 for a balanced decay profile, "
            "or 16.97 for a tighter decay profile.",
        )

        # New parameters should be added here

        self._is_train = BaseOptionsConfig.IS_TRAIN
