import os
import sys

from option.base_options import BaseOptions

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
            default="ddpm_35m",
            choices=[
                # -- DDPM Variant Names -- >
                "ddpm_35m",
                "ddpm_ema",
                "ddpm_power_law_ema",
                # -- DDPM CFGVariant Names -- >
                "cfg_ddpm",
                "cfg_ema_ddpm",
                "cfg_power_law_ema_ddpm",
                # -- DDPM CFG ++ Variant Names -- >
                "cfg_plus_ddpm",
                "cfg_plus_ddpm_ema",
                "cfg_plus_ddpm_power_law_ema",
            ],
            help="Name of the model to use",
        )

        self._parser.add_argument(
            "--lr", type=float, default=2e-5, help="Learning rate"
        )

        self._parser.add_argument(
            "--Time_steps_FD",
            type=int,
            default=1000,
            help="Number of time steps for the diffusion model",
        )

        self._parser.add_argument(
            "--nb_images", type=int, default=16, help="Number of images to generate"
        )

        self._parser.add_argument(
            "--unet_ch",
            type=int,
            default=128,
            help="Number of filters for the UNet",
        )

        self._parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "adamw"],
            help="Optimizer to use",
        )

        self._parser.add_argument(
            "--mean",
            type=float,
            default=0.5,
            help="Mean of the dataset",
        )

        self._parser.add_argument(
            "--std",
            type=float,
            default=0.5,
            help="Standard deviation of the dataset",
        )

        self._parser.add_argument(
            "--continue_train",
            type=bool,
            default=False,
            help="Continue training",
        )

        self._parser.add_argument(
            "--print_freq",
            type=int,
            default=20,
            help="Print frequency",
        )

        self._parser.add_argument(
            "--save_freq",
            type=int,
            default=4000,
            help="Checkpoint saving frequency of the model over epochs",
        )

        # ----- CFG parameters ------>>>>

        self._parser.add_argument(
            "--cfg_scale",
            type=float,
            default=1.0,
            help="Scale of the Gaussian noise",
        )

        self._parser.add_argument(
            "--label_usage",
            type=float,
            default=0.2,
            help="Percentage of labels to use",
        )

        self._parser.add_argument(
            "--num_classes",
            type=int,
            default=5,
            help="Number of classes",
        )

        # ----- CFG ++ parameters ------>>>>

        self._parser.add_argument(
            "--control_cfg_scale",
            type=bool,
            default=True,
            help="To control the scale weights for the CFG Plus DDPM",
        )

        # ----- EMA parameters ------>>>>

        # --ema_apply is set to false by default.
        # If the model_name is either "ddpm_ema" or "cfg_ema_ddpm" or "cfg_plus_ddpm_ema"
        # The flag will be set to true in the call_methods.py file.

        self._parser.add_argument(
            "--ema_apply",
            type=bool,
            default=False,
            help="Apply EMA for Diffusion Models",
        )

        self._parser.add_argument(
            "--ema_beta",
            type=float,
            default=0.999,
            help="Beta value for the EMA",
        )

        self._parser.add_argument(
            "--ema_start_step",
            type=int,
            default=30,  # 2000
            help="Step to start the EMA",
        )

        # ----- Power Law EMA parameters ------>>>>

        # --power_ema_apply is set to false by default.
        # If the model_name is either "ddpm_power_law_ema" or "cfg_power_law_ema_ddpm" or "cfg_plus_ddpm_power_law_ema"
        # The flag will be set to true in the call_methods.py file.

        self._parser.add_argument(
            "--power_ema_apply",
            type=bool,
            default=False,
            help="Apply Power Law EMA for Diffusion Models",
        )

        self._parser.add_argument(
            "--power_ema_gamma",
            type=float,
            default=6.94,
            choices=[
                6.94,
                16.97,
            ],  # Add more values if needed
            help="Gamma value for the Power Law EMA. Use 6.94 for a balanced decay profile, "
            "or 16.97 for a tighter decay profile.",
        )

        # New parameters should be added here

        self._is_train = True
