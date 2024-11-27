from dataclasses import dataclass
from option.enums import ModelNames


@dataclass
class BaseOptionsConfig:
    """
    This class contains the general configuration options that are
    used for training the models in 'train.py'.
    """

    # ----- Dataset parameters ------>>>>
    IMAGES_FOLDER: str = (
        "../Datasets/Topographies/raw/FiguresStacked 8X8_4X4_2X2 Embossed"
    )
    LABEL_PATH: str = "../Datasets/biology_data/TopoChip/AeruginosaWithClass.csv"
    DATASET_NAME: str = "biological"

    # ----- Dataloader Params ------>>>>
    N_EPOCHS: int = 40000
    IMG_TYPE: str = "png"
    IMG_SIZE: int = 32
    IN_CHANNELS: int = 1
    OUT_CHANNELS: int = 1
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    SEED: int = 101
    DATASET: str = "dataset"

    # ----- Artifacts Params ------>>>>
    SAVE_DIR: str = "./artifacts"
    IS_TRAIN: bool = True


@dataclass
class TrainOptionsConfig:
    """
    This class contains the configuration options that are
    used for training the models in 'train.py'.
    """

    # ----- Training parameters ------>>>>

    MODEL_NAME: str = ModelNames.DDPM
    LR: float = 2e-5
    TIME_STEPS_FD: int = 1000
    NB_IMAGES: int = 16
    UNET_CH: int = 128
    MEAN: float = 0.5
    STD: float = 0.5
    CONTINUE_TRAIN: bool = False
    PRINT_FREQ: int = 20
    SAVE_FREQ: int = 4000

    # ----- CFG parameters ------>>>>

    CFG_SCALE: float = 1.0
    LABEL_USAGE: float = 0.2
    NUM_CLASSES: int = 5

    # ----- CFG ++ parameters ------>>>>

    CONTROL_CFG_SCALE: bool = True

    # ----- EMA parameters ------>>>>

    # --ema_apply is set to false by default.
    # If the model_name is either "ddpm_ema" or "cfg_ema_ddpm" or "cfg_plus_ddpm_ema"
    # The flag will be set to true in the call_methods.py file.

    EMA_APPLY: bool = False
    EMA_BETA: float = 0.999
    EMA_START_STEP: int = 2000

    # ----- Power Law EMA parameters ------>>>>

    # --power_ema_apply is set to false by default.
    # If the model_name is either "ddpm_power_law_ema" or "cfg_power_law_ema_ddpm" or "cfg_plus_ddpm_power_law_ema"
    # The flag will be set to true in the call_methods.py file.

    POWER_EMA_APPLY: bool = False
    POWER_EMA_GAMMA: float = 6.94
