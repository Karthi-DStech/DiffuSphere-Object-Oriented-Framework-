from enum import StrEnum


class ModelNames(StrEnum):
    """
    This class contains the names of the diffusion models
    that are used for training and inference.
    """

    DDPM = "ddpm_35m"
    DDPMwithEMA = "ddpm_ema"
    DDPMwithPowerLawEMA = "ddpm_power_law_ema"

    CFG_DDPM = "cfg_ddpm"
    CFG_DDPM_EMA = "cfg_ema_ddpm"
    CFG_DDPM_PowerLawEMA = "cfg_power_law_ema_ddpm"

    CFG_Plus_DDPM = "cfg_plus_ddpm"
    CFG_Plus_DDPM_EMA = "cfg_plus_ddpm_ema"
    CFG_Plus_DDPM_PowerLawEMA = "cfg_plus_ddpm_power_law_ema"


class NetworkNames(StrEnum):
    """
    This class contains the names of the U-Net Architectures
    that are used for training and inference.
    """

    DDPM_Unet = "ddpm_unet"
    CFG_Unet = "cfg_unet"


class DatasetNames(StrEnum):
    """
    This class contains the names of the datasets that are
    used for training and inference.
    """

    MNIST = "mnist"
    BIOLOGICAL = "biological"


class OptimiserNames(StrEnum):
    """
    This class contains the names of the optimisers that are
    used for training and inference.
    """

    ADAM = "adam"
    ADAM_W = "adamw"


class TrainScriptParams(StrEnum):
    """
    This class contains the names of the parameters that are
    used for training the models in 'train.py'.
    """

    BATCH_SIZE = "batch_size"
    DATASET = "dataset"

    CFG_SCALE = "cfg_scale"
    LABEL_USAGE = "label_usage"
