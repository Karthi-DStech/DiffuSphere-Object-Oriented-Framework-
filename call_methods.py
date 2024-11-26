import argparse
from typing import Union

import torch

from data.datasets import BaseDataset
from model.models import BaseModel
from option.enums import ModelNames, NetworkNames, DatasetNames


def make_model(model_name: str, *args, **kwargs) -> Union[BaseModel, BaseModel]:
    """
    Creates a model from the given model name

    Parameters
    ----------
    model_name: str
        The name of the model to create
    *args: list
        The arguments to pass to the model constructor
    **kwargs: dict
        The keyword arguments to pass to the model constructor

    Returns
    -------
    model: BaseModel
        The created model
    """
    model = None

    #  ========= Make Model Instance for all DDPM Variants without Condition =========

    # --- DDPM ---

    if model_name.lower() == ModelNames.DDPM:
        from model.ddpm import DDPM

        model = DDPM(*args, **kwargs)
        model.train_method = model.train
        print("DDPM 35M will be trained using the 'DDPM.train method'...")

    # --- DDPM with EMA ---

    elif model_name.lower() == ModelNames.DDPMwithEMA:
        from model.ddpm import DDPM

        opt = kwargs["opt"]
        opt.ema_apply = True
        model = DDPM(*args, **kwargs)
        print(f"Final DDPM ema_apply setting: {opt.ema_apply}")
        model.train_method = model.train
        print("DDPM will be trained using the 'DDPM.train method' with EMA...")

    # --- DDPM with Power Law EMA ---

    elif model_name.lower() == ModelNames.DDPMwithPowerLawEMA:
        from model.ddpm import DDPM

        opt = kwargs["opt"]
        opt.power_ema_apply = True
        model = DDPM(*args, **kwargs)
        print(f"Final DDPM power_law_ema_apply setting: {opt.power_ema_apply}")
        model.train_method = model.train
        print(
            "DDPM will be trained using the 'DDPM.train method' with Power Law EMA..."
        )

    # ========= Make Model Instance for all DDPM Classifier Free Guidance Variants =========

    # --- DDPM with CFG ---

    elif model_name.lower() == ModelNames.CFG_DDPM:
        from model.ddpm import DDPMCFG

        model = DDPMCFG(*args, **kwargs)
        model.train_method = model.cfg_train
        print("DDPM CFG will be trained using the 'DDPMCFG.cfg_train method'...")

    # --- DDPM with CFG and EMA ---

    elif model_name.lower() == ModelNames.CFG_DDPM_EMA:
        from model.ddpm import DDPMCFG

        opt = kwargs["opt"]
        opt.ema_apply = True
        model = DDPMCFG(*args, **kwargs)
        print(f"Final CFG DDPM ema_apply setting: {opt.ema_apply}")
        model.train_method = model.cfg_train
        print(
            "DDPM CFG will be trained using the 'DDPMCFG.cfg_train method' with EMA..."
        )

    # --- DDPM with CFG and Power Law EMA ---

    elif model_name.lower() == ModelNames.CFG_DDPM_PowerLawEMA:
        from model.ddpm import DDPMCFG

        opt = kwargs["opt"]
        opt.power_ema_apply = True
        model = DDPMCFG(*args, **kwargs)
        print(f"Final DDPM CFG power_law_ema_apply setting: {opt.power_ema_apply}")
        model.train_method = model.cfg_train
        print(
            "DDPM CFG will be trained using the 'DDPMCFG.cfg_train method' with Power Law EMA..."
        )

    # ========= Make Model Instance for all DDPM Classifier Free Guidance ++ Variants =========

    # --- DDPM with CFG ++ ---

    elif model_name.lower() == ModelNames.CFG_Plus_DDPM:
        from model.ddpm import DDPMCFG

        model = DDPMCFG(*args, **kwargs)
        model.train_method = model.cfg_plus_train
        print(
            "DDPM CFG ++ will be trained using the 'DDPMCFG.cfg_plus_train method'..."
        )

    # --- DDPM with CFG ++ and EMA ---

    elif model_name.lower() == ModelNames.CFG_Plus_DDPM_EMA:
        from model.ddpm import DDPMCFG

        opt = kwargs["opt"]
        opt.ema_apply = True
        model = DDPMCFG(*args, **kwargs)
        print(f"Final DDPM CFG ++ ema_apply setting: {opt.ema_apply}")
        model.train_method = model.cfg_plus_train
        print(
            "DDPM CFG ++ will be trained using the 'DDPMCFG.cfg_plus_train method' with EMA..."
        )

    # --- DDPM with CFG ++ and Power Law EMA ---

    elif model_name.lower() == ModelNames.CFG_Plus_DDPM_PowerLawEMA:
        from model.ddpm import DDPMCFG

        opt = kwargs["opt"]
        opt.power_ema_apply = True
        model = DDPMCFG(*args, **kwargs)
        print(f"Final DDPM CFG ++ power_law_ema_apply setting: {opt.power_ema_apply}")
        model.train_method = model.cfg_plus_train
        print(
            "DDPM CFG ++ will be trained using the 'DDPMCFG.cfg_plus_train method' with Power Law EMA..."
        )

    else:
        raise ValueError(f"Invalid model name: {model_name}")
    print(f"Model {model_name} was created")
    return model


def make_network(network_name: str, *args, **kwargs) -> torch.nn.Module:
    """
    Creates a network from the given network name

    Parameters
    ----------
    network_name: str
        The name of the network to create
    *args: list
        The arguments to pass to the network constructor
    **kwargs: dict
        The keyword arguments to pass to the network constructor

    Returns
    -------
    network: torch.nn.Module
        The created network
    """
    network = None

    # ------ Network instance for DDPM ------

    if network_name.lower() == NetworkNames.DDPM_Unet:
        from model.unet import UNet

        network = UNet(*args, **kwargs)

    # ------ Network instance for DDPM with CFG and CFG ++ ------

    elif network_name.lower() == NetworkNames.CFG_Unet:
        from model.unet import UnetCFG

        network = UnetCFG(*args, **kwargs)

    else:
        raise ValueError(f"Invalid network name: {network_name}")
    print(f"Network {network_name} was created")
    return network


def make_dataset(dataset_name: str, opt: argparse.Namespace, *args, **kwargs):
    """
    Creates a dataset from the given dataset name

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to create
    opt: argparse.Namespace
        The training options
    *args: list
        The arguments to pass to the dataset constructor
    **kwargs: dict
        The keyword arguments to pass to the dataset constructor

    Returns
    -------
    dataset: BaseDataset
        The created dataset
    """
    dataset = None
    if dataset_name.lower() == DatasetNames.MNIST:
        from data.mnist import MNISTDataset, MNISTTest

        train_dataset = MNISTDataset(opt, *args, **kwargs)
        test_dataset = MNISTTest(opt, *args, **kwargs)
        dataset = (train_dataset, test_dataset)

    elif dataset_name.lower() == DatasetNames.BIOLOGICAL:
        from data.topographies import BiologicalObservation

        train_dataset = BiologicalObservation(opt, *args, **kwargs)
        dataset = (train_dataset,)

    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    for d in dataset:
        make_dataloader(
            d,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
        )
        d.print_dataloader_info()

    print(f"Dataset {dataset_name} was created")
    return dataset


def make_dataloader(dataset: BaseDataset, *args, **kwargs) -> None:
    """
    Creates a dataloader from the given dataset

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        The dataset to create the dataloader from
    *args: list
        The arguments to pass to the dataloader constructor
    **kwargs: dict
        The keyword arguments to pass to the dataloader constructor

    Returns
    -------
    None
    """
    dataset.dataloader = torch.utils.data.DataLoader(dataset, *args, **kwargs)
