import argparse
from typing import Union

import torch

from data.datasets import BaseDataset
from model.models import BaseModel


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

    if model_name.lower() == "ddpm_35m":
        from model.ddpm import DDPM

        model = DDPM(*args, **kwargs)
        model.train_method = model.train
        print("DDPM will be trained using the 'DDPM.train method'...")

    elif model_name.lower() == "cfg_ddpm":
        from model.ddpm import DDPMCFG

        model = DDPMCFG(*args, **kwargs)
        model.train_method = model.cfg_train
        print("DDPM will be trained using the 'DDPMCFG.cfg_train method'...")

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

    if network_name.lower() == "ddpm_unet":
        from model.unet import UNet

        network = UNet(*args, **kwargs)

    elif network_name.lower() == "cfg_unet":
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
    if dataset_name.lower() == "mnist":
        from data.mnist import MNISTDataset, MNISTTest

        train_dataset = MNISTDataset(opt, *args, **kwargs)
        test_dataset = MNISTTest(opt, *args, **kwargs)
        dataset = (train_dataset, test_dataset)

    elif dataset_name.lower() == "biological":
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
