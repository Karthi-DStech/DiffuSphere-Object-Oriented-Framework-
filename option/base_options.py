import argparse
import ast
import os
import sys
from typing import Dict
from option.enums import DatasetNames
from option.config import BaseOptionsConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseOptions:

    def __init__(self):

        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):

        self._parser.add_argument(
            "--images_folder",
            type=str,
            required=False,
            default=BaseOptionsConfig.IMAGES_FOLDER,
            help="path to the images",
        )
        self._parser.add_argument(
            "--label_path",
            type=str,
            required=False,
            default=BaseOptionsConfig.LABEL_PATH,
            help="path to the label csv file",
        )

        self._parser.add_argument(
            "--dataset_name",
            type=str,
            required=False,
            default=DatasetNames.BIOLOGICAL,
            help="dataset name",
            choices=["mnist", "biological"],
        )

        self._parser.add_argument(
            "--dataset_params",
            type=lambda x: ast.literal_eval(x),
            required=False,
            default={"mean": 0.5, "std": 0.5},
            help="mean and standard deviation of the dataset for normalisation",
        )

        self._parser.add_argument(
            "--n_epochs",
            type=int,
            required=False,
            default=BaseOptionsConfig.N_EPOCHS,
            help="number of epochs",
        )
        self._parser.add_argument(
            "--img_type",
            type=str,
            required=False,
            default=BaseOptionsConfig.IMG_TYPE,
            help="image type",
        )
        self._parser.add_argument(
            "--img_size",
            type=int,
            required=False,
            default=BaseOptionsConfig.IMG_SIZE,
            choices=[32, 64, 128, 256],
            help="image size",
        )

        self._parser.add_argument(
            "--in_channels",
            type=int,
            required=False,
            default=BaseOptionsConfig.IN_CHANNELS,
            help="number of input channels",
        )
        self._parser.add_argument(
            "--out_channels",
            type=int,
            required=False,
            default=BaseOptionsConfig.OUT_CHANNELS,
            help="number of output channels",
        )
        self._parser.add_argument(
            "--batch_size",
            type=int,
            required=False,
            default=BaseOptionsConfig.BATCH_SIZE,
            help="batch size",
        )
        self._parser.add_argument(
            "--num_workers",
            type=int,
            required=False,
            default=BaseOptionsConfig.NUM_WORKERS,
            help="number of workers",
        )

        self._parser.add_argument(
            "--seed",
            type=int,
            required=False,
            default=BaseOptionsConfig.SEED,
            help="random seed",
        )

        self._parser.add_argument(
            "--save_dir",
            type=str,
            default=BaseOptionsConfig.SAVE_DIR,
            help="Path to save the artifacts of the model",
        )

        self._initialized = True

    def parse(self) -> argparse.Namespace:

        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()
        self._opt.is_train = self._is_train

        args = vars(self._opt)
        self._print(args)

        return self._opt

    def _print(self, args: Dict) -> None:
        """
        Prints the arguments passed to the script

        Parameters
        ----------
        args: dict
            The arguments to print

        Returns
        -------
        None
        """
        print("------------ Options -------------")
        for k, v in args.items():
            print(f"{str(k)}: {str(v)}")
        print("-------------- End ---------------")
