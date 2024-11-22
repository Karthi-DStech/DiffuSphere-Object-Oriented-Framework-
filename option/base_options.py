import argparse
import ast
import os
import sys
from typing import Dict

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
            default="../Datasets/Topographies/raw/FiguresStacked 8X8_4X4_2X2 Embossed",
            help="path to the images",
        )
        self._parser.add_argument(
            "--label_path",
            type=str,
            required=False,
            default="../Datasets/biology_data/TopoChip/AeruginosaWithClass.csv",
            help="path to the label csv file",
        )

        self._parser.add_argument(
            "--dataset_name",
            type=str,
            required=False,
            default="biological",
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
            default=40000,
            help="number of epochs",
        )
        self._parser.add_argument(
            "--img_type", type=str, required=False, default="png", help="image type"
        )
        self._parser.add_argument(
            "--img_size",
            type=int,
            required=False,
            default=32,
            choices=[32, 64, 128, 256],
            help="image size",
        )

        self._parser.add_argument(
            "--in_channels",
            type=int,
            required=False,
            default=1,
            help="number of input channels",
        )
        self._parser.add_argument(
            "--out_channels",
            type=int,
            required=False,
            default=1,
            help="number of output channels",
        )
        self._parser.add_argument(
            "--batch_size", type=int, required=False, default=32, help="batch size"
        )
        self._parser.add_argument(
            "--num_workers",
            type=int,
            required=False,
            default=4,
            help="number of workers",
        )

        self._parser.add_argument(
            "--seed", type=int, required=False, default=101, help="random seed"
        )

        self._parser.add_argument(
            "--save_dir",
            type=str,
            default="./artifacts",
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
