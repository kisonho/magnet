import argparse, torch
from typing import Optional, Union

from .basic import Config as _Config

class Config(_Config):
    """
    Configurations for a single monai module
    
    - Properties:
        - batch_size: An `int` of batch size
        - data: A `str` of dataset directory
        - device: A `torch.device` to be trained on
        - epochs: An `int` of total training epochs
        - experiment: A `str` of experiment name
        - experiment_dir: A `str` of experiment directory
        - img_size: A tuple of the image size in `int`
        - output_model: A `str` of output model path
        - show_verbose: A `bool` flag of if showing training progress bar
        - training_split: An `int` of the split number of validation dataset during training
        - use_multi_gpus: A `bool` flag of if using multi GPUs
    """
    device: torch.device
    img_size: tuple[int, ...]
    modality: Optional[Union[int, list[int]]]
    show_verbose: bool

    def __init__(
        self, 
        data: str,
        model: str, 
        device: str = "cuda", 
        img_size: list[int] = [96, 96, 64],
        modality: Optional[list[int]] = None,
        show_verbose: bool = False
        ) -> None:

        """Constructor"""
        # initialize parameters
        super().__init__(data, model, device=device, show_verbose=show_verbose)
        if len(img_size) == 3: self.img_size = tuple(img_size)
        elif len(img_size) == 1: self.img_size = (img_size[0], img_size[0], img_size[0])
        else: raise ValueError(f'Image size must be in 3-dimension or 1-dimension, got length {len(img_size)}')
        if modality is not None:
            if len(modality) == 1: self.modality = modality[0]
            else: self.modality = modality
        else: self.modality = modality

        # assert properties
        for s in self.img_size: assert s > 0, f"Image size must be positive numbers, got {self.img_size}."

    def _show_settings(self) -> None:
        print(f"Dataset: path={self.data}, modality={self.modality}")
        print(f"Evaluation settings: img_size={self.img_size}")

    @staticmethod
    def get_parser(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        parser = _Config.get_parser(parser)
        testing_group = parser.add_argument_group("Testing Arguments")
        testing_group.add_argument("--img_size", type=int, nargs="+", default=[96, 96, 64], help="The image size, default is 96.")
        testing_group.add_argument("--modality", type=int, nargs="+", default=None, help="The target modality to load, default is None.")
        return parser
