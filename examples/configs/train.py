import argparse, logging, os, torchmanager, warnings
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
    batch_size: int
    epochs: int
    experiment: str
    img_size: tuple[int, ...]
    modality: Optional[Union[int, list[int]]]
    output_channels: Optional[int]
    pretrained_model: Optional[str]
    training_split: int
    
    @property
    def experiment_dir(self) -> str:
        return os.path.join("experiments", self.experiment)

    @property
    def output_model(self) -> str: return self.model

    def __init__(
        self, 
        data: str,
        model: str, 
        batch_size: int = 1, 
        device: str = "cuda", 
        epochs: int = 600, 
        experiment: str = "test.exp", 
        img_size: list[int] = [96, 96, 64],
        modality: Optional[list[int]] = None,
        pretrained_model: Optional[str] = None,
        show_verbose: bool = False,
        training_split: int = 4,
        use_multi_gpus: bool = False,
        out_channels: Optional[int] = None
        ) -> None:

        """Constructor"""
        # initialize parameters
        super().__init__(data, model, device=device, show_verbose=show_verbose, use_multi_gpus=use_multi_gpus)
        self.batch_size = batch_size
        self.epochs = epochs
        self.experiment = experiment if experiment.endswith(".exp") else f'{experiment}.exp'
        if len(img_size) == 3: self.img_size = tuple(img_size)
        elif len(img_size) == 1: self.img_size = (img_size[0], img_size[0], img_size[0])
        else: raise ValueError(f'Image size must be in 3-dimension or 1-dimension, got length {len(img_size)}')
        if modality is not None:
            if len(modality) == 1: self.modality = modality[0]
            else: self.modality = modality
        else: self.modality = modality
        self.pretrained_model = os.path.normpath(pretrained_model) if pretrained_model is not None else None
        self.training_split = training_split
        self.out_channels = out_channels

        # initialize log
        os.makedirs(self.experiment_dir, exist_ok=True)
        log_file = os.path.basename(self.experiment.replace(".exp", ".log"))
        log_path = os.path.join(self.experiment_dir, log_file)
        logging.basicConfig(level=logging.INFO, filename=log_path, format="%(message)s")
        warnings.filterwarnings("ignore")
        if self.show_verbose:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            console.setFormatter(formatter)
            logging.getLogger().addHandler(console)

        # assert properties
        assert torchmanager.version >= "1.0.4", f"Version v1.0.4+ is required for torchmanager, got {torchmanager.version}."
        assert self.batch_size > 0, f"Batch size must be positive, got {self.batch_size}."
        assert self.epochs > 0, f"Epochs must be positive, got {self.epochs}."
        for s in self.img_size: assert s > 0, f"Image size must be positive numbers, got {self.img_size}."

    def _show_settings(self) -> None:
        logging.info(f"Experiment {self.experiment}: pretrained_model={self.pretrained_model}, output_model_path={self.output_model}")
        logging.info(f"Dataset: path={self.data}, modality={self.modality}")
        logging.info(f"Training settings: epochs={self.epochs}, batch_size={self.batch_size}, img_size={self.img_size}")

    @staticmethod
    def get_parser(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:        
        # add training arguments
        parser = _Config.get_parser(parser)
        training_group = parser.add_argument_group("Training Arguments")
        training_group.add_argument("-b", "--batch_size", type=int, default=1, help="Training batch size, default is 1.")
        training_group.add_argument("-e", "--epochs", type=int, default=600, help="Training epochs, default is 600.")
        training_group.add_argument("-exp", "--experiment", type=str, default="test.exp", help="Name of the experiment, default is 'test.exp'.")
        training_group.add_argument("--img_size", type=int, nargs="+", default=[96, 96, 64], help="The image size, default is 96.")
        training_group.add_argument("--modality", type=int, nargs="+", default=None, help="The target modality to load, default is None.")
        training_group.add_argument("-pre", "--pretrained_model", type=str, default=None, help="Pretrained model path, default is None.")
        training_group.add_argument("-ts", "--training_split", type=int, default=4, help="The index to split the training data, creating a validation set, default is 4")
        training_group.add_argument("--out_channels", type=int, default=None, help="Defines the number of out_channels, over ruling the automatic detection")
        return parser
