import argparse, logging, os
from torchmanager.callbacks import Frequency
from typing import Any, Optional

from .train import Config as _Config

class Config(_Config):
    """
    Configurations for shared modules
    
    - Properties:
        - distillation_gamma: The gamma factor in `float` for self distillation
        - distillation_lambda: The lambda factor in `float` for self distillation
        - target_frequency: An optional target update `Frequency`
    """
    distillation_gamma: float
    distillation_lambda: float
    linear_scheduled_gamma: bool
    linear_scheduled_lambda: bool
    target_frequency: Optional[Frequency]
    temperature: float
    
    @property
    def experiment_dir(self) -> str:
        return os.path.join("experiments", self.experiment)

    def __init__(self, *args: Any, distillation_lambda: float = 1, linear_scheduled_lambda: bool = False, distillation_gamma: float = 1, linear_scheduled_gamma: bool = False, target_frequency: Optional[str] = None, temperature: float = 1, **kwargs: Any) -> None:
        """Constructor"""
        # initialize parameters
        super().__init__(*args, **kwargs)
        self.distillation_gamma = distillation_gamma
        self.distillation_lambda = distillation_lambda
        self.linear_scheduled_gamma = linear_scheduled_gamma
        self.linear_scheduled_lambda = linear_scheduled_lambda
        self.target_frequency = Frequency[target_frequency.upper()] if target_frequency is not None else target_frequency
        self.temperature = temperature

        assert self.distillation_lambda > 0, f"Distillation lambda must be a positive number, got {self.distillation_lambda}."
        assert self.temperature > 0, f"Temperature must be a positive number, got {self.temperature}."

    @staticmethod
    def get_parser(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        # add sharing arguments
        parser = _Config.get_parser(parser)
        sharing_group = parser.add_argument_group("Sharing Arguments")
        sharing_group.add_argument("-lambda", "--distillation_lambda", type=float, default=1, help="The lambda factor for self distillation, default is 1.")
        sharing_group.add_argument("--linear_scheduled_lambda", action="store_true", default=False, help="Boolean flag of if using linear schedule for lambda.")
        sharing_group.add_argument("-gamma", "--distillation_gamma", type=float, default=1, help="The gamma factor for self distillation, default is 1.")
        sharing_group.add_argument("--linear_scheduled_gamma", action="store_true", default=False, help="Boolean flag of if using linear schedule for gamma.")
        sharing_group.add_argument("--target_frequency", type=str, default=None, help="The frequency for target training loop, can be 'None', 'batch', or 'epoch', default is 'None'.")
        sharing_group.add_argument("-t", "--temperature", type=float, default=1, help="The softmax temperature for knowledge distillation KL-Div loss, default is 1.")
        return parser

    def _show_settings(self) -> None:
        super()._show_settings()
        logger = logging.getLogger("torchmanager")
        logger.info(f"Sharing settings: gamma={self.distillation_gamma}, linear_scheduled_gamma={self.linear_scheduled_gamma}, lambda={self.distillation_lambda}, linear_scheduled_lambda={self.linear_scheduled_lambda}, target_frequency={self.target_frequency.name if self.target_frequency is not None else self.target_frequency}, temperature={self.temperature}")
