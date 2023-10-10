import argparse, logging
from typing import Any

from .train import Config as _Config

class Config(_Config):
    patch_size: int

    def __init__(self, *args: Any, patch_size: int = 16, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        assert self.patch_size > 0, f"[Argument Error]: The patch_size must be a positive number, got {self.patch_size}."

    @staticmethod
    def get_parser(parser: argparse.ArgumentParser = ...) -> argparse.ArgumentParser:
        parser = _Config.get_parser(parser)
        training_group = parser.add_argument_group("QuilTR Arguments")
        training_group.add_argument("-p", "--patch_size", type=int, default=16, help="The transformer patch size, default is 16.")
        return parser

    def _show_settings(self) -> None:
        super()._show_settings()
        logging.info(f"QuilTR settings: patch_size={self.patch_size}")