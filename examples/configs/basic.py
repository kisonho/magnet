import abc, argparse, logging, monai, os, torch, torchmanager
from typing import Union

import data, magnet, nightly

class Config(abc.ABC):
    data: str
    device: Union[torch.device, list[torch.device]]
    model: str
    show_verbose: bool
    use_multi_gpus: bool

    def __init__(self, data: str, model: str, device: Union[str, list[str]] = "cuda", show_verbose: bool = True, use_multi_gpus: bool = False):
        self.data = os.path.normpath(data)
        if isinstance(device, list):
            self.device = [torch.device(d) for d in device]
        else:
            self.device = torch.device(device)
        self.model = os.path.normpath(model)
        self.show_verbose = show_verbose
        self.use_multi_gpus = use_multi_gpus if torch.cuda.is_available() else False

        # argument check
        assert torchmanager.version >= "1.0.4", f"Version v1.0.4+ is required for torchmanager, got {torchmanager.version}."

    @abc.abstractmethod
    def _show_settings(self) -> None: pass

    @staticmethod
    def get_parser(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        parser.add_argument("data", type=str, help="Data directory.")
        parser.add_argument("model", type=str, help="Directory for output model.")
        parser.add_argument("--device", type=str, nargs="+", default="cuda", help="The device that running with, default is 'cuda'.")
        parser.add_argument("--show_verbose", action="store_true", default=False, help="Flag to show progress bar during running.")
        parser.add_argument("--use_multi_gpus", action="store_true", default=False, help="Flag to use multi GPUs during running.")
        return parser

    def show_settings(self) -> None:
        self._show_settings()
        logger = logging.getLogger("torchmanager")
        logger.info(f"View settings: show_verbose={self.show_verbose}")
        logger.info(f"Device settings: device={self.device}, use_multi_gpus={self.use_multi_gpus}")
        logger.info(f"Environments: data={data.VERSION}, magnet={magnet.VERSION}, magnet-nightly={nightly.VERSION}, monai={monai.__version__}, torch={torch.__version__}, torchmanager={torchmanager.version}")
        logger.info("---------------------------------------")

    @classmethod
    def from_arguments(cls, parser: argparse.ArgumentParser = argparse.ArgumentParser()):
        parser = cls.get_parser()
        arguments = parser.parse_args().__dict__
        return cls(**arguments)
