import torchmanager as tm
from torchmanager.data import Dataset, DataLoader
from torchmanager.callbacks import Frequency
from torchmanager_core import torch
from torchmanager_core.typing import Any, Collection, Module

from .protocols import Targeting


class Manager(tm.Manager[Module]):
    """
    A training manager that trains with multiple mixed dataset

    * extends: `tm.Manager`
    * Compatibility: The `backward` method in this class that clips gradients will only be available in torchmanager with version higher than 1.2

    - Properties:
        - target: An `int` of targeted modality index
        - target_freq: A `torchmanager.train.LrScheduleFreq` of the target updating frequency
        - target_dict: A `dict` of available targets with the indices in `int` as keys and the names in `str` as values
    """

    __freq: Frequency | None
    __target: list[int] | int | None

    @property
    def target(self) -> list[int] | int | None:
        """The targeted modality index"""
        return self.__target

    @target.setter
    def target(self, t: list[int] | int | None) -> None:
        self.__target = t
        if isinstance(self.raw_model, Targeting):
            self.raw_model.target = t

    @property
    def target_freq(self) -> Frequency | None:
        """The target training frequency"""
        return self.__freq

    @property
    def target_dict(self) -> dict[int, str]:
        if isinstance(self.raw_model, Targeting):
            return self.raw_model.target_dict
        else:
            return {0: "all"}

    @target_dict.setter
    def target_dict(self, target_dict: dict[int, str]) -> None:
        if isinstance(self.raw_model, Targeting):
            self.raw_model.target_dict = target_dict

    def __init__(self, model: Module, optimizer: torch.optim.Optimizer | None = None, loss_fn: tm.losses.Loss | dict[str, tm.losses.Loss] | None = None, metrics: dict[str, tm.metrics.Metric] = {}, target_freq: Frequency | None = None) -> None:
        """
        Constructor

        - Parameters:
            - target_freq: The update training `Frequency`
        """
        super().__init__(model, optimizer, loss_fn, metrics)
        self.__target = None
        self.__freq = target_freq

    def _train(self, dataset: DataLoader[Any] | Dataset[Any], show_verbose: bool = False, **kwargs: Any) -> dict[str, float]:
        """
        The single training step for an epoch

        - Parameters:
            - dataset: A `SizedIterable` training dataset
            - show_verbose: A `bool` flag of if showing progress bar
        - Returns: A summary of `dict` with keys as `str` and values as `float`
        """
        # initialize summary
        summary: dict[str, Any] = {}

        # loop for targets
        if isinstance(dataset, Targeting) and self.target_freq == Frequency.EPOCH:
            for t in range(dataset.num_targets):
                # fetch dataset target
                if show_verbose:
                    print(f"Training target {t} (Dataset {dataset.target_dict[t]})...")
                dataset.target = t
                self.target = t

                # train model
                try:
                    subsummary = super()._train(dataset, show_verbose=show_verbose, **kwargs)
                except Exception as e:
                    raise RuntimeError(f"Training dataset '{dataset.target_dict[t]}' failed") from e
                for k, v in subsummary.items():
                    summary[f"{k}_{dataset.target_dict[t]}"] = v
            return summary
        elif self.target_freq == Frequency.EPOCH:
            for t in self.target_dict.keys():
                # fetch target
                self.target = t
                if show_verbose:
                    print(f"Training target {t} (Dataset {self.target_dict[t]})...")

                # train model
                subsummary = super()._train(dataset, show_verbose=show_verbose, **kwargs)
                for k, v in subsummary.items():
                    summary[f"{k}_{self.target_dict[t]}"] = v

            # reset target
            self.target = None
            return summary
        else:
            return super()._train(dataset, show_verbose=show_verbose, **kwargs)

    @torch.no_grad()
    def test(self, dataset: DataLoader[Any] | Dataset[Any] | Collection[Any] | dict[int | None, DataLoader[Any] | Dataset[Any] | Collection[Any]], show_verbose: bool = False, **kwargs: Any) -> dict[str, float]:
        # initialize
        summary: dict[str, float] = {}

        # check if dataset is dictionary
        if isinstance(dataset, dict):
            # test datasets in one epoch
            for modality, d in dataset.items():
                # fetch modality name
                name = self.target_dict[modality] if modality is not None else "all"

                # set target
                if show_verbose:
                    print(f"Target {modality} (Dataset {name}):")
                self.target = modality

                # test dataset
                try:
                    subsummary = super().test(d, show_verbose=show_verbose, **kwargs)
                except Exception as e:
                    raise RuntimeError(f"Testing dataset '{name}' failed") from e
                for k, v in subsummary.items():
                    summary[f"{k}_{name}"] = v

            # reset loss fn
            return summary
        else:
            return super().test(dataset, show_verbose=show_verbose, **kwargs)

    def train_step(self, x_train: torch.Tensor, y_train: Any) -> dict[str, float]:
        if self.target_freq == Frequency.BATCH:
            # initialize summary
            summary = {}

            # loop for each target
            for t in range(x_train.shape[1]):
                self.target = t
                subsummary = super().train_step(x_train, y_train)
                summary.update(subsummary)
            return summary
        else:
            return super().train_step(x_train, y_train)
