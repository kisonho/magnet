from torchmanager_core import abc
from torchmanager_core.typing import Any, Optional, Self, Sized
from torch.utils.data import Dataset

try:
    from monai.data.dataloader import DataLoader
except:
    from torch.utils.data import DataLoader


class MultiDataset(Dataset, abc.ABC):
    """
    An abstract dataset class with multiple datasets combined

    * Abstract class
    * extends: `torch.utils.data.Dataset`

    - Properties:
        - datasets: A `list` of multiple datasets in `Dataset`
    """

    __datasets: list[Dataset[Any]]

    @property
    def datasets(self) -> list[Dataset[Any]]:
        return self.__datasets

    def __init__(self, *datasets: Dataset[Any]) -> None:
        """
        - Parameters:
            - datasets: Multiple target datasets in `Dataset`
        """
        super().__init__()
        self.__datasets = list(datasets)

    def __add__(self, other: Dataset[Any]) -> Self:
        self.datasets.append(other)
        return self

    @abc.abstractmethod
    def __getitem__(self, index: Any) -> Any:
        return NotImplemented

    @abc.abstractmethod
    def __len__(self) -> int:
        return NotImplemented


class EquivalentDataset(MultiDataset):
    """
    TODO A dataset with equivalent amount of data in multiple datasets

    * extends: `MultiDataset`
    """

    def __init__(self, *datasets: Dataset[Any]) -> None:
        """
        - Parameters:
            - datasets: Multiple target datasets in `Dataset`
        """
        super().__init__(*datasets)

        # assert length
        if not isinstance(datasets[0], Sized):
            raise TypeError("There are datasets that do not confirm to `Sized` protocol given.")
        length = len(datasets[0])
        for d in self.datasets:
            if not isinstance(d, Sized):
                raise TypeError("There are datasets that do not confirm to `Sized` protocol given.")
            assert length == len(d), "The given datasets must contain equivalent size."

    def __add__(self, other: Dataset[Any]) -> Self:
        # check type
        if not isinstance(other, Sized):
            raise TypeError("The dataset does not confirm to `Sized` protocol given.")

        # check length
        if len(self.datasets) > 0:
            if not isinstance(self.datasets[0], Sized):
                raise TypeError("One of the dataset in current datasets does not confirm to `Sized` protocol.")
            assert len(self.datasets[0]) == len(other), "The given datasets must contain equivalent size with other datasets."

        # append
        super().__add__(other)
        return self

    def __getitem__(self, index: Any) -> Any:
        return [d[index] for d in self.datasets]

    def __len__(self) -> int:
        d = self.datasets[0]
        if not isinstance(d, Sized):
            raise TypeError("There are datasets that do not confirm to `Sized` protocol given.")
        return len(d)


class TargetedDataset(MultiDataset):
    """
    A dataset for mixed targets of datasets

    * extends: `MultiDataset`
    * implements: `managers.protocols.Targeting`

    - Properties:
        - datasets: A `list` of multiple datasets
        - target: An `int` of target dataset index in `datasets` list
    """

    __target: Optional[int]
    __target_dict: dict[int, str]

    @property
    def target(self) -> Optional[int]:
        return self.__target

    @target.setter
    def target(self, t: Optional[int]) -> None:
        if t is not None and (abs(t) >= len(self.datasets)) and t != 0:
            raise IndexError(f"Target {t} out of datasets range ({len(self.datasets)}).")
        self.__target = t

    @property
    def target_dict(self) -> dict[int, str]:
        assert len(self.datasets) == len(self.__target_dict), "The length of datasets does not equal to the dictionary."
        return self.__target_dict

    def __init__(self, *datasets: Dataset[Any], default_target: int = 0, target_dict: dict[int, str]) -> None:
        """
        - Parameters:
            - *datasets: Multiple target datasets in `Dataset`
            - default_target: An `int` of the default target modality
            - target_dict: A `dict` of the target modality as key in `int` and modality name as value in `str`
        """
        super().__init__(*datasets)
        self.target = default_target
        self.__target_dict = target_dict

    def __getitem__(self, index: Any) -> Any:
        if self.target is not None:
            return self.datasets[self.target][index]
        else:
            raise NotImplementedError("Returns data from all targets is not supported for `data.TargetingDataset`, use original `Dataset` class instead.")

    def __len__(self) -> int:
        d = self.datasets[self.target] if self.target is not None else self.datasets
        if not isinstance(d, Sized):
            raise TypeError(f"Dataset with target index {self.target} does not perform to Sized protocol")
        return len(d)


class TargetedDataLoader(DataLoader):  # type: ignore
    """
    The dataloader for `MixedDataset`

    * The `dataset` property must be a `MixedDataset`

    - Properties:
        - num_targets: An `int` of total target number
        - target: An `int` of target dataset index in the mixed dataset
        - target_dict: A `dict` of target index (keys in `int`) and name mapping (values in `str`)
    """

    dataset: TargetedDataset

    @property
    def num_targets(self) -> int:
        return len(self.dataset.datasets)

    @property
    def target(self) -> Optional[int]:
        return self.dataset.target

    @target.setter
    def target(self, t: Optional[int]) -> None:
        self.dataset.target = t

    @property
    def target_dict(self) -> dict[int, str]:
        return self.dataset.target_dict

    def __init__(self, dataset: TargetedDataset, *args: Any, **kwargs: Any):
        super().__init__(dataset, *args, **kwargs)
