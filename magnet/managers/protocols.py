from torchmanager_core import abc
from torchmanager_core.typing import Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class Targeting(Protocol):
    """Targeting protocol"""
    target_dict: dict[int, str]

    @property
    @abc.abstractmethod
    def num_targets(self) -> int:
        return NotImplemented

    @property
    @abc.abstractmethod
    def target(self) -> Optional[Union[list[int], int]]:
        return NotImplemented

    @target.setter
    @abc.abstractmethod
    def target(self, t: Optional[Union[list[int], int]]) -> None:
        pass

