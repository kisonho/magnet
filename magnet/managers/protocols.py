from torchmanager_core import abc
from torchmanager_core.typing import Protocol, runtime_checkable


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
    def target(self) -> list[int] | int | None:
        return NotImplemented

    @target.setter
    @abc.abstractmethod
    def target(self, t: list[int] | int | None) -> None:
        pass

