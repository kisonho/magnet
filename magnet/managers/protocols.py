from torchmanager_core import abc, torch
from torchmanager_core.typing import Protocol, runtime_checkable

@runtime_checkable
class SubResulting(Protocol):
    '''Protocol that contains sub-results'''
    @property
    @abc.abstractmethod
    def sub_results(self) -> torch.Tensor:
        return NotImplemented

@runtime_checkable
class Targeting(Protocol):
    '''Targeting protocol'''
    @property
    @abc.abstractmethod
    def num_targets(self) -> int:
        return NotImplemented

    @property
    @abc.abstractmethod
    def target(self) -> int:
        return NotImplemented

    @target.setter
    @abc.abstractmethod
    def target(self, t: int) -> None:
        pass

    @property
    @abc.abstractmethod
    def target_dict(self) -> dict[int, str]:
        return NotImplemented