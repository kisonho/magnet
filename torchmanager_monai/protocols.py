from torchmanager_core import abc, torch
from torchmanager_core.typing import Protocol, runtime_checkable

@runtime_checkable
class SubResulting(Protocol):
    """Protocol that contains sub-results"""
    @property
    @abc.abstractmethod
    def sub_results(self) -> torch.Tensor:
        return NotImplemented