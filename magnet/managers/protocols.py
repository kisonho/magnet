from torchmanager_core import abc
from torchmanager_core.typing import Optional, Protocol, Union, runtime_checkable
from torchmanager_monai.protocols import SubResulting


@runtime_checkable
class Targeting(Protocol):
    """Targeting protocol"""

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

    @property
    @abc.abstractmethod
    def target_dict(self) -> dict[Optional[int], str]:
        return NotImplemented
