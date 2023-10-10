from torchmanager_core import torch
from torchmanager_core.typing import Any, Iterable, Protocol


class DistillatedData(Protocol):
    """A distillated format protocol"""
    student: Any
    teacher: Any


class FeaturedData(Protocol):
    features: list[Iterable[torch.Tensor]]
    out: torch.Tensor
