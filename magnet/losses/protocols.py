from torchmanager_core import torch
from torchmanager_core.typing import Iterable, Protocol


class FeaturedData(Protocol):
    features: list[Iterable[torch.Tensor]]
    out: torch.Tensor
