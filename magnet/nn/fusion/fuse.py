from torchmanager_core import abc, torch
from torchmanager_core.typing import Any


class Fusion(torch.nn.Module, abc.ABC):
    """
    Abstract fusion block

    * extends: `torch.nn.Module`
    * Abstract class
    
    - Methods to implement:
        - fuse: The fuse method that fuse a `list` of inputs to a single one
    """
    def forward(self, x_in: list[Any]) -> list[Any]:
        # fuse
        y = self.fuse(x_in)

        # return results
        return [y] + x_in if self.training else [y]

    @abc.abstractmethod
    def fuse(self, x_in: list[Any]) -> Any:
        """
        Fuse the input to a single one

        - Parameters:
            - x_in: A `list` of inputs with multi-modalities
        - Returns: A fused output
        """
        return NotImplemented


class MeanFusion(Fusion):
    """
    Fusion at the middle for encoders with multiple outputs
    
    * extends: `Fusion`
    """
    def forward(self, x_in: list[tuple[torch.Tensor, ...] | None]) -> list[tuple[torch.Tensor, ...]]:
        return super().forward(x_in)
    
    def fuse(self, x_in: list[tuple[torch.Tensor, ...] | None]) -> tuple[torch.Tensor, ...]:
        # initialize
        assert len(x_in) > 0, "Fused inputs must have at least one target."

        # initialize x to fuse
        num_features = next((len(x) for x in x_in if x is not None), 0)
        x_to_fuse: list[list[torch.Tensor]] = [[] for _ in range(num_features)]

        # loop for each target
        for x in x_in:
            # check if x is given
            if x is None:
                continue

            # loop each feature
            for i, f in enumerate(x):
                x_to_fuse[i].append(f.unsqueeze(1))

        # mean fusion
        y: tuple[torch.Tensor, ...] = tuple([torch.cat(x, dim=1).mean(1) for x in x_to_fuse])
        return y


class MeanSingleFusion(Fusion):
    """
    Fusion at the middle for encoders with single output
    """
    def forward(self, x_in: list[torch.Tensor]) -> list[torch.Tensor]:
        return super().forward(x_in)

    def fuse(self, x_in: list[torch.Tensor]) -> torch.Tensor:
        # initialize
        assert len(x_in) > 0, "Fused inputs must have at least one target."
        x_to_fuse: list[torch.Tensor] = [x.unsqueeze(1) for x in x_in]
        y = torch.cat(x_to_fuse, dim=1).mean(1)
        return y


MidFusion = MeanFusion
MidSingleFusion = MeanSingleFusion
