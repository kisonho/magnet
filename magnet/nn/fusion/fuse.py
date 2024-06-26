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


class MeanAttentionFusion(MeanFusion):
    """
    Fusion at the middle for encoders with features parsed through attention convolutions

    * extends: `MeanFusion`

    - Properties:
        - attention_convs: A `torch.nn.ModuleList` of attention convolutions
    """
    attention_convs: torch.nn.ModuleList

    def __init__(self, num_features: int, num_attention_features: list[int]) -> None:
        super().__init__()
        self.attention_convs = torch.nn.ModuleList([torch.nn.ModuleList([torch.nn.Conv3d(f, f, 1) for f in num_attention_features]) for _ in range(num_features)])

    def fuse(self, x_in: list[tuple[torch.Tensor, ...] | None]) -> tuple[torch.Tensor, ...]:
        # get number of features
        num_features = next((len(x) for x in x_in if x is not None), 0)

        # ensure number of features equal to number of attention convs
        assert num_features == len(self.attention_convs), "Number of features and attention convs must be equal."

        # initialize attentioned x_in
        attnetiond_x_in: list[tuple[torch.Tensor, ...] | None] = [None for _ in range(num_features)]

        # loop for each target
        for i, x in enumerate(x_in):
            # check if x is given
            if x is None:
                continue

            # apply attention
            target_convs = self.attention_convs[i]
            assert isinstance(target_convs, torch.nn.ModuleList), "Attention convs must be a ModuleList."
            attnetiond_x_in[i] = tuple([target_convs[j](f) for j, f in enumerate(x)])
        return super().fuse(x_in)


class MeanSingleFusion(Fusion):
    """
    Fusion at the middle for encoders with single output

    * extends: `Fusion`
    """
    def forward(self, x_in: list[torch.Tensor]) -> list[torch.Tensor]:
        return super().forward(x_in)

    def fuse(self, x_in: list[torch.Tensor]) -> torch.Tensor:
        # initialize
        assert len(x_in) > 0, "Fused inputs must have at least one target."
        x_to_fuse: list[torch.Tensor] = [x.unsqueeze(1) for x in x_in]
        y = torch.cat(x_to_fuse, dim=1).mean(1)
        return y


class MeanAttentionSingleFusion(MeanSingleFusion):
    """
    Fusion at the middle for encoders with single output parsed through attention convolutions

    * extends: `MeanSingleFusion`

    - Properties:
        - attention_convs: A `torch.nn.ModuleList` of attention convolutions
    """
    attention_convs: torch.nn.ModuleList

    def __init__(self, num_attention_features: int) -> None:
        super().__init__()
        self.attention_convs = torch.nn.ModuleList([torch.nn.Conv3d(num_attention_features, num_attention_features, 1)])

    def fuse(self, x_in: list[torch.Tensor]) -> torch.Tensor:
        # apply attention
        attentioned_target: list[torch.Tensor] = [self.attention_convs[0](f) for f in x_in]
        return super().fuse(attentioned_target)


MidFusion = MeanFusion
MidSingleFusion = MeanSingleFusion
