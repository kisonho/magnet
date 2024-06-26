from monai.networks.nets.unetr import UNETR
from torchmanager_core import torch
from torchmanager_core.view import warnings
from torchmanager_core.typing import Sequence


class UNETRWithMultiModality(UNETR):
    """
    A UNETR for multi-modality

    * extends `monai.networks.nets.unetr.UNETR`

    - Parameters:
        - copy_modality: An optional `bool` flag to copy the first modality to others or copy zeros to others, a `None` will result as not copying
        - num_targets: The total number of ta
    """

    copy_modality: bool | None
    target_dict: dict[int, str]

    @property
    def num_targets(self) -> int:
        return self.encoder1.layer.conv1.in_channels

    def __init__(self, in_channels: int, out_channels: int, img_size: Sequence[int] | int, feature_size: int = 16, hidden_size: int = 768, mlp_dim: int = 3072, num_heads: int = 12, pos_embed: str = "conv", norm_name: tuple[str, ...] | str = "instance", conv_block: bool = True, res_block: bool = True, dropout_rate: float = 0, spatial_dims: int = 3, copy_modality: bool | None = None, target_dict: dict[int, str] = {}) -> None:
        super().__init__(in_channels, out_channels, img_size, feature_size, hidden_size, mlp_dim, num_heads, pos_embed, norm_name, conv_block, res_block, dropout_rate, spatial_dims)
        self.copy_modality = copy_modality
        self.target_dict = target_dict

    def forward(self, x_in: torch.Tensor):
        if not self.training and self.copy_modality is not None:
            # copy modalities
            x_copy = x_in.mean(dim=1).unsqueeze(1) if self.copy_modality is True else torch.zeros_like(x_in[:, :1, ...])
            x: list[torch.Tensor] = []

            # handle missing modalities
            for i in range(self.num_targets):
                if i not in self.target_dict:
                    x.append(x_copy)
                else:
                    x.append(x_in[:, i, ...].unsqueeze(1))

            # concat
            x_in = torch.cat(x, dim=1)
        return super().forward(x_in)


class UNETRWithDictOutput(UNETRWithMultiModality):
    """
    A UNETR that returns a dictionary during training

    * extends: `UNETRWithMultiModality`
    """

    def __init__(self, in_channels: int, out_channels: int, img_size: Sequence[int] | int, feature_size: int = 16, hidden_size: int = 768, mlp_dim: int = 3072, num_heads: int = 12, pos_embed: str = "conv", norm_name: tuple[str, ...] | str = "instance", conv_block: bool = True, res_block: bool = True, dropout_rate: float = 0, spatial_dims: int = 3, copy_modality: bool | None = None, target_dict: dict[int, str] = {}) -> None:
        super().__init__(in_channels, out_channels, img_size, feature_size, hidden_size, mlp_dim, num_heads, pos_embed, norm_name, conv_block, res_block, dropout_rate, spatial_dims, copy_modality, target_dict)
        warnings.warn("The class `UNETRWithDictOutput` has been deprecated from v1.1 and will be removed in v2.0", DeprecationWarning)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        # repeat modality for input tensor
        return {"out": super().forward(x_in)} if self.training else super().forward(x_in)
