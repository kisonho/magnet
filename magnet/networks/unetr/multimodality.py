from torchmanager_core import torch
from torchmanager_core.view import warnings
from torchmanager_core.typing import Optional, Union, Sequence

from monai.networks.nets.unetr import UNETR


class UNETRWithMultiModality(UNETR):
    """
    A UNETR for multi-modality

    * extends `monai.networks.nets.unetr.UNETR`

    - Parameters:
        - copy_modality: An optional `bool` flag to copy the first modality to others or copy zeros to others, a `None` will result as not copying
    """

    copy_modality: Optional[bool]

    def __init__(self, in_channels: int, out_channels: int, img_size: Union[Sequence[int], int], feature_size: int = 16, hidden_size: int = 768, mlp_dim: int = 3072, num_heads: int = 12, pos_embed: str = "conv", norm_name: Union[tuple, str] = "instance", conv_block: bool = True, res_block: bool = True, dropout_rate: float = 0, spatial_dims: int = 3, copy_modality: Optional[bool] = None) -> None:
        super().__init__(in_channels, out_channels, img_size, feature_size, hidden_size, mlp_dim, num_heads, pos_embed, norm_name, conv_block, res_block, dropout_rate, spatial_dims)
        self.copy_modality = copy_modality

    def forward(self, x_in: torch.Tensor):
        if not self.training and self.copy_modality is not None:
            x_copy = x_in[:, :1] if self.copy_modality is True else torch.zeros_like(x_in[:, :1, ...])
            x_in = x_in[:, :1, ...]
            for _ in range(1, self.encoder1.layer.conv1.in_channels):
                x_in = torch.cat([x_in, x_copy], dim=1)
        return super().forward(x_in)


class UNETRWithDictOutput(UNETRWithMultiModality):
    """
    A UNETR that returns a dictionary during training

    * extends: `UNETRWithMultiModality`
    """

    def __init__(self, in_channels: int, out_channels: int, img_size: Union[Sequence[int], int], feature_size: int = 16, hidden_size: int = 768, mlp_dim: int = 3072, num_heads: int = 12, pos_embed: str = "conv", norm_name: Union[tuple, str] = "instance", conv_block: bool = True, res_block: bool = True, dropout_rate: float = 0, spatial_dims: int = 3, copy_modality: Optional[bool] = None) -> None:
        super().__init__(in_channels, out_channels, img_size, feature_size, hidden_size, mlp_dim, num_heads, pos_embed, norm_name, conv_block, res_block, dropout_rate, spatial_dims, copy_modality)
        warnings.warn("The class `UNETRWithDictOutput` has been deprecated from v1.1 and will be removed in v2.0", DeprecationWarning)

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        # repeat modality for input tensor
        return {"out": super().forward(x_in)} if self.training else super().forward(x_in)
