import torch

from .decoders import UNetDecoder
from .encoders import UNetEncoder


class UNet(torch.nn.Module):
    encoder: UNetEncoder
    decoder: UNetDecoder

    def __init__(self, in_channels: int, /, num_classes: int, *, basic_dims: int = 16, dim_mults: list[int] = [1, 2, 4, 8]) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels, basic_dims=basic_dims, dim_mults=dim_mults)
        self.decoder = UNetDecoder(num_classes, basic_dims=basic_dims, dim_mults=dim_mults)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        y: torch.Tensor = self.decoder(features)
        return y