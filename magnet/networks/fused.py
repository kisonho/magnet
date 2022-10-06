import torch
from typing import Optional, Sequence, Union

from .encoders import UNETREncoder
from .decoders import UNETRDecoder

class FusedUNETR(torch.nn.Module):
    """
    UNETR with Fused architecture

    - Properties:
        - copy_modality: An optional `bool` flag of if copying or using zeros for the remaining modalities during testing
        - decoder: A `UNETRDecoder`
    """
    copy_modality: Optional[bool]
    decoder: UNETRDecoder
    fuse_dropout: float
    encoders: torch.nn.ModuleList
    enc1_conv: torch.nn.ModuleList
    enc2_conv: torch.nn.ModuleList
    enc3_conv: torch.nn.ModuleList
    enc4_conv: torch.nn.ModuleList
    dec4_conv: torch.nn.ModuleList
    target_modality: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        copy_modality: Optional[bool] = None,
        fuse_dropout: float = 1,
        target_modality: int = 0,
    ) -> None:
        super().__init__()

        # assert parameters
        assert in_channels > 0, f"[Parameters Error]: Input channels must be a positive number, got {in_channels}."
        assert out_channels > 0, f"[Parameters Error]: Input channels must be a positive number, got {out_channels}."
        assert target_modality < in_channels, f"[Parameters Error]: Target modality should be in range [0, {in_channels}), got {target_modality}."
        assert fuse_dropout > 0 and fuse_dropout <= 1, f"[Parameters Error]: The fuse dropout ratio should be in range (0, 1], got {fuse_dropout}."
        if not isinstance(img_size, Sequence): img_size = (img_size, img_size, img_size)
        
        # multi modality parameters
        self.copy_modality = copy_modality
        self.fuse_dropout = fuse_dropout
        self.target_modality = target_modality

        # initial encoders
        self.encoders = torch.nn.ModuleList([])
        self.enc1_conv = torch.nn.ModuleList([torch.nn.Conv3d(feature_size, feature_size, (1, 1, 1)) for _ in range(in_channels)])
        self.enc2_conv = torch.nn.ModuleList([torch.nn.Conv3d(feature_size * 2, feature_size * 2, (1, 1, 1)) for _ in range(in_channels)])
        self.enc3_conv = torch.nn.ModuleList([torch.nn.Conv3d(feature_size * 4, feature_size * 4, (1, 1, 1)) for _ in range(in_channels)])
        self.enc4_conv = torch.nn.ModuleList([torch.nn.Conv3d(feature_size * 8, feature_size * 8, (1, 1, 1)) for _ in range(in_channels)])
        self.dec4_conv = torch.nn.ModuleList([torch.nn.Conv3d(hidden_size, hidden_size, (1, 1, 1)) for _ in range(in_channels)])

        # loop for each channel
        for _ in range(in_channels):
            # encoder
            encoder = UNETREncoder(1, img_size, feature_size=feature_size, hidden_size=hidden_size, mlp_dim=mlp_dim, num_heads=num_heads, pos_embed=pos_embed, norm_name=norm_name, conv_block=conv_block, res_block=res_block, dropout_rate=dropout_rate, spatial_dims=spatial_dims)
            self.encoders.append(encoder)

        # initial decoder
        self.decoder = UNETRDecoder(hidden_size, out_channels, feature_size=feature_size, norm_name=norm_name, res_block=res_block, spatial_dims=spatial_dims)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # copy modality
        if self.copy_modality is None and x_in.shape[1] > 1:
            assert x_in.shape[1] == len(self.encoders), "Number of modality (input channels) does not equal to number of encoders."
        elif self.copy_modality is None and x_in.shape[1] == 1:
            if len(self.encoders) < 1: raise RuntimeError("No encoders found in list.")
            enc1, enc2, enc3, enc4, dec4 = self.encoders[self.target_modality](x_in)
            num_modalities = len(self.encoders)
            y = self.decoder(enc1 * num_modalities, enc2 * num_modalities, enc3 * num_modalities, enc4 * num_modalities, dec4 * num_modalities)
            return y
        else:
            x_copy = x_in[:, self.target_modality:self.target_modality+1] if self.copy_modality is True else torch.zeros_like(x_in[:, self.target_modality:self.target_modality+1, ...])
            x_in = x_in[:, self.target_modality:self.target_modality+1, ...]
            for i in range(0, self.target_modality): x_in = torch.concat([x_copy, x_in], dim=1)
            for i in range(self.target_modality, len(self.encoders)): x_in = torch.concat([x_in, x_copy], dim=1)

        # initialize output of encoders
        enc1s = 0
        enc2s = 0
        enc3s = 0
        enc4s = 0
        dec4s = 0
        forwarded_encoder = False

        # forward encoders
        for i, encoder in enumerate(self.encoders):
            if (i == len(self.encoders) - 1 and not forwarded_encoder) or float(torch.randn(())) <= self.fuse_dropout or not self.training:
                x = x_in [:, i:i+1, ...]
                assert isinstance(encoder, UNETREncoder), "[Runtime Error]: The encoder is not a valid `UNETREncoder`."
                enc1, enc2, enc3, enc4, dec4 = encoder(x)
                enc1s += self.enc1_conv[i](enc1)
                enc2s += self.enc2_conv[i](enc2)
                enc3s += self.enc3_conv[i](enc3)
                enc4s += self.enc4_conv[i](enc4)
                dec4s += self.dec4_conv[i](dec4)
                forwarded_encoder = True

        # forward decoder
        y = self.decoder(enc1s, enc2s, enc3s, enc4s, dec4s)
        return y

class FusedUNETRWithDictOutput(FusedUNETR):
    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        # repeat modality for input tensor
        return {"out": super().forward(x_in)} if self.training else super().forward(x_in)