import torch


class UNetDecoder(torch.nn.Module):
    """
    The decoder for 3D UNet

    - Properties:
        - blocks: A `torch.nn.ModuleList` of 3D UNet decoder blocks
        - seg_layer: A `torch.nn.Conv3d` segmentation layer
    """
    blocks: torch.nn.ModuleList
    seg_layer: torch.nn.Conv3d

    def __init__(self, num_classes: int, *, basic_dims: int = 16, dim_mults: list[int] = [1, 2, 4, 8]) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([])

        # loop for dimension multiplies
        for mults in dim_mults[::-1][:-1]:
            # build decoder level: upsampling -> c1 -> c2 -> c3
            # upsampling = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            c1 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(basic_dims * mults, int(basic_dims * mults / 2), kernel_size=3, padding=1, output_padding=1, stride=2),
                torch.nn.InstanceNorm3d(int(basic_dims * mults / 2)),
                torch.nn.LeakyReLU(negative_slope=0.2),
            )
            c2 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(basic_dims * mults, int(basic_dims * mults / 2), kernel_size=3, padding=1),
                torch.nn.InstanceNorm3d(int(basic_dims * mults / 2)),
                torch.nn.LeakyReLU(negative_slope=0.2),
            )
            c3 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(int(basic_dims * mults / 2), int(basic_dims * mults / 2), kernel_size=3, padding=1),
                torch.nn.InstanceNorm3d(int(basic_dims * mults / 2)),
                torch.nn.LeakyReLU(negative_slope=0.2),
            )
            self.blocks.append(torch.nn.ModuleList([c1, c2, c3]))

        # segmentation layer
        self.seg_layer = torch.nn.Conv3d(in_channels=basic_dims, out_channels=num_classes, kernel_size=3, stride=1, padding=1, bias=True)

    def __call__(self, x_in: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return super().__call__(x_in)

    def forward(self, x_in: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # unpack inputs
        features = tuple(reversed(x_in))
        x = features[0]

        # loop for blocks
        for i, b in enumerate(self.blocks):
            assert isinstance(b, torch.nn.ModuleList) and len(b) == 3, "Block is not a valid UNet decoder block."
            [c1, c2, c3] = b
            x = c1(x)
            x = torch.cat((x, features[i+1]), dim=1)
            x = c2(x)
            x = c3(x)
        return self.seg_layer(x)
