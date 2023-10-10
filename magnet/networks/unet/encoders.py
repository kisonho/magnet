import torch


class UNetEncoder(torch.nn.Module):
    """
    The encoder for 3D UNet

    - Properties:
        - blocks: A `torch.nn.ModuleList` of 3D UNet encoder blocks
    """
    blocks: torch.nn.ModuleList

    def __init__(self, in_channels: int, /, *, basic_dims: int = 16, dim_mults: list[int] = [1, 2, 4, 8]) -> None:
        """
        Constructor

        - Parameters:
            - in_channels: An `int` of the input channel
            - basic_dims: An `int` of the basic dimension
            - dim_mults: A `list` of dimension multiplies based on the `basic_dims` in `int`
            - with_fuse_conv: A `bool` flag of if add a 1x1x1 Conv3d layer at the end
        """
        super().__init__()
        self.blocks = torch.nn.ModuleList()

        # loop for dimension multiplies
        for i, mults in enumerate(dim_mults):
            # build encoder level: c1 -> c2 -> c3
            c1 = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, basic_dims * mults, kernel_size=3, padding=1, stride=2 if i > 0 else 1),
                torch.nn.InstanceNorm3d(basic_dims),
                torch.nn.LeakyReLU(negative_slope=0.2),
            )
            c2 = torch.nn.Sequential(
                torch.nn.Conv3d(basic_dims * mults, basic_dims * mults, kernel_size=3, padding=1),
                torch.nn.InstanceNorm3d(basic_dims),
                torch.nn.LeakyReLU(negative_slope=0.2),
            )
            c3 = torch.nn.Sequential(
                torch.nn.Conv3d(basic_dims * mults, basic_dims * mults, kernel_size=3, padding=1),
                torch.nn.InstanceNorm3d(basic_dims),
                torch.nn.LeakyReLU(negative_slope=0.2),
            )

            # add to blocks
            self.blocks.append(torch.nn.ModuleList([c1, c2, c3]))
            in_channels = basic_dims * mults

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return super().__call__(x)

    def forward(self, x_in: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # initialize outputs
        y: list[torch.Tensor] = []
        x = x_in

        # call for blocks
        for b in self.blocks:
            assert isinstance(b, torch.nn.ModuleList) and len(b) == 3, "Block is not a valid UNet encoder block."
            c1, c2, c3 = b
            x = c1(x)
            x_block: torch.Tensor = c2(x)
            x_block = c3(x_block)
            y_block = x + x_block
            x = y_block
            y.append(x)
        return tuple(y)


class UNetEncoderWithFuseConv(UNetEncoder):
    fuse_convs: torch.nn.ModuleList

    def __init__(self, in_channels: int, /, *, basic_dims: int = 16, dim_mults: list[int] = [1, 2, 4, 8]) -> None:
        super().__init__(in_channels, basic_dims=basic_dims, dim_mults=dim_mults)
        self.fuse_convs = torch.nn.ModuleList()

        # loop for dimension multiplies
        for mults in dim_mults:
            # fuse conv
            fuse_conv = torch.nn.Conv3d(basic_dims * mults, basic_dims * mults, (1, 1, 1), bias=False)
            self.fuse_convs.append(fuse_conv)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # initialize
        features = super().forward(x)
        y: list[torch.Tensor] = []

        # loop for each features
        for i, feature in enumerate(features):
            f: torch.Tensor = self.fuse_convs[i](feature)
            y.append(f)
        return tuple(y)
