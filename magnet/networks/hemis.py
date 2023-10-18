import numpy as np
from torchmanager_core import torch, deprecated


@deprecated("v2.1", "v2.3")
class HeMISFuse(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=1)
        x_var = x.var(dim=1) if x.shape[1] > 1 else (x_mean * 0)
        return torch.cat([x_mean, x_var], dim=1)


@deprecated("v2.1", "v2.3")
class HeMIS(torch.nn.Module):
    decoder: torch.nn.Sequential
    encoders: torch.nn.ModuleList
    fusion: HeMISFuse

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        encoders: list[torch.nn.Sequential] = []

        # build encoders
        for _ in range(in_channels):
            conv1 = torch.nn.Conv2d(1, 48, kernel_size=5, padding=2)
            relu1 = torch.nn.ReLU()
            conv2 = torch.nn.Conv2d(48, 48, kernel_size=5, padding=2)
            relu2 = torch.nn.ReLU()
            pool = torch.nn.MaxPool2d(2, stride=1, padding=1, dilation=2)
            encoder = torch.nn.Sequential(conv1, relu1, conv2, relu2, pool)
            encoders.append(encoder)

        # initialize encoders and fusion
        self.encoders = torch.nn.ModuleList(encoders)
        self.fusion = HeMISFuse()

        # initialize decoder
        deconv2 = torch.nn.Conv2d(96, 16, kernel_size=5, padding=2)
        relu3 = torch.nn.ReLU()
        deconv1 = torch.nn.Conv2d(16, num_classes, kernel_size=21, padding=10)
        self.decoder = torch.nn.Sequential(deconv2, relu3, deconv1)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # format x_in
        if len(x_in.shape) == 4:
            x_in.unsqueeze(1)
        elif len(x_in.shape) != 5:
            raise TypeError(f"Input tensor must be in 4 shapes as [B, X, Y, Z] for one modality or 5 shapes as [B, M, X, Y, Z] for multi-modalities, got {x_in.shape}.")
        
        # initialize
        y_2d: list[torch.Tensor] = []

        # forward to 2d
        for i in range(x_in.shape[-1]):
            x = x_in[..., i]
            x = self.forward_2d(x).unsqueeze(dim=2)
            x = torch.permute(x, (0, 1, 3, 4, 2))
            y_2d.append(x)

        # concat y
        y = torch.cat(y_2d, dim=-1)
        return y

    def forward_2d(self, x_in: torch.Tensor) -> torch.Tensor:
        # initialize
        features: list[torch.Tensor] = []

        # loop for each modality
        for modality in range(x_in.shape[1]):
            # initialize dropping
            drop_current = (np.random.rand() > 0.5) and self.training

            # check if need to drop current modality
            if not drop_current:
                f: torch.Tensor = self.encoders[modality](x_in[:, modality, ...].unsqueeze(1))
                f = f.unsqueeze(1)
                features.append(f)
        
        # check chosen modalities are not empty
        if len(features) < 1:
            f: torch.Tensor = self.encoders[-1](x_in[:, -1, ...].unsqueeze(1))
            f = f.unsqueeze(1)
            features.append(f)
        
        # fuse features
        x = torch.cat(features, dim=1)
        x = self.fusion(x)

        # forward to decoder
        y: torch.Tensor = self.decoder(x)
        assert x_in.shape == y.shape, RuntimeError(f"Input shape {x_in.shape} is not the same as output shape {y.shape}.")
        return y
