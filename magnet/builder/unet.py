import copy

from magnet import MAGNET2
from magnet.networks.unet import UNetEncoder, UNetEncoderWithFuseConv, UNetDecoder
from magnet.nn.fusion import MeanFusion


def build_v2_unet(in_channels: int, num_classes: int, target_dict: dict[int, str], copy_encoder: bool = False, return_features: bool = True) -> MAGNET2[UNetEncoder, MeanFusion, UNetDecoder]:
    """
    Function to load a MAGNET v2 with 3D UNet backbone

    - Parameters:
        - in_channels: An `int` of input number of channels as modalities
        - num_classes: An `int` of the number of output classes
        - img_size: Either a `Sequence` of image size in `int` or an `int` of single image size
        - target_dict: A `dict` of target index as key in `int` and name of target as value in `str`
        - copy_encoder: A `bool` flag of if copying encoder to other modalities so that all encoders have the same initialized weights
        - return_features: A `bool` flag of if returning features during training
    - Returns: A `MAGNET` with `.networks.unetr.UNETRWithDictOutput` as its target modules
    """
    # initialize
    if not in_channels > 0:
        raise ValueError(f"The input channels must be a positive number, got {in_channels}.")
    if not num_classes > 0:
        raise ValueError(f"The number of classes must be a positive number, got {num_classes}.")
    encoders: list[UNetEncoder] = []
    encoder = UNetEncoderWithFuseConv(1) if copy_encoder else None
    
    # initialize encoders
    for _ in range(in_channels):
        if encoder is not None:
            e = copy.deepcopy(encoder)
        else:
            e = UNetEncoderWithFuseConv(1)
        encoders.append(e)

    # initialize fusion
    fusion = MeanFusion()

    # initialize decoder
    decoder = UNetDecoder(num_classes)

    # initialize magnet
    return MAGNET2(*encoders, fusion=fusion, decoder=decoder, return_features=return_features, target_dict=target_dict)
