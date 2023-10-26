import copy
from torchmanager_core.typing import Sequence, Union

from .nn import MAGNET, MAGNET2, share_modules
from .nn.fusion import MidFusion
from .networks.unet import UNetDecoder, UNetEncoderWithFuseConv as UNetEncoder
from .networks.unetr import UNETR, UNETRDecoder, UNETREncoderWithFusionConv as UNETREncoder


def build_v1(in_channels: int, num_classes: int, img_size: Union[Sequence[int], int], target_dict: dict[int, str], copy_modality: bool = False) -> MAGNET[UNETR]:
    """
    Function to load a MAGNET

    - Parameters:
        - in_channels: An `int` of input number of channels as modalities
        - num_classes: An `int` of the number of output classes
        - img_size: Either a `Sequence` of image size in `int` or an `int` of single image size
        - target_dict: A `dict` of target index as key in `int` and name of target as value in `str`
        - copy_modality: A `bool` flag of if copying model to other modalities so that all models have the same initialized weights
    - Returns: A `MAGNET` with `.networks.unetr.UNETRWithDictOutput` as its target modules
    """
    # initialize
    if not in_channels > 0:
        raise ValueError(f"The input channels must be a positive number, got {in_channels}.")
    if not num_classes > 0:
        raise ValueError(f"The number of classes must be a positive number, got {num_classes}.")
    if isinstance(img_size, Sequence):
        for size in img_size:
            if not size > 0:
                raise ValueError(f"The image size must be all positive numbers, got {img_size}.")
    elif not img_size > 0:
        raise ValueError(f"The image size must be a positive number, got {img_size}.")
    models: list[UNETR] = []
    init_model = UNETR(1, num_classes, img_size=img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0) if copy_modality else None

    # initialize models for all modalities
    for _ in range(in_channels):
        if copy_modality:
            assert init_model is not None, "Build model failed."
            model = copy.deepcopy(init_model)
        else:
            model = UNETR(1, num_classes, img_size=img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)
        models.append(model)

    # share decoders
    cnns = ["decoder5", "decoder4", "decoder3", "decoder2", "out"]
    shared_modules = {k: m for k, m in models[0].named_children() if k in cnns}
    for cnn in cnns:
        if cnn not in shared_modules:
            raise NameError("Sharing decoder in UNETR failed.")
    magnet = share_modules(models, shared_modules, target_dict=target_dict)
    return magnet


def build_v2(in_channels: int, num_classes: int, img_size: Union[Sequence[int], int], target_dict: dict[int, str], copy_encoder: bool = False, return_features: bool = True) -> MAGNET2[UNETREncoder]:
    """
    Function to load a MAGNET v2 with UNETR backbone

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
    if isinstance(img_size, Sequence):
        for size in img_size:
            if not size > 0:
                raise ValueError(f"The image size must be all positive numbers, got {img_size}.")
    elif not img_size > 0:
        raise ValueError(f"The image size must be a positive number, got {img_size}.")
    encoders: list[UNETREncoder] = []
    encoder = UNETREncoder(1, img_size=img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0) if copy_encoder else None
    
    # initialize encoders
    for _ in range(in_channels):
        if copy_encoder:
            assert encoder is not None, "Built encoder failed."
            e = copy.deepcopy(encoder)
        else:
            e = UNETREncoder(1, img_size=img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)
        encoders.append(e)

    # initialize fusion
    fusion = MidFusion()

    # initialize decoder
    decoder = UNETRDecoder(768, num_classes, feature_size=16, norm_name="instance", res_block=True)

    # initialize magnet
    return MAGNET2(*encoders, fusion=fusion, decoder=decoder, return_features=return_features, target_dict=target_dict)


def build_v2_unet(in_channels: int, num_classes: int, target_dict: dict[int, str], copy_encoder: bool = False, return_features: bool = True) -> MAGNET2[UNetEncoder]:
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
    encoder = UNetEncoder(1) if copy_encoder else None
    
    # initialize encoders
    for _ in range(in_channels):
        if encoder is not None:
            e = copy.deepcopy(encoder)
        else:
            e = UNetEncoder(1)
        encoders.append(e)

    # initialize fusion
    fusion = MidFusion()

    # initialize decoder
    decoder = UNetDecoder(num_classes)

    # initialize magnet
    return MAGNET2(*encoders, fusion=fusion, decoder=decoder, return_features=return_features, target_dict=target_dict)


build = build_v2
