from typing import Sequence, Union

from .nn import MAGNET, share_modules
from .unetr import UNETRWithDictOutput as UNETR

def load(in_channels: int, num_classes: int, img_size: Union[Sequence[int], int], target_dict: dict[int, str]) -> MAGNET[UNETR]:
    '''
    Function to load a MAGNET

    - Parameters:
        - in_channels: An `int` of input number of channels as modalities
        - num_classes: An `int` of the number of output classes
        - img_size: Either a `Sequence` of image size in `int` or an `int` of single image size
        - target_dict: A `dict` of target index as key in `int` and name of target as value in `str`
    - Returns: A `MAGNET` with `.unetr.UNETRWithDictOutput` as its target modules
    '''
    # initialize
    if not in_channels > 0: raise ValueError(f"The input channels must be a positive number, got {in_channels}.")
    if not num_classes > 0: raise ValueError(f"The number of classes must be a positive number, got {num_classes}.")
    if isinstance(img_size, Sequence):
        for size in img_size:
            if not size > 0: raise ValueError(f"The image size must be all positive numbers, got {img_size}.")
    elif not img_size > 0: raise ValueError(f"The image size must be a positive number, got {img_size}.")
    models: list[UNETR] = []
    
    # initialize models for all modalities
    for _ in range(in_channels):
        model = UNETR(1, num_classes, img_size=img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)
        models.append(model)

    # share decoders
    cnns = ["decoder5", "decoder4", "decoder3", "decoder2", "out"]
    shared_modules = {k: m for k, m in models[0].named_children() if k in cnns}
    for cnn in cnns:
        if cnn not in shared_modules: raise NameError("Sharing decoder in UNETR failed.")
    magnet = share_modules(models, shared_modules, target_dict=target_dict)
    return magnet