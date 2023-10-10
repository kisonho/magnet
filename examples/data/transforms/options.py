# pyright: reportPrivateImportUsage=false
import torch
from monai import transforms
from monai.transforms.compose import Compose
from typing import Any, NamedTuple, Optional, Union

from .general import ConvertLabel, CropStructure, NormToOne
from .modality import CopyModality, SetModality

class TransformOptions(NamedTuple):
    load_imaged: bool = True
    """must be `True`"""
    add_channeld: Optional[list[str]] = None
    as_channel_firstd: Optional[list[str]] = None
    center_spatial_cropd: Optional[Union[tuple[int, int, int], int]] = None
    convert_label: bool = False
    spacingd: Optional[tuple[tuple[int, ...], str]] = None
    """`tuple`[pixdim (`tuple`), mode (`int`)]"""
    copy_modality: Optional[tuple[int, bool]] = None
    set_modality: Optional[Union[int, list[int]]] = None
    normalize_intensityd: bool = False
    norm_to_one: bool = False
    crop_structure: Optional[tuple[int, int]] = None
    """`tuple`[id (`int`), pad (`int`)] or `False`"""
    grid_patchd: Optional[tuple[int, ...]] = None
    spatial_padd: bool = False
    orientationd: bool = False
    rand_crop_by_pos_neg_labeld: Optional[int] = None
    rand_flipd: bool = False
    rand_rotate_90d: bool = False
    rand_scale_intensityd: bool = False
    rand_shift_intensityd: bool = False
    to_tensord: bool = False

    @classmethod
    def from_dict(cls, td: dict[str, Any]):
        """Compability from `dict`"""
        if not td['Spacingd']: td['Spacingd'] = None
        if not td['CropStructure']: td['CropStructure'] = None
        if not td['RandCropByPosNegLabeld']: td['RandCropByPosNegLabeld'] = None
        return cls(**td)

def load_transforms(transform_options: TransformOptions, img_size: tuple[int, ...], keys: list[str], mapping_dict: Optional[dict[int, int]] = None) -> tuple[Compose, Compose]:
    """
    Load transforms via options

    - Parameters:
        - transform_options: A `TransformOptions` of transform settings
        - img_size: A `tuple` of target image size in `int`
        - keys: A `list` of input keys in `str`
        - mapping_dict: A `dict` of label value mapping
    """
    # load transforms
    training_transforms: list[transforms.Transform] = []

    if transform_options.load_imaged:
        training_transforms.append(transforms.LoadImaged(keys))

    if transform_options.add_channeld:
        training_transforms.append(transforms.AddChannelD(transform_options.add_channeld))

    if transform_options.as_channel_firstd:
        training_transforms.append(transforms.AsChannelFirstd(transform_options.as_channel_firstd))

    if transform_options.center_spatial_cropd is not None:
        training_transforms.append(transforms.CenterSpatialCropd(keys, transform_options.center_spatial_cropd))

    if transform_options.convert_label:
        assert mapping_dict is not None, "Label value must be given to convert the label."
        training_transforms.append(ConvertLabel(keys[1], mapping_dict))

    if transform_options.spacingd is not None:
        training_transforms.append(transforms.Spacingd(keys=keys, pixdim=transform_options.spacingd[0], mode=transform_options.spacingd[1]))

    if transform_options.orientationd:
        training_transforms.append(transforms.Orientationd(keys=keys, axcodes="RAS"))

    if transform_options.copy_modality is not None:
        mode, copy_modality = transform_options.copy_modality
        training_transforms.append(CopyModality(mode=mode, key=keys[0], copy_modality=copy_modality))

    if transform_options.set_modality is not None:
        training_transforms.append(SetModality(mode=transform_options.set_modality, key=keys[0]))

    if transform_options.normalize_intensityd:
        training_transforms.append(transforms.NormalizeIntensityd(keys=keys[0], nonzero=True, channel_wise=True))

    if transform_options.norm_to_one:
        training_transforms.append(NormToOne(key=keys[0]))

    if transform_options.crop_structure is not None:
        training_transforms.append(CropStructure(struct_id=transform_options.crop_structure[0], pad=transform_options.crop_structure[1], keys=keys, source_key=keys[1]))

    validation_transforms = training_transforms.copy()

    if transform_options.rand_crop_by_pos_neg_labeld is not None:
        training_transforms.append(transforms.RandCropByPosNegLabeld(keys=keys, image_key='image', label_key='label', neg=0, spatial_size=img_size, num_samples=4))

    if transform_options.rand_flipd:
        training_transforms.append(transforms.RandFlipd(keys=keys, spatial_axis=[i for i in range(3)], prob=0.50))

    if transform_options.rand_rotate_90d:
        training_transforms.append(transforms.RandRotate90d(keys=keys, prob=0.1, max_k=3))

    if transform_options.rand_scale_intensityd:
        training_transforms.append(transforms.RandScaleIntensityd(keys=keys[0], factors=0.1, prob=1.0))

    if transform_options.rand_shift_intensityd:
        training_transforms.append(transforms.RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=1.0))

    if transform_options.grid_patchd is not None:
        training_transforms.append(transforms.GridPatchd(keys, patch_size=transform_options.grid_patchd))

    if transform_options.to_tensord:
        to_tensor = transforms.ToTensord(keys=keys[0], dtype=torch.float)
        training_transforms.append(to_tensor)
        validation_transforms.append(to_tensor)

    training_transform = transforms.Compose(training_transforms)
    testing_transform = transforms.Compose(validation_transforms)
    return training_transform, testing_transform