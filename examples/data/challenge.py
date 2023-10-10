# pyright: reportPrivateImportUsage=false
import glob
import os
from monai.data import CacheDataset

from .transforms import (
    AddChanneld,
    Compose,
    ConvertLabel,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)

def load(data_dir: str, img_size: tuple[int, ...] = (96, 96, 96), scale_intensity_ranged: bool = False, show_verbose: bool = False) -> tuple[CacheDataset, CacheDataset, int]:
    """
    Load dataset

    - Parameters:
        - data_dir: A `str` of data directory
        - img_size: A `tuple` of image size `int`
        - show_verbose: A `bool` of flag to show loading progress
    - Returns: A `tuple` of training `DataLoader`, validation `DataLoader`, and the number of classes in `int`
    """
    # load images and labels
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    mapping_dict = {
        205: 1,
        420: 2,
        500: 3,
        550: 4,
        600: 5,
        820: 6,
        850: 7,
        421: 0
    }
    train_data_dicts, val_data_dicts = data_dicts[4:], data_dicts[:4]

    # transforms
    train_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ConvertLabel(keys='label', mapping=mapping_dict),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if scale_intensity_ranged:
        train_transforms += [
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),  # for ct onlys
        ]
    train_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=img_size, mode='reflect'),  # only pad when size < img_size
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size,  # 16*n
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="label", select_fn=lambda x: x > 0),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
    train_transforms = Compose(train_transforms) # type: ignore
    val_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ConvertLabel(keys='label', mapping=mapping_dict),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if scale_intensity_ranged:
        val_transforms += [
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ]
    val_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
    val_transforms = Compose(val_transforms) # type: ignore

    # get datasets
    train_ds = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        cache_num=2,
        cache_rate=1.0,
        progress=show_verbose
    )
    val_ds = CacheDataset(
        data=val_data_dicts,
        transform=val_transforms,
        progress=show_verbose
    )
    return train_ds, val_ds, 8
