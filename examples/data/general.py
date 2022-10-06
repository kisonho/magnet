# pyright: reportPrivateImportUsage=false
from typing import Any, Union, Tuple

import glob, numpy as np, os, torch
from monai.data import CacheDataset
from torch.utils.data import Subset

from .transforms import TransformOptions, load_transforms

def load(image_dir: str, label_dir: str, img_size: Union[int, tuple[int, ...]], train_split: int, transform_options: Union[TransformOptions, dict[str, Any]], show_verbose: bool = False) -> Tuple[CacheDataset, CacheDataset, int, int]:
    """
    Load dataset for a general case

    - Parameters:
        - image_dir: A `str` of an image data directory
        - label_dir: A `str` of a label data directory
        - img_size: Either an `int` or a `tuple` defining the model input size
        - train_split: An `int` defining where to split the training data to create a validation set (counting backwards)
        - transform_options: A `TransformOptions` defining the transforms, or a `dict` that includes for the data set following the toggles:
            - 'LoadImaged': Must be `True`
            - 'AddChanneld': `True` or `False`
            - 'ConvertLabel': `True` or `False`
            - 'Spacingd': `list`[pixdim (`tuple`), mode (`int`)] or `False`
            - 'SetModality': mode (`int`) or `False`
            - 'NormalizeIntensityd': `True` or `False`
            - 'NormToOne': `True` or `False`
            - 'CropStructure': `list`[id (`int`), pad (`int`)] or `False`
            - 'SpatialPadd': `True` or `False` 
            - 'Orientationd': `True` or `False`
            - 'RandCropByPosNegLabeld': NumSamples (`int`) or `False`
            - 'RandFlipd': `True` or `False`
            - 'RandRotate90d': `True` or `False`
            - 'RandShiftIntensityd': `True` or `False`
            - 'ToTensord': Must be `True`
        - show_verbose: A `bool` flag of if showing verbose during loading datasets
    - Returns: A `tuple` of training `DataLoader`, validation `DataLoader`, an `int` of the input channels, and the number of classes in `int`
    """
    # load images and labels and join in a dictionary
    train_images_mri = sorted(
        glob.glob(os.path.join(image_dir, "*.nii.gz")))
    train_labels_mri = sorted(
        glob.glob(os.path.join(label_dir, "*.nii.gz")))
    img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size, img_size)
    keys = ['image', 'label']
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
    data_dicts = [
        {keys[0]: image_name, keys[1]: label_name}
        for image_name, label_name in zip(train_images_mri, train_labels_mri)
    ]

    # Split into training and validation sets based off of option `train_split`
    train_data_dicts, val_data_dicts = data_dicts[train_split:], data_dicts[:train_split]
    
    # Creating the sequence of transformations based off of transformation option dictionary
    td = transform_options if isinstance(transform_options, TransformOptions) else TransformOptions.from_dict(transform_options) # Renaming the options to reduce size
    train_transforms, val_transforms = load_transforms(td, img_size, keys, mapping_dict)

    train_dataset = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        cache_num=2,
        cache_rate=1.0,
        progress=show_verbose
    )
    val_dataset = CacheDataset(
        data=val_data_dicts,
        transform=val_transforms,
        cache_num=2,
        cache_rate=1.0,
        progress=show_verbose
    )

    # get input channels and number of classes
    data = train_dataset[0] if len(train_dataset) > 0 else val_dataset[0]
    assert not isinstance(data, Subset), 'Fetch data failed, should not be a `Subset` with `int` index.'
    x: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]] = data
    if isinstance(x, list): x = x[0]
    input_channels = x[keys[0]].shape[0]
    num_classes = len(np.unique(x[keys[1]]))
    return train_dataset, val_dataset, input_channels, num_classes
