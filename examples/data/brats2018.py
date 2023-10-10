from typing import Union, Optional, Tuple

import glob, logging, os, random as rand, torch
from monai.data.dataset import CacheDataset, Dataset
from monai.transforms.transform import MapTransform

from . import transforms
from .transforms import SetModality


class RoundLable(MapTransform):
    def __init__(self, key):
        self.k = key
    def __call__(self, d):
        d = dict(d)
        d[self.k] = torch.round(d[self.k])
        return d


def load(
    train_image_dir: str, 
    train_label_dir: str, 
    img_size: Union[int, tuple[int, ...]], 
    train_split: Optional[int] = None, 
    validation_image_dir: Optional[str] = None,
    validation_label_dir: Optional[str] = None,
    
    for_testing: bool = False,
    set_modality: Optional[Union[int, list[int]]] = None,

    show_verbose: bool = False,
    ndim: int = 3,
    search_key: str = '*.nii.gz',
    chached: bool = False,
    cache_num: Union[int, tuple[int, ...]] = 1,
    num_workers: int = 0,
    logger: Optional[logging.Logger] = None
    ) -> Tuple[Dataset, Dataset]:
    if logger: logger.info(f'Loading images from single volume from source {train_image_dir}...')

    # index 0 is the image volume, index 1 is the ground truth
    keys = ['image', 'label']

    # Load the training images/labels in a dictionary
    train_images = sorted(glob.glob(os.path.join(train_image_dir, search_key)))
    train_labels = sorted(glob.glob(os.path.join(train_label_dir, search_key)))
    img_size = img_size if isinstance(img_size, tuple) else tuple([img_size for _ in range(ndim)])
    train_dict = [{keys[0]: img, keys[1]: lab} for img, lab in zip(train_images, train_labels)]
    
    # If this is for training, shuffled the list for randomness
    if not for_testing: rand.shuffle(train_dict)

    # If a validation data directory is defined, create a dictionary for them
    if (validation_image_dir and validation_label_dir):
        val_images = sorted(glob.glob(os.path.join(validation_image_dir, search_key)))
        val_labels = sorted(glob.glob(os.path.join(validation_label_dir, search_key)))
        val_dict = [{keys[0]: img, keys[1]: lab} for img, lab in zip(val_images, val_labels)]

    # If no validation dir is defined, split the training one into two as defined by train_split
    else:
        if not isinstance(train_split, int): raise ValueError(f'If a validation image and label directory are not specified, use train_split to create a validation set from the trianing cohort wtih n=train_split cases')
        val_dict = train_dict[-train_split:]
        train_dict = train_dict[:-train_split]

    # Creating the sequence of transformations for validation
    validation_transform_list = [
        transforms.LoadImaged(keys=keys),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.CropForegroundd(keys=keys, source_key=keys[0]),
        transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
    ]
    if set_modality is not None:
        validation_transform_list.append(SetModality(mode=set_modality, key='image'))

    # Specific for training
    training_transform_list = [
        transforms.RandCropByPosNegLabeld(keys=keys, image_key='image', label_key='label', neg=0, spatial_size=img_size, num_samples=1),
        transforms.RandScaleIntensityd(keys=keys[0], factors=0.1, prob=0.1),
        transforms.RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=0.1),
        transforms.RandFlipd(keys=keys, spatial_axis=[i for i in range(3)], prob=0.50),
        transforms.RandRotate90d(keys=keys, prob=0.1, max_k=3),
    ]
    training_transform_list = validation_transform_list + training_transform_list

    # Rounding because of floating points
    training_transform_list.append(RoundLable(key='label'))
    validation_transform_list.append(RoundLable(key='label'))

    # Turning into tensors
    training_transform_list.append(transforms.ToTensord(keys=keys))
    validation_transform_list.append(transforms.ToTensord(keys=keys))

    # wrap to compose
    train_transforms = transforms.Compose(training_transform_list)
    val_transforms = transforms.Compose(validation_transform_list)
    cache_num = cache_num if isinstance(cache_num, tuple) else tuple([cache_num, cache_num])

    # load dataset
    if not chached:
        train_dataset = Dataset(
            data=train_dict,
            transform=train_transforms,
        )

        val_dataset = Dataset(
            data=val_dict,
            transform=val_transforms,
        )

    else:
        train_dataset = CacheDataset(
            data=train_dict,
            transform=train_transforms,
            cache_num=cache_num[0],
            num_workers=num_workers,
            cache_rate=1.0,
            progress=show_verbose
        )
        val_dataset = CacheDataset(
            data=val_dict,
            transform=val_transforms,
            cache_num=cache_num[1],
            num_workers=num_workers,
            cache_rate=1.0,
        )
    return train_dataset, val_dataset
