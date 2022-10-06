# pyright: reportPrivateImportUsage=false
import itk, torch
from typing import Union

from .datasets.iseg import ImageType, ISeg
from .transforms import TransformOptions, load_transforms

def load(root_dir: str, img_size: tuple[int, int, int], transform_options: TransformOptions, split: list[int] = [6, 1, 3], img_type: ImageType = ImageType.IMG) -> tuple[ISeg, ISeg, ISeg, int, int]:
    '''
    Load iSeg2017 dataset

    - Parameters:
        - root_dir: A `str` of the root dataset directory
        - img_size: An `int` or a `tuple` of `int` of image sizes
        - transform_options: A `TransformOptions` of the preprocessing options
        - modality: A `ImageModality` of the target modality
        - split: A `list` of split amount for training, validation, and testing
        - type: The target image type in `ImageType` to read
    - Returns: A `tuple` of training `ISeg2017`, validation `ISeg2017`, testing `ISeg2017`, number of input channels in `int`, and number of classes in `int`
    '''
    # initialize
    itk.ProcessObject.SetGlobalWarningDisplay(False) # type: ignore
    keys = ['image', 'label']
    mapping_dict = {
        0: 0,
        10: 1,
        150: 2,
        250: 3,
    }

    # load transforms
    training_transform, testing_transform = load_transforms(transform_options, img_size, keys, mapping_dict)

    # load dataset
    train_val_amount = split[0] + split[1]
    train_val_dataset = ISeg(root_dir, data_prefix=[f"subject-{i}" for i in range(1, train_val_amount+1)], shuffle=True, transforms=training_transform, type=img_type)
    testing_dataset = ISeg(root_dir, data_prefix=[f"subject-{i}" for i in range(11 - split[-1], 11)], transforms=testing_transform, type=img_type)
    train_dataset, val_dataset = train_val_dataset.split(split[0] / train_val_amount)
    val_dataset.transforms = testing_transform

    # get input channels and number of classes
    x: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]] = testing_dataset[0]
    if isinstance(x, list): x = x[0]
    input_channels = x[keys[0]].shape[0]
    num_classes = int(x[keys[1]].max()) + 1
    return train_dataset, val_dataset, testing_dataset, input_channels, num_classes
