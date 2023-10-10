from __future__ import annotations

import imageio, numpy as np, os
from enum import Enum
from monai.transforms.compose import Compose
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Any, Optional, Union

class ImageType(Enum):
    HDR = ".hdr"
    IMG = ".img"

class ISeg(Dataset):
    '''
    The iSeg dataset

    - Properties:
        - data: A `list` of the prefix for all data in the given dataset without modality, i.e. \'subject-1\'
        - modality: The target `ImageModality`
    '''
    __data: list[str]
    __root_dir: str
    __type: ImageType
    transforms: Optional[Compose]
    '''Target modality, the modality will be set to the first channel in the image'''

    @property
    def data(self) -> list[str]:
        '''The prefix for all data in the given dataset without modality, i.e. \'subject-1\''''
        return self.__data

    def __init__(self, root_dir: str, data_prefix: list[str] = [f"subject-{i}" for i in range(1, 11)], shuffle: bool = False, transforms: Optional[Compose] = None, type: ImageType = ImageType.IMG) -> None:
        '''
        Constructor

        - Parameters:
            - root_dir: A `str` of the root directory
            - data_prefix: A `list` of the prefix for all data in the given dataset without modality, i.e. \'subject-1\'
            - shuffle: A `bool` flag of if shuffling the dataset
            - transorms: An optional `Compose` to transform the input image
            - type: An `ImageType` of the data extension
        '''
        super().__init__()
        self.__data = data_prefix
        self.__root_dir = os.path.normpath(root_dir)
        self.__type = type
        self.transforms = transforms
        if shuffle: np.random.shuffle(self.__data)

    def __getitem__(self, i: int) -> Union[list[dict[str, Any]], dict[str, Any]]:
        # read label
        label_file = f"{self.__data[i]}-label{self.__type.value}"
        label_path = os.path.join(self.__root_dir, label_file)
        label: NDArray[np.int_] = imageio.imread(label_path)

        # reat t1 and t2 images
        t1_file = f"{self.__data[i]}-T1{self.__type.value}"
        t1_file_path = os.path.join(self.__root_dir, t1_file)
        t1_img = imageio.imread(t1_file_path)
        t2_file = f"{self.__data[i]}-T2{self.__type.value}"
        t2_file_path = os.path.join(self.__root_dir, t2_file)
        t2_img = imageio.imread(t2_file_path)
        image = [t1_img, t2_img]

        # wrap data
        data = {
            'image': np.array(image),
            'label': np.array([label]),
        }

        # transform image
        if self.transforms is not None:
            data: Union[list[dict[str, Any]], dict[str, Any]] = self.transforms(data) # type: ignore
        return data

    def __len__(self) -> int:
        return len(self.__data)

    def split(self, p: float) -> tuple[ISeg, ISeg]:
        '''
        Split current dataset into two `ISeg` dataset
        
        - Parameters:
            - p: A `float` of split ratio
        - Returns A `tuple` of two `ISeg` dataset with the first one contains p ratio of current and another contains 1-p ratio of current
        '''
        # initialize
        assert p > 0 and p < 1, f"Split ratio must be in range (0,1), got {p}."
        split_amount = int(len(self) * p)

        # split data
        first_dataset = ISeg(self.__root_dir, data_prefix=[d for d in self.__data[:split_amount]], transforms=self.transforms, type=self.__type)
        second_dataset = ISeg(self.__root_dir, data_prefix=[d for d in self.__data[split_amount:]], transforms=self.transforms, type=self.__type)
        return first_dataset, second_dataset