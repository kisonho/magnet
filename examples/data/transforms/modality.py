import torch
from monai.transforms.transform import MapTransform
from typing import Any, Union


class CopyModality(MapTransform):
    """
    Takes a monai style data dictionary and highlights a single channel. Used for multichannel images where only one is wanted but others will be replaced as zeros or the one that wanted.
    copy_modality : the bool flag that if copy the wanted channel image
    key : the dictionary key that defines the array of interest
    mode : the index of the channel to be looked at
    """
    copy_modality: bool
    key: str
    mode: int

    def __init__(self, mode: int, key: str, copy_modality: bool = False) -> None:
        self.copy_modality = copy_modality
        self.key = key
        self.mode = mode

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for m in range(len(data[self.key])):
            data[self.key][m] = data[self.key][self.mode]
            if not self.copy_modality:
                data[self.key][m] *= 0
        return data


class SetModality(MapTransform):
    """
    Takes a monai style data dictionary and highlights a single channel. Used for multichannel images where only several is wanted
    key : the dictionary key that defines the array of interest
    mode : the index of the channel to be looked at
    """

    def __init__(self, mode: Union[int, list[int]], key:str) -> None:
        self.key = key
        self.mode = mode
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.mode, list):
            # initialize modalities
            modality_list: list[torch.Tensor] = []

            # loop for each modality
            for m in self.mode:
                d = self.take_modality(data, m)
                modality_list.append(torch.tensor(d))
            data[self.key] = torch.cat(modality_list)
        else: data[self.key] = self.take_modality(data, self.mode)
        return data

    def take_modality(self, data: dict[str, Any], modality: int) -> Any:
        """
        Takes a single modality from given data

        - Parameters:
            - data: A `dict` of data with key as `str` and `Any` type of value
            - modality: The target modality in `int`
        - Returns: `Any` type of target modality data
        """
        d = dict(data)
        return d[self.key][modality, ...][None, ...]