# pyright: reportPrivateImportUsage=false
import numpy as np, torch
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.transforms import CropForegroundd, MapTransform, Transform
from typing import Any, Union

# convert the label pixel value 
class ConvertLabel(MapTransform):
    __label_map: dict[int, int]

    def __init__(self, keys: KeysCollection, mapping: dict[int, int], allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.__label_map = mapping

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            for k, v in self.__label_map.items():
                d[key][d[key]==k] = v
        return d

class Skip:
    def __init__(self) -> None:
        """
        Simple class that just acts as a placeholder for toggle options
        """
        self.PlaceHolder = True

    def __call__(self, data:dict) -> dict:
        return data

class NormToOne(MapTransform):
    def __init__(self, key:str) -> None:
        """
        Takes a monai style data dictionary and normalizes values to 1 when they're scaled by a consistent factor (i.e. converting a 0-255 tiff image into a 0-1 array)
        key : the data that needs to be normalized
        """
        self.k = key

    def __call__(self, data:dict) -> dict:
        d = dict(data)
        _max = np.max(d[self.k])# if isinstance(d[self.k], np.array) else torch.max(d[self.k])
        d[self.k] = d[self.k] / _max
        return d

class CropStructure(Transform):
    def __init__(self, struct_id:int, pad:int, keys:list, source_key:str) -> None:
        """
        Takes a monai style data dictionary and crops all data down to a single structure with a variable padding size on either side
        struct_id : the numerical value that represents the structure
        pad : the amount to pad around the substructure by
        keys : the keys that represent the data in the dictionary
        source_key : the key the cropping is based off of
        """
        super().__init__()
        self.id = struct_id
        self.k = source_key
        self.cropper = CropForegroundd(keys=keys, source_key=source_key, k_divisible=pad)

    def __call__(self, data:dict) -> dict:
        d = dict(data)
        d[self.k] = np.where(d[self.k] == self.id, 1, 0)# if isinstance(d[self.k], np.array) else torch.where(d[self.k] == self.id, 1, 0)
        d = self.cropper(d)
        return d

def isolate_structure(data:Union[np.ndarray, torch.Tensor], labels:Union[list[int], int]) -> Union[np.ndarray, torch.Tensor]:
    """
    Inputs: 
        Data: np.ndarray or torch.Tensor with integer labels on the same channel. Can be any shape.
        labels: list[int] or int representing the label(s) that are to be isolated
    Outputs:
        np.ndarray or torch.Tensor with the same size depending on input with only the selected label(s)
    """
    assert (isinstance(data, np.ndarray) or isinstance(data, torch.Tensor))
    package = np if isinstance(data, np.ndarray) else torch
    _array = package.zeros(data.shape)
    if isinstance(_array, torch.Tensor): _array = _array.type(torch.LongTensor)
    if isinstance(labels, int): labels = [labels]
    for label in labels:
        _array = package.where(data == label, label, _array)
    return _array

def isolate_channel(data:Union[np.ndarray, torch.Tensor], labels:Union[list[int], int]) -> Union[np.ndarray, torch.Tensor]:
    """
    Inputs: 
        Data: np.ndarray or torch.Tensor with a binary label on each channe. Must be have shape: [channel, ...]
        labels: list[int] or int representing the channel(s) to be isolated
    Outputs:
        np.ndarray or torch.Tensor with the same size depending on input with only the selected channel(s)
    """
    assert (isinstance(data, np.ndarray) or isinstance(data, torch.Tensor))
    cat = torch.cat if isinstance(data, torch.Tensor) else np.concatenate
    if isinstance(labels, int): labels = [labels]
    _array = None
    for label in labels:
        if _array is None: _array = data[label, ...][None, ...]
        else: _array = cat((_array, data[label, ...][None, ...]), 0)
    return _array    

class IsolateStructure(Transform):
    """
    Dictionary transform for use in Monai pipelines to isolate a structure of a given ground truth, working off of the function isolate_structure
    Inputs:
        labels:Union[list[int], int] a list of or single integer representing the structure that should be isolated
        key:str the key that will have the isolation done to it
    """
    def __init__(self, labels:Union[list[int], int], key:str) -> None:
        super().__init__()
        self.labels = labels
        self.k = key
    
    def __call__(self, data:dict[str, Union[np.ndarray, torch.Tensor]]) -> dict:
        d = dict(data)
        d[self.k] = isolate_structure(d[self.k], self.labels)
        return d

class IsolateChannel(Transform):
    """
    Dictionary transform for use in Monai pipelines to isolate a channel of a given ground truth, working off of the function isolate_channel
    Inputs:
        labels:Union[list[int], int] a list of or single integer representing the channels that should be isolated
        key:str the key that will have the isolation done to it
    """
    def __init__(self, labels:Union[list[int], int], key=str) -> None:
        super().__init__()
        self.labels = labels
        self.k = key

    def __call__(self, data: dict[str, Union[np.ndarray, torch.Tensor]]) -> dict:
        d = dict(data)
        d[self.k] = isolate_channel(d[self.k], self.labels)
        return d
