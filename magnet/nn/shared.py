from torchmanager_core import torch
from torchmanager_core.typing import Any, Generic, Module, NamedTuple, Optional, Union, TypeVar, overload

from .fusion import Fusion

E = TypeVar('E', bound=torch.nn.Module)
"""The type of modality specific `Module` encoder"""
F = TypeVar('F', bound=Optional[Fusion])
"""The type of optional `Fusion` module"""
D = TypeVar('D', bound=Optional[torch.nn.Module])
"""The type of optional shared `torch.nn.Module` decoder"""


class MAGNET(torch.nn.Module, Generic[Module]):
    """
    Modules that shared same architecture

    * extends: `torch.nn.Module`
    * generic: The type of target `Module`
    * implements: `managers.protocols.Targeting`

    - Properties:
        - fuse_none: A `bool` flag of if adding none target modalities for fusion
        - num_targets: An `int` of the total number of modalities in MAGNET
        - target: An optional `int` of the target index in the module list
        - target_dict: A `dict` of available target modality as key in `int` and name of target as value in `str`
        - target_modules: A `torch.nn.ModuleList` of all modules
    """

    __fuse_none: bool
    target: Optional[Union[list[int], int]]
    target_dict: dict[int, str]
    target_modules: torch.nn.ModuleList

    @property
    def fuse_none(self) -> bool:
        try:
            return self.__fuse_none
        except AttributeError:
            self.__fuse_none = False
            return False
        
    @fuse_none.setter
    def fuse_none(self, value: bool) -> None:
        self.__fuse_none = value

    @property
    def num_targets(self) -> int:
        return len(self.target_modules)

    def __init__(self, *modules: Module, fuse_none: bool = False, target_dict: dict[int, str] = {}) -> None:
        """
        Constructor

        - Parameters:
            - *modules: `torch.nn.Module` that shares architecture
            - shared_params: A `list` of shared parameters in modules
            - target_dict: A `dict` of available target index as key in `int` and name of target as value in `str`
        """
        super().__init__()
        self.fuse_none = fuse_none
        self.target = None
        self.target_modules = torch.nn.ModuleList(list(modules))
        self.target_dict = target_dict

    def forward(self, x_in: Union[torch.Tensor, list[torch.Tensor]], *args: Any, **kwargs: Any) -> Any:
        # check num modalitites
        num_target = len(x_in) if isinstance(x_in, list) else x_in.shape[1]

        # check input type
        if num_target > 1:
            # initialize output
            preds: list[Any] = []
            target = self.target if isinstance(self.target, list) else self.target_dict
            has_target_missing = len(target) == num_target
            i = 0

            # forward each modality
            for t in self.target_dict:
                if t in target and not has_target_missing:
                    x = x_in[:, t, ...].unsqueeze(1) if isinstance(x_in, torch.Tensor) else x_in[t]
                    y_pred: Union[torch.Tensor, dict[str, torch.Tensor]] = self.target_modules[t](x, *args, **kwargs)
                    preds.append(y_pred)
                elif t in target and has_target_missing:
                    x = x_in[:, i, ...].unsqueeze(1) if isinstance(x_in, torch.Tensor) else x_in[i]
                    y_pred: Union[torch.Tensor, dict[str, torch.Tensor]] = self.target_modules[t](x, *args, **kwargs)
                    preds.append(y_pred)
                    i += 1
                elif self.fuse_none:
                    preds.append(None)

            # fuse
            return self.fuse(preds)
        elif self.target is None:
            return self.target_modules[0](x_in, *args, **kwargs)
        else:
            assert not isinstance(self.target, list), "Multiple targets are assigned but only one modality is given to the input."
            return self.target_modules[self.target](x_in, *args, **kwargs)

    @overload
    def fuse(self, preds: list[torch.Tensor]) -> torch.Tensor:
        ...

    @overload
    def fuse(self, preds: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        ...

    def fuse(self, preds: list[Any]) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        # check predictions
        assert len(preds) > 0, "Results must contain at least one predictions"

        # check predictions type
        if isinstance(preds[0], torch.Tensor):
            preds_tensor: list[torch.Tensor] = preds
            preds_tensor = [p.unsqueeze(1) for p in preds_tensor]
            y = torch.concat(preds_tensor, dim=1) if self.training else torch.concat(preds_tensor, dim=1).sum(dim=1)
        elif isinstance(preds[0], dict):
            preds_dict: list[dict[str, torch.Tensor]] = preds
            preds_dict_unsqueezed: dict[str, list[torch.Tensor]] = {name: [p[name].unsqueeze(1) for p in preds_dict] for name in preds[0].keys()}
            y = {name: torch.concat(v, dim=1) for name, v in preds_dict_unsqueezed.items()} if self.training else {name: torch.concat(v, dim=1).sum(dim=1) for name, v in preds_dict_unsqueezed.items()}
        else:
            return NotImplemented
        
        # select modality
        if self.training and self.target is not None and isinstance(y, torch.Tensor):
            return torch.cat([y.sum(dim=1).unsqueeze(1), y[:, self.target, ...].unsqueeze(1)], dim=1)
        elif self.training and self.target is not None and isinstance(y, dict):
            return {name: torch.cat([v.sum(dim=1).unsqueeze(1), v[:, self.target, ...].unsqueeze(1)], dim=1) for name, v in y.items()}
        else:
            return y


class FeaturedData(NamedTuple):
    """The output data with features"""
    features: Any
    out: Any
    additional_out: Any = None


class MAGNET2(MAGNET[E], Generic[E, F, D]):
    """
    Modules that shared same architecture

    * extends: `magnet.MAGNET`
    * generic: A modality specific encoder `E`, a fusion module `F`, and a decoder `D`

    - Parameters:
        - encodrs: A `torch.nn.ModuleList` of modality specific encoders
        - fusion: An optional `Fusion` model to fuse the target modalities
        - decoder: An optional `Decoder` model for the shared decoder
    """
    fusion: F
    decoder: D
    return_features: bool

    @property
    def num_targets(self) -> int:
        return len(self.target_modules)

    @property
    def encoders(self) -> torch.nn.ModuleList:
        return self.target_modules

    def __init__(self, *encoders: E, fusion: F = None, decoder: D = None, fuse_none: bool = False, return_features: bool = True, target_dict: dict[int, str] = {}) -> None:
        """
        Constructor

        - Parameters:
            - *encoders: Target `Module` encoders
            - fusion: An optional `Fusion` module to fuse the target modalities
            - decoder: An optional `torch.nn.Module` module for the shared fused decoder
            - fuse_none: A `bool` flag of if adding none target modalities for fusion
            - return_features: A `bool` flag of if returning features during training
            - target_dict: A `dict` of available target index as key in `int` and name of target as value in `str`
        """
        super().__init__(*encoders, fuse_none=fuse_none, target_dict=target_dict)
        self.fusion = fusion
        self.decoder = decoder
        self.return_features = return_features

    def forward(self, x_in: Union[torch.Tensor, list[torch.Tensor], dict[str, Any]], *args: Any, **kwargs: Any) -> Union[FeaturedData, Union[torch.Tensor, list[Optional[torch.Tensor]], tuple[Union[torch.Tensor, list[Optional[torch.Tensor]]], list[Any]]]]:
        # unpack inputs
        if isinstance(x_in, dict):
            x = x_in['image']
            assert isinstance(x, torch.Tensor), "Input image is not a valid `torch.Tensor`."
            target: Optional[list[int]] = x_in['targets'] if 'targets' in x_in else None
            self.target = target
        else:
            x = x_in

        # forward encoders and fuse
        x = super().forward(x, *args, **kwargs)

        # check input modalities
        if self.target is None or isinstance(self.target, list):
            # initialize
            y_targets: list[Optional[torch.Tensor]] = []
            additional_out_targets: Optional[list[Any]] = []

            # loop for each modality
            for x_target in x:
                # check if target modality is given
                if x_target is None:
                    y_targets.append(None)
                    continue

                # forward decoder
                y_target: Union[torch.Tensor, tuple[torch.Tensor, Any]] = x_target if self.decoder is None else self.decoder(x_target)

                # unwrap output
                if isinstance(y_target, torch.Tensor):
                    y_targets.append(y_target.unsqueeze(1))
                else:
                    y_target, additional_out = y_target
                    y_targets.append(y_target.unsqueeze(1))
                    additional_out_targets.append(additional_out)
            y = torch.cat([y for y in y_targets if y is not None], dim=1)
        else:
            # forward decoder
            y_target: Union[torch.Tensor, tuple[torch.Tensor, Any]] = x if self.decoder is None else self.decoder(x)

            # unwrap output
            if isinstance(y_target, torch.Tensor):
                y = y_target.unsqueeze(1)
                additional_out_targets = None
            else:
                y, additional_out = y_target
                additional_out_targets = [additional_out]
            y_targets = [y]

        # decision level fusion if not fused
        if not self.training:
            y = y.sum(1)
        elif self.fuse_none:
            y = y_targets

        # return formatted outputs
        return FeaturedData(x, y, additional_out=additional_out_targets) if self.training and self.return_features else (y, additional_out_targets) if additional_out_targets is not None else y

    def fuse(self, preds: list[Any]) -> Any:
        if self.fusion is not None:
            return self.fusion(preds)
        elif isinstance(preds[0], tuple):
            return preds
        else:
            return super().fuse(preds)


def share_modules(models: list[Module], shared_modules: dict[str, torch.nn.Module], target_dict: dict[int, str] = {}) -> MAGNET[Module]:
    """
    Share modules with shared modules in attribution name

    - Parameters:
        - models: A target `list` of `Module`
        - shared_modules: A `dict` of shared `torch.nn.Module` with name in `str`
        - target_dict: A `dict` of target index as key in `int` and name of target as value in `str`
    - Returns: A `TargetingModule` with given `models` that shares `shared_modules`
    """
    # loop for shared modules
    for k, m in shared_modules.items():
        # loop for target modules
        for model in models:
            if not hasattr(model, k):
                raise NameError(f"One or more given models do not have layer named {k}.")
            setattr(model, k, m)
    return MAGNET(*models, target_dict=target_dict)
