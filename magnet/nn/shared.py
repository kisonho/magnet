from torchmanager_core import torch
from torchmanager_core.typing import Any, Generic, Module, NamedTuple, Optional, Union, overload

from .fusion import Fusion


class MAGNET(torch.nn.Module, Generic[Module]):
    """
    Modules that shared same architecture

    * extends: `torch.nn.Module`
    * implements: `..managers.protocols.Targeting`

    - Properties:
        - num_targets: An `int` of the total number of modalities in MAGNET
        - target: An optional `int` of the target index in the module list
        - target_dict: A `dict` of available target modality as key in `int` and name of target as value in `str`
        - target_modules: A `torch.nn.ModuleList` of all modules
    """

    target: Optional[Union[list[int], int]]
    target_dict: dict[int, str]
    target_modules: torch.nn.ModuleList

    @property
    def num_targets(self) -> int:
        return len(self.target_modules)

    def __init__(self, *modules: Module, target_dict: dict[int, str] = {}) -> None:
        """
        Constructor

        - Parameters:
            - *modules: `torch.nn.Module` that shares architecture
            - shared_params: A `list` of shared parameters in modules
            - target_dict: A `dict` of available target index as key in `int` and name of target as value in `str`
        """
        super().__init__()
        self.target = None
        self.target_modules = torch.nn.ModuleList(list(modules))
        self.target_dict = target_dict

    def forward(self, x_in: Union[torch.Tensor, list[torch.Tensor]], *args: Any, **kwargs: Any) -> Any:
        # check input type
        if not isinstance(x_in, torch.Tensor):
            # initialize output
            preds: list[Any] = []
            target = self.target if isinstance(self.target, list) else self.target_dict

            # forward each modality
            for i, t in enumerate(target):
                if len(x_in) > len(self.target_dict):
                    x = x_in[t]
                else:
                    x = x_in[i]
                y_pred: Union[torch.Tensor, dict[str, torch.Tensor]] = self.target_modules[t](x, *args, **kwargs)
                preds.append(y_pred)

            # fuse
            return self.fuse(preds)
        elif x_in.shape[1] > 1:
            # initialize output
            preds: list[Any] = []
            target = self.target if isinstance(self.target, list) else self.target_dict

            # forward each modality
            for i, t in enumerate(target):
                if x_in.shape[1] > len(target):
                    x = x_in[:, t : t + 1, ...]
                else:
                    x = x_in[:, i : i + 1, ...]
                if x.shape[1] < 1:
                    raise ValueError(f"No modality '{self.target_dict[t]}' (id={i}) found in the input.")
                y_pred: Union[torch.Tensor, dict[str, torch.Tensor]] = self.target_modules[t](x, *args, **kwargs)
                preds.append(y_pred)

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
    features: Any
    out: Any


class MAGNET2(MAGNET[Module]):
    """
    Modules that shared same architecture

    * extends: `magnet.MAGNET`

    - Parameters:
        - encodrs: A `torch.nn.ModuleList` of modality specific encoders
        - fusion: An optional `Fusion` model to fuse the target modalities
        - decoder: An optional `Decoder` model for the shared decoder
    """
    fusion: Optional[Fusion]
    decoder: Optional[torch.nn.Module]
    return_features: bool

    @property
    def num_targets(self) -> int:
        return len(self.target_modules)

    @property
    def encoders(self) -> torch.nn.ModuleList:
        return self.target_modules

    def __init__(self, *encoders: Module, fusion: Optional[Fusion] = None, decoder: Optional[torch.nn.Module] = None, return_features: bool = True, target_dict: dict[int, str] = {}) -> None:
        """
        Constructor

        - Parameters:
            - *encoders: Target `Module` encoders
            - fusion: An optional `Fusion` module to fuse the target modalities
            - decoder: An optional `torch.nn.Module` module for the shared decoder
            - return_features: A `bool` flag of if returning features during training
            - target_dict: A `dict` of available target index as key in `int` and name of target as value in `str`
        """
        super().__init__(*encoders, target_dict=target_dict)
        self.fusion = fusion
        self.decoder = decoder
        self.return_features = return_features

    def forward(self, x_in: Union[torch.Tensor, list[torch.Tensor], dict[str, Any]], *args: Any, **kwargs: Any) -> Union[FeaturedData, torch.Tensor]:
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
            y_targets: list[torch.Tensor] = []

            # loop for each modality
            for x_target in x:
                y_target: torch.Tensor = x_target if self.decoder is None else self.decoder(x_target)
                y_targets.append(y_target.unsqueeze(1))
            y = torch.cat(y_targets, dim=1)
        else:
            y_target: torch.Tensor = x if self.decoder is None else self.decoder(x)
            assert isinstance(y_target, torch.Tensor), "The output from the decoder must be a valid `torch.Tensor`."
            y = y_target.unsqueeze(1)

        # decision level fusion if not fused
        if not self.training:
            y = y.sum(1)

        # return formatted outputs
        if self.training and self.return_features:
            out = FeaturedData(x, y)
        else:
            out = y
        return out

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
