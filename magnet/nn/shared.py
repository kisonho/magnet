import torch
from torchmanager_core.typing import Any, Generic, Module, Optional, Union


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

    target: Optional[int]
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

    def forward(self, x_in: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        # check if input is in multi-modality mode
        if x_in.shape[1] > 1 and self.target is None:
            # initialize output
            preds: list[Any] = []

            # loop for each modality
            for i, t in enumerate(self.target_dict):
                # forward each modality
                if x_in.shape[1] > len(self.target_dict):
                    x = x_in[:, t : t + 1, ...]
                else:
                    x = x_in[:, i : i + 1, ...]
                if x.shape[1] < 1:
                    raise ValueError(f"No modality '{self.target_dict[t]}' (id={i}) found in the input.")
                y_pred: Union[torch.Tensor, dict[str, torch.Tensor]] = self.target_modules[t](x, *args, **kwargs)

                # append to preds
                preds.append(y_pred)

            # fuse
            return self.fuse(preds)
        elif x_in.shape[1] > 1 and self.target is not None:
            # forward target input
            t = self.target
            x = x_in[:, t : t + 1, ...]
            y = self.target_modules[t](x, *args, **kwargs)
            return y
        elif self.target is None:
            return self.target_modules[0](x_in, *args, **kwargs)
        else:
            return self.target_modules[self.target](x_in, *args, **kwargs)

    def fuse(self, preds: list[Any]) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        # check predictions
        assert len(preds) > 0, "Results must contain at least one predictions"
        y: Union[torch.Tensor, dict[str, torch.Tensor]]

        # check predictions type
        if isinstance(preds[0], torch.Tensor):
            preds_tensor: list[torch.Tensor] = preds
            preds_tensor = [p.unsqueeze(1) for p in preds_tensor]
            return torch.concat(preds_tensor, dim=1) if self.training else torch.concat(preds_tensor, dim=1).sum(dim=1)
        elif isinstance(preds[0], dict):
            preds_dict: list[dict[str, torch.Tensor]] = preds
            preds_dict_unsqueezed: dict[str, list[torch.Tensor]] = {name: [p[name].unsqueeze(1) for p in preds_dict] for name in preds[0].keys()}
            return {name: torch.concat(v, dim=1) for name, v in preds_dict_unsqueezed.items()} if self.training else {name: torch.concat(v, dim=1).sum(dim=1) for name, v in preds_dict_unsqueezed.items()}
        else:
            return NotImplemented


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
