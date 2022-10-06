import torch
from torchmanager_core.typing import Any, Generic, Module, Union

class MAGNET(torch.nn.Module, Generic[Module]):
    """
    Modules that shared same architecture

    * extends: `torch.nn.Module`
    * implements: `Targeting`, `ParametersSharing`
    
    - Properties:
        - num_targets: An `int` of the total number of modalities in MAGNET
        - shared_parameters: A `list` of shared parameters
        - target: An `int` of the target index in the module list
        - target_dict: A `dict` of available target modality as key in `int` and name of target as value in `str`
        - target_modules: A `torch.nn.ModuleList` of all modules
    """
    target: int
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
        self.target = 0
        self.target_modules = torch.nn.ModuleList(list(modules))
        self.target_dict = target_dict

    def forward(self, x_in: torch.Tensor, *args: Any, **kwargs: Any) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        # check if input is in multi-modality mode
        if x_in.shape[1] > 1:
            # initialize output
            preds: list[torch.Tensor] = []
            preds_dict: dict[str, list[torch.Tensor]] = {}

            # loop for each modality
            for t in self.target_dict:
                # forward each modality
                x = x_in[:, t:t+1, ...]
                y_pred: Union[torch.Tensor, dict[str, torch.Tensor]] = self.target_modules[t](x, *args, **kwargs)
                
                # check prediction type
                if isinstance(y_pred, dict):
                    # check output types
                    assert len(preds) == 0, "Outputs for modalities must be in same format."

                    # loop for items in dictionary
                    for key, pred in y_pred.items():
                        if key not in preds_dict: preds_dict[key] = [pred.unsqueeze(1)]
                        else: preds_dict[key].append(pred)
                else:
                    assert len(preds_dict) == 0, "Outputs for modalities must be in same format."
                    preds.append(y_pred.unsqueeze(1))

            # calculate mean or max
            assert len(preds) > 0 or len(preds_dict) > 0, "Fetch output failed, no modality was passed."
            y = torch.concat(preds, dim=1).mean(dim=1) if len(preds) > 0 else {key: torch.cat(values, dim=1).mean(dim=1) for key, values in preds_dict.items()}
            return y
        else: return self.target_modules[self.target](x_in, *args, **kwargs)

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
            if not hasattr(model, k): raise NameError(f"One or more given models do not have layer named {k}.")
            setattr(model, k, m)
    return MAGNET(*models, target_dict=target_dict)
