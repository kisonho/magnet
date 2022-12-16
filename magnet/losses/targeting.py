from torchmanager.losses import MultiLosses
from torchmanager_core import torch, _raise
from torchmanager_core.typing import Any


class TargetingLoss(MultiLosses):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # switch mode
        if self.training:
            # initialize
            loss = 0
            num_target = input.shape[1]
            assert len(self.losses) >= num_target, _raise(ValueError(f"Main losses must have the same or more targets, got {len(self.losses)} and {num_target}."))
            
            # loop for each modality
            for t in range(input.shape[1]):
                loss += self.forward_target(input, target, t)
        else:
            loss = self.losses[0](input, target)

        # return summed loss
        assert isinstance(loss, torch.Tensor), "Loss is not a valid `torch.Tensor`."
        return loss

    def forward_target(self, input: torch.Tensor, target: Any, t: int) -> torch.Tensor:
        x = input[:, t, ...]
        return self.losses[t](x, target)
