from torchmanager.losses import Loss, MultiLosses
from torchmanager_core import torch, _raise
from torchmanager_core.typing import Any, Optional


class MAGLoss(MultiLosses):
    modality: Optional[int]

    def __init__(self, losses: list[Loss], modality: Optional[int] = None, target: Optional[str] = None, weight: float = 1) -> None:
        super().__init__(losses, target, weight)
        self.modality = modality
        
        # check modality targeted
        if self.modality is not None:
            assert len(self.losses) == 1, "Only one loss function needed when modality is targeted."

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # switch mode
        if self.training and self.modality is None:
            # initialize
            loss = 0
            num_target = input.shape[1]
            assert len(self.losses) >= num_target, _raise(ValueError(f"Main losses must have the same or more targets, got {len(self.losses)} and {num_target}."))
            
            # loop for each modality
            for t in range(input.shape[1]):
                loss += self.forward_target(input, target, t)
        elif self.training and self.modality is not None:
            return self.forward_target(input, target, self.modality)
        else:
            loss = self.losses[0](input, target)

        # return summed loss
        assert isinstance(loss, torch.Tensor), "Loss is not a valid `torch.Tensor`."
        return loss

    def forward_target(self, input: torch.Tensor, target: Any, modality: int) -> torch.Tensor:
        """
        Forward the target modality loss

        - Parameters:
            - input: A `torch.Tensor` of the predictions
            - target: `Any` type of labels
            - modality: An `int` of the modality index
        - Returns:
            The loss scalar in `torch.Tensor` for the target modality
        """
        x = input[:, modality, ...]
        return self.losses[modality](x, target) if self.modality is None else self.losses[0](x, target)
