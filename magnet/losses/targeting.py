from torchmanager.losses import Loss
from torchmanager_core import torch, _raise
from torchmanager_core.typing import Any, Optional, Union


class MAGLoss(Loss):
    losses: torch.nn.ModuleList
    modality: Optional[int]

    def __init__(self, losses: list[Loss], modality: Optional[int] = None, target: Optional[str] = None, weight: float = 1) -> None:
        super().__init__(target=target, weight=weight)
        self.losses = torch.nn.ModuleList(losses)
        self.modality = modality
        
        # check modality targeted
        if self.modality is not None:
            assert len(self.losses) == 1, "Only one loss function needed when modality is targeted."

    def forward(self, input: Union[list[Optional[torch.Tensor]], torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        # switch mode
        if self.training and self.modality is None:
            # initialize
            loss = 0
            num_target = input.shape[1] if isinstance(input, torch.Tensor) else len(input)
            assert len(self.losses) >= num_target, f"Main losses must have the same or more targets, got {len(self.losses)} and {num_target}."

            # loop for each modality
            for t in range(num_target):
                # check input type
                if isinstance(input, torch.Tensor):
                    x = input
                elif input[t] is None:
                    continue
                else:
                    x = input[t]
                    assert x is not None, _raise(ValueError(f"Input at index {t} is None."))

                # forward loss for modality t
                loss += self.forward_target(x, target, t)
        elif self.training and self.modality is not None:
            assert isinstance(input, torch.Tensor), _raise(TypeError(f"Input must be a `torch.Tensor` when modality is targeted, got {type(input)}"))
            loss = self.forward_target(input, target, self.modality)
        else:
            loss = self.losses[0](input, target)

        # return summed loss
        assert isinstance(loss, torch.Tensor), _raise(TypeError("Loss is not a valid `torch.Tensor`."))
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

    def reset(self) -> None:
        for l in self.losses:
            if isinstance(l, Loss):
                l.reset()
        return super().reset()
