import torch
from torchmanager.metrics import metric


@metric
def whole_tumor_dice(input: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    # argmax
    input = input.argmax(dim=1)
    target = target.argmax(dim=1)

    # label 1
    o1 = (input == 1).float()
    t1 = (target == 1).float()

    # label 2
    o2 = (input == 2).float()
    t2 = (target == 2).float()

    # label 3
    o3 = (input == 3).float()
    t3 = (target == 3).float()

    # whole tumor
    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    return intersect_whole / denominator_whole


@metric
def tumor_core_dice(input: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    # argmax
    input = input.argmax(dim=1)
    target = target.argmax(dim=1)

    # label 1
    o1 = (input == 1).float()
    t1 = (target == 1).float()

    # label 3
    o3 = (input == 3).float()
    t3 = (target == 3).float()

    # tumor core
    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    return intersect_core / denominator_core


@metric
def enhancing_tumor_dice(input: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8, false_alert_num: int = 500) -> torch.Tensor:
    # argmax
    input = input.argmax(dim=1)
    target = target.argmax(dim=1)

    # label 3
    o3 = (input == 3).float()
    t3 = (target == 3).float()

    # post processing:
    if torch.sum(o3) < false_alert_num:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect_enhancing = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator_enhancing = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    return intersect_enhancing / denominator_enhancing
