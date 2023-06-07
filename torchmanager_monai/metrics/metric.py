from torchmanager.metrics import * # type: ignore

from monai.metrics.metric import CumulativeIterationMetric as _CumulativeIterationMetric
from torchmanager_core import torch
from torchmanager_core.typing import Optional

class CumulativeIterationMetric(Metric):
    """
    The basic metric wrap for monai `CumulativeIterationMetric`

    * extends: `Metric`
    * implements: `SubResulting`

    - Properties:
        - sub_results: A `torch.Tensor` of the iteratively sub-results at given dim
    """
    __dim: int
    _metric_fn: _CumulativeIterationMetric

    @property
    def result(self) -> torch.Tensor:
        # get results
        try:
            results = self._metric_fn.aggregate()
            assert isinstance(results, torch.Tensor), f"Results type ({type(results)}) are not a valid `torch.Tensor`."
            return torch.nanmean(results)
        except:
            return torch.tensor(torch.nan)

    @property
    def sub_results(self) -> torch.Tensor:
        # get results
        try:
            results = self._metric_fn.aggregate()
            assert isinstance(results, torch.Tensor), f"Results type ({type(results)}) are not a valid `torch.Tensor`."

            # get sub dice
            subdice = torch.nanmean(results, dim=self.__dim, keepdim=True)
            subdice = subdice.unsqueeze(-1)
            return subdice
        except:
            return torch.tensor(torch.nan)

    def __init__(self, metric_fn: _CumulativeIterationMetric, dim: int = 0, target: Optional[str] = None) -> None:
        """
        Constructor

        - Parameters:
            - metric_fn: A `_CumulativeIterationMetric` of the main monai metric function
            - dim: An `int` of the dimension of subresult
            - target: An optional `str` of target name in `input` and `target` during direct calling
        """
        super().__init__(metric_fn, target=target) # type: ignore
        self.__dim = dim

    def reset(self) -> None:
        self._metric_fn.reset()
        super().reset()