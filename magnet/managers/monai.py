from monai.data.utils import decollate_batch
from monai.inferers.utils import sliding_window_inference
from torchmanager.data import DataLoader, Dataset
from torchmanager_core import devices, torch, view
from torchmanager_core.typing import Any, Callable, Collection, Module, Optional, Sequence, SizedIterable, Union
from torchmanager_monai import Manager as _Manager
from torchmanager_monai.protocols import SubResulting

from .targeting import Manager as _TargetingManager


class Manager(_Manager[Module], _TargetingManager[Module]):
    """
    A `TargetingManager` with monai compatibility wrap

    * extends: `torchmanager_monai.Manager`, `.targeting.Manager`
    """

    _post_labels: list[Callable[[torch.Tensor], torch.Tensor]]
    _post_predicts: list[Callable[[torch.Tensor], torch.Tensor]]
    _roi_size: Sequence[int]
    return_dict: bool

    def __init__(self, model: Module, post_labels: list[Callable[[torch.Tensor], torch.Tensor]] = [], post_predicts: list[Callable[[torch.Tensor], torch.Tensor]] = [], return_dict: bool = False, roi_size: Sequence[int] = (96, 96, 96), **kwargs: Any) -> None:
        """
        Constructor

        - Parameters:
            - model: A `TargetingModule` to be trained
            - post_labels: A `list` of post `Callable[[torch.Tensor], torch.Tensor]`s for labels
            - post_predicts: A `list` of post `Callable[[torch.Tensor], torch.Tensor]`s for predictions
            - roi_size: A `Sequence` of sliding window size in `int` during the testing
        """
        _TargetingManager.__init__(self, model, **kwargs)
        self._post_labels = post_labels
        self._post_predicts = post_predicts
        self._roi_size = roi_size
        self.return_dict = return_dict

    @torch.no_grad()
    def predict(self, dataset: SizedIterable, device: Optional[Union[torch.device, list[torch.device]]] = None, use_multi_gpus: bool = False, show_verbose: bool = False) -> list[torch.Tensor]:
        # find available device
        cpu, device, target_devices = devices.search(None if use_multi_gpus else device)
        if device == cpu and len(target_devices) < 2:
            use_multi_gpus = False
        devices.set_default(target_devices[0])

        # move model
        if use_multi_gpus and not isinstance(self.model, torch.nn.parallel.DataParallel):
            raw_model = self.model
            self.model, use_multi_gpus = devices.data_parallel(self.model, devices=target_devices)
        else:
            raw_model = None

        # initialize predictions
        self.model.eval()
        predictions: list[torch.Tensor] = []
        if len(dataset) == 0:
            return predictions
        progress_bar = view.tqdm(total=len(dataset)) if show_verbose else None
        self.to(device)
        t = self.target if self.target is not None else 0

        # loop the dataset
        for data in dataset:
            x, _ = self.unpack_data(data)
            if use_multi_gpus is not True:
                x = x.to(device)
            val_outputs = sliding_window_inference(x, self._roi_size, 1, self.model)
            y: list[torch.Tensor] = decollate_batch(val_outputs)  # type: ignore
            y = [self._post_predicts[t](val_pred_tensor).unsqueeze(0) for val_pred_tensor in y]
            y_post = torch.cat(y)
            predictions.append(y_post)
            if progress_bar is not None:
                progress_bar.update()

        # reset model and loss
        if raw_model is not None:
            self.model = raw_model.to(cpu)
        devices.empty_cache()
        return predictions

    @torch.no_grad()
    def test(self, dataset: Union[DataLoader[Any], Dataset[Any], dict[str, Any]], show_verbose: bool = False, **kwargs: Any) -> dict[str, float]:
        # initialize
        summary: dict[str, float] = {}

        # check if dataset is dictionary
        if isinstance(dataset, dict):
            # test target dataset
            for target, (name, d) in enumerate(dataset.items()):
                # normal test
                self.target = target
                try:
                    subsummary = _TargetingManager.test(self, d, show_verbose=show_verbose, **kwargs)
                except Exception as e:
                    raise RuntimeError(f"Testing dataset '{name}' failed") from e
                for k, v in subsummary.items():
                    summary[f"{k}_{name}"] = v

                # get sub results
                for k, m in self.compiled_metrics.items():
                    if isinstance(m, SubResulting):
                        subresults: list[float] = m.sub_results.reshape([-1]).detach().tolist()
                        k = k.replace("val_", "")
                        for i, r in enumerate(subresults):
                            summary[f"{k}_{i}_{name}"] = r
        else:
            summary = _TargetingManager.test(self, dataset, show_verbose=show_verbose, **kwargs)

            # get sub results
            for k, m in self.compiled_metrics.items():
                if isinstance(m, SubResulting):
                    subresults: list[float] = m.sub_results.reshape([-1]).detach().tolist()
                    k = k.replace("val_", "")
                    for i, r in enumerate(subresults):
                        summary[f"{k}_{i}"] = r

        # reset loss fn
        return summary

    def test_step(self, x_test: torch.Tensor, y_test: torch.Tensor) -> dict[str, float]:
        # initialize
        summary: dict[str, float] = {}
        t = self.target if self.target is not None else 0

        # forward pass
        val_outputs = sliding_window_inference(x_test, self._roi_size, 1, self.model)
        val_labels_list: list[torch.Tensor] = decollate_batch(y_test)  # type: ignore
        y_test = torch.cat([l.unsqueeze(0) for l in val_labels_list])
        y_list: list[torch.Tensor] = decollate_batch(val_outputs)  # type: ignore
        y_out = torch.cat([o.unsqueeze(0) for o in y_list])

        # repack output
        if self.return_dict:
            y = {"out": y_out}
        else:
            y = y_out

        # calculate loss
        if self.loss_fn is not None:
            summary["loss"] = float(self.loss_fn(y, y_test))

        # post process for metrics evaluation
        y_test_post = [self._post_labels[t](val_label_tensor).unsqueeze(0) for val_label_tensor in val_labels_list]
        y_post = [self._post_predicts[t](val_pred_tensor).unsqueeze(0) for val_pred_tensor in y_out]
        y_test = torch.cat(y_test_post)
        y = torch.cat(y_post).to(y_test.device)

        # repack output
        if self.return_dict:
            y = {"out": y}
        else:
            y = y

        # forward metrics
        for name, fn in self.compiled_metrics.items():
            if name.startswith("val_"):
                name = name.replace("val_", "")
            elif "loss" in name:
                continue
            try:
                fn(y, y_test)
                summary[name] = float(fn.result.detach())
            except Exception as metric_error:
                runtime_error = RuntimeError(f"Cannot fetch metric '{name}'.")
                raise runtime_error from metric_error
        return summary

    def unpack_data(self, data: dict[str, Any]) -> tuple[Any, Union[dict[str, Any], Any]]:
        if self.return_dict:
            image, label = _Manager.unpack_data(self, data)
            label = {"out": label}
            return image, label
        else:
            return _Manager.unpack_data(self, data)
