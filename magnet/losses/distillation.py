from torchmanager.losses import KLDiv, Loss
from torchmanager_core import torch, view, _raise
from torchmanager_core.typing import Any, Iterable, Optional, Union

from .protocols import FeaturedData
from .targeting import MAGLoss


class PixelWiseKLDiv(KLDiv):
    """The pixel wise KL-Divergence loss for semantic segmentation"""
    def __init__(self, *args: Any, target: Optional[str] = None, weight: float = 1, **kwargs: Any) -> None:
        super().__init__(*args, target=target, weight=weight, reduction="none", **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = super().forward(input, target)
        return loss.sum(1).mean()


class MAGSelfDistillationLoss(MAGLoss):
    """
    The targeting loss with self distillation

    - Properties:
        - distillation_losses: A `torch.nn.ModuleList` of `Loss` functions for self distillation
    """
    distillation_losses: torch.nn.ModuleList

    def __init__(self, main_losses: list[Loss], distillation_losses: list[Loss], modality: Optional[int] = None, target: Optional[str] = None) -> None:
        super().__init__(main_losses, modality=modality, target=target)
        self.distillation_losses = torch.nn.ModuleList(distillation_losses)
        view.warnings.warn("The `MAGSelfDistillationLoss` and `MAGFeatureDistillationLoss` have be combined into `MAGMSLoss`. They will be deprecated in v2.3.")

        # check modality target
        if self.modality is not None and self.distillation_losses is not None:
            assert len(self.distillation_losses) == 1, "Only one distillation loss function needed when modality is targeted."

    def forward_target(self, input: torch.Tensor, target: Any, modality: int) -> torch.Tensor:
        # main loss
        x_all = input[:, 0, ...]
        loss = super().forward_target(input, target, modality)

        # distillation loss
        if self.modality is None and modality > 0:
            # fetch modality input
            x = input[:, modality, ...]
            loss += self.distillation_losses[modality - 1](x, x_all)
        elif self.modality is not None:
            # fetch modality input
            x = input[:, modality, ...]
            loss += self.distillation_losses[modality - 1](x, x_all)
        return loss

    def reset(self) -> None:
        for fn in self.distillation_losses:
            if isinstance(fn, Loss):
                fn.reset()
        super().reset()


class MAGFeatureDistillationLoss(Loss):
    """
    The targeting loss with feature distillation

    * The features losses will be 0 if input format is not `FeaturedData`

    - Properties:
        - features_dtype: A `torch.dtype` of the features type
        - features_losses: An optional `list` of `loss` functions for feature distillation
        - main_loss: An optional `list` of `Loss` functions for the main loss function
        - modality: An optional `int` of target modality
    """
    features_dtype: torch.dtype
    features_losses: torch.nn.ModuleList
    main_loss: torch.nn.Module
    modality: Optional[int]
    normalize_features: bool

    def __init__(self, loss_fn: Loss, feature_losses: list[Loss], features_dtype: torch.dtype = torch.float32, modality: Optional[int] = None, normalize_features: bool = False, target: Optional[str] = None, weight: float = 1) -> None:
        super().__init__(target=target, weight=weight)
        self.features_dtype =features_dtype
        self.features_losses = torch.nn.ModuleList(feature_losses)
        self.main_loss = loss_fn
        self.modality = modality
        self.normalize_features = normalize_features
        view.warnings.warn("The `MAGSelfDistillationLoss` and `MAGFeatureDistillationLoss` have be combined into `MAGMSLoss`. They will be deprecated in v2.3.")

    def forward(self, input: Union[FeaturedData, torch.Tensor], target: Any) -> torch.Tensor:
        if isinstance(input, torch.Tensor):
            return self.main_loss(input, target)
        else:
            loss = self.main_loss(input.out, target)
            next_out = next((x for x in input.out if x is not None), None) if isinstance(input.out, list) else input.out
            assert next_out is not None, "At least one output must be given."
            loss += self.forward_features(input.features, next_out.shape[0])
            return loss

    def forward_features(self, features: list[Optional[Union[torch.Tensor, Iterable[torch.Tensor]]]], batch_size: int) -> torch.Tensor:
        """
        Forward features losses

        - Parameters:
            - features: A `list` of `Iterable` features in `torch.Tensor`
            - batch_size: An `int` of the batch size
        - Returns: A `torch.Tensor` of summed features losses
        """
        # initialize feature loss
        loss = 0
        assert features[0] is not None, "The fused feature must be given."
        f_all = [features[0].view([batch_size, -1])] if isinstance(features[0], torch.Tensor) else [f.view([batch_size, -1]) for f in features[0]]
        f_all = torch.cat(f_all, dim=1).to(self.features_dtype)
        if self.normalize_features:
            f_all = torch.nn.functional.normalize(f_all, dim=1)

        # feature target feature distillation loss
        if self.modality is None:
            for t, feature in enumerate(features[1:]):
                # check if feature is available
                if feature is None:
                    continue

                # flatten feature
                feature = [feature.view([batch_size, -1])] if isinstance(feature, torch.Tensor) else [f.view([batch_size, -1]) for f in feature]
                feature = torch.cat(feature, dim=1).to(self.features_dtype)

                # normalize feature
                if self.normalize_features:
                    feature = torch.nn.functional.normalize(feature, dim=1)

                # calculate feature loss
                feature_loss = self.features_losses[t](feature, f_all)
                loss += feature_loss
        else:
            target_features = features[self.modality + 1]
            assert target_features is not None, "The target feature must be given."
            feature = [target_features.view([batch_size, -1])] if isinstance(target_features, torch.Tensor) else [f.view([batch_size, -1]) for f in target_features]
            feature = torch.cat(feature, dim=1).to(self.features_dtype)
            if self.normalize_features:
                feature = torch.nn.functional.normalize(feature, dim=1)
            feature_loss: torch.Tensor = self.features_losses[self.modality](feature, f_all)
            loss += feature_loss

        # return features losses
        assert isinstance(loss, torch.Tensor), _raise(TypeError("Fetch features losses failed."))
        return loss

    def reset(self) -> None:
        # reset feature losses
        for fn in self.features_losses:
            if isinstance(fn, Loss):
                fn.reset()

        # reset main loss
        if isinstance(self.main_loss, Loss):
            self.main_loss.reset()
        super().reset()


class MAGMSLoss(MAGLoss):
    """
    The targeting loss with feature distillation

    * The features losses will be 0 if input format is not `FeaturedData`

    - Properties:
        - distillation_losses: An optional `list` of `Loss` functions for self distillation
        - features_losses: An optional `list` of `loss` functions for feature distillation
        - features_dtype: A `torch.dtype` of the features type
        - modality: An optional `int` of target modality
        - normalize_features: A `bool` flag of if normalize features
    """
    distillation_losses: torch.nn.ModuleList
    features_dtype: torch.dtype
    features_losses: Optional[torch.nn.ModuleList]
    modality: Optional[int]
    normalize_features: bool

    def __init__(self, losses: list[Loss], distillation_loss: list[Loss], feature_losses: Optional[list[Loss]] = None, *, features_dtype: torch.dtype = torch.float32, modality: Optional[int] = None, normalize_features: bool = False, target: Optional[str] = None, weight: float = 1) -> None:
        super().__init__(losses, modality=modality, target=target, weight=weight)
        self.distillation_losses = torch.nn.ModuleList(distillation_loss)
        self.features_dtype =features_dtype
        self.features_losses = torch.nn.ModuleList(feature_losses) if feature_losses is not None else None
        self.modality = modality
        self.normalize_features = normalize_features

    def forward(self, input: Union[FeaturedData, list[Optional[torch.Tensor]], torch.Tensor], target: Any) -> torch.Tensor:
        # check input type
        if isinstance(input, FeaturedData):
            # forward main loss along with features losses
            loss = super().forward(input.out, target)
            next_out = next((x for x in input.out if x is not None), None) if isinstance(input.out, list) else input.out
            assert next_out is not None, "At least one output must be given."
            loss += self.forward_features(input.features, next_out.shape[0]) if self.features_losses is not None else 0
        else:
            # forward main loss only
            loss = super().forward(input, target)
        return loss

    def forward_features(self, features: list[Optional[Union[torch.Tensor, Iterable[torch.Tensor]]]], batch_size: int) -> torch.Tensor:
        """
        Forward features losses

        - Parameters:
            - features: A `list` of `Iterable` features in `torch.Tensor`
            - batch_size: An `int` of the batch size
        - Returns: A `torch.Tensor` of summed features losses
        """
        # initialize feature loss
        loss = 0

        # fetch fused feature
        assert features[0] is not None, "The fused feature must be given."
        f_all = [features[0].view([batch_size, -1])] if isinstance(features[0], torch.Tensor) else [f.view([batch_size, -1]) for f in features[0]]
        f_all = torch.cat(f_all, dim=1).to(self.features_dtype)

        # normalize fused feature
        if self.normalize_features:
            f_all = torch.nn.functional.normalize(f_all, dim=1)
        assert self.features_losses is not None, "Features losses are not given."

        # feature target feature distillation loss
        if self.modality is None:
            for t, feature in enumerate(features[1:]):
                # check if feature is available
                if feature is None:
                    continue

                # flatten feature
                feature = [feature.view([batch_size, -1])] if isinstance(feature, torch.Tensor) else [f.view([batch_size, -1]) for f in feature]
                feature = torch.cat(feature, dim=1).to(self.features_dtype)

                # normalize feature
                if self.normalize_features:
                    feature = torch.nn.functional.normalize(feature, dim=1)

                # calculate feature loss
                feature_loss = self.features_losses[t](feature, f_all)
                loss += feature_loss
        else:
            target_features = features[self.modality + 1]
            assert target_features is not None, "The target feature must be given."
            feature = [target_features.view([batch_size, -1])] if isinstance(target_features, torch.Tensor) else [f.view([batch_size, -1]) for f in target_features]
            feature = torch.cat(feature, dim=1).to(self.features_dtype)
            if self.normalize_features:
                feature = torch.nn.functional.normalize(feature, dim=1)
            loss += self.features_losses[self.modality](feature, f_all)

        # return features losses
        assert isinstance(loss, torch.Tensor), _raise(TypeError("Fetch features losses failed."))
        return loss

    def forward_target(self, input: torch.Tensor, target: Any, modality: int) -> torch.Tensor:
        # main loss
        x_all = input[:, 0, ...]
        loss = super().forward_target(input, target, modality)

        # distillation loss
        if self.modality is None and modality > 0:
            # fetch modality input
            x = input[:, modality, ...]
            loss += self.distillation_losses[modality-1](x, x_all)
        elif self.modality is not None:
            # fetch modality input
            x = input[:, modality, ...]
            loss += self.distillation_losses[0](x, x_all)
        return loss

    def reset(self) -> None:
        # reset distillation losses
        for fn in self.distillation_losses:
            if isinstance(fn, Loss):
                fn.reset()

        # reset feature losses
        if self.features_losses is not None:
            for fn in self.features_losses:
                if isinstance(fn, Loss):
                    fn.reset()
        super().reset()
