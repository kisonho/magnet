from torchmanager.losses import * # type: ignore

from .distillation import PixelWiseKLDiv, MAGFeatureDistillationLoss, MAGSelfDistillationLoss
from .targeting import MAGLoss