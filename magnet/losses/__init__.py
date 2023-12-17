from torchmanager.losses import * # type: ignore

from .distillation import PixelWiseKLDiv, MAGFeatureDistillationLoss, MAGSelfDistillationLoss, MAGMSLoss
from .targeting import MAGLoss
