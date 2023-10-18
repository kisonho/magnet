from . import data, networks, nn
from .builder import MAGNET, MAGNET2, build, build_v1, build_v2, build_v2_unet
from .managers import MonaiTargetingManager as MonaiManager, TargetingManager as Manager

VERSION = "2.0.1"
