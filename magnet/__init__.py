from . import data, losses, networks, nn
from .managers import MonaiTargetingManager as MonaiManager, TargetingManager as Manager
from .nn import MAGNET, MAGNET2

try:
    from .builder import build, build_v1, build_v2, build_v2_unet
except ImportError:
    build = build_v1 = build_v2 = build_v2_unet = NotImplemented

VERSION = "2.1"
