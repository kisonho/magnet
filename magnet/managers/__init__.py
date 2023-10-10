from .targeting import Manager as TargetingManager

try:
    from .monai import SegmentationManager as MonaiTargetingManager
except ImportError:
    MonaiTargetingManager = NotImplemented