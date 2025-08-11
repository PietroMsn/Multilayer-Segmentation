from .builder import build_criteria, LOSSES

from .misc import (
    CrossEntropyLoss,
    SmoothCELoss,
    DiceLoss,
    FocalLoss,
    BinaryFocalLoss,
    MultiHeadLoss,
)
from .lovasz import LovaszLoss
