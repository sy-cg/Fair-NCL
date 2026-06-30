"""Method layer for Fair-NCL experiments.

Backbones live in :mod:`models`. This package contains method-level wrappers,
augmentation policies, and shared training utilities.
"""

from .registry import METHOD_NAMES, create_method

__all__ = ["METHOD_NAMES", "create_method"]
