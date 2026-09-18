"""Bobs Latent Optimizer - ComfyUI custom node package entry point.

ComfyUI imports this package and reads NODE_CLASS_MAPPINGS and
NODE_DISPLAY_NAME_MAPPINGS from it. Both are re-exported from the
implementation module rather than redefined here: the two dicts are matched by
key, so keeping a second copy in this file is how the display names silently
stopped applying before 1.3.0.
"""

from .Bobs_Latent_Optimizer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
