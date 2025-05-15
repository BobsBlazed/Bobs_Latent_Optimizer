# __init__.py

"""
@author: BobsBlazed
@title: Bobs Latent Optimizer
@nickname: BobsLatent
@description: Nodes to generate empty latent images optimized for FLUX, SDXL, and SD3 models.
"""

# Import the node mappings from your main node file
from .Bobs_Latent_Optimizer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# A dictionary that contains all nodes you want to be exposed in the UI
# NOTE: It is usually set to NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = NODE_CLASS_MAPPINGS

# A dictionary that contains the friendly/user friendly category names
NODE_DISPLAY_NAME_MAPPINGS = NODE_DISPLAY_NAME_MAPPINGS

# If you had a separate web directory with frontend JavaScript for your nodes,
# you would specify it here. Since these nodes seem to use standard ComfyUI widgets,
# you likely don't need a custom web directory.
# WEB_DIRECTORY = "./web"

print("✨ Bobs Latent Optimizer nodes loaded! ✨")
