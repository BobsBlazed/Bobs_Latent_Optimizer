from .Bobs_Latent_Optimizer import BobsLatentNode
from .Bobs_Latent_Optimizer import BobsLatentNodeAdvanced

NODE_CLASS_MAPPINGS = {
    "BobsLatentNode": BobsLatentNode,
    "BobsLatentNodeAdvanced": BobsLatentNodeAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Bobs-Latent-Optimizer": "Bobs Latent Optimizer",
    "Bobs-Advanced-Latent-Optimizer": "Bobs Latent Optimizer (Advanced)"
}

print("✨ Bobs Latent Optimizer nodes loaded! ✨")
