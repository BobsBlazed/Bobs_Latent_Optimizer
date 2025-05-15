from .Bobs_Latent_Optimizer import BobsLatentNode import BobsLatentNodeAdvanced

# --- NODE MAPPINGS ---
NODE_CLASS_MAPPINGS = {
    "BobsLatentNode": BobsLatentNode,
    "BobsLatentNodeAdvanced": BobsLatentNodeAdvanced
}

# --- NODE DISPLAY NAMES ---
NODE_DISPLAY_NAME_MAPPINGS = {
    "Bobs-Latent-Optimizer": "Bobs Latent Optimizer",
    "Bobs-Advanced-Latent-Optimizer": "Bobs Latent Optimizer (Advanced)"
}
