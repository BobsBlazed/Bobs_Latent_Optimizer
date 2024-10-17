
from nodes import EmptyLatentImage
import torch

class BobsFluxSDXLLatentNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ("STRING", {"default": "1:1"}),
                "mp_size": (["1", "1.25", "1.5", "1.75", "2"], {"default": "1"}),
                "upscale_by": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": .01
                }),
                "mode": (["FLUX", "SDXL"], {"default": "FLUX"})
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("latent", "tile_width", "tile_height", "upscale_by")
    FUNCTION = "generate"

    def generate(self, aspect_ratio, mp_size, upscale_by, mode):
        try:
            ratio_w, ratio_h = map(int, aspect_ratio.split(":"))
        except ValueError:
            raise ValueError(f"Invalid aspect ratio format: {aspect_ratio}. Please use the format 'x:y'.")

        mp_to_size = {
            "1": 1024 * 1024,
            "1.25": 1280 * 1024,
            "1.5": 1440 * 1024,
            "1.75": 1600 * 1024,
            "2": 1920 * 1080
        }

        target_area = mp_to_size[mp_size]
        aspect_ratio_multiplier = (ratio_w / ratio_h)

        target_width = int((target_area * aspect_ratio_multiplier) ** 0.5)
        target_height = int(target_width / aspect_ratio_multiplier)

        if mode == "SDXL":
            target_width = (target_width // 64) * 64
            target_height = (target_height // 64) * 64

        latent = EmptyLatentImage().generate(target_width, target_height, 1)[0]

        tile_width = int(target_width * upscale_by) // 2
        tile_height = int(target_height * upscale_by) // 2

        return (
            latent,
            tile_width,
            tile_height,
            upscale_by
        )

NODE_CLASS_MAPPINGS = {
    "BobsFluxSDXLLatentNode": BobsFluxSDXLLatentNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BobsFluxSDXLLatentNode": "Bobs FLUX/SDXL Latent Optimizer"
}
