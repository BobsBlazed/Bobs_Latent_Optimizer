"""Bobs Latent Optimizer - empty latent generation with model-aware sizing.

Generates empty latents whose pixel dimensions are legal for the selected model
family, and derives sensible tile dimensions for a downstream tiled upscaler.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)

# ComfyUI keeps freshly allocated latents on an "intermediate" device so they do
# not pin VRAM before the sampler needs them. Fall back to CPU when the node is
# imported outside of ComfyUI (tests, tooling).
try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None


MP_BASE_AREA = 1024 * 1024

VAE_SCALE_FACTOR = 8

MAX_TILE_DIM = 2048

# Per-model latent channel count and pixel alignment.
#
# `channels` is the channel count of the model's VAE: SDXL is the only 4-channel
# family here. FLUX, SD3/3.5, Qwen-Image and Wan all use 16-channel VAEs.
#
# `align` must stay a multiple of VAE_SCALE_FACTOR so that
# pixel_size // VAE_SCALE_FACTOR is exact and the reported pixel dimensions
# actually describe the tensor we return.
MODEL_SPECS = {
    "FLUX": {"channels": 16, "align": 64},
    "SDXL": {"channels": 4, "align": 64},
    "SD3": {"channels": 16, "align": 64},
    "QWEN": {"channels": 16, "align": 16},
    "WAN": {"channels": 16, "align": 16},
}

MODEL_TYPES = list(MODEL_SPECS.keys())

# Discrete area presets. These are approximate megapixel labels mapped to common
# standard resolution areas rather than exact multiples of 1MP.
MP_SIZE_TO_AREA = {
    "0.25": 512 * 512,
    "0.5": 768 * 768,
    "1": 1024 * 1024,
    "1.25": 1280 * 1024,
    "1.5": 1440 * 1080,
    "1.75": 1664 * 1088,
    "2": 1920 * 1080,
    "2.5": 1536 * 1536,
    "3": 1792 * 1792,
    "4": 2048 * 2048,
}

MP_SIZES = list(MP_SIZE_TO_AREA.keys())

_ASPECT_SEPARATORS = (":", "/", "x", "X", ",")


def round_to_nearest_multiple(value, multiple):
    """Round `value` to the nearest positive multiple of `multiple`."""
    if multiple <= 0:
        return int(round(value))
    return int(round(value / multiple)) * multiple


def parse_aspect_ratio(aspect_ratio):
    """Parse an aspect ratio string into a width/height multiplier.

    Accepts "16:9", "16/9", "16x9", "16,9" and decimal components such as
    "1.5:1". A bare number ("1.777") is treated as the ratio itself.
    """
    if isinstance(aspect_ratio, (int, float)):
        parts = [str(aspect_ratio)]
    else:
        text = str(aspect_ratio).strip()
        if not text:
            raise ValueError("Aspect ratio is empty. Use a format like '1:1' or '16:9'.")
        parts = [text]
        for separator in _ASPECT_SEPARATORS:
            if separator in text:
                parts = text.split(separator)
                break

    try:
        numbers = [float(part.strip()) for part in parts]
    except ValueError:
        raise ValueError(
            f"Invalid aspect ratio: {aspect_ratio!r}. Use 'width:height' with numeric "
            "components, for example '1:1', '16:9' or '3:2'."
        )

    if len(numbers) == 1:
        ratio = numbers[0]
    elif len(numbers) == 2:
        width, height = numbers
        if height == 0:
            raise ValueError(
                f"Invalid aspect ratio: {aspect_ratio!r}. The height component cannot be zero."
            )
        ratio = width / height
    else:
        raise ValueError(
            f"Invalid aspect ratio: {aspect_ratio!r}. Expected two components, got {len(numbers)}."
        )

    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(
            f"Invalid aspect ratio: {aspect_ratio!r}. The ratio must be a positive number."
        )
    return ratio


def compute_base_dimensions(target_area, aspect_ratio_multiplier, align):
    """Return (width, height) in pixels covering ~`target_area`, aligned to `align`.

    Both dimensions are clamped to at least one full alignment step so tiny
    megapixel targets cannot produce a zero-sized latent.
    """
    if target_area <= 0:
        raise ValueError(f"Target area must be positive, got {target_area}.")

    width = math.sqrt(target_area * aspect_ratio_multiplier)
    height = width / aspect_ratio_multiplier

    minimum = max(align, VAE_SCALE_FACTOR)
    width = max(minimum, round_to_nearest_multiple(width, align))
    height = max(minimum, round_to_nearest_multiple(height, align))
    return width, height


def compute_tile_dimensions(width, height, upscale_by, max_tile_dim=MAX_TILE_DIM):
    """Suggest tile dimensions for the upscaled pixel output.

    Aims for a 2x2 grid, adding tiles along an axis only when a 2x2 tile would
    exceed `max_tile_dim`. Tile dimensions are aligned up to a multiple of
    VAE_SCALE_FACTOR because tiled VAE/upscaler nodes expect that.

    Returns (tile_width, tile_height, tiles_x, tiles_y).
    """
    upscaled_width = max(1, int(width * upscale_by))
    upscaled_height = max(1, int(height * upscale_by))
    max_tile_dim = max(VAE_SCALE_FACTOR, int(max_tile_dim))

    def axis_tiles(total):
        tiles = 2
        if -(-total // tiles) > max_tile_dim:
            tiles = -(-total // max_tile_dim)
        return max(1, tiles)

    tiles_x = axis_tiles(upscaled_width)
    tiles_y = axis_tiles(upscaled_height)

    def tile_size(total, tiles):
        size = -(-total // tiles)
        # Round the tile up to the VAE stride, but never past the whole image.
        size = -(-size // VAE_SCALE_FACTOR) * VAE_SCALE_FACTOR
        return max(VAE_SCALE_FACTOR, min(size, total))

    return (
        tile_size(upscaled_width, tiles_x),
        tile_size(upscaled_height, tiles_y),
        tiles_x,
        tiles_y,
    )


def _latent_device():
    if model_management is not None:
        return model_management.intermediate_device()
    return torch.device("cpu")


class _BobsLatentBase:
    """Shared sizing, tiling and tensor allocation for both node variants."""

    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("latent", "tile_width", "tile_height", "upscale_by", "width", "height")
    OUTPUT_TOOLTIPS = (
        "Empty latent batch sized for the selected model.",
        "Suggested tile width for a tiled upscaler operating on the upscaled pixel output.",
        "Suggested tile height for a tiled upscaler operating on the upscaled pixel output.",
        "The upscale factor, passed through unchanged for convenience.",
        "Base image width in pixels.",
        "Base image height in pixels.",
    )
    FUNCTION = "generate"
    CATEGORY = "latent/generate"

    @staticmethod
    def _shared_inputs():
        return {
            "upscale_by": (
                "FLOAT",
                {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": (
                        "Upscale factor for the FINAL output image. Used only to compute the "
                        "tile dimensions; the generated latent is NOT upscaled."
                    ),
                },
            ),
            "model_type": (
                MODEL_TYPES,
                {
                    "default": "FLUX",
                    "tooltip": (
                        "Model family. Sets latent channels (SDXL=4, FLUX/SD3/QWEN/WAN=16) and "
                        "pixel alignment (64 for FLUX/SDXL/SD3, 16 for QWEN/WAN)."
                    ),
                },
            ),
            "batch_size": (
                "INT",
                {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "Number of latents in the batch."},
            ),
        }

    @staticmethod
    def _optional_inputs():
        return {
            "max_tile_size": (
                "INT",
                {
                    "default": MAX_TILE_DIM,
                    "min": 256,
                    "max": 8192,
                    "step": 64,
                    "tooltip": (
                        "Largest tile edge allowed before the tile grid is subdivided further. "
                        "Lower this if your upscaler runs out of VRAM."
                    ),
                },
            ),
        }

    def _build(self, aspect_ratio, target_area, upscale_by, model_type, batch_size, max_tile_size):
        spec = MODEL_SPECS.get(model_type)
        if spec is None:
            raise ValueError(
                f"Unknown model_type {model_type!r}. Expected one of {', '.join(MODEL_TYPES)}."
            )

        aspect_ratio_multiplier = parse_aspect_ratio(aspect_ratio)
        width, height = compute_base_dimensions(target_area, aspect_ratio_multiplier, spec["align"])

        latent_width = width // VAE_SCALE_FACTOR
        latent_height = height // VAE_SCALE_FACTOR
        channels = spec["channels"]

        try:
            samples = torch.zeros(
                [batch_size, channels, latent_height, latent_width],
                device=_latent_device(),
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not allocate latent of shape "
                f"[{batch_size}, {channels}, {latent_height}, {latent_width}] for {model_type}: {error}"
            )

        tile_width, tile_height, tiles_x, tiles_y = compute_tile_dimensions(
            width, height, upscale_by, max_tile_size
        )

        logger.info(
            "Bobs Latent Optimizer: %s %dx%d px (latent %dx%d, %d channels, batch %d) -> "
            "upscaled %dx%d px in a %dx%d grid of %dx%d tiles",
            model_type,
            width,
            height,
            latent_width,
            latent_height,
            channels,
            batch_size,
            int(width * upscale_by),
            int(height * upscale_by),
            tiles_x,
            tiles_y,
            tile_width,
            tile_height,
        )

        return ({"samples": samples}, tile_width, tile_height, upscale_by, width, height)


class BobsLatentNode(_BobsLatentBase):
    """Generate an empty latent from an aspect ratio and a preset megapixel area.

    Pixel dimensions are rounded to the nearest alignment step for the selected
    model family, and the latent is allocated with that family's channel count.
    Also returns tile dimensions for a downstream tiled upscaler, targeting a
    2x2 grid unless that would push a tile past `max_tile_size`.
    """

    DESCRIPTION = (
        "Empty latent sized for FLUX / SDXL / SD3 / Qwen / Wan from an aspect ratio and a "
        "preset megapixel area, plus suggested tile dimensions for tiled upscaling."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "aspect_ratio": (
                "STRING",
                {
                    "default": "1:1",
                    "tooltip": "Aspect ratio of the base image, e.g. '1:1', '16:9', '3:2'.",
                },
            ),
            "mp_size": (
                MP_SIZES,
                {
                    "default": "1",
                    "tooltip": (
                        "Approximate megapixel area of the base image. Values map to common "
                        "standard resolution areas (1 = 1024x1024, 4 = 2048x2048)."
                    ),
                },
            ),
        }
        required.update(cls._shared_inputs())
        return {"required": required, "optional": cls._optional_inputs()}

    def generate(self, aspect_ratio, mp_size, upscale_by, model_type, batch_size, max_tile_size=MAX_TILE_DIM):
        target_area = MP_SIZE_TO_AREA.get(mp_size)
        if target_area is None:
            raise ValueError(
                f"Unknown mp_size {mp_size!r}. Expected one of {', '.join(MP_SIZES)}."
            )
        return self._build(aspect_ratio, target_area, upscale_by, model_type, batch_size, max_tile_size)


class BobsLatentNodeAdvanced(_BobsLatentBase):
    """Same as Bobs Latent Optimizer, but with a continuous megapixel target.

    Use this when you want an exact area rather than one of the presets.
    """

    DESCRIPTION = (
        "Empty latent sized for FLUX / SDXL / SD3 / Qwen / Wan from an aspect ratio and a "
        "continuous megapixel target, plus suggested tile dimensions for tiled upscaling."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "aspect_ratio": (
                "STRING",
                {
                    "default": "1:1",
                    "tooltip": "Aspect ratio of the base image, e.g. '1:1', '16:9', '3:2'.",
                },
            ),
            "mp_size_float": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 16.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": (
                        f"Target area in megapixels, where 1.0 = {MP_BASE_AREA} pixels "
                        "(1024x1024). 4.0 is a 2048x2048 area."
                    ),
                },
            ),
        }
        required.update(cls._shared_inputs())
        return {"required": required, "optional": cls._optional_inputs()}

    def generate(self, aspect_ratio, mp_size_float, upscale_by, model_type, batch_size, max_tile_size=MAX_TILE_DIM):
        if mp_size_float <= 0:
            raise ValueError(f"mp_size_float must be greater than zero, got {mp_size_float}.")
        return self._build(
            aspect_ratio, mp_size_float * MP_BASE_AREA, upscale_by, model_type, batch_size, max_tile_size
        )


NODE_CLASS_MAPPINGS = {
    "BobsLatentNode": BobsLatentNode,
    "BobsLatentNodeAdvanced": BobsLatentNodeAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BobsLatentNode": "Bobs Latent Optimizer",
    "BobsLatentNodeAdvanced": "Bobs Latent Optimizer (Advanced)",
}
