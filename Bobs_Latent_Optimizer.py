"""Bobs Latent Optimizer - empty latent generation with model-aware sizing.

Generates empty latents whose pixel dimensions, channel count and rank are legal
for the selected model family, and derives sensible tile dimensions for a
downstream tiled upscaler.

Every entry in MODEL_SPECS is taken from ComfyUI's own `comfy/latent_formats.py`
(`latent_channels`, `latent_dimensions`, `spacial_downscale_ratio`,
`temporal_downscale_ratio`) and cross-checked against the matching
`Empty*Latent*` node in `nodes.py` / `comfy_extras/`.
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

MAX_TILE_DIM = 2048

# Mirrors MAX_RESOLUTION in ComfyUI's nodes.py. Every built-in latent node caps
# its width/height widgets at this, so emitting anything larger produces a
# latent that fails later - in the sampler or VAE decode - a long way from the
# aspect ratio that actually caused it.
MAX_DIMENSION = 16384

# Model table.
#
#   channels  - latent channel count (latent_formats.latent_channels)
#   vae_scale - spatial downscale from pixels to latent (spacial_downscale_ratio)
#   align     - pixel alignment; MUST be a multiple of vae_scale so that
#               pixel_size // vae_scale is exact and the reported pixel
#               dimensions actually describe the tensor we return
#   temporal  - temporal downscale (temporal_downscale_ratio); video models only
#   dims      - 2 for image models ([B,C,H,W]), 3 for video ([B,C,T,H,W])
#   starts_from_empty - False for models that are never driven from an empty
#               latent (refiners, restorers). Defaults to True.
#
# Video latents use ComfyUI's frame formula: ((length - 1) // temporal) + 1.
#
# Two entries deliberately deviate from the format they map to. QWEN and
# COSMOS_PREDICT2 both use latent_formats.Wan21, which declares
# latent_dimensions=3 and temporal_downscale_ratio=4, because they share Wan's
# VAE. Both are still-image models though - ComfyUI's own Qwen workflows use
# EmptySD3LatentImage, which is 4-D - so we follow the workflow rather than the
# shared format and treat them as 2-D.
MODEL_SPECS = {
    # --- 4-channel SD-family VAE ---
    "SD15": {"channels": 4, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "SD21": {"channels": 4, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "SDXL": {"channels": 4, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "PIXART": {"channels": 4, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "AURAFLOW": {"channels": 4, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "HUNYUAN_DIT": {"channels": 4, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    # --- 16-channel image models ---
    "SD3": {"channels": 16, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "FLUX": {"channels": 16, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "CHROMA": {"channels": 16, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "HIDREAM": {"channels": 16, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "LUMINA2": {"channels": 16, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "OMNIGEN2": {"channels": 16, "vae_scale": 8, "align": 64, "temporal": 1, "dims": 2},
    "QWEN": {"channels": 16, "vae_scale": 8, "align": 16, "temporal": 1, "dims": 2},
    "COSMOS_PREDICT2": {"channels": 16, "vae_scale": 8, "align": 16, "temporal": 1, "dims": 2},
    # --- high-channel / high-compression image models ---
    "FLUX2": {"channels": 128, "vae_scale": 16, "align": 64, "temporal": 1, "dims": 2},
    "HUNYUAN_IMAGE": {"channels": 64, "vae_scale": 32, "align": 64, "temporal": 1, "dims": 2},
    # --- pixel-space image models: no VAE at all, so the "latent" IS the image.
    # Alignment of 16 matches the step on ComfyUI's own pixel-space latent node.
    "CHROMA_RADIANCE": {"channels": 3, "vae_scale": 1, "align": 16, "temporal": 1, "dims": 2},
    "HIDREAM_O1": {"channels": 3, "vae_scale": 1, "align": 16, "temporal": 1, "dims": 2},
    "ZIMAGE_PIXEL": {"channels": 3, "vae_scale": 1, "align": 16, "temporal": 1, "dims": 2},
    "PIXELDIT": {"channels": 3, "vae_scale": 1, "align": 16, "temporal": 1, "dims": 2},
    # --- video models (5-D latents; use the `length` input) ---
    "WAN": {"channels": 16, "vae_scale": 8, "align": 16, "temporal": 4, "dims": 3},
    "WAN22": {"channels": 48, "vae_scale": 16, "align": 32, "temporal": 4, "dims": 3},
    "HUNYUAN_VIDEO": {"channels": 16, "vae_scale": 8, "align": 16, "temporal": 4, "dims": 3},
    "HUNYUAN_VIDEO_15": {"channels": 32, "vae_scale": 16, "align": 32, "temporal": 4, "dims": 3},
    "COSMOS": {"channels": 16, "vae_scale": 8, "align": 16, "temporal": 8, "dims": 3},
    "COGVIDEOX": {"channels": 16, "vae_scale": 8, "align": 16, "temporal": 4, "dims": 3},
    "MOCHI": {"channels": 12, "vae_scale": 8, "align": 16, "temporal": 6, "dims": 3},
    "LTXV": {"channels": 128, "vae_scale": 32, "align": 32, "temporal": 8, "dims": 3},
    # --- shape-only: these models are never driven from an empty latent.
    # SeedVR2 restores existing video (its preprocess node takes an IMAGE), and
    # the HunyuanImage 2.1 refiner consumes the base model's latent. Included so
    # the shapes are available, but selecting them logs a warning.
    "SEEDVR2": {
        "channels": 16, "vae_scale": 8, "align": 16, "temporal": 1, "dims": 3,
        "starts_from_empty": False,
    },
    "HUNYUAN_IMAGE_REFINER": {
        "channels": 64, "vae_scale": 8, "align": 16, "temporal": 1, "dims": 3,
        "starts_from_empty": False,
    },
}

MODEL_TYPES = list(MODEL_SPECS.keys())

VIDEO_MODEL_TYPES = [name for name, spec in MODEL_SPECS.items() if spec["dims"] == 3]

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

TILE_ALIGN = 8

# "," is deliberately NOT a separator. In decimal-comma locales "1,5" means 1.5,
# and treating the comma as a separator would silently read that as 1:5 = 0.2 -
# a 7.5x wrong aspect ratio with no error. Rejecting it is the safer default.
_ASPECT_SEPARATORS = (":", "/", "x", "X")


def round_to_nearest_multiple(value, multiple):
    """Round `value` to the nearest positive multiple of `multiple`."""
    if multiple <= 0:
        return int(round(value))
    return int(round(value / multiple)) * multiple


def parse_aspect_ratio(aspect_ratio):
    """Parse an aspect ratio string into a width/height multiplier.

    Accepts "16:9", "16/9", "16x9" and decimal components such as "1.5:1".
    A bare number ("1.777") is treated as the ratio itself. A comma is not a
    separator - see _ASPECT_SEPARATORS.
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


def compute_base_dimensions(target_area, aspect_ratio_multiplier, align, max_dim=MAX_DIMENSION):
    """Return (width, height) in pixels covering ~`target_area`, aligned to `align`.

    Dimensions are held between one full alignment step and `max_dim` (ComfyUI's
    resolution ceiling). Both bounds distort the requested aspect ratio or area,
    so each one warns when it bites rather than adjusting the result silently.
    """
    if target_area <= 0:
        raise ValueError(f"Target area must be positive, got {target_area}.")
    if align <= 0:
        raise ValueError(f"Alignment must be positive, got {align}.")

    ceiling = (int(max_dim) // align) * align
    if ceiling < align:
        raise ValueError(
            f"max_dim ({max_dim}) is smaller than one alignment step ({align})."
        )

    width = math.sqrt(target_area * aspect_ratio_multiplier)
    height = width / aspect_ratio_multiplier

    # Scale down before aligning so the aspect ratio survives the ceiling; only
    # then clamp, which can still distort it when the other side is at the floor.
    overshoot = max(width / ceiling, height / ceiling, 1.0)
    if overshoot > 1.0:
        logger.warning(
            "Bobs Latent Optimizer: %.0fx%.0f px exceeds the %d px limit; scaling down "
            "by %.2fx. The latent will be smaller than the megapixel target you asked for.",
            width,
            height,
            ceiling,
            overshoot,
        )
        width /= overshoot
        height /= overshoot

    aligned_width = round_to_nearest_multiple(width, align)
    aligned_height = round_to_nearest_multiple(height, align)

    if aligned_width < align or aligned_height < align:
        logger.warning(
            "Bobs Latent Optimizer: %dx%d px is below the %d px minimum for this model; "
            "raising it. The aspect ratio will not match what you asked for.",
            aligned_width,
            aligned_height,
            align,
        )

    width = min(ceiling, max(align, aligned_width))
    height = min(ceiling, max(align, aligned_height))
    return width, height


def compute_latent_frames(length, temporal):
    """Latent temporal size for `length` pixel-space frames.

    Matches ComfyUI's video latent nodes: ((length - 1) // temporal) + 1.
    """
    length = max(1, int(length))
    if temporal <= 1:
        return length
    return ((length - 1) // temporal) + 1


def compute_tile_dimensions(width, height, upscale_by, max_tile_dim=MAX_TILE_DIM):
    """Suggest tile dimensions for the upscaled pixel output.

    Aims for a 2x2 grid, adding tiles along an axis only when a 2x2 tile would
    exceed `max_tile_dim`. Tile dimensions are aligned up to a multiple of
    TILE_ALIGN because tiled VAE/upscaler nodes expect that.

    Returns (tile_width, tile_height, tiles_x, tiles_y).
    """
    upscaled_width = max(1, int(width * upscale_by))
    upscaled_height = max(1, int(height * upscale_by))
    max_tile_dim = max(TILE_ALIGN, int(max_tile_dim))

    def axis_tiles(total):
        tiles = 2
        if -(-total // tiles) > max_tile_dim:
            tiles = -(-total // max_tile_dim)
        return max(1, tiles)

    tiles_x = axis_tiles(upscaled_width)
    tiles_y = axis_tiles(upscaled_height)

    def tile_size(total, tiles):
        size = -(-total // tiles)
        # Round the tile up to the upscaler stride, but never past the whole image.
        size = -(-size // TILE_ALIGN) * TILE_ALIGN
        return max(TILE_ALIGN, min(size, total))

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


def _latent_dtype():
    """Intermediate dtype, or None when it should not be passed.

    ComfyUI's image latent nodes (EmptyLatentImage, EmptySD3LatentImage) pass
    dtype=intermediate_dtype(); its video latent nodes (Wan, Hunyuan, Mochi,
    LTXV) pass only device. We follow that split per model family rather than
    applying one rule to both.
    """
    if model_management is None:
        return None
    return model_management.intermediate_dtype()


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
                        "Model family. Sets latent channels, VAE downscale and pixel alignment. "
                        "Video families (" + ", ".join(VIDEO_MODEL_TYPES) + ") produce a 5-D "
                        "latent and use the `length` input. See the README for the full table."
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
            "length": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": (
                        "Number of video frames. Only used by video model families; ignored "
                        "(with a warning) for image models."
                    ),
                },
            ),
        }

    def _build(
        self,
        aspect_ratio,
        target_area,
        upscale_by,
        model_type,
        batch_size,
        max_tile_size,
        length,
    ):
        spec = MODEL_SPECS.get(model_type)
        if spec is None:
            raise ValueError(
                f"Unknown model_type {model_type!r}. Expected one of {', '.join(MODEL_TYPES)}."
            )

        if not spec.get("starts_from_empty", True):
            logger.warning(
                "Bobs Latent Optimizer: %s is not normally driven from an empty latent - it "
                "consumes an existing image or latent. This gives you a correctly shaped zero "
                "tensor, but it is probably not the input that model wants.",
                model_type,
            )

        aspect_ratio_multiplier = parse_aspect_ratio(aspect_ratio)
        width, height = compute_base_dimensions(target_area, aspect_ratio_multiplier, spec["align"])

        vae_scale = spec["vae_scale"]
        latent_width = width // vae_scale
        latent_height = height // vae_scale
        channels = spec["channels"]

        if spec["dims"] == 3:
            frames = compute_latent_frames(length, spec["temporal"])
            shape = [batch_size, channels, frames, latent_height, latent_width]
        else:
            frames = None
            if length > 1:
                logger.warning(
                    "Bobs Latent Optimizer: length=%d ignored - %s is an image model and "
                    "produces a 4-D latent. Pick a video model (%s) to use length.",
                    length,
                    model_type,
                    ", ".join(VIDEO_MODEL_TYPES),
                )
            shape = [batch_size, channels, latent_height, latent_width]

        allocate_kwargs = {"device": _latent_device()}
        if spec["dims"] == 2:
            dtype = _latent_dtype()
            if dtype is not None:
                allocate_kwargs["dtype"] = dtype

        try:
            samples = torch.zeros(shape, **allocate_kwargs)
        except Exception as error:
            raise RuntimeError(
                f"Could not allocate latent of shape {shape} for {model_type}: {error}"
            )

        tile_width, tile_height, tiles_x, tiles_y = compute_tile_dimensions(
            width, height, upscale_by, max_tile_size
        )

        logger.info(
            "Bobs Latent Optimizer: %s %dx%d px%s -> latent %s (/%d, %d channels) | "
            "upscaled %dx%d px in a %dx%d grid of %dx%d tiles",
            model_type,
            width,
            height,
            "" if frames is None else f" x {int(length)} frames",
            tuple(shape),
            vae_scale,
            channels,
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
    model family, and the latent is allocated with that family's channel count,
    VAE downscale and rank. Also returns tile dimensions for a downstream tiled
    upscaler, targeting a 2x2 grid unless that would push a tile past
    `max_tile_size`.
    """

    DESCRIPTION = (
        "Empty latent sized for a wide range of image and video model families from an "
        "aspect ratio and a preset megapixel area, plus suggested tile dimensions for "
        "tiled upscaling."
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

    def generate(
        self,
        aspect_ratio,
        mp_size,
        upscale_by,
        model_type,
        batch_size,
        max_tile_size=MAX_TILE_DIM,
        length=1,
    ):
        target_area = MP_SIZE_TO_AREA.get(mp_size)
        if target_area is None:
            raise ValueError(
                f"Unknown mp_size {mp_size!r}. Expected one of {', '.join(MP_SIZES)}."
            )
        return self._build(
            aspect_ratio, target_area, upscale_by, model_type, batch_size, max_tile_size, length
        )


class BobsLatentNodeAdvanced(_BobsLatentBase):
    """Same as Bobs Latent Optimizer, but with a continuous megapixel target.

    Use this when you want an exact area rather than one of the presets.
    """

    DESCRIPTION = (
        "Empty latent sized for a wide range of image and video model families from an "
        "aspect ratio and a continuous megapixel target, plus suggested tile dimensions "
        "for tiled upscaling."
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

    def generate(
        self,
        aspect_ratio,
        mp_size_float,
        upscale_by,
        model_type,
        batch_size,
        max_tile_size=MAX_TILE_DIM,
        length=1,
    ):
        if mp_size_float <= 0:
            raise ValueError(f"mp_size_float must be greater than zero, got {mp_size_float}.")
        return self._build(
            aspect_ratio,
            mp_size_float * MP_BASE_AREA,
            upscale_by,
            model_type,
            batch_size,
            max_tile_size,
            length,
        )


NODE_CLASS_MAPPINGS = {
    "BobsLatentNode": BobsLatentNode,
    "BobsLatentNodeAdvanced": BobsLatentNodeAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BobsLatentNode": "Bobs Latent Optimizer",
    "BobsLatentNodeAdvanced": "Bobs Latent Optimizer (Advanced)",
}
