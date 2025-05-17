import torch
import math # Needed for sqrt

# Helper for rounding to nearest multiple
def round_to_nearest_multiple(value, multiple):
    """Rounds a value to the nearest multiple of 'multiple'."""
    if multiple <= 0:
        return value # Avoid division by zero or non-positive multiple
    # Use int(round()) for standard rounding behavior
    return int(round(value / multiple) * multiple)

# Define the base unit for Megapixels as 1024*1024 pixels for calculation purposes
MP_BASE_AREA = 1024 * 1024

# --- BobsLatentNode ---
class BobsLatentNode:
    """
    Generates an empty latent image optimized for FLUX, SDXL, or SD3 models
    based on aspect ratio, approximate discrete megapixel size options, and batch size.
    Calculates dimensions by rounding to the NEAREST multiple of 64 for model compatibility.
    Handles the correct number of latent channels (4 for SDXL/SD3, 16 for FLUX).
    Calculates tile dimensions for a 2x2 tiling grid of the *upscaled pixel output*
    to help optimize tiled upscale times.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ("STRING", {"default": "1:1", "tooltip": "Target image aspect ratio (e.g., '1:1', '16:9', '3:2'). This determines the shape of the BASE latent image."}),
                # Extended MP size options to include values mapping up to 2048x2048 area (~4MP base)
                "mp_size": (["0.25", "0.5", "1", "1.25", "1.5", "1.75", "2", "2.5", "3", "4"], {"default": "1", "tooltip": "Approximate target megapixel area for the BASE latent image. These options map to common standard resolution areas (e.g., 1 is 1024x1024 area, 4 is 2048x2048 area)."}),
                "upscale_by": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": .01,
                    "tooltip": "Desired upscale factor for the FINAL output image. Used to calculate tiling dimensions for a 2x2 grid in pixel space. Does NOT upscale the generated latent."
                }),
                "model_type": (["FLUX", "SDXL", "SD3"], {"default": "FLUX", "tooltip": "Select the target model type to set base resolution rounding rules and latent channel count (FLUX=16, SDXL/SD3=4)."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "Number of latent images in the batch."})
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("latent", "tile_width", "tile_height", "upscale_by")
    FUNCTION = "generate"
    CATEGORY = "latent/generate"

    def generate(self, aspect_ratio, mp_size, upscale_by, model_type, batch_size):

        # --- 1. Calculate Target BASE Pixel Dimensions ---
        try:
            ratio_w, ratio_h = map(int, aspect_ratio.split(":"))
            if ratio_h == 0: # Prevent division by zero
                 raise ValueError("Height ratio cannot be zero.")
            aspect_ratio_multiplier = (ratio_w / ratio_h)
        except ValueError as e:
            raise ValueError(f"Invalid aspect ratio format: {aspect_ratio}. Please use the format 'x:y' where x and y are integers (e.g., '1:1', '16:9'). Detail: {e}")


        # Map MP size string to a target area in pixels.
        # These values are chosen based on common standard resolution areas and the FLUX training size.
        mp_to_area = {
            "0.25": 512 * 512,   # Common ~0.25MP resolution area
            "0.5": 768 * 768,    # Common ~0.5MP resolution area
            "1": 1024 * 1024,    # Common 1MP resolution area
            "1.25": 1280 * 1024, # Approx 1.25MP area (SXGA)
            "1.5": 1440 * 1080,  # Common resolution area (~1.5MP)
            "1.75": 1664 * 1088, # Approx 1.75MP area
            "2": 1920 * 1080,    # Common ~2MP resolution area
            "2.5": 1536 * 1536,  # Common resolution area (~2.25MP)
            "3": 1792 * 1792,  # Common resolution area (~3MP)
            "4": 2048 * 2048,  # FLUX training resolution area (~4MP base)
        }

        target_area = mp_to_area.get(mp_size, 1024 * 1024) # Default to 1MP area if mp_size is somehow invalid

        # Calculate initial base dimensions based on area and aspect ratio (using floats)
        # Initial_W = sqrt(Area * aspect_ratio_multiplier)
        # Initial_H = Initial_W / aspect_ratio_multiplier
        initial_target_width_float = math.sqrt(target_area * aspect_ratio_multiplier)
        initial_target_height_float = initial_target_width_float / aspect_ratio_multiplier # Use width for recalculation

        # --- 2. Apply Model-Specific BASE Sizing and Get Latent Channels ---
        # Pixel dimensions must be divisible by VAE scale * latent patch size (usually 8 * 8 = 64)
        vae_scale_factor = 8 # Standard for these models (8x downsampling)

        # Default latent channels to 4 (SDXL/SD3)
        latent_channels = 4

        # Round initial dimensions to the NEAREST multiple of 64
        target_width = round_to_nearest_multiple(initial_target_width_float, 64)
        target_height = round_to_nearest_multiple(initial_target_height_float, 64)


        if model_type == "SDXL" or model_type == "SD3":
            # SDXL/SD3 use 4 channels (already default)

            if model_type == "SD3":
                # SD3 has specific target area scaling logic based on a 1024x1024 reference
                # Apply scaling AFTER initial rounding to 64
                target_area_sd3_ref = 1024 * 1024
                current_area = target_width * target_height
                # Prevent division by zero if current_area is 0
                if current_area > 0:
                     scaling_factor = (target_area_sd3_ref / current_area) ** 0.5
                     target_width = int(target_width * scaling_factor) # Apply scaling
                     target_height = int(target_height * scaling_factor)
                     # Re-round to 64 after scaling
                     target_width = round_to_nearest_multiple(target_width, 64)
                     target_height = round_to_nearest_multiple(target_height, 64)
                else:
                     print(f"Warning: Calculated base dimensions were zero after initial rounding, skipping SD3 scaling.")


        elif model_type == "FLUX":
            # FLUX uses 16 latent channels
            latent_channels = 16
            # No special SD3-like scaling needed for FLUX

        # Ensure minimum valid base pixel dimensions (must be at least 64 for 64x64 divisibility)
        min_dim = 64
        if target_width < min_dim:
             print(f"Warning: Calculated base width was {target_width}. Clamping to minimum {min_dim}.")
             target_width = min_dim
        if target_height < min_dim:
             print(f"Warning: Calculated base height was {target_height}. Clamping to minimum {min_dim}.")
             target_height = min_dim

        # --- 3. Generate Empty Latent Image at BASE Resolution with Correct Channels ---
        # Calculate latent dimensions from final pixel dimensions
        latent_width = target_width // vae_scale_factor
        latent_height = target_height // vae_scale_factor

        try:
            # Manually create a zero tensor with the correct batch size, channels, and latent dimensions
            latent_tensor = torch.zeros([batch_size, latent_channels, latent_height, latent_width])
            # ComfyUI's latent format is a dictionary containing the samples tensor
            latent = {"samples": latent_tensor}

        except Exception as e:
             # Updated error message to reflect manual tensor creation
             raise RuntimeError(f"Error creating empty latent tensor with shape [{batch_size}, {latent_channels}, {latent_height}, {latent_width}] for {model_type}: {e}")

        print(f"Generated {model_type} base latent: Batch={batch_size}, Channels={latent_channels}, Latent Size=({latent_width}x{latent_height}), Base Pixel Size=({target_width}x{target_height})")


        # --- 4. Calculate Tile Dimensions for 2x2 Tiling of the FINAL Upscaled Pixel Output ---
        # tile dimensions are for a 2x2 grid in the upscaled pixel space.
        # Each tile is half the size of the total upscaled dimension.
        upscaled_total_width = int(target_width * upscale_by)
        upscaled_total_height = int(target_height * upscale_by)

        tile_width = upscaled_total_width // 2
        tile_height = upscaled_total_height // 2

        # Ensure tile dimensions are at least 1
        if tile_width <= 0:
             print(f"Warning: Calculated tile width was {tile_width}. Clamping to minimum 1.")
             tile_width = 1
        if tile_height <= 0:
             print(f"Warning: Calculated tile height was {tile_height}. Clamping to minimum 1.")
             tile_height = 1

        print(f"Calculated tile dimensions for 2x2 grid of upscaled pixel output (upscale_by {upscale_by}): {tile_width}x{tile_height}")


        # --- 5. Return Outputs ---
        return (
            latent, # The generated base latent image (now a dictionary)
            tile_width, # Suggested tile width for a 2x2 grid of the upscaled image
            tile_height, # Suggested tile height for a 2x2 grid of the upscaled image
            upscale_by # The upscale factor used for calculation
        )

# --- New Advanced Node: BobsLatentNodeAdvanced ---
class BobsLatentNodeAdvanced:
    """
    Generates an empty latent image optimized for FLUX, SDXL, or SD3 models
    based on aspect ratio, a continuous float megapixel size, and batch size.
    Calculates dimensions by rounding to the NEAREST multiple of 64 for model compatibility.
    Handles the correct number of latent channels (4 for SDXL/SD3, 16 for FLUX).
    Calculates tile dimensions for a 2x2 tiling grid of the *upscaled pixel output*
    to help optimize tiled upscale times.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ("STRING", {"default": "1:1", "tooltip": "Target image aspect ratio (e.g., '1:1', '16:9', '3:2'). This determines the shape of the BASE latent image."}),
                # mp_size as a float, range up to 4.0 to cover 2048x2048 area
                "mp_size_float": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01, "display": "number", "tooltip": f"Target megapixel area (in millions of pixels, based on {int(MP_BASE_AREA/1000000)} MP = {MP_BASE_AREA} pixels) for the BASE latent image. Range 0.01 to 4.0 (4.0 is 2048x2048 area)."}),
                "upscale_by": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": .01,
                    "tooltip": "Desired upscale factor for the FINAL output image. Used to calculate tiling dimensions for a 2x2 grid in pixel space. Does NOT upscale the generated latent."
                }),
                "model_type": (["FLUX", "SDXL", "SD3"], {"default": "FLUX", "tooltip": "Select the target model type to set base resolution rounding rules and latent channel count (FLUX=16, SDXL/SD3=4)."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1, "tooltip": "Number of latent images in the batch."})
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("latent", "tile_width", "tile_height", "upscale_by")
    FUNCTION = "generate"
    CATEGORY = "latent/generate"

    def generate(self, aspect_ratio, mp_size_float, upscale_by, model_type, batch_size):

        # --- 1. Calculate Target BASE Pixel Dimensions ---
        try:
            ratio_w, ratio_h = map(int, aspect_ratio.split(":"))
            if ratio_h == 0: # Prevent division by zero
                 raise ValueError("Height ratio cannot be zero.")
            aspect_ratio_multiplier = (ratio_w / ratio_h)
        except ValueError as e:
            raise ValueError(f"Invalid aspect ratio format: {aspect_ratio}. Please use the format 'x:y' where x and y are integers (e.g., '1:1', '16:9'). Detail: {e}")

        # Calculate target area directly from the float mp_size (based on MP_BASE_AREA)
        target_area = mp_size_float * MP_BASE_AREA

        # Calculate initial base dimensions based on area and aspect ratio (using floats)
        initial_target_width_float = math.sqrt(target_area * aspect_ratio_multiplier)
        initial_target_height_float = initial_target_width_float / aspect_ratio_multiplier # Use width for recalculation

        # --- 2. Apply Model-Specific BASE Sizing and Get Latent Channels ---
        vae_scale_factor = 8 # Standard for these models (8x downsampling)

        # Default latent channels to 4 (SDXL/SD3)
        latent_channels = 4

        # Round initial dimensions to the NEAREST multiple of 64
        target_width = round_to_nearest_multiple(initial_target_width_float, 64)
        target_height = round_to_nearest_multiple(initial_target_height_float, 64)


        if model_type == "SDXL" or model_type == "SD3":
            # SDXL/SD3 use 4 channels (already default)

            if model_type == "SD3":
                # SD3 has specific target area scaling logic based on a 1024x1024 reference
                # Apply scaling AFTER initial rounding to 64
                target_area_sd3_ref = 1024 * 1024
                current_area = target_width * target_height
                if current_area > 0:
                     scaling_factor = (target_area_sd3_ref / current_area) ** 0.5
                     target_width = int(target_width * scaling_factor) # Apply scaling
                     target_height = int(target_height * scaling_factor)
                     # Re-round to 64 after scaling
                     target_width = round_to_nearest_multiple(target_width, 64)
                     target_height = round_to_nearest_multiple(target_height, 64)
                else:
                     print(f"Warning: Calculated base dimensions were zero after initial rounding, skipping SD3 scaling.")


        elif model_type == "FLUX":
            # FLUX uses 16 latent channels
            latent_channels = 16
            # No special SD3-like scaling needed for FLUX

        # Ensure minimum valid base pixel dimensions (must be at least 64 for 64x64 divisibility)
        min_dim = 64
        if target_width < min_dim:
             print(f"Warning: Calculated base width was {target_width}. Clamping to minimum {min_dim}.")
             target_width = min_dim
        if target_height < min_dim:
             print(f"Warning: Calculated base height was {target_height}. Clamping to minimum {min_dim}.")
             target_height = min_dim

        # --- 3. Generate Empty Latent Image at BASE Resolution with Correct Channels ---
        # Calculate latent dimensions from final pixel dimensions
        latent_width = target_width // vae_scale_factor
        latent_height = target_height // vae_scale_factor

        try:
            # Manually create a zero tensor with the correct batch size, channels, and latent dimensions
            latent_tensor = torch.zeros([batch_size, latent_channels, latent_height, latent_width])
            # ComfyUI's latent format is a dictionary containing the samples tensor
            latent = {"samples": latent_tensor}

        except Exception as e:
             # Updated error message to reflect manual tensor creation
             raise RuntimeError(f"Error creating empty latent tensor with shape [{batch_size}, {latent_channels}, {latent_height}, {latent_width}] for {model_type}: {e}")

        print(f"Generated {model_type} base latent: Batch={batch_size}, Channels={latent_channels}, Latent Size=({latent_width}x{latent_height}), Base Pixel Size=({target_width}x{target_height})")


        # --- 4. Calculate Tile Dimensions for 2x2 Tiling of the FINAL Upscaled Pixel Output ---
        # tile dimensions are for a 2x2 grid in the upscaled pixel space.
        # Each tile is half the size of the total upscaled dimension.
        upscaled_total_width = int(target_width * upscale_by)
        upscaled_total_height = int(target_height * upscale_by)

        tile_width = upscaled_total_width // 2
        tile_height = upscaled_total_height // 2

        # Ensure tile dimensions are at least 1
        if tile_width <= 0:
             print(f"Warning: Calculated tile width was {tile_width}. Clamping to minimum 1.")
             tile_width = 1
        if tile_height <= 0:
             print(f"Warning: Calculated tile height was {tile_height}. Clamping to minimum 1.")
             tile_height = 1

        print(f"Calculated tile dimensions for 2x2 grid of upscaled pixel output (upscale_by {upscale_by}): {tile_width}x{tile_height}")


        # --- 5. Return Outputs ---
        return (
            latent, # The generated base latent image (now a dictionary)
            tile_width, # Suggested tile width for a 2x2 grid of the upscaled image
            tile_height, # Suggested tile height for a 2x2 grid of the upscaled image
            upscale_by # The upscale factor used for calculation
        )

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
