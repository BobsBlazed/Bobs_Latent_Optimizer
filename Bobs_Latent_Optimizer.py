# --- START OF FILE Bobs_Latent_Optimizer.py ---

import torch
import math

def round_to_nearest_multiple(value, multiple):
    """Rounds a value to the nearest multiple of 'multiple'."""
    if multiple <= 0:
        return value
    return int(round(value / multiple) * multiple)

MP_BASE_AREA = 1024 * 1024

# --- BobsLatentNode ---
class BobsLatentNode:
    """
    Generates an empty latent image optimized for various models (FLUX, SDXL, SD3, QWEN, WAN)
    based on aspect ratio, approximate discrete megapixel size options, and batch size.
    Calculates dimensions by rounding to the NEAREST multiple appropriate for the selected model.
    Handles the correct number of latent channels (4 for most, 16 for FLUX).
    Calculates tile dimensions for tiling of the *upscaled pixel output*
    to help optimize tiled upscale times. Aims for a 2x2 grid (4 tiles) unless
    individual tiles would exceed 2048x2048, in which case more tiles are used.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ("STRING", {"default": "1:1", "tooltip": "Target image aspect ratio (e.g., '1:1', '16:9', '3:2'). This determines the shape of the BASE latent image."}),
                "mp_size": (["0.25", "0.5", "1", "1.25", "1.5", "1.75", "2", "2.5", "3", "4"], {"default": "1", "tooltip": "Approximate target megapixel area for the BASE latent image. These options map to common standard resolution areas (e.g., 1 is 1024x1024 area, 4 is 2048x2048 area)."}),
                "upscale_by": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": .01,
                    "tooltip": "Desired upscale factor for the FINAL output image. Used to calculate tiling dimensions in pixel space. Does NOT upscale the generated latent."
                }),
                "model_type": (["FLUX", "SDXL", "SD3", "QWEN", "WAN"], {"default": "FLUX", "tooltip": "Select model to set resolution rounding rules and latent channels (FLUX=16, QWEN=round to 28, others=4 channels & round to 64)."}),
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


        mp_to_area = {
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

        target_area = mp_to_area.get(mp_size, 1024 * 1024)

        initial_target_width_float = math.sqrt(target_area * aspect_ratio_multiplier)
        initial_target_height_float = initial_target_width_float / aspect_ratio_multiplier

        # --- 2. Apply Model-Specific BASE Sizing and Get Latent Channels ---
        vae_scale_factor = 8
        
        if model_type == "FLUX":
            latent_channels = 16
            target_width = round_to_nearest_multiple(initial_target_width_float, 64)
            target_height = round_to_nearest_multiple(initial_target_height_float, 64)
        
        elif model_type == "QWEN":
            latent_channels = 4
            target_width = round_to_nearest_multiple(initial_target_width_float, 28)
            target_height = round_to_nearest_multiple(initial_target_height_float, 28)
            
        elif model_type == "SDXL" or model_type == "WAN":
            latent_channels = 4
            target_width = round_to_nearest_multiple(initial_target_width_float, 64)
            target_height = round_to_nearest_multiple(initial_target_height_float, 64)
        
        elif model_type == "SD3":
            latent_channels = 4
            # First, round to 64 as a base
            temp_width = round_to_nearest_multiple(initial_target_width_float, 64)
            temp_height = round_to_nearest_multiple(initial_target_height_float, 64)
            
            # Then apply SD3 specific scaling logic
            target_area_sd3_ref = 1024 * 1024
            current_area = temp_width * temp_height
            if current_area > 0:
                 scaling_factor = (target_area_sd3_ref / current_area) ** 0.5
                 target_width = int(temp_width * scaling_factor)
                 target_height = int(temp_height * scaling_factor)
                 # And re-round to 64
                 target_width = round_to_nearest_multiple(target_width, 64)
                 target_height = round_to_nearest_multiple(target_height, 64)
            else:
                 print(f"Warning: Calculated base dimensions were zero after initial rounding, skipping SD3 scaling.")
                 target_width = temp_width
                 target_height = temp_height
        
        min_dim = 64
        if target_width < min_dim:
             print(f"Warning: Calculated base width was {target_width}. Clamping to minimum {min_dim}.")
             target_width = min_dim
        if target_height < min_dim:
             print(f"Warning: Calculated base height was {target_height}. Clamping to minimum {min_dim}.")
             target_height = min_dim

        # --- 3. Generate Empty Latent Image at BASE Resolution with Correct Channels ---
        latent_width = target_width // vae_scale_factor
        latent_height = target_height // vae_scale_factor

        try:
            latent_tensor = torch.zeros([batch_size, latent_channels, latent_height, latent_width])
            latent = {"samples": latent_tensor}
        except Exception as e:
             raise RuntimeError(f"Error creating empty latent tensor with shape [{batch_size}, {latent_channels}, {latent_height}, {latent_width}] for {model_type}: {e}")

        print(f"Generated {model_type} base latent: Batch={batch_size}, Channels={latent_channels}, Latent Size=({latent_width}x{latent_height}), Base Pixel Size=({target_width}x{target_height})")

        # --- 4. Calculate Tile Dimensions for Tiling of the FINAL Upscaled Pixel Output ---
        upscaled_total_width = int(target_width * upscale_by)
        upscaled_total_height = int(target_height * upscale_by)

        MAX_TILE_DIM = 2048

        num_tiles_w = 2
        num_tiles_h = 2

        hypothetical_tile_w_for_2x2 = (upscaled_total_width + num_tiles_w - 1) // num_tiles_w
        hypothetical_tile_h_for_2x2 = (upscaled_total_height + num_tiles_h - 1) // num_tiles_h
        
        if hypothetical_tile_w_for_2x2 > MAX_TILE_DIM:
            num_tiles_w = int(math.ceil(upscaled_total_width / MAX_TILE_DIM))
        
        if hypothetical_tile_h_for_2x2 > MAX_TILE_DIM:
            num_tiles_h = int(math.ceil(upscaled_total_height / MAX_TILE_DIM))

        if num_tiles_w < 1: num_tiles_w = 1
        if num_tiles_h < 1: num_tiles_h = 1
            
        # Calculate final tile dimensions using integer ceiling division
        tile_width = (upscaled_total_width + num_tiles_w - 1) // num_tiles_w
        tile_height = (upscaled_total_height + num_tiles_h - 1) // num_tiles_h
        
        # Ensure tile dimensions are at least 1 (final safety clamp)
        if tile_width <= 0:
             print(f"Warning: Calculated tile width was {tile_width}. Clamping to minimum 1.")
             tile_width = 1
        if tile_height <= 0:
             print(f"Warning: Calculated tile height was {tile_height}. Clamping to minimum 1.")
             tile_height = 1
        
        print(f"Upscaled image size: {upscaled_total_width}x{upscaled_total_height}")
        print(f"Tiling grid: {num_tiles_w}x{num_tiles_h} ({num_tiles_w * num_tiles_h} tiles)")
        print(f"Calculated tile dimensions for upscaled pixel output (upscale_by {upscale_by}): {tile_width}x{tile_height}")

        # --- 5. Return Outputs ---
        return (
            latent,
            tile_width,
            tile_height,
            upscale_by
        )

# --- Advanced Node: BobsLatentNodeAdvanced ---
class BobsLatentNodeAdvanced:
    """
    Generates an empty latent image optimized for various models (FLUX, SDXL, SD3, QWEN, WAN)
    based on aspect ratio, a continuous float megapixel size, and batch size.
    Calculates dimensions by rounding to the NEAREST multiple appropriate for the selected model.
    Handles the correct number of latent channels (4 for most, 16 for FLUX).
    Calculates tile dimensions for tiling of the *upscaled pixel output*
    to help optimize tiled upscale times. Aims for a 2x2 grid (4 tiles) unless
    individual tiles would exceed 2048x2048, in which case more tiles are used.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ("STRING", {"default": "1:1", "tooltip": "Target image aspect ratio (e.g., '1:1', '16:9', '3:2'). This determines the shape of the BASE latent image."}),
                "mp_size_float": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01, "display": "number", "tooltip": f"Target megapixel area (in millions of pixels, based on {int(MP_BASE_AREA/1000000)} MP = {MP_BASE_AREA} pixels) for the BASE latent image. Range 0.01 to 4.0 (4.0 is 2048x2048 area)."}),
                "upscale_by": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": .01,
                    "tooltip": "Desired upscale factor for the FINAL output image. Used to calculate tiling dimensions in pixel space. Does NOT upscale the generated latent."
                }),
                "model_type": (["FLUX", "SDXL", "SD3", "QWEN", "WAN"], {"default": "FLUX", "tooltip": "Select model to set resolution rounding rules and latent channels (FLUX=16, QWEN=round to 28, others=4 channels & round to 64)."}),
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

        target_area = mp_size_float * MP_BASE_AREA

        initial_target_width_float = math.sqrt(target_area * aspect_ratio_multiplier)
        initial_target_height_float = initial_target_width_float / aspect_ratio_multiplier

        # --- 2. Apply Model-Specific BASE Sizing and Get Latent Channels ---
        vae_scale_factor = 8
        
        if model_type == "FLUX":
            latent_channels = 16
            target_width = round_to_nearest_multiple(initial_target_width_float, 64)
            target_height = round_to_nearest_multiple(initial_target_height_float, 64)
        
        elif model_type == "QWEN":
            latent_channels = 4
            target_width = round_to_nearest_multiple(initial_target_width_float, 28)
            target_height = round_to_nearest_multiple(initial_target_height_float, 28)
            
        elif model_type == "SDXL" or model_type == "WAN":
            latent_channels = 4
            target_width = round_to_nearest_multiple(initial_target_width_float, 64)
            target_height = round_to_nearest_multiple(initial_target_height_float, 64)
        
        elif model_type == "SD3":
            latent_channels = 4
            # First, round to 64 as a base
            temp_width = round_to_nearest_multiple(initial_target_width_float, 64)
            temp_height = round_to_nearest_multiple(initial_target_height_float, 64)
            
            # Then apply SD3 specific scaling logic
            target_area_sd3_ref = 1024 * 1024
            current_area = temp_width * temp_height
            if current_area > 0:
                 scaling_factor = (target_area_sd3_ref / current_area) ** 0.5
                 target_width = int(temp_width * scaling_factor)
                 target_height = int(temp_height * scaling_factor)
                 # And re-round to 64
                 target_width = round_to_nearest_multiple(target_width, 64)
                 target_height = round_to_nearest_multiple(target_height, 64)
            else:
                 print(f"Warning: Calculated base dimensions were zero after initial rounding, skipping SD3 scaling.")
                 target_width = temp_width
                 target_height = temp_height

        min_dim = 64
        if target_width < min_dim:
             print(f"Warning: Calculated base width was {target_width}. Clamping to minimum {min_dim}.")
             target_width = min_dim
        if target_height < min_dim:
             print(f"Warning: Calculated base height was {target_height}. Clamping to minimum {min_dim}.")
             target_height = min_dim

        # --- 3. Generate Empty Latent Image at BASE Resolution with Correct Channels ---
        latent_width = target_width // vae_scale_factor
        latent_height = target_height // vae_scale_factor

        try:
            latent_tensor = torch.zeros([batch_size, latent_channels, latent_height, latent_width])
            latent = {"samples": latent_tensor}
        except Exception as e:
             raise RuntimeError(f"Error creating empty latent tensor with shape [{batch_size}, {latent_channels}, {latent_height}, {latent_width}] for {model_type}: {e}")

        print(f"Generated {model_type} base latent: Batch={batch_size}, Channels={latent_channels}, Latent Size=({latent_width}x{latent_height}), Base Pixel Size=({target_width}x{target_height})")

        # --- 4. Calculate Tile Dimensions for Tiling of the FINAL Upscaled Pixel Output ---
        upscaled_total_width = int(target_width * upscale_by)
        upscaled_total_height = int(target_height * upscale_by)

        MAX_TILE_DIM = 2048

        num_tiles_w = 2
        num_tiles_h = 2

        hypothetical_tile_w_for_2x2 = (upscaled_total_width + num_tiles_w - 1) // num_tiles_w
        hypothetical_tile_h_for_2x2 = (upscaled_total_height + num_tiles_h - 1) // num_tiles_h
        
        if hypothetical_tile_w_for_2x2 > MAX_TILE_DIM:
            num_tiles_w = int(math.ceil(upscaled_total_width / MAX_TILE_DIM))
        
        if hypothetical_tile_h_for_2x2 > MAX_TILE_DIM:
            num_tiles_h = int(math.ceil(upscaled_total_height / MAX_TILE_DIM))
        
        if num_tiles_w < 1: num_tiles_w = 1
        if num_tiles_h < 1: num_tiles_h = 1
            
        tile_width = (upscaled_total_width + num_tiles_w - 1) // num_tiles_w
        tile_height = (upscaled_total_height + num_tiles_h - 1) // num_tiles_h
        
        if tile_width <= 0:
             print(f"Warning: Calculated tile width was {tile_width}. Clamping to minimum 1.")
             tile_width = 1
        if tile_height <= 0:
             print(f"Warning: Calculated tile height was {tile_height}. Clamping to minimum 1.")
             tile_height = 1
        
        print(f"Upscaled image size: {upscaled_total_width}x{upscaled_total_height}")
        print(f"Tiling grid: {num_tiles_w}x{num_tiles_h} ({num_tiles_w * num_tiles_h} tiles)")
        print(f"Calculated tile dimensions for upscaled pixel output (upscale_by {upscale_by}): {tile_width}x{tile_height}")

        # --- 5. Return Outputs ---
        return (
            latent,
            tile_width,
            tile_height,
            upscale_by
        )

# --- NODE MAPPINGS ---
NODE_CLASS_MAPPINGS = {
    "BobsLatentNode": BobsLatentNode,
    "BobsLatentNodeAdvanced": BobsLatentNodeAdvanced
}

# --- NODE DISPLAY NAMES ---
NODE_DISPLAY_NAME_MAPPINGS = {
    "BobsLatentNode": "Bobs Latent Optimizer",
    "BobsLatentNodeAdvanced": "Bobs Latent Optimizer (Advanced)"
}

# --- END OF FILE Bobs_Latent_Optimizer.py ---
