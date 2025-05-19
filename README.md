# Bobs Latent Optimizer for ComfyUI

A set of custom nodes for ComfyUI designed to generate optimized empty latent images, ideal for models like FLUX, SDXL, and SD3. These nodes help you easily define your desired aspect ratio and approximate image resolution (in megapixels) and will automatically calculate dimensions that are compatible with the selected model.

A key feature is the intelligent calculation of **tile dimensions for upscaling workflows**. The nodes aim to provide optimal tile sizes (width and height) for a subsequent tiled upscaler (like Ultimate SD Upscale, Tiled VAE Decode, etc.), helping to improve performance and manage VRAM usage during high-resolution upscales.

## Features

*   **Aspect Ratio Control:** Easily define common aspect ratios like "1:1", "16:9", "3:2", etc.
*   **Megapixel-Based Sizing:**
    *   **Standard Node:** Choose from a predefined list of approximate megapixel areas (e.g., 0.5MP, 1MP, 4MP).
    *   **Advanced Node:** Use a continuous float input for precise megapixel targets.
*   **Model-Specific Optimizations:**
    *   Automatically rounds base pixel dimensions to the nearest multiple of 64 for model compatibility.
    *   Sets the correct number of latent channels (16 for FLUX, 4 for SDXL/SD3).
    *   Applies SD3-specific target area scaling logic if "SD3" model type is selected.
*   **Batch Size Support:** Generate batches of latent images.
*   **Optimized Tiling Calculation for Upscalers:**
    *   Calculates `tile_width` and `tile_height` for a *subsequent* upscaling step.
    *   Aims for a **2x2 grid (4 tiles)** for the upscaled image by default.
    *   If a 2x2 grid would result in individual tiles larger than **2048x2048 pixels**, the number of tiles is increased (e.g., to 3x2, 2x3, 3x3 etc.) to ensure no single tile dimension exceeds 2048.
    *   This provides sensible tile sizes to feed into tiled upscaler nodes, potentially speeding up generation and reducing VRAM strain.
*   **Outputs Ready for Workflow:** Provides the generated latent, calculated tile dimensions, and the upscale factor for easy integration.

## Nodes

### 1. Bobs Latent Optimizer (`BobsLatentNode`)

The standard node for generating optimized latent images.

*   **Key Difference:** Uses a dropdown menu (`mp_size`) with predefined approximate megapixel sizes (e.g., "1" for a 1024x1024 area, "4" for a 2048x2048 area).

### 2. Bobs Latent Optimizer (Advanced) (`BobsLatentNodeAdvanced`)

The advanced version offering finer control over the target resolution.

*   **Key Difference:** Uses a float input (`mp_size_float`) for specifying the target megapixel area directly (e.g., 1.0 for 1MP, 0.75 for 0.75MP).

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    *   `cd ComfyUI/custom_nodes/`
2.  Clone this repository:
    *   `git clone https://github.com/BobsBlazed/Bobs_Latent_Optimizer.git`
3.  Restart ComfyUI.
    The nodes "Bobs Latent Optimizer" and "Bobs Latent Optimizer (Advanced)" should now be available in the "latent/generate" category.

## Usage

Both nodes share a similar set of inputs and outputs, with the main difference being how the target megapixel size is specified.

### Inputs

*   **`aspect_ratio` (STRING):** The target aspect ratio for the base latent image (e.g., "1:1", "16:9", "4:3").
*   **`mp_size` (STRING list - Standard Node only):** Approximate target megapixel area for the base latent image, chosen from a list.
*   **`mp_size_float` (FLOAT - Advanced Node only):** Precise target megapixel area for the base latent image (e.g., 1.0 = 1024x1024 pixels).
*   **`upscale_by` (FLOAT):** The desired upscale factor for the *final output image*. This value is **crucial** as it's used to calculate the `tile_width` and `tile_height` for a subsequent tiled upscaler. This node *does not* perform the upscale itself.
*   **`model_type` (STRING list):** Select the target model (FLUX, SDXL, SD3) to apply appropriate rounding rules and latent channel counts.
*   **`batch_size` (INT):** Number of latent images to generate in the batch.

### Outputs

*   **`latent` (LATENT):** The generated empty latent image(s) as a dictionary (`{"samples": tensor}`).
*   **`tile_width` (INT):** The calculated suggested width for each tile of the *upscaled pixel output*.
*   **`tile_height` (INT):** The calculated suggested height for each tile of the *upscaled pixel output*.
*   **`upscale_by` (FLOAT):** The input upscale factor, passed through for convenience in your workflow.

### Example Workflow

These nodes are typically used at the beginning of an image generation workflow, before the KSampler. The key is to connect the `tile_width` and `tile_height` outputs to a tiled upscaler node that you might use *after* your initial generation and VAE decode.

```
[Bobs Latent Optimizer] ----> latent (to KSampler)
                         |
                         |---> tile_width  -----\
                         |                     |
                         |---> tile_height -----+--> [Your Tiled Upscaler Node] (e.g., Ultimate SD Upscale inputs: tile_width, tile_height)
                         |                                   (or Tiled VAE Decode etc.)
                         |
                         ----> upscale_by ------> (Potentially to your Tiled Upscaler Node if it takes a scale factor directly)

[KSampler] --------------> VAE --------------> [Tiled Upscaler Node]
(using latent from above)  (decode)             (using tile_width, tile_height from above)
```

**Why is this useful for tiling?**

Instead of manually calculating or guessing tile sizes for your upscaler, this node provides sensible defaults based on your desired final resolution (`target_base_resolution * upscale_by`) and the 2048x2048 per-tile limit. This can prevent issues like:
*   Tiles being too large, causing VRAM errors.
*   Tiles being unnecessarily small, potentially leading to more processing overhead or seam issues if not handled well by the upscaler.
*   Inconsistent tiling strategies.

By aiming for a 2x2 grid (4 tiles total) unless the image is very large, it strikes a balance for common upscaling scenarios.

## Benefits

*   **Simplified Resolution Setup:** No need to manually calculate pixel dimensions that are multiples of 64.
*   **Model Compatibility:** Ensures your latent images are correctly sized and have the right channel count for FLUX, SDXL, or SD3.
*   **Optimized Upscaling:** Provides intelligent tile dimensions for downstream tiled upscalers, potentially improving performance and VRAM management.
*   **Consistent Workflows:** Standardizes how you define resolutions and prepare for tiled upscaling.

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to open an issue or submit a pull request.
