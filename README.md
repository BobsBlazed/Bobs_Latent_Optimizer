# Bobs Latent Optimizer for ComfyUI

Custom nodes to generate empty latent images specifically optimized for FLUX, SDXL, and SD3 models within ComfyUI. These nodes ensure the correct latent channel count, handle base resolution rounding for model compatibility, and provide calculated tile dimensions to assist in optimizing tiled upscaling workflows.

## Features

*   **Model-Specific Latent Generation:** Automatically generates empty latent images with the correct channel count:
    *   **FLUX:** 16 latent channels
    *   **SDXL / SD3:** 4 latent channels
*   **Flexible Base Resolution Control:** Define the base latent image dimensions using aspect ratio combined with an approximate Megapixel (MP) area.
*   **Accurate Rounding:** Rounds the calculated base pixel dimensions to the **nearest** multiple of 64 (required for VAE compatibility) to best match the requested aspect ratio and area.
*   **SD3 Compatibility:** Includes specific resolution scaling logic for SD3 models.
*   **Tiled Upscale Optimization:** Calculates and outputs suggested `tile_width` and `tile_height` dimensions designed for a 2x2 tiling grid of your *final upscaled pixel output*. This helps minimize the number of tiles needed for efficiency.
*   **Batching Support:** Generate multiple latents simultaneously using the `batch_size` input.
*   **Two Node Variants:** Choose between a node with discrete, commonly used MP size options and an advanced node with a continuous float MP size input for finer control.

## Installation

1.  Navigate to your ComfyUI installation's `custom_nodes` directory.
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository into the `custom_nodes` directory.
    ```bash
    git clone https://github.com/BobsBlazed/Bobs_Latent_Optimizer.git
    ```
3.  Restart your ComfyUI server. The new nodes should appear under the `latent/generate` category.

## Usage

1.  Add the desired node (`Bobs Latent Optimizer (Discrete MP)` or `Bobs Advanced Latent Optimizer (Float MP)`) from the `latent/generate` category.
2.  Connect the `latent` output to the `latent_image` input of your KSampler node.
3.  Set the `aspect_ratio`, `mp_size` (or `mp_size_float`), `upscale_by`, and `model_type` inputs according to your desired base resolution and target model.
4.  Use the `tile_width` and `tile_height` outputs to configure a downstream tiling or tiled upscaling node (e.g., nodes from WAS Node Suite, Ultimate SD Upscale, etc.). These dimensions represent the size of each tile if you were to divide the *final upscaled pixel image* into a 2x2 grid.

## Node Details

### Bobs Latent Optimizer (Discrete MP)

This node provides a selection of commonly used Megapixel areas as discrete options.

*   **Display Name:** `Bobs Latent Optimizer (Discrete MP)`
*   **Category:** `latent/generate`
*   **Inputs:**
    *   `aspect_ratio` (`STRING`): Target image aspect ratio (e.g., '1:1', '16:9', '3:2'). Determines the shape of the BASE latent image. Default: '1:1'.
    *   `mp_size` (`["0.25", "0.5", "1", "1.25", "1.5", "1.75", "2", "2.5", "3", "4"]`): Approximate target megapixel area for the BASE latent image. These options map to common standard resolution areas (e.g., 1 is 1024x1024 area, 4 is 2048x2048 area). Default: '1'.
    *   `upscale_by` (`FLOAT`): Desired upscale factor for the FINAL output image. Used to calculate tiling dimensions for a 2x2 grid in pixel space. Does NOT upscale the generated latent. Default: 2.0.
    *   `model_type` (`["FLUX", "SDXL", "SD3"]`): Select the target model type to set base resolution rounding rules and latent channel count (FLUX=16, SDXL/SD3=4). Default: 'FLUX'.
    *   `batch_size` (`INT`): Number of latent images in the batch. Default: 1.
*   **Outputs:**
    *   `latent` (`LATENT`): The generated empty base latent image (with correct channels and dimensions).
    *   `tile_width` (`INT`): Suggested tile width for a 2x2 grid of the final upscaled pixel image.
    *   `tile_height` (`INT`): Suggested tile height for a 2x2 grid of the final upscaled pixel image.
    *   `upscale_by` (`FLOAT`): The upscale factor used for calculation (passed through).

### Bobs Advanced Latent Optimizer (Float MP)

This node allows for precise control over the Megapixel area using a continuous float input.

*   **Display Name:** `Bobs Advanced Latent Optimizer Advanced (Float MP)`
*   **Category:** `latent/generate`
*   **Inputs:**
    *   `aspect_ratio` (`STRING`): Target image aspect ratio (e.g., '1:1', '16:9', '3:2'). Determines the shape of the BASE latent image. Default: '1:1'.
    *   `mp_size_float` (`FLOAT`): Target megapixel area (in millions of pixels, based on 1 MP = 1024\*1024 pixels) for the BASE latent image. Range 0.01 to 4.0 (4.0 is 2048x2048 area). Default: 1.0.
    *   `upscale_by` (`FLOAT`): Desired upscale factor for the FINAL output image. Used to calculate tiling dimensions for a 2x2 grid in pixel space. Does NOT upscale the generated latent. Default: 2.0.
    *   `model_type` (`["FLUX", "SDXL", "SD3"]`): Select the target model type to set base resolution rounding rules and latent channel count (FLUX=16, SDXL/SD3=4). Default: 'FLUX'.
    *   `batch_size` (`INT`): Number of latent images in the batch. Default: 1.
*   **Outputs:**
    *   `latent` (`LATENT`): The generated empty base latent image (with correct channels and dimensions).
    *   `tile_width` (`INT`): Suggested tile width for a 2x2 grid of the final upscaled pixel image.
    *   `tile_height` (`INT`): Suggested tile height for a 2x2 grid of the final upscaled pixel image.
    *   `upscale_by` (`FLOAT`): The upscale factor used for calculation (passed through).

## Technical Details

*   **Latent Channel Count:** FLUX models require a latent image with 16 channels, unlike SDXL/SD3 which use 4. This node bypasses the default 4-channel generation of `EmptyLatentImage` and manually creates a zero tensor with the correct number of channels based on the selected `model_type`.
*   **Base Resolution Rounding:** Diffusion models trained on latent representations require the corresponding pixel dimensions to be divisible by a specific factor (typically 64, as the VAE downsamples by 8, and latent space is often processed in 8x8 patches). These nodes calculate initial dimensions based on area and aspect ratio and then round them to the **nearest** multiple of 64 to satisfy this requirement while staying as close as possible to the requested size.
*   **Tiling Dimensions:** The `tile_width` and `tile_height` outputs are calculated to represent the dimensions needed for a 2x2 grid of the *final output image after upscaling*. The logic `(target_dimension * upscale_by) // 2` effectively gives you half of the total upscaled dimension, which is the size of each tile in a 2x2 arrangement. These values are designed to be fed into tiled upscaling nodes.
*   **Megapixel (MP) Definition:** The MP size inputs in both nodes (discrete options and float value) are interpreted as target areas based on `1 MP = 1024 * 1024` pixels. The discrete options map to specific common resolution areas (e.g., 1920x1080 for "2 MP", 2048x2048 for "4 MP"), while the float input allows for a continuous range based on this 1024x1024 unit.

## Credits

*   Developed by BobsBlazed.
*   Inspired by the standard ComfyUI `EmptyLatentImage` node and the requirements of FLUX models.
*   Improvements guided by collaborative analysis and testing.
