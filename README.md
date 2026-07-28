# Bobs Latent Optimizer for ComfyUI

A pair of custom nodes for ComfyUI that generate empty latents sized correctly for FLUX, SDXL, SD3, Qwen-Image and Wan. You give them an aspect ratio and a target megapixel area; they work out pixel dimensions that are legal for the selected model family and allocate the latent with that family's channel count.

They also calculate **tile dimensions for upscaling workflows**, so you can feed sensible `tile_width` / `tile_height` values straight into a tiled upscaler (Ultimate SD Upscale, Tiled VAE Decode, etc.) instead of guessing.

## Features

*   **Aspect Ratio Control:** `1:1`, `16:9`, `3:2`, `13:19`, `85:110`, and so on. `16/9`, `16x9` and decimal components like `1.5:1` are accepted too.
*   **Megapixel-Based Sizing:**
    *   **Standard Node:** pick from a list of approximate megapixel areas (0.25MP … 4MP).
    *   **Advanced Node:** a continuous float for an exact target area.
*   **Model-Specific Optimizations:**
    *   Rounds base pixel dimensions to the nearest alignment step for the chosen model (64 for FLUX/SDXL/SD3, 16 for Qwen/Wan).
    *   Allocates the correct latent channel count — 4 for SDXL, 16 for FLUX, SD3, Qwen-Image and Wan.
    *   Guarantees pixel dimensions stay divisible by the VAE stride, so the reported size always matches the tensor.
*   **Batch Size Support:** generate batches of latents.
*   **Optimized Tiling Calculation for Upscalers:**
    *   Targets a **2x2 grid (4 tiles)** for the upscaled image.
    *   Subdivides further along an axis only when a 2x2 tile would exceed the tile cap (**2048px** by default, adjustable via `max_tile_size`).
    *   Tile dimensions are rounded up to a multiple of 8 so tiled VAE nodes are happy.
*   **Allocation on ComfyUI's intermediate device**, matching the behaviour of the built-in `EmptyLatentImage`.

## Nodes

### 1. Bobs Latent Optimizer (`BobsLatentNode`)

Uses a dropdown (`mp_size`) of predefined approximate megapixel areas — e.g. `"1"` for a 1024x1024 area, `"4"` for a 2048x2048 area.

### 2. Bobs Latent Optimizer (Advanced) (`BobsLatentNodeAdvanced`)

Uses a float (`mp_size_float`) for the target area directly, where `1.0` = 1048576 pixels.

## Installation

Install from the [ComfyUI Registry](https://registry.comfy.org/) via ComfyUI-Manager, or manually:

1.  `cd ComfyUI/custom_nodes/`
2.  `git clone https://github.com/BobsBlazed/Bobs_Latent_Optimizer.git`
3.  Restart ComfyUI.

The nodes appear under the **latent/generate** category.

## Usage

### Inputs

*   **`aspect_ratio` (STRING):** target aspect ratio for the base image, e.g. `"1:1"`, `"16:9"`, `"4:3"`.
*   **`mp_size` (list — Standard node):** approximate target megapixel area.
*   **`mp_size_float` (FLOAT — Advanced node):** exact target megapixel area (`1.0` = 1024x1024 pixels).
*   **`upscale_by` (FLOAT):** the upscale factor for your *final* image. Used to compute `tile_width` / `tile_height`. **This node does not perform the upscale.**
*   **`model_type` (list):** `FLUX`, `SDXL`, `SD3`, `QWEN` or `WAN` — selects alignment and latent channels.
*   **`batch_size` (INT):** number of latents in the batch.
*   **`max_tile_size` (INT, optional):** largest tile edge before the grid subdivides further. Default 2048; lower it if your upscaler runs out of VRAM.

### Outputs

*   **`latent` (LATENT):** the empty latent batch, as `{"samples": tensor}`.
*   **`tile_width` (INT):** suggested tile width for the *upscaled pixel output*.
*   **`tile_height` (INT):** suggested tile height for the *upscaled pixel output*.
*   **`upscale_by` (FLOAT):** passed through unchanged.
*   **`width` (INT):** base image width in pixels.
*   **`height` (INT):** base image height in pixels.

### Model reference

| `model_type` | Latent channels | Pixel alignment |
| ------------ | --------------- | --------------- |
| FLUX         | 16              | 64              |
| SDXL         | 4               | 64              |
| SD3 / SD3.5  | 16              | 64              |
| QWEN         | 16              | 16              |
| WAN          | 16              | 16              |

> **Note on WAN:** Wan is a video model and its samplers expect a 5-D latent with a temporal axis. These nodes emit a 4-D image latent, so the `WAN` option is useful for Wan-compatible *image-space* sizing and channel count, not as a drop-in replacement for a Wan video latent node.

### Example Workflow

These nodes sit at the start of a generation workflow, before the KSampler. Connect `tile_width` and `tile_height` to the tiled upscaler you use *after* your initial generation and VAE decode.

```
[Bobs Latent Optimizer] ----> latent (to KSampler)
                         |
                         |---> tile_width  -----\
                         |                      |
                         |---> tile_height -----+--> [Your Tiled Upscaler Node]
                         |                                (Ultimate SD Upscale, Tiled VAE Decode, …)
                         |
                         ----> upscale_by ------> (if your upscaler takes a scale factor directly)

[KSampler] --------------> VAE --------------> [Tiled Upscaler Node]
(using latent from above)  (decode)             (using tile_width, tile_height from above)
```

**Why is this useful for tiling?**

Rather than guessing tile sizes, the node derives them from your desired final resolution (`base_resolution * upscale_by`) and the per-tile cap. That avoids:

*   tiles large enough to cause VRAM errors,
*   tiles unnecessarily small, adding processing overhead and seam risk,
*   inconsistent tiling between workflows.

## Development

The sizing and tiling math is exposed as plain functions (`parse_aspect_ratio`, `compute_base_dimensions`, `compute_tile_dimensions`), and the test suite stubs `torch`, so it runs without a ComfyUI or PyTorch install:

```
python -m unittest discover -s tests -v
```

## Changelog

### 1.3.0

*   **Fixed:** node display names never showed up in ComfyUI — the keys in `NODE_DISPLAY_NAME_MAPPINGS` did not match the class-mapping keys.
*   **Fixed:** SD3, Qwen and Wan latents were allocated with 4 channels. All three use 16-channel VAEs, so those latents were unusable with their samplers.
*   **Fixed:** Qwen rounded pixel dimensions to multiples of 28, which is not divisible by the VAE stride of 8 — the reported pixel size did not match the latent that was produced. Qwen now aligns to 16.
*   **Fixed:** SD3 rescaled every result back to roughly 1MP, so `mp_size` / `mp_size_float` were silently ignored for that model.
*   **Fixed:** negative or non-integer aspect ratios crashed with an opaque `math domain error` instead of a helpful message.
*   **Added:** `width` and `height` outputs (appended, so existing workflows keep working).
*   **Added:** optional `max_tile_size` input.
*   **Added:** `16/9`, `16x9` and decimal aspect-ratio formats.
*   **Changed:** tile dimensions are rounded up to a multiple of 8 and never exceed the image itself.
*   **Changed:** latents are allocated on ComfyUI's intermediate device, matching `EmptyLatentImage`.
*   **Changed:** the two node classes now share one implementation; output goes through `logging` instead of `print`.
*   **Added:** unit test suite and a CI workflow.

## Contributing

Contributions, issues, and feature requests are welcome — please open an issue or a pull request.
