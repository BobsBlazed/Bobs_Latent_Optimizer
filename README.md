# Bobs Latent Optimizer for ComfyUI

A pair of custom nodes for ComfyUI that generate empty latents sized correctly for **30 model families** — SD1.5 through Flux2, Qwen-Image, HiDream, HunyuanImage, and video models like Wan, LTXV and Mochi. You give them an aspect ratio and a target megapixel area; they work out pixel dimensions that are legal for the selected model family and allocate the latent with that family's channel count, VAE downscale and rank.

They also calculate **tile dimensions for upscaling workflows**, so you can feed sensible `tile_width` / `tile_height` values straight into a tiled upscaler (Ultimate SD Upscale, Tiled VAE Decode, etc.) instead of guessing.

## Features

*   **Aspect Ratio Control:** `1:1`, `16:9`, `3:2`, `13:19`, `85:110`, and so on. `16/9`, `16x9` and decimal components like `1.5:1` are accepted too.
*   **Megapixel-Based Sizing:**
    *   **Standard Node:** pick from a list of approximate megapixel areas (0.25MP … 4MP).
    *   **Advanced Node:** a continuous float for an exact target area.
*   **Model-Specific Optimizations:**
    *   Rounds base pixel dimensions to the nearest alignment step for the chosen model.
    *   Allocates the correct latent channel count — anywhere from 3 (Chroma Radiance, pixel space) to 128 (Flux2, LTXV).
    *   Applies the correct VAE downscale — 1x, 8x, 16x or 32x depending on the family.
    *   Guarantees pixel dimensions stay divisible by that downscale, so the reported size always matches the tensor.
*   **Video Model Support:** video families emit a proper 5-D latent (`[B, C, T, H, W]`) using ComfyUI's frame formula `((length - 1) // temporal_downscale) + 1`.
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
*   **`model_type` (list):** the model family — selects latent channels, VAE downscale, alignment and rank. See the table below.
*   **`batch_size` (INT):** number of latents in the batch.
*   **`max_tile_size` (INT, optional):** largest tile edge before the grid subdivides further. Default 2048; lower it if your upscaler runs out of VRAM.
*   **`length` (INT, optional):** number of video frames. Only used by video families; image families ignore it and log a warning.

### Outputs

*   **`latent` (LATENT):** the empty latent batch, as `{"samples": tensor}`.
*   **`tile_width` (INT):** suggested tile width for the *upscaled pixel output*.
*   **`tile_height` (INT):** suggested tile height for the *upscaled pixel output*.
*   **`upscale_by` (FLOAT):** passed through unchanged.
*   **`width` (INT):** base image width in pixels.
*   **`height` (INT):** base image height in pixels.

### Model reference

Every row is taken from ComfyUI's own `comfy/latent_formats.py` (`latent_channels`, `latent_dimensions`, `spacial_downscale_ratio`, `temporal_downscale_ratio`) and cross-checked against the matching `Empty*Latent*` node, so the shapes match what the samplers actually expect.

#### Image families — latent is `[B, C, H/s, W/s]`

| `model_type`      | Channels | VAE downscale | Alignment | Covers |
| ----------------- | -------- | ------------- | --------- | ------ |
| `SD15`            | 4        | 8             | 64        | SD 1.5, SVD, Stable Zero123 |
| `SD21`            | 4        | 8             | 64        | SD 2.0 / 2.1 |
| `SDXL`            | 4        | 8             | 64        | SDXL, Playground v2.5, SSD-1B, Segmind Vega, KOALA |
| `PIXART`          | 4        | 8             | 64        | PixArt-α, PixArt-Σ |
| `AURAFLOW`        | 4        | 8             | 64        | AuraFlow |
| `HUNYUAN_DIT`     | 4        | 8             | 64        | HunyuanDiT |
| `SD3`             | 16       | 8             | 64        | SD3, SD3.5 |
| `FLUX`            | 16       | 8             | 64        | FLUX.1 dev/schnell, Kontext, Inpaint |
| `CHROMA`          | 16       | 8             | 64        | Chroma |
| `HIDREAM`         | 16       | 8             | 64        | HiDream-I1 |
| `LUMINA2`         | 16       | 8             | 64        | Lumina Image 2.0, Z-Image |
| `OMNIGEN2`        | 16       | 8             | 64        | OmniGen2 |
| `QWEN`            | 16       | 8             | 16        | Qwen-Image |
| `COSMOS_PREDICT2` | 16       | 8             | 16        | Cosmos Predict2 (text-to-image) |
| `FLUX2`           | 128      | 16            | 64        | FLUX.2, Ideogram4, MageFlow, ErnieImage, Lens |
| `HUNYUAN_IMAGE`   | 64       | 32            | 64        | HunyuanImage 2.1 |

Pixel-space families have no VAE at all, so the "latent" *is* the image. Alignment of 16 matches the step on ComfyUI's own pixel-space latent node.

| `model_type`      | Channels | VAE downscale | Alignment | Covers |
| ----------------- | -------- | ------------- | --------- | ------ |
| `CHROMA_RADIANCE` | 3        | 1             | 16        | Chroma Radiance |
| `HIDREAM_O1`      | 3        | 1             | 16        | HiDream O1 (distinct from `HIDREAM`) |
| `ZIMAGE_PIXEL`    | 3        | 1             | 16        | Z-Image pixel space |
| `PIXELDIT`        | 3        | 1             | 16        | PixelDiT T2I, PiD |

#### Video families — latent is `[B, C, T, H/s, W/s]`

`T` is derived from the `length` input as `((length - 1) // temporal) + 1`.

| `model_type`       | Channels | VAE downscale | Temporal | Alignment | Covers |
| ------------------ | -------- | ------------- | -------- | --------- | ------ |
| `WAN`              | 16       | 8             | 4        | 16        | Wan 2.1 (T2V, I2V, VACE, Camera, …), Krea2, JoyImage, Anima |
| `WAN22`            | 48       | 16            | 4        | 32        | Wan 2.2 T2V |
| `HUNYUAN_VIDEO`    | 16       | 8             | 4        | 16        | HunyuanVideo, I2V, Skyreels, Kandinsky5 |
| `HUNYUAN_VIDEO_15` | 32       | 16            | 4        | 32        | HunyuanVideo 1.5, SR distilled |
| `COSMOS`           | 16       | 8             | 8        | 16        | Cosmos 1.0 T2V / I2V |
| `COGVIDEOX`        | 16       | 8             | 4        | 16        | CogVideoX T2V / I2V / Inpaint |
| `MOCHI`            | 12       | 8             | 6        | 16        | Genmo Mochi |
| `LTXV`             | 128      | 32            | 8        | 32        | LTX-Video, LTX-AV |

#### Shape-only families

These two are included so their shapes are available, but **neither is normally driven from an empty latent** — selecting one logs a warning. SeedVR2 restores existing video (its preprocess node takes an `IMAGE`), and the HunyuanImage 2.1 refiner consumes the base model's latent.

| `model_type`            | Channels | VAE downscale | Temporal | Alignment |
| ----------------------- | -------- | ------------- | -------- | --------- |
| `SEEDVR2`               | 16       | 8             | 1        | 16        |
| `HUNYUAN_IMAGE_REFINER` | 64       | 8             | 1        | 16        |

#### Two deliberate deviations from upstream

`QWEN` and `COSMOS_PREDICT2` map to `latent_formats.Wan21`, which declares `latent_dimensions = 3` and `temporal_downscale_ratio = 4` — they inherit it because they share Wan's VAE. Both are still-image models, and ComfyUI's own Qwen workflows build their latent with `EmptySD3LatentImage`, which is 4-D. This node follows the workflow rather than the shared format and treats both as 2-D. The test suite keeps the verbatim upstream values and the override in separate tables, so the deviation stays visible rather than being absorbed into the "transcribed from ComfyUI" claim.

**Not included:** Stable Cascade needs *two* latents (stage C and stage B) from one node, which doesn't fit this node's single-`LATENT` output. Audio and 3D formats (StableAudio, Hunyuan3D, ACEStep, TripoSplat) aren't image/video latents at all.

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

The sizing and tiling math is exposed as plain functions (`parse_aspect_ratio`, `compute_base_dimensions`, `compute_tile_dimensions`, `compute_latent_frames`), and the test suite stubs `torch`, so it runs without a ComfyUI or PyTorch install:

```
python -m unittest discover -s tests -v
```

## Changelog

### 1.5.0

*   **Added 7 more model families**, taking the total from 23 to 30 and closing the gap against ComfyUI `master`:
    *   Pixel-space image: `HIDREAM_O1`, `ZIMAGE_PIXEL`, `PIXELDIT` (3 channels, no VAE downscale).
    *   Video: `HUNYUAN_VIDEO_15` (32ch, 16×), `COGVIDEOX`.
    *   Shape-only: `SEEDVR2` and `HUNYUAN_IMAGE_REFINER`, which are **not** empty-latent workflows — selecting either logs a warning explaining that the model consumes an existing image or latent.
*   **Fixed a provenance overclaim in the test suite.** `COMFY_REFERENCE` was documented as transcribed from ComfyUI's `latent_formats.py`, but the `QWEN` and `COSMOS_PREDICT2` rows silently encoded a judgement call (treating them as 2-D despite `latent_formats.Wan21` declaring 3-D). The table now holds verbatim upstream values, with the deviation isolated in `INTENTIONAL_OVERRIDES` alongside its justification, plus a test asserting each override still genuinely deviates — so it becomes dead code to delete if upstream ever agrees. The node's behaviour is unchanged; only the claim about where the numbers came from is now accurate.
*   **Documented which models each family covers.** Newer models that reuse an existing format (Ideogram4, MageFlow, ErnieImage and Lens on Flux2; Krea2, JoyImage and Anima on Wan21; LongCat and Kandinsky5Image on Flux) already worked but weren't discoverable — they're now named in the reference tables.

### 1.4.0

*   **Added 18 model families**, taking the total from 5 to 23. New image families: `SD15`, `SD21`, `PIXART`, `AURAFLOW`, `HUNYUAN_DIT`, `CHROMA`, `HIDREAM`, `LUMINA2`, `OMNIGEN2`, `COSMOS_PREDICT2`, `FLUX2`, `HUNYUAN_IMAGE`, `CHROMA_RADIANCE`. New video families: `WAN22`, `HUNYUAN_VIDEO`, `COSMOS`, `MOCHI`, `LTXV`.
*   **Added proper video latent support.** Video families now emit a 5-D latent `[B, C, T, H, W]` using ComfyUI's frame formula, driven by a new optional `length` input. Previously there was no way to produce a usable video latent.
*   **Added per-model VAE downscale.** The downscale factor was hardcoded to 8, which cannot express Flux2 (16x), HunyuanImage 2.1 (32x), LTXV (32x), Wan 2.2 (16x) or Chroma Radiance (pixel space, 1x).
*   **Changed:** `WAN` now produces a 5-D latent instead of a 4-D one. The 4-D latent was the documented limitation in 1.3.0 and was not usable with Wan samplers — this is the fix, but it is a visible change if you were relying on the old shape.
*   Every channel count, downscale and temporal ratio is now verified against ComfyUI's `latent_formats.py` by a test.

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
