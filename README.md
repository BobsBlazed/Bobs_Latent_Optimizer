# Bobs Latent Optimizer for ComfyUI

A pair of custom nodes for ComfyUI that generate empty latents sized correctly for **30 model families** — SD1.5 through Flux2, Qwen-Image, HiDream, HunyuanImage, and video models like Wan, LTXV and Mochi. You give them an aspect ratio and a target megapixel area; they work out pixel dimensions that are legal for the selected model family and allocate the latent with that family's channel count, VAE downscale and rank.

They also calculate **tile dimensions for upscaling workflows**, so you can feed sensible `tile_width` / `tile_height` values straight into a tiled upscaler (Ultimate SD Upscale, Tiled VAE Decode, etc.) instead of guessing.

## Features

*   **Aspect Ratio Control:** `1:1`, `16:9`, `3:2`, `13:19`, `85:110`, and so on. `16/9`, `16x9` and decimal components like `1.5:1` are accepted too. A comma is deliberately rejected — `1,5` is a decimal in many locales, so guessing would risk silently reading it as `1:5`.
*   **Megapixel-Based Sizing:**
    *   **Standard Node:** pick from a list of preset areas (0.25 … 4). The labels are approximate — see [Megapixel presets](#megapixel-presets) for what each one actually resolves to.
    *   **Advanced Node:** a continuous float (0.01 – 16.0) for an exact target area.
*   **Model-Specific Optimizations:**
    *   Rounds base pixel dimensions to the nearest alignment step for the chosen model.
    *   Allocates the correct latent channel count — anywhere from 3 (Chroma Radiance, pixel space) to 128 (Flux2, LTXV).
    *   Applies the correct VAE downscale — 1x, 8x, 16x or 32x depending on the family.
    *   Guarantees pixel dimensions stay divisible by that downscale, so the reported size always matches the tensor.
    *   Keeps dimensions within ComfyUI's `MAX_RESOLUTION` (16384) and above one alignment step, warning whenever either bound changes what you asked for.
*   **Video Model Support:** video families emit a proper 5-D latent (`[B, C, T, H, W]`) using ComfyUI's frame formula `((length - 1) // temporal_downscale) + 1`.
*   **Batch Size Support:** generate batches of latents.
*   **Optimized Tiling Calculation for Upscalers:**
    *   Targets a **2x2 grid (4 tiles)** for the upscaled image.
    *   Subdivides further along an axis only when a 2x2 tile would exceed the tile cap (**2048px** by default, adjustable via `max_tile_size`).
    *   Tile dimensions are rounded up to a multiple of 8 so tiled VAE nodes are happy.
*   **Allocation that matches ComfyUI's own nodes** — on the intermediate device, and with `intermediate_dtype()` for image families (video families take only `device`, exactly as ComfyUI's video latent nodes do).

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

| Input | Type | Range | Default | Notes |
| --- | --- | --- | --- | --- |
| `aspect_ratio` | STRING | — | `1:1` | `16:9`, `16/9`, `16x9`, `1.5:1`, or a bare number like `1.777`. A comma is rejected (see below). |
| `mp_size` | list | 0.25 – 4 | `1` | Standard node only. Preset areas — see the table below. |
| `mp_size_float` | FLOAT | 0.01 – 16.0 | `1.0` | Advanced node only. `1.0` = 1048576 px (1024x1024). |
| `upscale_by` | FLOAT | 1.0 – 10.0 | `2.0` | Upscale factor for your *final* image, used only to compute the tile size. **This node does not upscale anything.** |
| `model_type` | list | 30 families | `FLUX` | Selects channels, VAE downscale, alignment and rank. See [Model reference](#model-reference). |
| `batch_size` | INT | 1 – 64 | `1` | Number of latents in the batch. |
| `max_tile_size` | INT (optional) | 256 – 8192 | `2048` | Largest tile edge before the grid subdivides further. Lower it if your upscaler runs out of VRAM. |
| `length` | INT (optional) | 1 – 4096 | `1` | Video frames. Video families only; image families ignore it and warn. |

#### Megapixel presets

The `mp_size` labels map to common standard resolution areas rather than exact multiples of 1MP, so a couple of them don't match their label. Use the **Advanced** node with `mp_size_float` if you need the exact number.

| `mp_size` | Area (px) | Actual MP | Equivalent to |
| --------- | --------- | --------- | ------------- |
| `0.25` | 262144 | 0.25 | 512 x 512 |
| `0.5` | 589824 | **0.56** | 768 x 768 |
| `1` | 1048576 | 1.00 | 1024 x 1024 |
| `1.25` | 1310720 | 1.25 | 1280 x 1024 |
| `1.5` | 1555200 | 1.48 | 1440 x 1080 |
| `1.75` | 1810432 | 1.73 | 1664 x 1088 |
| `2` | 2073600 | 1.98 | 1920 x 1080 |
| `2.5` | 2359296 | **2.25** | 1536 x 1536 |
| `3` | 3211264 | 3.06 | 1792 x 1792 |
| `4` | 4194304 | 4.00 | 2048 x 2048 |

These are the areas *before* model alignment; the final dimensions are rounded to the family's alignment step, so the delivered area shifts by a percent or two either way.

### Outputs

*   **`latent` (LATENT):** the empty latent batch, as `{"samples": tensor}`.
*   **`tile_width` (INT):** suggested tile width for the *upscaled pixel output*.
*   **`tile_height` (INT):** suggested tile height for the *upscaled pixel output*.
*   **`upscale_by` (FLOAT):** passed through unchanged.
*   **`width` (INT):** base image width in pixels.
*   **`height` (INT):** base image height in pixels.

### Warnings you might see

The node logs a one-line INFO summary for every run (dimensions, latent shape, tile grid) and warns in four cases. Each warning means the result differs from what you literally asked for — none are cosmetic.

| Warning | Cause | What to do |
| --- | --- | --- |
| `...px exceeds the 16384 px limit; scaling down` | An aspect ratio extreme enough to push a dimension past ComfyUI's `MAX_RESOLUTION`. The area is reduced to fit. | Lower `mp_size` or use a less extreme ratio if you wanted the full area. |
| `...px is below the N px minimum for this model; raising it` | The short side rounded below one alignment step. The aspect ratio will not match what you asked for. | Raise `mp_size`, or accept the distortion. |
| `length=N ignored - X is an image model` | `length` was set on an image family, which produces a 4-D latent. | Pick a video family, or leave `length` at 1. |
| `X is not normally driven from an empty latent` | `SEEDVR2` or `HUNYUAN_IMAGE_REFINER` was selected. Both consume an existing image or latent. | You almost certainly want a different node; these are provided for their shapes only. |

An `aspect_ratio` the node cannot parse raises a `ValueError` with the offending value rather than warning — including a comma (`1,5`), which is ambiguous between `1.5` and `1:5` in different locales.

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

### Example workflows

#### Image, with tiled upscaling

The node sits at the start of the workflow, before the KSampler. Connect `tile_width` and `tile_height` to the tiled upscaler you use *after* generation and VAE decode.

```
[Bobs Latent Optimizer]
  model_type: FLUX          ----> latent ---------> [KSampler] ---> [VAE Decode] ---+
  aspect_ratio: 16:9          |                                                     |
  mp_size: 1                  |---> tile_width  ---\                                |
  upscale_by: 2.0             |                     +--> [Tiled Upscaler] <---------+
                              |---> tile_height ---/     (Ultimate SD Upscale,
                              |                           Tiled VAE Decode, …)
                              |---> upscale_by -------> (if it takes a scale factor)
                              |
                              |---> width, height ----> (labels, filenames, debug)
```

With those settings the base image is 1344x768, and the tiles come back as 1344x768 — a 2x2 grid over the 2688x1536 upscaled output.

#### Video

Video families need the `length` input and emit a 5-D latent. Nothing else changes.

```
[Bobs Latent Optimizer]
  model_type: WAN           ----> latent ---------> [Wan sampler] ---> [VAE Decode]
  aspect_ratio: 16:9          |
  mp_size: 0.5                |---> width, height -> (for a matching video combine node)
  length: 81
```

`length: 81` with Wan's temporal downscale of 4 gives 21 latent frames — `((81 - 1) // 4) + 1`.

**Why is the tiling useful?**

Rather than guessing tile sizes, the node derives them from your desired final resolution (`base_resolution * upscale_by`) and the per-tile cap. That avoids:

*   tiles large enough to cause VRAM errors,
*   tiles unnecessarily small, adding processing overhead and seam risk,
*   inconsistent tiling between workflows.

## Upgrading from 1.2.x

Saved workflows keep loading — all five original `model_type` values still exist, and the two new outputs were appended so existing output links are unchanged. Four things do behave differently:

*   **`SD3` and `QWEN` produce different latents.** Both were allocated with 4 channels; both use 16-channel VAEs. `QWEN` also aligned to 28, which isn't divisible by the VAE stride of 8, so the reported pixel size didn't describe the tensor — it now aligns to 16. These were bugs; the new output is the correct one.
*   **`SD3` now honours `mp_size`.** It used to rescale every result back to roughly 1MP, so a 4MP selection silently gave you 1MP.
*   **`WAN` now emits a 5-D latent** `[B, C, T, H, W]` instead of 4-D, and takes a `length` input. The old 4-D shape was not usable with Wan samplers.
*   **`16,9` no longer parses.** Use `16:9`. A comma is ambiguous with decimal-comma locales, where `1,5` means 1.5.

**`FLUX` and `SDXL` are unchanged** — same dimensions, same channel counts, same tiles.

## Development

The sizing and tiling math is exposed as plain functions (`parse_aspect_ratio`, `compute_base_dimensions`, `compute_tile_dimensions`, `compute_latent_frames`), and the test suite stubs `torch`, so it runs without a ComfyUI or PyTorch install:

```
python -m unittest discover -s tests -v
```

CI runs the same suite on Python 3.9 and 3.12.

### Adding a model family

1.  Find the model in ComfyUI's `comfy/supported_models.py` to get its `latent_format`, then read that class in `comfy/latent_formats.py` for `latent_channels`, `latent_dimensions`, `spacial_downscale_ratio` and `temporal_downscale_ratio`.
2.  Add a row to `MODEL_SPECS` in `Bobs_Latent_Optimizer.py`. **`align` must be a multiple of `vae_scale`** — that invariant is what keeps `width // vae_scale` exact, and a test enforces it.
3.  Add the same verbatim upstream values to `COMFY_REFERENCE` in the test suite. If you need to deviate from upstream, put the deviation in `INTENTIONAL_OVERRIDES` with a justification rather than editing the reference table — a test asserts each override still genuinely conflicts, so it becomes obvious dead code if upstream ever agrees.
4.  Cross-check against the matching `Empty*Latent*` node in `nodes.py` / `comfy_extras/` where one exists. Not every family has one; say so in the comment if you couldn't verify.

### Releasing

Bump `version` in `pyproject.toml` and merge to `main` — the publish workflow triggers on that path and pushes to the registry. It refuses to run from any other branch, including via manual dispatch, because publishing from a feature branch ships unmerged code and burns the version number.

## Changelog

### 1.5.1

Fixes from a review of the 1.3.0–1.5.0 work. No shape changes — `FLUX 16:9 @1MP` is still `1344x768`, and every model's channel count, downscale and rank are untouched.

*   **Fixed a silent wrong answer in aspect-ratio parsing.** `,` was treated as a separator, so in decimal-comma locales `1,5` (meaning 1.5) was read as `1:5` = 0.2 — a 7.5× wrong ratio with no error. Before 1.3.0 this raised a clear `ValueError`; the comma is now rejected again. **If you were relying on `16,9`, use `16:9`.**
*   **Fixed dimensions exceeding ComfyUI's resolution limit.** There was no upper bound, so an extreme aspect ratio could emit sizes far past `MAX_RESOLUTION` (16384) — `1000:1` at 1MP produced 32384px wide. That allocates a small latent but fails much later, in the sampler or VAE decode, a long way from the aspect ratio that caused it. Dimensions are now scaled to fit the ceiling with the aspect ratio preserved where possible.
*   **Fixed silent clamping.** Both the minimum and the new maximum distort the requested ratio or area, so each now warns. The pre-1.3.0 code warned on the minimum clamp and the rewrite had dropped it.
*   **Fixed a fidelity gap against `EmptyLatentImage`.** Image families now allocate with `dtype=comfy.model_management.intermediate_dtype()`, matching `EmptyLatentImage` and `EmptySD3LatentImage`. Video families still pass only `device`, matching ComfyUI's video latent nodes.
*   Tests grow from 45 to 52, covering the resolution ceiling, both clamp warnings, comma rejection, and the dtype split — the gaps that let the two clamping bugs through.

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
