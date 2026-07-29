"""Tests for the pure sizing/tiling math and the node wiring.

torch is stubbed so the suite runs without a ComfyUI install.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class _FakeTensor:
        def __init__(self, shape, device=None, dtype=None):
            self.shape = tuple(shape)
            self.device = device
            # Sentinel so tests can tell "dtype was not passed" from "passed None".
            self.dtype_was_passed = dtype is not _MISSING
            self.dtype = None if dtype is _MISSING else dtype

    class _Missing:
        pass

    _MISSING = _Missing()

    torch_stub.zeros = lambda shape, device=None, dtype=_MISSING: _FakeTensor(
        shape, device, dtype
    )
    torch_stub.device = lambda name: name
    sys.modules["torch"] = torch_stub

import Bobs_Latent_Optimizer as blo  # noqa: E402

# Several tests deliberately pass `length` to image models to prove it is
# ignored; silence the resulting warnings. assertLogs re-enables the level it
# needs for the test that actually asserts on the warning.
import logging  # noqa: E402

blo.logger.setLevel(logging.CRITICAL)


# Ground truth transcribed VERBATIM from ComfyUI's comfy/latent_formats.py, as
# (latent_channels, spacial_downscale_ratio, temporal_downscale_ratio,
# latent_dimensions) for the format each model maps to in supported_models.py,
# cross-checked against the matching Empty*Latent* node where one exists.
# Keyed by this node's model_type. Nothing here is a judgement call.
COMFY_REFERENCE = {
    "SD15": (4, 8, 1, 2),
    "SD21": (4, 8, 1, 2),
    "SDXL": (4, 8, 1, 2),
    "PIXART": (4, 8, 1, 2),
    "AURAFLOW": (4, 8, 1, 2),
    "HUNYUAN_DIT": (4, 8, 1, 2),
    "SD3": (16, 8, 1, 2),
    "FLUX": (16, 8, 1, 2),
    "CHROMA": (16, 8, 1, 2),
    "HIDREAM": (16, 8, 1, 2),
    "LUMINA2": (16, 8, 1, 2),
    "OMNIGEN2": (16, 8, 1, 2),
    # QwenImage and CosmosT2IPredict2 map to latent_formats.Wan21, which is 3-D.
    "QWEN": (16, 8, 4, 3),
    "COSMOS_PREDICT2": (16, 8, 4, 3),
    "FLUX2": (128, 16, 1, 2),
    "HUNYUAN_IMAGE": (64, 32, 1, 2),
    "CHROMA_RADIANCE": (3, 1, 1, 2),
    "HIDREAM_O1": (3, 1, 1, 2),
    "ZIMAGE_PIXEL": (3, 1, 1, 2),
    "PIXELDIT": (3, 1, 1, 2),
    "WAN": (16, 8, 4, 3),
    "WAN22": (48, 16, 4, 3),
    "HUNYUAN_VIDEO": (16, 8, 4, 3),
    "HUNYUAN_VIDEO_15": (32, 16, 4, 3),
    "COSMOS": (16, 8, 8, 3),
    "COGVIDEOX": (16, 8, 4, 3),
    "MOCHI": (12, 8, 6, 3),
    "LTXV": (128, 32, 8, 3),
    "SEEDVR2": (16, 8, 1, 3),
    "HUNYUAN_IMAGE_REFINER": (64, 8, 1, 3),
}

# Deliberate deviations from the table above, with the reason. Keeping these
# separate is the point: it stops the reference table from quietly absorbing a
# judgement call and pretending it came from upstream.
#
# Qwen-Image and Cosmos Predict2 (text-to-image) share Wan's VAE, so they
# inherit latent_formats.Wan21 and its 3-D / temporal-4 declaration. Both are
# still-image models: ComfyUI's own Qwen workflows build their latent with
# EmptySD3LatentImage, which is 4-D. We follow the workflow, not the format.
INTENTIONAL_OVERRIDES = {
    "QWEN": {"temporal": 1, "dims": 2},
    "COSMOS_PREDICT2": {"temporal": 1, "dims": 2},
}

# Models included for their shape only - they are never driven from an empty
# latent, so selecting them warns.
NOT_EMPTY_LATENT_WORKFLOWS = {"SEEDVR2", "HUNYUAN_IMAGE_REFINER"}


class TestModelSpecs(unittest.TestCase):
    def test_specs_match_comfyui_latent_formats(self):
        self.assertEqual(set(blo.MODEL_SPECS), set(COMFY_REFERENCE))
        for model, (channels, vae_scale, temporal, dims) in COMFY_REFERENCE.items():
            expected = {
                "channels": channels,
                "vae_scale": vae_scale,
                "temporal": temporal,
                "dims": dims,
            }
            expected.update(INTENTIONAL_OVERRIDES.get(model, {}))
            spec = blo.MODEL_SPECS[model]
            for key, value in expected.items():
                self.assertEqual(spec[key], value, f"{model}.{key}")

    def test_overrides_actually_deviate_from_upstream(self):
        # If upstream ever changes to agree with us, this override is dead code
        # and should be deleted rather than left implying a conflict.
        for model, override in INTENTIONAL_OVERRIDES.items():
            channels, vae_scale, temporal, dims = COMFY_REFERENCE[model]
            upstream = {
                "channels": channels,
                "vae_scale": vae_scale,
                "temporal": temporal,
                "dims": dims,
            }
            self.assertTrue(
                any(upstream[k] != v for k, v in override.items()),
                f"{model} override no longer deviates from upstream",
            )

    def test_shape_only_models_are_flagged(self):
        for model, spec in blo.MODEL_SPECS.items():
            expected = model not in NOT_EMPTY_LATENT_WORKFLOWS
            self.assertEqual(spec.get("starts_from_empty", True), expected, model)

    def test_alignment_is_a_multiple_of_the_vae_scale(self):
        # This is the invariant that keeps width // vae_scale exact.
        for model, spec in blo.MODEL_SPECS.items():
            self.assertEqual(spec["align"] % spec["vae_scale"], 0, model)
            self.assertGreater(spec["align"], 0, model)

    def test_video_model_list_is_derived_correctly(self):
        expected = set()
        for model, (_, _, _, dims) in COMFY_REFERENCE.items():
            dims = INTENTIONAL_OVERRIDES.get(model, {}).get("dims", dims)
            if dims == 3:
                expected.add(model)
        self.assertEqual(set(blo.VIDEO_MODEL_TYPES), expected)


class TestParseAspectRatio(unittest.TestCase):
    def test_common_formats(self):
        for text in ("16:9", "16/9", "16x9", "16X9"):
            self.assertAlmostEqual(blo.parse_aspect_ratio(text), 16 / 9, msg=text)

    def test_decimal_components_and_bare_number(self):
        self.assertAlmostEqual(blo.parse_aspect_ratio("1.5:1"), 1.5)
        self.assertAlmostEqual(blo.parse_aspect_ratio("1.777"), 1.777)

    def test_whitespace_is_tolerated(self):
        self.assertAlmostEqual(blo.parse_aspect_ratio(" 3 : 2 "), 1.5)

    def test_rejects_bad_input(self):
        for text in ("", "1:0", "abc", "-16:9", "1:2:3", "0:1"):
            with self.assertRaises(ValueError, msg=text):
                blo.parse_aspect_ratio(text)

    def test_comma_is_rejected_rather_than_silently_misread(self):
        # Regression: "," used to be a separator, so a decimal-comma locale
        # typing "1,5" (meaning 1.5) silently got 1:5 = 0.2 instead. An error is
        # the only safe answer when the input is genuinely ambiguous.
        for text in ("1,5", "16,9"):
            with self.assertRaises(ValueError, msg=text):
                blo.parse_aspect_ratio(text)


class TestComputeBaseDimensions(unittest.TestCase):
    def test_square_1mp_at_align_64(self):
        self.assertEqual(blo.compute_base_dimensions(1024 * 1024, 1.0, 64), (1024, 1024))

    def test_dimensions_are_aligned_for_every_model(self):
        for model, spec in blo.MODEL_SPECS.items():
            for ratio in (1.0, 16 / 9, 9 / 16, 3 / 2, 0.37):
                width, height = blo.compute_base_dimensions(1024 * 1024, ratio, spec["align"])
                self.assertEqual(width % spec["align"], 0, (model, ratio))
                self.assertEqual(height % spec["align"], 0, (model, ratio))
                self.assertEqual(width % spec["vae_scale"], 0, (model, ratio))
                self.assertEqual(height % spec["vae_scale"], 0, (model, ratio))

    def test_tiny_target_area_clamps_instead_of_collapsing(self):
        self.assertEqual(blo.compute_base_dimensions(16, 1.0, 64), (64, 64))

    def test_area_is_approximately_preserved(self):
        width, height = blo.compute_base_dimensions(4 * 1024 * 1024, 16 / 9, 64)
        self.assertAlmostEqual(width / height, 16 / 9, delta=0.03)
        self.assertAlmostEqual((width * height) / (4 * 1024 * 1024), 1.0, delta=0.05)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            blo.compute_base_dimensions(0, 1.0, 64)
        with self.assertRaises(ValueError):
            blo.compute_base_dimensions(1024, 1.0, 0)
        with self.assertRaises(ValueError):
            # max_dim below one alignment step has no valid answer.
            blo.compute_base_dimensions(1024 * 1024, 1.0, 64, max_dim=32)

    def test_never_exceeds_comfyui_resolution_limit(self):
        # ComfyUI caps width/height at MAX_RESOLUTION (16384); anything larger
        # blows up later in the sampler or VAE decode, far from the cause.
        for model, spec in blo.MODEL_SPECS.items():
            for ratio in ("100:1", "1000:1", "1:1000"):
                multiplier = blo.parse_aspect_ratio(ratio)
                for area in (1024 * 1024, 16 * 1024 * 1024):
                    width, height = blo.compute_base_dimensions(
                        area, multiplier, spec["align"]
                    )
                    self.assertLessEqual(width, blo.MAX_DIMENSION, (model, ratio))
                    self.assertLessEqual(height, blo.MAX_DIMENSION, (model, ratio))
                    self.assertEqual(width % spec["align"], 0, (model, ratio))
                    self.assertEqual(height % spec["align"], 0, (model, ratio))

    def test_ceiling_scaling_preserves_the_ratio_when_it_can(self):
        # Only the width is over the limit here, so the ratio should survive.
        multiplier = blo.parse_aspect_ratio("4:1")
        width, height = blo.compute_base_dimensions(64 * 1024 * 1024, multiplier, 64)
        self.assertLessEqual(width, blo.MAX_DIMENSION)
        self.assertAlmostEqual(width / height, 4.0, delta=0.05)

    def test_hitting_the_ceiling_warns(self):
        with self.assertLogs(blo.logger, level="WARNING") as captured:
            blo.compute_base_dimensions(16 * 1024 * 1024, 1000.0, 64)
        self.assertTrue(any("exceeds the" in line for line in captured.output))

    def test_hitting_the_floor_warns(self):
        with self.assertLogs(blo.logger, level="WARNING") as captured:
            blo.compute_base_dimensions(1024, 1000.0, 64)
        self.assertTrue(any("below the" in line for line in captured.output))

    def test_ordinary_sizes_do_not_warn(self):
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        previous = blo.logger.level
        blo.logger.setLevel(logging.WARNING)
        blo.logger.addHandler(handler)
        try:
            for ratio in (1.0, 16 / 9, 9 / 16, 3 / 2, 2 / 3):
                for area in (512 * 512, 1024 * 1024, 2048 * 2048):
                    blo.compute_base_dimensions(area, ratio, 64)
        finally:
            blo.logger.removeHandler(handler)
            blo.logger.setLevel(previous)

        self.assertEqual([r.getMessage() for r in records], [])


class TestComputeLatentFrames(unittest.TestCase):
    def test_image_models_pass_length_through(self):
        self.assertEqual(blo.compute_latent_frames(1, 1), 1)
        self.assertEqual(blo.compute_latent_frames(17, 1), 17)

    def test_matches_comfyui_formula(self):
        # ComfyUI: ((length - 1) // temporal) + 1
        for temporal in (4, 6, 8):
            for length in (1, 2, 5, 17, 33, 121):
                self.assertEqual(
                    blo.compute_latent_frames(length, temporal),
                    ((length - 1) // temporal) + 1,
                    (temporal, length),
                )

    def test_wan_reference_values(self):
        # 81 frames is the common Wan 2.1 default -> 21 latent frames.
        self.assertEqual(blo.compute_latent_frames(81, 4), 21)
        self.assertEqual(blo.compute_latent_frames(1, 4), 1)

    def test_length_is_clamped_to_at_least_one(self):
        self.assertEqual(blo.compute_latent_frames(0, 4), 1)
        self.assertEqual(blo.compute_latent_frames(-5, 4), 1)


class TestComputeTileDimensions(unittest.TestCase):
    def test_default_grid_is_2x2(self):
        tile_w, tile_h, tiles_x, tiles_y = blo.compute_tile_dimensions(1024, 1024, 2.0)
        self.assertEqual((tiles_x, tiles_y), (2, 2))
        self.assertEqual((tile_w, tile_h), (1024, 1024))

    def test_grid_subdivides_when_tiles_would_exceed_the_cap(self):
        tile_w, tile_h, tiles_x, tiles_y = blo.compute_tile_dimensions(2048, 2048, 4.0)
        self.assertGreater(tiles_x, 2)
        self.assertGreater(tiles_y, 2)
        self.assertLessEqual(tile_w, blo.MAX_TILE_DIM)
        self.assertLessEqual(tile_h, blo.MAX_TILE_DIM)

    def test_tiles_cover_the_whole_upscaled_image(self):
        for width, height, upscale in ((1024, 1024, 2.0), (1920, 1080, 3.5), (512, 768, 1.0)):
            tile_w, tile_h, tiles_x, tiles_y = blo.compute_tile_dimensions(width, height, upscale)
            self.assertGreaterEqual(tile_w * tiles_x, int(width * upscale))
            self.assertGreaterEqual(tile_h * tiles_y, int(height * upscale))

    def test_tiles_are_stride_aligned(self):
        tile_w, tile_h, _, _ = blo.compute_tile_dimensions(1080, 1080, 1.7)
        self.assertEqual(tile_w % blo.TILE_ALIGN, 0)
        self.assertEqual(tile_h % blo.TILE_ALIGN, 0)

    def test_custom_cap_is_respected(self):
        tile_w, tile_h, _, _ = blo.compute_tile_dimensions(2048, 2048, 2.0, max_tile_dim=512)
        self.assertLessEqual(tile_w, 512)
        self.assertLessEqual(tile_h, 512)

    def test_tile_never_exceeds_the_image(self):
        tile_w, tile_h, _, _ = blo.compute_tile_dimensions(64, 64, 1.0)
        self.assertLessEqual(tile_w, 64)
        self.assertLessEqual(tile_h, 64)


class TestNodes(unittest.TestCase):
    def test_latent_shape_matches_reported_pixel_size(self):
        node = blo.BobsLatentNode()
        for model, spec in blo.MODEL_SPECS.items():
            latent, _, _, _, width, height = node.generate("16:9", "1", 2.0, model, 2, length=17)
            shape = latent["samples"].shape
            self.assertEqual(shape[0], 2, model)
            self.assertEqual(shape[1], spec["channels"], model)
            latent_h, latent_w = shape[-2], shape[-1]
            self.assertEqual(latent_w * spec["vae_scale"], width, model)
            self.assertEqual(latent_h * spec["vae_scale"], height, model)

    def test_latent_rank_matches_the_model_family(self):
        node = blo.BobsLatentNode()
        for model, spec in blo.MODEL_SPECS.items():
            latent, _, _, _, _, _ = node.generate("1:1", "1", 2.0, model, 1, length=17)
            expected_rank = 5 if spec["dims"] == 3 else 4
            self.assertEqual(len(latent["samples"].shape), expected_rank, model)

    def test_video_models_use_the_comfyui_frame_formula(self):
        node = blo.BobsLatentNode()
        for model in blo.VIDEO_MODEL_TYPES:
            temporal = blo.MODEL_SPECS[model]["temporal"]
            for length in (1, 17, 81):
                latent, _, _, _, _, _ = node.generate(
                    "1:1", "1", 2.0, model, 1, length=length
                )
                self.assertEqual(
                    latent["samples"].shape[2],
                    ((length - 1) // temporal) + 1,
                    (model, length),
                )

    def test_image_models_ignore_length_and_warn(self):
        node = blo.BobsLatentNode()
        with self.assertLogs(blo.logger, level="WARNING") as captured:
            latent, _, _, _, _, _ = node.generate("1:1", "1", 2.0, "FLUX", 1, length=48)
        self.assertEqual(len(latent["samples"].shape), 4)
        self.assertTrue(any("length=48 ignored" in line for line in captured.output))

    def test_dtype_is_passed_for_image_families_only(self):
        # Matches ComfyUI: EmptyLatentImage / EmptySD3LatentImage pass
        # dtype=intermediate_dtype(); the video latent nodes pass only device.
        # Outside ComfyUI there is no intermediate_dtype(), so nothing is passed.
        node = blo.BobsLatentNode()
        expect_dtype = blo._latent_dtype() is not None
        for model, spec in blo.MODEL_SPECS.items():
            latent, _, _, _, _, _ = node.generate("1:1", "1", 2.0, model, 1)
            samples = latent["samples"]
            self.assertEqual(
                samples.dtype_was_passed, expect_dtype and spec["dims"] == 2, model
            )

    def test_pixel_space_models_have_no_downscale(self):
        # vae_scale of 1 means the latent dims equal the pixel dims.
        node = blo.BobsLatentNode()
        for model in ("CHROMA_RADIANCE", "HIDREAM_O1", "ZIMAGE_PIXEL", "PIXELDIT"):
            latent, _, _, _, width, height = node.generate("1:1", "1", 2.0, model, 1)
            _, channels, latent_h, latent_w = latent["samples"].shape
            self.assertEqual(channels, 3, model)
            self.assertEqual((latent_w, latent_h), (width, height), model)

    def test_shape_only_models_warn_when_selected(self):
        node = blo.BobsLatentNode()
        for model in sorted(NOT_EMPTY_LATENT_WORKFLOWS):
            with self.assertLogs(blo.logger, level="WARNING") as captured:
                node.generate("1:1", "1", 2.0, model, 1)
            self.assertTrue(
                any("not normally driven from an empty latent" in line for line in captured.output),
                model,
            )

    def test_ordinary_models_do_not_warn(self):
        # assertNoLogs is 3.10+, and this suite supports 3.9, so capture manually.
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        previous = blo.logger.level
        blo.logger.setLevel(logging.WARNING)
        blo.logger.addHandler(handler)
        try:
            node = blo.BobsLatentNode()
            for model in ("FLUX", "SDXL", "WAN", "LTXV"):
                node.generate("1:1", "1", 2.0, model, 1)
        finally:
            blo.logger.removeHandler(handler)
            blo.logger.setLevel(previous)

        self.assertEqual([r.getMessage() for r in records], [])

    def test_high_compression_models(self):
        node = blo.BobsLatentNode()
        # HunyuanImage 2.1: 64 channels at 32x downscale.
        latent, _, _, _, width, height = node.generate("1:1", "1", 2.0, "HUNYUAN_IMAGE", 1)
        _, channels, latent_h, latent_w = latent["samples"].shape
        self.assertEqual(channels, 64)
        self.assertEqual((latent_w, latent_h), (width // 32, height // 32))
        # Flux2: 128 channels at 16x downscale.
        latent, _, _, _, width, height = node.generate("1:1", "1", 2.0, "FLUX2", 1)
        _, channels, latent_h, latent_w = latent["samples"].shape
        self.assertEqual(channels, 128)
        self.assertEqual((latent_w, latent_h), (width // 16, height // 16))

    def test_mp_size_is_honoured_for_every_model(self):
        # Regression: SD3 used to rescale every result back to ~1MP.
        node = blo.BobsLatentNode()
        for model in blo.MODEL_TYPES:
            _, _, _, _, small_w, small_h = node.generate("1:1", "1", 2.0, model, 1)
            _, _, _, _, big_w, big_h = node.generate("1:1", "4", 2.0, model, 1)
            self.assertGreater(big_w * big_h, small_w * small_h, model)
            self.assertAlmostEqual((big_w * big_h) / (2048 * 2048), 1.0, delta=0.05, msg=model)

    def test_upscale_by_passes_through(self):
        result = blo.BobsLatentNode().generate("1:1", "1", 3.25, "FLUX", 1)
        self.assertAlmostEqual(result[3], 3.25)

    def test_advanced_node_matches_preset_node_at_equal_area(self):
        preset = blo.BobsLatentNode().generate("16:9", "1", 2.0, "FLUX", 1)
        advanced = blo.BobsLatentNodeAdvanced().generate("16:9", 1.0, 2.0, "FLUX", 1)
        self.assertEqual(preset[4:], advanced[4:])

    def test_advanced_node_supports_video_models(self):
        latent, _, _, _, _, _ = blo.BobsLatentNodeAdvanced().generate(
            "16:9", 1.0, 2.0, "WAN", 1, length=81
        )
        self.assertEqual(latent["samples"].shape[1:3], (16, 21))

    def test_advanced_node_rejects_non_positive_area(self):
        with self.assertRaises(ValueError):
            blo.BobsLatentNodeAdvanced().generate("1:1", 0.0, 2.0, "FLUX", 1)

    def test_unknown_selections_raise(self):
        with self.assertRaises(ValueError):
            blo.BobsLatentNode().generate("1:1", "1", 2.0, "NOPE", 1)
        with self.assertRaises(ValueError):
            blo.BobsLatentNode().generate("1:1", "7", 2.0, "FLUX", 1)

    def test_max_tile_size_override(self):
        _, tile_w, tile_h, _, _, _ = blo.BobsLatentNode().generate(
            "1:1", "4", 2.0, "FLUX", 1, max_tile_size=512
        )
        self.assertLessEqual(tile_w, 512)
        self.assertLessEqual(tile_h, 512)

    def test_input_types_declare_every_generate_argument(self):
        for cls in (blo.BobsLatentNode, blo.BobsLatentNodeAdvanced):
            spec = cls.INPUT_TYPES()
            declared = set(spec["required"]) | set(spec.get("optional", {}))
            params = set(cls.generate.__code__.co_varnames[1:cls.generate.__code__.co_argcount])
            self.assertEqual(declared, params, cls.__name__)

    def test_model_dropdown_lists_every_spec(self):
        for cls in (blo.BobsLatentNode, blo.BobsLatentNodeAdvanced):
            choices = cls.INPUT_TYPES()["required"]["model_type"][0]
            self.assertEqual(list(choices), list(blo.MODEL_SPECS), cls.__name__)

    def test_default_model_is_still_flux(self):
        for cls in (blo.BobsLatentNode, blo.BobsLatentNodeAdvanced):
            opts = cls.INPUT_TYPES()["required"]["model_type"][1]
            self.assertEqual(opts["default"], "FLUX", cls.__name__)

    def test_return_metadata_is_consistent(self):
        for cls in (blo.BobsLatentNode, blo.BobsLatentNodeAdvanced):
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES), cls.__name__)
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.OUTPUT_TOOLTIPS), cls.__name__)
            result = cls().generate("1:1", "1" if cls is blo.BobsLatentNode else 1.0, 2.0, "FLUX", 1)
            self.assertEqual(len(result), len(cls.RETURN_TYPES), cls.__name__)

    def test_display_names_cover_every_registered_node(self):
        self.assertEqual(set(blo.NODE_CLASS_MAPPINGS), set(blo.NODE_DISPLAY_NAME_MAPPINGS))

    def test_package_exports_match_the_module(self):
        # Import __init__.py the way ComfyUI does: as a package, so the relative
        # import inside it is exercised.
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "bobs_latent_optimizer_pkg",
            os.path.join(root, "__init__.py"),
            submodule_search_locations=[root],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        try:
            spec.loader.exec_module(package)
            self.assertEqual(set(package.NODE_CLASS_MAPPINGS), set(blo.NODE_CLASS_MAPPINGS))
            self.assertEqual(
                package.NODE_DISPLAY_NAME_MAPPINGS, blo.NODE_DISPLAY_NAME_MAPPINGS
            )
        finally:
            sys.modules.pop(spec.name, None)


if __name__ == "__main__":
    unittest.main()
