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
        def __init__(self, shape):
            self.shape = tuple(shape)

    torch_stub.zeros = lambda shape, device=None: _FakeTensor(shape)
    torch_stub.device = lambda name: name
    sys.modules["torch"] = torch_stub

import Bobs_Latent_Optimizer as blo  # noqa: E402


class TestParseAspectRatio(unittest.TestCase):
    def test_common_formats(self):
        for text in ("16:9", "16/9", "16x9", "16,9"):
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


class TestComputeBaseDimensions(unittest.TestCase):
    def test_square_1mp_at_align_64(self):
        self.assertEqual(blo.compute_base_dimensions(1024 * 1024, 1.0, 64), (1024, 1024))

    def test_dimensions_are_aligned_and_divisible_by_vae_stride(self):
        for model, spec in blo.MODEL_SPECS.items():
            for ratio in (1.0, 16 / 9, 9 / 16, 3 / 2, 0.37):
                width, height = blo.compute_base_dimensions(1024 * 1024, ratio, spec["align"])
                self.assertEqual(width % spec["align"], 0, (model, ratio))
                self.assertEqual(height % spec["align"], 0, (model, ratio))
                self.assertEqual(width % blo.VAE_SCALE_FACTOR, 0, (model, ratio))
                self.assertEqual(height % blo.VAE_SCALE_FACTOR, 0, (model, ratio))

    def test_tiny_target_area_clamps_instead_of_collapsing(self):
        width, height = blo.compute_base_dimensions(16, 1.0, 64)
        self.assertEqual((width, height), (64, 64))

    def test_area_is_approximately_preserved(self):
        width, height = blo.compute_base_dimensions(4 * 1024 * 1024, 16 / 9, 64)
        self.assertAlmostEqual(width / height, 16 / 9, delta=0.03)
        self.assertAlmostEqual((width * height) / (4 * 1024 * 1024), 1.0, delta=0.05)

    def test_rejects_non_positive_area(self):
        with self.assertRaises(ValueError):
            blo.compute_base_dimensions(0, 1.0, 64)


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

    def test_tiles_are_vae_stride_aligned(self):
        tile_w, tile_h, _, _ = blo.compute_tile_dimensions(1080, 1080, 1.7)
        self.assertEqual(tile_w % blo.VAE_SCALE_FACTOR, 0)
        self.assertEqual(tile_h % blo.VAE_SCALE_FACTOR, 0)

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
        for model in blo.MODEL_TYPES:
            latent, _, _, _, width, height = node.generate("16:9", "1", 2.0, model, 2)
            batch, channels, latent_h, latent_w = latent["samples"].shape
            self.assertEqual(batch, 2, model)
            self.assertEqual(channels, blo.MODEL_SPECS[model]["channels"], model)
            self.assertEqual(latent_w * blo.VAE_SCALE_FACTOR, width, model)
            self.assertEqual(latent_h * blo.VAE_SCALE_FACTOR, height, model)

    def test_sdxl_is_the_only_four_channel_family(self):
        node = blo.BobsLatentNode()
        for model in blo.MODEL_TYPES:
            latent, _, _, _, _, _ = node.generate("1:1", "1", 2.0, model, 1)
            expected = 4 if model == "SDXL" else 16
            self.assertEqual(latent["samples"].shape[1], expected, model)

    def test_mp_size_is_honoured_for_every_model(self):
        # Regression: SD3 used to rescale every result back to ~1MP.
        node = blo.BobsLatentNode()
        for model in blo.MODEL_TYPES:
            _, _, _, _, small_w, small_h = node.generate("1:1", "1", 2.0, model, 1)
            _, _, _, _, big_w, big_h = node.generate("1:1", "4", 2.0, model, 1)
            self.assertGreater(big_w * big_h, small_w * small_h, model)
            self.assertAlmostEqual((big_w * big_h) / (2048 * 2048), 1.0, delta=0.05, msg=model)

    def test_upscale_by_passes_through(self):
        node = blo.BobsLatentNode()
        result = node.generate("1:1", "1", 3.25, "FLUX", 1)
        self.assertAlmostEqual(result[3], 3.25)

    def test_advanced_node_matches_preset_node_at_equal_area(self):
        preset = blo.BobsLatentNode().generate("16:9", "1", 2.0, "FLUX", 1)
        advanced = blo.BobsLatentNodeAdvanced().generate("16:9", 1.0, 2.0, "FLUX", 1)
        self.assertEqual(preset[4:], advanced[4:])

    def test_advanced_node_rejects_non_positive_area(self):
        with self.assertRaises(ValueError):
            blo.BobsLatentNodeAdvanced().generate("1:1", 0.0, 2.0, "FLUX", 1)

    def test_unknown_selections_raise(self):
        with self.assertRaises(ValueError):
            blo.BobsLatentNode().generate("1:1", "1", 2.0, "NOPE", 1)
        with self.assertRaises(ValueError):
            blo.BobsLatentNode().generate("1:1", "7", 2.0, "FLUX", 1)

    def test_max_tile_size_override(self):
        node = blo.BobsLatentNode()
        _, tile_w, tile_h, _, _, _ = node.generate("1:1", "4", 2.0, "FLUX", 1, max_tile_size=512)
        self.assertLessEqual(tile_w, 512)
        self.assertLessEqual(tile_h, 512)

    def test_input_types_declare_every_generate_argument(self):
        for cls in (blo.BobsLatentNode, blo.BobsLatentNodeAdvanced):
            spec = cls.INPUT_TYPES()
            declared = set(spec["required"]) | set(spec.get("optional", {}))
            params = set(cls.generate.__code__.co_varnames[1:cls.generate.__code__.co_argcount])
            self.assertEqual(declared, params, cls.__name__)

    def test_return_metadata_is_consistent(self):
        for cls in (blo.BobsLatentNode, blo.BobsLatentNodeAdvanced):
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES), cls.__name__)
            self.assertEqual(len(cls.RETURN_TYPES), len(cls.OUTPUT_TOOLTIPS), cls.__name__)
            result = cls().generate("1:1", "1" if cls is blo.BobsLatentNode else 1.0, 2.0, "FLUX", 1)
            self.assertEqual(len(result), len(cls.RETURN_TYPES), cls.__name__)

    def test_display_names_cover_every_registered_node(self):
        self.assertEqual(
            set(blo.NODE_CLASS_MAPPINGS), set(blo.NODE_DISPLAY_NAME_MAPPINGS)
        )

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
