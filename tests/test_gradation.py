import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import gradation
from engine import Engine, adjustments, read_image, settings
from gradation import deband, read_gain_map, roll_off, to_uint8

XMP = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
       b'<rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0" '
       b'hdrgm:GainMapMin="0" hdrgm:GainMapMax="2.5" hdrgm:Gamma="1"/></rdf:RDF></x:xmpmeta>')


def blur(a, r=6):
    kernel = np.ones(2*r+1)/(2*r+1)
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode='valid'), 1, a)


class GradationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def test_dither_is_positional_clean_at_extremes_and_prevents_banding(self):
        ramp = torch.linspace(.40, .43, 512).repeat(64, 1)[..., None].expand(64, 512, 3).contiguous()
        stretched = (ramp-.40)*8+.3  # an edit that stretches ~8 codes over the whole width
        a, b = to_uint8(stretched), to_uint8(stretched)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(to_uint8(stretched[:32], 0), a[:32])
        np.testing.assert_array_equal(to_uint8(stretched[32:], 32), a[32:])
        ideal = stretched[..., 0].numpy()*255
        plain = to_uint8(stretched, dither=False)[..., 0].astype(float)
        dithered = a[..., 0].astype(float)
        # Viewed at normal distance (local average), dither tracks the smooth ramp far better.
        self.assertLess(np.abs(blur(dithered.mean(0, keepdims=True))-blur(ideal.mean(0, keepdims=True))).max(),
                        np.abs(blur(plain.mean(0, keepdims=True))-blur(ideal.mean(0, keepdims=True))).max()*.6)
        extremes = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]])
        np.testing.assert_array_equal(to_uint8(extremes), [[[0, 0, 0], [255, 255, 255]]])

    def test_deband_flattens_code_steps_but_keeps_texture(self):
        codes = np.repeat(np.arange(100, 108), 40)[None].repeat(48, 0)
        steps = torch.from_numpy(codes/255).float()[..., None].expand(48, 320, 3).contiguous()
        smooth = deband(steps, 12, .6)[..., 0].numpy()*255
        self.assertLess(np.abs(np.diff(smooth[24])).max(), .5)
        texture = .5+.04*torch.from_numpy(np.random.default_rng(1).standard_normal((64, 64, 3), dtype=np.float32))
        self.assertGreater(float(deband(texture, 8, .6).std()), float(texture.std())*.9)

    def test_roll_off_is_monotonic_bounded_and_identity_below_knee(self):
        values = torch.linspace(0, 1.6, 400)[:, None, None].expand(400, 1, 3).contiguous()
        np.testing.assert_array_equal(roll_off(values, 0).numpy(), values.numpy())
        shaped = roll_off(values, .6, headroom=1.6)[:, 0, 0]
        self.assertTrue(bool((shaped[1:] >= shaped[:-1]-1e-6).all()))
        self.assertLessEqual(float(shaped.max()), 1+1e-5)
        self.assertAlmostEqual(float(shaped[-1]), 1., places=4)
        below = values[:, 0, 0] < .84
        torch.testing.assert_close(shaped[below], values[below, 0, 0])

    def test_roll_off_leaves_negative_shadows_alone(self):
        # Strong contrast pushes deep shadows below zero before the final clamp.
        shadows = torch.tensor([[[-.08, -.02, -.05], [-.01, .02, -.03], [0., 0., 0.]]])
        torch.testing.assert_close(roll_off(shadows, .8), shadows)
        pixels = np.random.default_rng(6).integers(0, 30, (64, 64, 3), dtype=np.uint8)
        pixels[20:30, 20:30] = 250
        image = Image.fromarray(pixels)
        edit = {'contrast': 80, 'exposure': -.5}
        plain = np.asarray(self.engine.render(image, edit, 'holes')).astype(int)
        graded = np.asarray(self.engine.render(image, {**edit, 'gradation': 100}, 'holes')).astype(int)
        dark = pixels.max(-1) < 30
        self.assertLessEqual(int(np.abs(graded-plain)[dark].max()), 3)

    def test_tiled_gradation_render_matches_single_pass(self):
        pixels = np.random.default_rng(4).integers(90, 140, (405, 70, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        p = settings({'gradation': 60, 'exposure': .3})
        result = np.asarray(self.engine.render(image, p, 'grad-tiles')).astype(int)
        x = torch.from_numpy(pixels.copy()).float().to(self.engine.device)/255
        x = deband(x, gradation.deband_radius(image.size), .6)
        expected = to_uint8(adjustments(x, p), 0).astype(int)
        self.assertLessEqual(np.abs(result-expected).max(), 1)

    def test_ultra_hdr_gain_map_recovers_clipped_highlight_detail(self):
        with tempfile.TemporaryDirectory() as folder:
            primary = np.full((120, 160, 3), 110, np.uint8)
            primary[:, 80:] = 255  # clipped sky: flat white in the SDR image
            gain = np.zeros((30, 40), np.uint8)
            gain[:, 20:] = np.linspace(0, 255, 20).astype(np.uint8)  # HDR still has a gradient
            path = Path(folder)/'ultra.jpg'
            Image.fromarray(primary).save(path, 'MPO', save_all=True, xmp=XMP, quality=95,
                                          append_images=[Image.fromarray(gain).convert('RGB')])
            image = read_image(path)
            self.assertEqual(image.info.get('onbit_gain_label'), 'Ultra HDR 게인맵')
            self.assertAlmostEqual(float(image.info['onbit_gain'].max()), 2.5, places=1)
            self.assertEqual(self.engine.recommend(image, 'ultra')['gradation'], 50)
            before = np.asarray(self.engine.render(image, {}, 'ultra'))[:, 100:, 0].astype(float)
            after = np.asarray(self.engine.render(image, {'gradation': 60}, 'ultra'))[:, 100:, 0].astype(float)
            self.assertLess(before.std(), .5)
            self.assertGreater(after[60, -5]-after[60, 5], 8)  # recorded gradation is back
            # A stereo MPO without gain-map metadata is left alone.
            stereo = Path(folder)/'stereo.jpg'
            Image.fromarray(primary).save(stereo, 'MPO', save_all=True,
                                          append_images=[Image.fromarray(primary)])
            self.assertNotIn('onbit_gain', read_image(stereo).info)

    def test_gain_map_follows_photo_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'rotated.jpg'
            exif = Image.Exif()
            exif[0x0112] = 6
            gain = np.zeros((20, 40), np.uint8)
            gain[:, :4] = 255  # bright stripe on the stored left edge
            Image.new('RGB', (80, 40), 'gray').save(path, 'MPO', save_all=True, xmp=XMP, exif=exif,
                                                   append_images=[Image.fromarray(gain).convert('RGB')])
            image = read_image(path)
            self.assertEqual(image.size, (40, 80))
            stops = image.info['onbit_gain']
            self.assertEqual(stops.shape, (40, 20))
            # Rotating 90° clockwise moves the stored left edge to the top.
            self.assertGreater(stops[:2].mean(), 2)
            self.assertLess(stops[-2:].mean(), .1)

    def test_apple_heic_gain_map_is_read_from_auxiliary_image(self):
        gain = Image.fromarray(np.tile(np.linspace(0, 255, 40, dtype=np.uint8), (30, 1)))
        fake = SimpleNamespace(info={'aux': {'urn:com:apple:photo:2020:aux:hdrgainmap': [7]}},
                               get_aux_image=lambda i: SimpleNamespace(to_pillow=lambda: gain.copy()))
        with patch('pillow_heif.open_heif', return_value=fake):
            stops, label = read_gain_map('photo.heic', 'HEIF', (80, 60))
        self.assertEqual(label, 'Apple HDR 게인맵')
        self.assertAlmostEqual(float(stops.max()), gradation.APPLE_HEADROOM_STOPS, places=2)
        self.assertLess(float(stops[:, 0].max()), .01)
        fake.info = {'aux': {'urn:com:apple:photo:2020:aux:depth': [3]}}
        with patch('pillow_heif.open_heif', return_value=fake):
            self.assertIsNone(read_gain_map('photo.heic', 'HEIF', (80, 60)))

    def test_untouched_photo_stays_bit_identical(self):
        pixels = np.random.default_rng(9).integers(0, 256, (97, 61, 3), dtype=np.uint8)
        np.testing.assert_array_equal(np.asarray(self.engine.render(Image.fromarray(pixels), {}, 'same')), pixels)


if __name__ == '__main__':
    unittest.main(verbosity=2)
