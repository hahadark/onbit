from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from batch import BatchItem, BatchOptions, file_key, run_batch
from engine import Engine, read_image, settings
from tone import color_profile, match_plan, rgb_to_lab, lab_to_rgb, light_stats, sample_rgb


class ToneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def scene(self):
        y, x = np.mgrid[0:96, 0:144].astype(np.float32)
        pixels = np.stack([.18+x/230, .16+y/170, .22+(x+y)/500], -1)
        return Image.fromarray((np.clip(pixels, 0, 1)*255).round().astype(np.uint8))

    def test_oklab_roundtrip_and_endpoints(self):
        rgb = torch.cat([torch.rand(200, 3), torch.zeros(1, 3), torch.ones(1, 3)])
        self.assertLess(float((rgb-lab_to_rgb(rgb_to_lab(rgb))).abs().max()), .00002)

    def test_matching_reduces_reference_distance_without_changing_size(self):
        image = self.scene()
        array = np.asarray(image).astype(float)
        target = Image.fromarray(np.clip(array*np.array([1.04, 1.02, .91])+8, 0, 255).astype(np.uint8))
        source, reference = color_profile(image), color_profile(target)
        result = self.engine.match(image, match_plan(source, reference, 100))
        before = np.linalg.norm(np.array(source['center'])-reference['center'])
        after = np.linalg.norm(np.array(color_profile(result)['center'])-reference['center'])
        self.assertLess(after, before*.6)
        self.assertEqual(result.size, image.size)
        np.testing.assert_array_equal(np.asarray(self.engine.match(image, match_plan(source, reference, 0))), np.asarray(image))

    def test_brightness_opt_out_and_flat_images_are_stable(self):
        image = self.scene()
        target = Image.new('RGB', image.size, (205, 190, 180))
        plan = match_plan(color_profile(image), color_profile(target), 100, False)
        result = self.engine.match(image, plan)
        before_l = rgb_to_lab(torch.from_numpy(np.asarray(image).copy()/255).float())[..., 0]
        after_l = rgb_to_lab(torch.from_numpy(np.asarray(result).copy()/255).float())[..., 0]
        self.assertLess(float((before_l-after_l).abs().mean()), .004)
        for color in ('black', 'white', '#808080'):
            flat = Image.new('RGB', (48, 32), color)
            out = np.asarray(self.engine.match(flat, match_plan(color_profile(flat), color_profile(target), 100)))
            self.assertTrue(np.isfinite(out).all())
            self.assertLessEqual(float(out.std((0, 1)).max()), .01)
            if color in ('black', 'white'):
                np.testing.assert_array_equal(out, np.asarray(flat))

    def test_natural_recommendations_are_bounded_and_repeatable(self):
        for i, image in enumerate([self.scene(), Image.new('RGB', (64, 48), 'black'), Image.new('RGB', (64, 48), 'white')]):
            p = self.engine.recommend(image, f'natural-{i}')
            self.assertEqual(p, self.engine.recommend(image, f'natural-{i}'))
            self.assertLessEqual(p['ai'], 35)
            self.assertLessEqual(abs(p['exposure']), .7)
            self.assertLessEqual(abs(p['highlights']), 30)
            self.assertLessEqual(abs(p['temperature']), 12)
            self.assertLessEqual(abs(p['tint']), 10)
            self.assertLessEqual(p['saturation'], 0)
            self.assertEqual(self.engine.render(image, p, f'natural-{i}').size, image.size)

    def test_batch_keeps_corrected_reference_and_records_per_photo_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref, target = root/'기준.png', root/'대상.png'
            self.scene().save(ref)
            Image.fromarray((np.asarray(self.scene())*.8).astype(np.uint8)).save(target)
            reference_settings = settings({'exposure': .1, 'temperature': 8})
            expected = self.engine.render(read_image(ref), reference_settings, file_key(ref))
            options = BatchOptions(str(root/'out'), format='PNG', correction='match', reference_path=str(ref),
                                   reference_adjustments=reference_settings, match_strength=100)
            result = run_batch([BatchItem(str(ref), ref.name), BatchItem(str(target), target.name)], options, self.engine)
            self.assertEqual((result['succeeded'], result['failed']), (2, 0), result)
            np.testing.assert_array_equal(np.asarray(read_image(result['files'][0]['output'])), np.asarray(expected))
            self.assertTrue(result['files'][0]['reference_unchanged'])
            self.assertIn('match', result['files'][1])
            self.assertNotEqual(result['files'][0]['output'], str(ref))
            auto = run_batch([BatchItem(str(target), target.name)], BatchOptions(str(root/'auto'), correction='auto'), self.engine)
            self.assertEqual(auto['succeeded'], 1, auto)
            self.assertLessEqual(auto['files'][0]['adjustments']['ai'], 35)

    def test_cuda_and_cpu_matching_agree(self):
        if self.engine.device.type != 'cuda':
            self.skipTest('CUDA unavailable')
        image = self.scene()
        plan = match_plan(color_profile(image), color_profile(Image.new('RGB', image.size, '#c7beb0')), 65)
        gpu = np.asarray(self.engine.match(image, plan)).astype(int)
        device = self.engine.device
        try:
            self.engine.device = torch.device('cpu')
            cpu = np.asarray(self.engine.match(image, plan)).astype(int)
        finally:
            self.engine.device = device
        self.assertLessEqual(np.abs(gpu-cpu).max(), 1)


if __name__ == '__main__':
    unittest.main()
