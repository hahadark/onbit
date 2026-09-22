import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from face import FaceRetoucher
from engine import Engine, box_filter, settings, read_image, smooth_skin
from batch import BatchItem, BatchOptions, run_batch


class FaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.portrait = read_image(Path(__file__).parent/'fixtures/astronaut.png')

    def test_skin_setting_is_bounded_and_persisted(self):
        self.assertEqual(settings({'skin': 42})['skin'], 42)
        with self.assertRaises(ValueError):
            settings({'skin': 101})

    def test_skin_mask_is_soft_and_stays_inside_face_box(self):
        retoucher = FaceRetoucher()
        if not retoucher.available:
            self.skipTest('OpenCV is optional in the source test environment')
        image = Image.new('RGB', (160, 120), (40, 80, 120))
        pixels = np.asarray(image).copy()
        pixels[25:95, 45:115] = (190, 135, 110)
        image = Image.fromarray(pixels)
        mask = retoucher.geometric_mask(image, [(40, 20, 80, 80)])
        self.assertGreater(int(mask[67, 64]), 0)
        self.assertEqual(int(mask[5, 5]), 0)
        self.assertEqual(int(mask[51, 80]), 0)  # eye band
        self.assertEqual(int(mask[81, 80]), 0)  # mouth
        self.assertLessEqual(int(mask.max()), 255)

    def test_ai_parsing_masks_real_skin_but_not_eyes_lips_or_hair(self):
        face = self.engine.face
        self.assertIsNotNone(face.detector, face.error)
        self.assertIsNotNone(face.parser, face.parser_error)
        self.assertEqual(face.mode, 'AI 얼굴 인식 + 피부 영역 분할')
        mask = self.engine._skin_mask(self.portrait, 'parsing')
        # astronaut.png: cheeks/forehead are skin; eyes, lips and hair are not.
        self.assertGreater(int(mask[125, 205]), 200)   # left cheek
        self.assertGreater(int(mask[88, 225]), 200)    # forehead below the fringe
        self.assertLess(int(mask[102, 200]), 60)       # left eye
        self.assertLess(int(mask[102, 247]), 60)       # right eye
        self.assertLess(int(mask[146, 220]), 60)       # lips / teeth
        self.assertLess(int(mask[35, 225]), 30)        # hair
        self.assertEqual(int(mask[300, 60]), 0)        # flag in the background

    def test_turned_and_tilted_faces_are_found(self):
        tilted = self.portrait.rotate(28, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(200, 200, 200))
        self.assertGreaterEqual(self.engine.face_count(tilted, 'tilted'), 1)
        close_up = self.portrait.crop((150, 30, 310, 190)).resize((1400, 1400), Image.Resampling.LANCZOS)
        self.assertEqual(self.engine.face_count(close_up, 'close-up'), 1)
        # A face turned towards the edge of the frame, cut at the far cheek.
        turned = self.portrait.crop((190, 40, 330, 200))
        self.assertGreaterEqual(self.engine.face_count(turned, 'turned'), 1)

    def test_real_face_and_group_detected_with_unicode_model_path(self):
        self.assertTrue(self.engine.face.available, self.engine.face.error)
        boxes = self.engine._face_boxes(self.portrait, 'astronaut')
        self.assertTrue(any(160 < x < 200 and 45 < y < 85 for x, y, w, h in boxes), boxes)
        group = Image.new('RGB', (1024, 512))
        group.paste(self.portrait, (0, 0))
        group.paste(self.portrait, (512, 0))
        self.assertGreaterEqual(self.engine.face_count(group, 'group'), 2)

    def test_skin_changes_face_only_and_zero_is_identity(self):
        mask = self.engine._skin_mask(self.portrait, 'astronaut')
        source = np.asarray(self.portrait)
        result = np.asarray(self.engine.render(self.portrait, {'skin': 80}, 'astronaut'))
        self.assertGreater(float(np.abs(result.astype(float)-source)[mask > 0].mean()), .1)
        np.testing.assert_array_equal(result[mask == 0], source[mask == 0])
        np.testing.assert_array_equal(np.asarray(self.engine.render(self.portrait, {}, 'astronaut')), source)
        blank = Image.new('RGB', (140, 140), (180, 130, 100))
        np.testing.assert_array_equal(np.asarray(self.engine.render(blank, {'skin': 100}, 'blank')), np.asarray(blank))

    def test_retouch_matches_untiled_calculation_at_tile_boundaries(self):
        pixels = np.random.default_rng(4).integers(60, 200, (405, 70, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        with patch.object(self.engine, '_skin_mask', return_value=np.full((405, 70), 255, np.uint8)), \
                patch.object(self.engine, '_skin_radius', return_value=7):
            result = np.asarray(self.engine.render(image, {'skin': 70}, 'seams'))
        x = torch.from_numpy(pixels.copy()).float()/255
        expected = torch.lerp(x, smooth_skin(x, 7), .7*.9)
        expected = ((expected+.7*.04*(1-expected)).clamp(0, 1)*255).round().byte().numpy()
        self.assertLessEqual(np.abs(result.astype(int)-expected.astype(int)).max(), 1)

    def test_box_filter_matches_direct_mean_and_guided_filter_keeps_edges(self):
        x = torch.from_numpy(np.random.default_rng(2).random((23, 17, 3), dtype=np.float32))
        r = 3
        direct = torch.stack([torch.stack([x[max(0, i-r):i+r+1, max(0, j-r):j+r+1].mean((0, 1))
                                           for j in range(17)]) for i in range(23)])
        torch.testing.assert_close(box_filter(x, r), direct, atol=1e-5, rtol=1e-5)
        # Fine texture is flattened, a strong step edge survives.
        texture = .6+.02*torch.from_numpy(np.random.default_rng(3).standard_normal((64, 64, 1), dtype=np.float32))
        edge = torch.zeros(64, 64, 1)
        edge[:, 32:] = .5
        self.assertLess(float(smooth_skin(texture, 6).std()), float(texture.std())*.5)
        smoothed = smooth_skin(edge+.2, 6)
        self.assertGreater(float(smoothed[:, 40:].mean()-smoothed[:, :24].mean()), .45)

    def test_skin_effect_scales_with_resolution(self):
        changes = []
        for size in (512, 2048):
            image = self.portrait.resize((size, size), Image.Resampling.LANCZOS)
            mask = self.engine._skin_mask(image, f'scale-{size}')
            source = np.asarray(image).astype(float)
            result = np.asarray(self.engine.render(image, {'skin': 50}, f'scale-{size}')).astype(float)
            changes.append(float(np.abs(result-source)[mask > 0].mean()))
        self.assertGreater(changes[1], changes[0]*.6, changes)
        self.assertGreater(changes[1], 1.2, changes)

    def test_original_drives_mask_and_batch_preserves_skin_setting(self):
        with patch.object(self.engine, '_skin_mask', wraps=self.engine._skin_mask) as mask:
            self.engine.render(self.portrait, {'skin': 60, 'temperature': 100}, 'original-mask')
            self.assertIs(mask.call_args.args[0], self.portrait)
        self.assertEqual(self.engine.recommend(self.portrait, 'astronaut')['skin'], 30)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'인물.png'
            self.portrait.save(path)
            result = run_batch([BatchItem(str(path), path.name, settings({'skin': 60}))],
                               BatchOptions(str(Path(folder)/'output'), format='PNG'), self.engine)
            self.assertEqual(result['succeeded'], 1, result)
            self.assertEqual(result['files'][0]['adjustments']['skin'], 60)

    def test_missing_detector_does_not_silently_export_unretouched_photo(self):
        with patch.object(self.engine.face, 'cascade', None), patch.object(self.engine.face, 'detector', None):
            with self.assertRaisesRegex(ValueError, '피부 보정'):
                self.engine.render(self.portrait, {'skin': 50}, 'missing')


if __name__ == '__main__':
    unittest.main(verbosity=2)
