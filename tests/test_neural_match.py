"""Real pretrained inference, integration, persistence and transform invariants."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from batch import BatchItem, BatchOptions, measure_look, resolve_match, run_batch
from desktop_store import Library
from engine import Engine, match_layer, read_image
from neural_match import NeuralMatcher, MODEL_ID
from tone import apply_match, rgb_to_lab


class NeuralMatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.reference = read_image(Path(__file__).resolve().parents[1] / 'samples/demo.jpg').resize((480, 320))
        cls.source = Image.fromarray((np.asarray(cls.reference) * [.72, .78, .87]).astype(np.uint8))

    def test_real_model_improves_controlled_cast_and_depends_on_both_images(self):
        plan = self.engine.neural_match(self.source, self.reference)
        self.assertEqual(plan['model'], MODEL_ID)
        result = np.clip(np.asarray(self.source) @ np.array(plan['matrix']).T, 0, 255)
        before = np.abs(np.asarray(self.source).astype(float) - np.asarray(self.reference)).mean()
        after = np.abs(result - np.asarray(self.reference)).mean()
        self.assertLess(after, before * .65)
        other = Image.fromarray((np.asarray(self.reference) * [.95, .80, .70]).astype(np.uint8))
        other_ref = self.engine.neural_match(self.source, other)
        other_source = self.engine.neural_match(other, self.reference)
        self.assertGreater(np.linalg.norm(np.array(plan['matrix']) - other_ref['matrix']), .02)
        self.assertGreater(np.linalg.norm(np.array(plan['matrix']) - other_source['matrix']), .02)
        identity = self.engine.neural_match(self.reference, self.reference)
        np.testing.assert_array_equal(identity['matrix'], np.eye(3))

    def test_cpu_gpu_agree(self):
        cpu = NeuralMatcher().plan(self.source, self.reference, 'cpu')
        gpu = self.engine.neural_match(self.source, self.reference)
        np.testing.assert_allclose(cpu['matrix'], gpu['matrix'], atol=2e-4, rtol=2e-4)

    def test_strength_brightness_skin_and_tiling(self):
        rgb = torch.tensor([[[.30, .23, .19], [.19, .30, .23]], [[.20, .25, .30], [.4, .4, .4]]])
        plan = {'method': 'neural', 'matrix': [[1.2, 0, 0], [0, 1.1, 0], [0, 0, .9]],
                'strength': 100, 'brightness': True}
        torch.testing.assert_close(apply_match(rgb, dict(plan, strength=0)), rgb)
        full = apply_match(rgb, plan)
        torch.testing.assert_close(apply_match(rgb, dict(plan, strength=50)), (rgb + full) / 2)
        torch.testing.assert_close(torch.cat([apply_match(row[None], plan) for row in rgb]), full)
        colour_only = apply_match(rgb, dict(plan, brightness=False))
        torch.testing.assert_close(rgb_to_lab(colour_only)[..., 0], rgb_to_lab(rgb)[..., 0], atol=1e-6, rtol=1e-5)
        protected = rgb_to_lab(apply_match(rgb, plan, torch.ones(2, 2, 1)))
        torch.testing.assert_close(protected[..., 1:], rgb_to_lab(rgb)[..., 1:], atol=1e-6, rtol=1e-5)
        self.assertGreater(float((protected[..., 0]-rgb_to_lab(rgb)[..., 0]).abs().mean()), .01)
        for matrix in ([[1, 2, 3]], [[float('nan')]*3]*3, [[99]*3]*3):
            with self.assertRaises(ValueError):
                match_layer({'plan': dict(plan, matrix=matrix)})

    def test_missing_model_fails_visibly_without_statistical_fallback(self):
        with tempfile.TemporaryDirectory() as folder, patch('neural_match.MODEL_PATH', Path(folder)/'missing.pth'):
            with self.assertRaisesRegex(ValueError, '모델'):
                NeuralMatcher().plan(self.source, self.reference, 'cpu')
        with self.assertRaisesRegex(ValueError, '기준 사진'):
            resolve_match(self.engine, '', {}, '', 0, {'method': 'neural', 'reference_look': None})

    def test_viewer_batch_reference_edits_lut_and_saved_layer_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src, ref = root/'target.png', root/'reference.png'
            self.source.save(src)
            self.reference.save(ref)
            cube = root/'warm.cube'
            cube.write_text('LUT_3D_SIZE 2\n0.04 0 0\n1 0 0\n0.04 1 0\n1 1 0\n'
                            '0.04 0 1\n1 0 1\n0.04 1 1\n1 1 1\n', encoding='utf-8')
            look = {'settings': {'exposure': .1}, 'lut_path': str(cube), 'lut_strength': 65}
            pending = {'method': 'neural', 'reference': str(ref), 'reference_look': look,
                       'strength': 80, 'brightness': True, 'protect_skin': False}
            resolved = resolve_match(self.engine, str(src), {}, '', 0, pending)
            changed = resolve_match(self.engine, str(src), {}, '', 0,
                                     dict(pending, reference_look={'settings': {}}))
            self.assertGreater(np.linalg.norm(np.array(resolved['plan']['matrix'])-changed['plan']['matrix']), .01)
            preview = self.engine.render(self.source, {}, 'integration', match=resolved)
            options = BatchOptions(str(root/'out'), format='PNG', correction='match', match_method='neural',
                                   reference_path=str(ref), reference_look=look, match_strength=80,
                                   match_protect_skin=False, parallel_gpu=True)
            batch = run_batch([BatchItem(str(src), src.name), BatchItem(str(ref), ref.name)], options, self.engine)
            self.assertEqual(batch['succeeded'], 2, batch)
            exported = read_image(batch['files'][0]['output'])
            np.testing.assert_array_equal(np.asarray(preview), np.asarray(exported))
            expected_ref = measure_look(self.engine, str(ref), **{'adjustments': look['settings'],
                                        'lut_path': str(cube), 'lut_strength': 65}, method='neural')
            np.testing.assert_array_equal(np.asarray(read_image(batch['files'][1]['output'])), np.asarray(expected_ref))
            library = Library(root/'library')
            record = library.add([src])[0]
            record['match'] = resolved
            library.save()
            stored = Library(root/'library').items[0]['match']
            self.assertEqual(stored, resolved)
            saved = run_batch([BatchItem(str(src), src.name, match=stored)],
                              BatchOptions(str(root/'saved'), format='PNG'), self.engine)
            np.testing.assert_array_equal(np.asarray(preview), np.asarray(read_image(saved['files'][0]['output'])))


if __name__ == '__main__':
    unittest.main()
