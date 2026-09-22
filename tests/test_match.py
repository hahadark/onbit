"""Reference colour matching: tone curve + 3-zone colour, skin protection, stored layers."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from batch import BatchItem, BatchOptions, run_batch, resolve_match
from desktop import MainWindow
from desktop_store import Library
from engine import Engine, match_layer, read_image
from tone import color_profile, match_plan


def grade(image, gain, lift, shadows, highs):
    x = np.asarray(image).astype(np.float32)/255
    y = (x @ np.array([.2126, .7152, .0722], np.float32))[..., None]
    x = np.clip(x*gain+lift, 0, 1)+(1-y)**2*np.array(shadows, np.float32)+y**2*np.array(highs, np.float32)
    return Image.fromarray((np.clip(x, 0, 1)*255).round().astype(np.uint8))


def distance(image, reference):
    a, b = color_profile(image), color_profile(reference)
    curve = np.abs(np.array(a['quantiles'])-b['quantiles']).mean()
    zones = np.mean([np.linalg.norm(np.array(p[1:])-q[1:]) for p, q in zip(a['zones'], b['zones'])])
    return curve, zones


class MatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.scene = read_image(Path(__file__).resolve().parents[1]/'samples/demo.jpg').resize((480, 320))
        cls.portrait = read_image(Path(__file__).parent/'fixtures/astronaut.png')
        cls.app = QApplication.instance() or QApplication([])

    def layer(self, target, reference, **options):
        plan = match_plan(self.engine.profile(target), self.engine.profile(reference), 100)
        return {'plan': plan, 'strength': 100, 'brightness': True, 'protect_skin': True, **options}

    def test_curve_and_zone_colours_follow_the_reference(self):
        target = grade(self.scene, .72, 0, (0, 0, .05), (0, 0, .03))
        reference = grade(self.scene, 1.2, .03, (-.02, .03, .06), (.10, .03, -.06))
        result = self.engine.render(target, {}, 'curve', match=self.layer(target, reference))
        (curve_before, zones_before), (curve_after, zones_after) = distance(target, reference), distance(result, reference)
        self.assertLess(curve_after, curve_before*.15)
        self.assertLess(zones_after, zones_before*.25)
        half = self.engine.render(target, {}, 'curve', match=self.layer(target, reference, strength=50))
        self.assertLess(distance(result, reference)[0], distance(half, reference)[0])
        self.assertLess(distance(half, reference)[0], curve_before)

    def test_zero_strength_and_brightness_opt_out(self):
        target = grade(self.scene, .7, 0, (0, 0, 0), (0, 0, 0))
        reference = grade(self.scene, 1.25, .02, (0, 0, 0), (.08, 0, -.05))
        self.assertIsNone(match_layer(self.layer(target, reference, strength=0)))
        np.testing.assert_array_equal(
            np.asarray(self.engine.render(target, {}, 'zero', match=self.layer(target, reference, strength=0))),
            np.asarray(target))
        colour_only = self.engine.render(target, {}, 'colour', match=self.layer(target, reference, brightness=False))
        before = color_profile(target)['quantiles']
        self.assertLess(np.abs(np.array(color_profile(colour_only)['quantiles'])-before).mean(), .01)
        with self.assertRaisesRegex(ValueError, '매칭 정보'):
            match_layer({'plan': {'curve_x': [0, 1]}, 'strength': 50})

    def test_skin_keeps_its_colour_while_the_scene_follows(self):
        reference = grade(self.portrait, 1.0, 0, (0, .02, .08), (0, .03, .1))  # strongly cool look
        mask = self.engine._skin_mask(self.portrait, 'skin-match') > 200
        background = self.engine._skin_mask(self.portrait, 'skin-match') == 0
        source = np.asarray(self.portrait).astype(float)
        protected = np.asarray(self.engine.render(self.portrait, {}, 'skin-match',
                                                  match=self.layer(self.portrait, reference))).astype(float)
        open_ = np.asarray(self.engine.render(self.portrait, {}, 'skin-match',
                                              match=self.layer(self.portrait, reference, protect_skin=False))).astype(float)
        self.assertLess(np.abs(protected-source)[mask].mean(), 1.5)
        self.assertGreater(np.abs(open_-source)[mask].mean(), 4)
        self.assertGreater(np.abs(protected-source)[background].mean(), 4)

    def test_fast_reduced_decode_for_match_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'큰 사진.jpg'
            self.scene.resize((4000, 2667)).save(path, quality=90)
            small = read_image(path, max_side=1024)
            self.assertLessEqual(max(small.size), 2048)
            self.assertGreaterEqual(max(small.size), 1024)
            self.assertEqual(read_image(path).size, (4000, 2667))

    def test_export_measures_pending_matches_and_skips_missing_references(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference, target = root/'기준.png', root/'대상.png'
            grade(self.scene, 1.2, .02, (0, 0, 0), (.08, .02, -.05)).save(reference)
            grade(self.scene, .7, 0, (0, 0, .04), (0, 0, 0)).save(target)
            look = {'settings': dict.fromkeys(('ai', 'exposure'), 0.), 'lut_path': '', 'lut_strength': 0, 'match': None}
            pending = {'reference': str(reference), 'plan': None, 'strength': 100, 'brightness': True,
                       'protect_skin': True, 'reference_look': look}
            resolved = resolve_match(self.engine, str(target), {}, '', 0, pending)
            self.assertIsNotNone(resolved['plan'])
            self.assertNotIn('reference_look', resolved)
            self.assertIsNone(resolve_match(self.engine, str(target), {}, '', 0, {**pending, 'reference_look': None}))
            self.assertIsNone(match_layer(dict(pending, plan=None)))
            options = BatchOptions(str(root/'out'), format='PNG')
            plain = run_batch([BatchItem(str(target), target.name)], options, self.engine)
            matched = run_batch([BatchItem(str(target), target.name, match=pending)], options, self.engine)
            self.assertEqual((plain['succeeded'], matched['succeeded']), (1, 1), matched)
            a = np.asarray(read_image(plain['files'][0]['output'])).astype(float)
            b = np.asarray(read_image(matched['files'][0]['output'])).astype(float)
            self.assertGreater(b.mean(), a.mean()+10)

    def test_batch_saved_mode_applies_the_stored_layer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = grade(self.scene, .7, 0, (0, 0, .04), (0, 0, 0))
            path = root/'대상.png'
            target.save(path)
            layer = self.layer(target, grade(self.scene, 1.2, .02, (0, 0, 0), (.08, .02, -.05)))
            options = BatchOptions(str(root/'out'), format='PNG')
            plain = run_batch([BatchItem(str(path), path.name)], options, self.engine)
            matched = run_batch([BatchItem(str(path), path.name, match=layer)], options, self.engine)
            self.assertEqual((plain['succeeded'], matched['succeeded']), (1, 1))
            self.assertIn('match', matched['files'][0])
            a = np.asarray(read_image(plain['files'][0]['output'])).astype(float)
            b = np.asarray(read_image(matched['files'][0]['output'])).astype(float)
            self.assertGreater(b.mean(), a.mean()+10)

    def wait_for(self, predicate, seconds=40):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Desktop worker did not finish within timeout')

    def test_editor_reference_match_flow_is_saved_and_reversible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference, target = root/'기준.png', root/'대상.png'
            grade(self.scene, 1.2, .03, (0, .02, .05), (.1, .03, -.05)).save(reference)
            grade(self.scene, .7, 0, (0, 0, .05), (0, 0, 0)).save(target)
            window = MainWindow(Library(root/'session'))
            try:
                window.add_paths([str(reference), str(target)])
                self.wait_for(lambda: window.engine is not None and window.displayed_path == str(reference))
                self.assertFalse(window.match_current_button.isEnabled())
                window.set_match_reference()
                self.assertTrue(window.list.item(0).text().startswith('★ '))
                self.assertIn('기준.png', window.match_reference_label.text())
                window.list.setCurrentRow(1)
                self.wait_for(lambda: window.displayed_path == str(target) and window.preview_worker is None)
                self.assertTrue(window.match_current_button.isEnabled())
                before = window.viewer.after.copy()
                window.match_photos(False)
                self.assertIsNone(window.current()['match']['plan'])
                self.wait_for(lambda: window.current()['match'].get('plan') is not None)
                self.wait_for(lambda: window.preview_worker is None and not window.preview_pending)
                self.assertNotEqual(window.viewer.after, before)
                record = window.current()
                self.assertEqual(record['match']['strength'], 80)
                self.assertTrue(record['match']['protect_skin'])
                self.assertTrue(window.match_skin_check.isChecked())
                window.match_strength_spin.setValue(45)
                self.assertEqual(record['match']['strength'], 45)
                window.match_brightness_check.setChecked(False)
                self.assertFalse(record['match']['brightness'])
                window.save_library()
                reloaded = Library(root/'session')
                self.assertEqual(reloaded.reference_path, str(reference.resolve()))
                self.assertEqual(reloaded.items[1]['match']['strength'], 45)
                self.assertIsNotNone(match_layer(reloaded.items[1]['match']))
                self.assertIsNone(reloaded.items[0]['match'])
                window.clear_match()
                self.assertIsNone(record['match'])
                self.assertFalse(window.match_strength_slider.isEnabled())
            finally:
                window.close()
                self.wait_for(lambda: all(w is None or not w.isRunning() for w in
                                          [window.loader, window.preview_worker, window.thumb_worker,
                                           window.auto_worker, window.detail_worker]))


    def test_strength_is_adjustable_from_the_reference_and_for_the_whole_group(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root/'기준.png', root/'a.png', root/'b.png']
            grade(self.scene, 1.2, .03, (0, .02, .05), (.1, .03, -.05)).save(paths[0])
            grade(self.scene, .7, 0, (0, 0, .05), (0, 0, 0)).save(paths[1])
            grade(self.scene, .8, 0, (0, .02, 0), (0, 0, 0)).save(paths[2])
            window = MainWindow(Library(root/'session'))
            try:
                window.add_paths([str(p) for p in paths])
                self.wait_for(lambda: window.engine is not None and window.displayed_path == str(paths[0]))
                # Mark the reference and match the checked photos without leaving it.
                window.set_match_reference()
                started = time.monotonic()
                window.match_photos(True)
                self.assertLess(time.monotonic()-started, .5)
                a, b = window.library.items[1], window.library.items[2]
                self.assertTrue(a['match'] and b['match'])
                self.assertIsNone(a['match']['plan'])
                self.assertIsNone(b['match']['plan'])
                self.assertTrue(window.match_strength_slider.isEnabled())
                self.assertIn('2장', window.match_state_label.text())
                self.assertFalse(window.match_clear_button.isEnabled())
                window.match_strength_slider.setValue(35)
                self.assertEqual((a['match']['strength'], b['match']['strength']), (35, 35))
                window.list.setCurrentRow(1)
                self.wait_for(lambda: window.displayed_path == str(paths[1]))
                # Viewing a photo measures only that photo.
                self.wait_for(lambda: a['match'].get('plan') is not None)
                self.assertIsNone(b['match']['plan'])
                window.match_group_check.setChecked(False)
                window.match_strength_spin.setValue(90)
                self.assertEqual((a['match']['strength'], b['match']['strength']), (90, 35))
                window.match_group_check.setChecked(True)
                window.match_skin_check.setChecked(False)
                self.assertFalse(a['match']['protect_skin'] or b['match']['protect_skin'])
            finally:
                window.close()
                self.wait_for(lambda: all(w is None or not w.isRunning() for w in
                                          [window.loader, window.preview_worker, window.thumb_worker,
                                           window.auto_worker, window.detail_worker]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
