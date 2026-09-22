"""Zoomed detail views: region renders line up with the full render, and the viewer gets sharper."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from desktop import MainWindow
from desktop_store import Library
from engine import Engine, read_image, settings
from tone import match_plan

XMP = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
       b'<rdf:Description xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0" '
       b'hdrgm:GainMapMin="0" hdrgm:GainMapMax="2.5" hdrgm:Gamma="1"/></rdf:RDF></x:xmpmeta>')


class ZoomDetailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()
        cls.portrait = read_image(Path(__file__).parent/'fixtures/astronaut.png')
        cls.app = QApplication.instance() or QApplication([])

    def assert_region_matches(self, image, p, key, box, match=None, margin=24):
        full = np.asarray(self.engine.render(image, p, key, lut_image=image, match=match)).astype(int)
        crop = image.crop(box)
        part = np.asarray(self.engine.render(crop, p, key, lut_image=image, match=match, region=box)).astype(int)
        x0, y0, x1, y1 = box
        expected = full[y0:y1, x0:x1]
        # Filters near the crop edge see less context; compare the interior.
        inner = (slice(margin, -margin), slice(margin, -margin))
        diff = np.abs(part[inner]-expected[inner])
        # Dither noise sits at different positions in a crop, so allow about one code value on average.
        self.assertLessEqual(float(diff.mean()), .8, diff.mean())
        self.assertLessEqual(int(np.percentile(diff, 99.5)), 3)

    def test_region_render_lines_up_with_full_render_for_skin_match_and_gradation(self):
        p = settings({'ai': 30, 'exposure': .2, 'skin': 60, 'gradation': 50})
        reference = self.portrait.point(lambda v: min(255, int(v*1.15)))
        match = {'plan': match_plan(self.engine.profile(self.portrait), self.engine.profile(reference), 100),
                 'strength': 80, 'brightness': True, 'protect_skin': True}
        self.assert_region_matches(self.portrait, p, 'zoom-face', (150, 40, 330, 220), match)

    def test_region_render_samples_the_gain_map_at_the_right_place(self):
        with tempfile.TemporaryDirectory() as folder:
            primary = np.full((240, 320, 3), 120, np.uint8)
            primary[:, 160:] = 255
            gain = np.zeros((60, 80), np.uint8)
            gain[:, 40:] = np.linspace(0, 255, 40).astype(np.uint8)
            path = Path(folder)/'ultra.jpg'
            Image.fromarray(primary).save(path, 'MPO', save_all=True, xmp=XMP, quality=95,
                                          append_images=[Image.fromarray(gain).convert('RGB')])
            image = read_image(path)
            self.assertIn('onbit_gain', image.info)
            self.assert_region_matches(image, settings({'gradation': 60}), 'zoom-gain', (120, 40, 300, 200), margin=12)

    def wait_for(self, predicate, seconds=40):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Desktop worker did not finish within timeout')

    def test_zooming_in_renders_the_visible_area_at_full_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'큰 사진.png'
            self.portrait.resize((3072, 3072), Image.Resampling.LANCZOS).save(path)
            window = MainWindow(Library(Path(folder)/'session'))
            window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
            window.resize(1380, 900)
            window.show()
            try:
                window.add_paths([str(path)])
                self.wait_for(lambda: window.engine is not None and window.displayed_path == str(path)
                              and window.preview_worker is None)
                self.assertIsNone(window.viewer.detail_region)
                window.viewer.change_zoom(6)
                self.wait_for(lambda: window.viewer.detail_region is not None)
                viewer = window.viewer
                x0, y0, x1, y1 = viewer.detail_region
                preview_pixels = viewer.after.width()*(x1-x0)
                self.assertGreater(viewer.detail_after.width(), preview_pixels*1.5)
                self.assertEqual(viewer.detail_after.size(), viewer.detail_before.size())
                # A settings change makes the old detail stale until the new one arrives.
                window.controls['exposure'][1].setValue(.4)
                self.wait_for(lambda: window.preview_worker is None and not window.preview_pending)
                self.wait_for(lambda: viewer.detail_region is not None and window.detail_worker is None)
                window.viewer.fit()
            finally:
                window.close()
                self.wait_for(lambda: all(w is None or not w.isRunning() for w in
                                          [window.loader, window.preview_worker, window.thumb_worker,
                                           window.auto_worker, window.detail_worker]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
