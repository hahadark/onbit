from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from batch import BatchItem, BatchOptions, render_item
from cube_lut import load_cube
from desktop_store import Library
from engine import Engine, read_image, settings


def write_cube(path, lift=.1):
    rows = ['TITLE "화사한 테스트"', 'LUT_3D_SIZE 2', 'DOMAIN_MIN 0 0 0', 'DOMAIN_MAX 1 1 1']
    for blue in (0., 1.):
        for green in (0., 1.):
            for red in (0., 1.):
                rows.append(f'{min(1, red+lift)} {min(1, green+lift)} {min(1, blue+lift)}')
    Path(path).write_text('\n'.join(rows), encoding='utf-8')


class LutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def test_cube_parser_and_gpu_render_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'화사하게.cube'
            write_cube(path)
            cube = load_cube(path)
            self.assertEqual((cube.title, cube.size, cube.values.shape), ('화사한 테스트', 2, (3, 2, 2, 2)))
            image = Image.fromarray(np.full((24, 32, 3), 64, dtype=np.uint8))
            full = np.asarray(self.engine.render(image, settings(), 'lut-full',
                                                 custom_lut_path=str(path), custom_lut_strength=100))
            half = np.asarray(self.engine.render(image, settings(), 'lut-half',
                                                 custom_lut_path=str(path), custom_lut_strength=50))
            self.assertGreater(full.mean(), half.mean())
            self.assertGreater(half.mean(), 64)
            with self.assertRaisesRegex(ValueError, '0~100'):
                self.engine.render(image, settings(), 'bad-strength', custom_lut_path=str(path), custom_lut_strength=101)

    def test_invalid_cube_and_library_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root/'bad.cube'
            invalid.write_text('LUT_3D_SIZE 2\n0 0 0', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '8개'):
                load_cube(invalid)
            photo = root/'사진.png'
            Image.new('RGB', (12, 8), 'gray').save(photo)
            lut = root/'화사.cube'
            write_cube(lut)
            library = Library(root/'session')
            record = library.add([photo])[0]
            record['lut_path'], record['lut_strength'] = str(lut), 65
            library.save()
            recovered = Library(root/'session').items[0]
            self.assertEqual((recovered['lut_path'], recovered['lut_strength']), (str(lut), 65))

    def test_batch_applies_saved_lut_and_none_skips_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo, lut = root/'사진.png', root/'화사.cube'
            Image.new('RGB', (32, 20), (64, 64, 64)).save(photo)
            write_cube(lut)
            item = BatchItem(str(photo), photo.name, settings(), str(lut), 100)
            _, saved, details = render_item(item, BatchOptions(str(root/'out'), correction='saved'), self.engine)
            _, plain, plain_details = render_item(item, BatchOptions(str(root/'out'), correction='none'), self.engine)
            self.assertGreater(np.asarray(saved).mean(), np.asarray(plain).mean())
            self.assertEqual(details['lut']['strength'], 100)
            self.assertNotIn('lut', plain_details)


if __name__ == '__main__':
    unittest.main(verbosity=2)
