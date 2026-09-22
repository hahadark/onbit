import hashlib
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from batch import (BatchItem, BatchOptions, configure_auto_adjustments, resized_dimensions,
                   run_batch, write_image)
from engine import Engine, read_image, settings
from desktop_store import Library


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'한글 사진.png'
        pixels = np.zeros((80, 120, 3), dtype=np.uint8)
        pixels[..., 0] = np.arange(120, dtype=np.uint8)[None, :]*2
        pixels[..., 1] = np.arange(80, dtype=np.uint8)[:, None]*3
        pixels[..., 2] = 100
        Image.fromarray(pixels).save(self.source)
        self.item = BatchItem(str(self.source), self.source.name, settings({'exposure': .25}))

    def test_heic_roundtrip_rotation_and_all_output_formats(self):
        heic = self.root/'아이폰.HEIC'
        original = read_image(self.source)
        exif = Image.Exif()
        exif[274] = 6
        original.save(heic, format='HEIF', quality=100, exif=exif.tobytes())
        rotated = read_image(heic)
        self.assertEqual(rotated.size, (80, 120))
        for fmt in ['JPEG', 'PNG', 'WEBP', 'HEIC', 'TIFF', 'BMP']:
            result = run_batch([BatchItem(str(heic), heic.name)],
                               BatchOptions(str(self.root/fmt), format=fmt, resize='long_edge', width=60,
                                            correction='none'), self.engine)
            self.assertEqual(result['failed'], 0, result)
            image = read_image(result['files'][0]['output'])
            self.assertEqual(image.size, (40, 60))

    def test_resize_modes_and_no_upscale(self):
        for kwargs, expected in [({'resize':'long_edge','width':50},(50,33)),
                                  ({'resize':'fit','width':70,'height':20},(30,20)),
                                  ({'resize':'percent','percent':25},(30,20)),
                                  ({'resize':'long_edge','width':500},(120,80)),
                                  ({'resize':'percent','percent':200,'upscale':True},(240,160))]:
            self.assertEqual(resized_dimensions((120,80),BatchOptions(str(self.root),**kwargs)),expected)
        with self.assertRaises(ValueError):
            resized_dimensions((6000,4000),BatchOptions(str(self.root),resize='percent',percent=400,upscale=True))

    def test_heic_primary_image_is_used(self):
        path = self.root/'multiple.heic'
        first = Image.new('RGB',(64,40),'red')
        primary = Image.new('RGB',(80,48),'green')
        first.save(path,format='HEIF',save_all=True,append_images=[primary],primary_index=1,quality=100)
        decoded = read_image(path)
        self.assertEqual(decoded.size,(80,48))
        mean = np.asarray(decoded).mean((0,1))
        self.assertGreater(mean[1],mean[0]+50)

    def test_multi_picture_jpeg_reads_primary_not_auxiliary(self):
        path = self.root/'다중 이미지.JPG'
        primary = Image.new('RGB', (128, 96), (220, 30, 20))
        auxiliary = Image.new('RGB', (32, 24), 'blue')
        primary.save(path, format='MPO', save_all=True, append_images=[auxiliary], quality=100)
        with Image.open(path) as image:
            self.assertEqual(image.format, 'MPO')
        decoded = read_image(path)
        self.assertEqual(decoded.size, (128, 96))
        self.assertGreater(float(np.asarray(decoded)[..., 0].mean()), 210)

    def test_duplicates_are_unique_and_original_is_unchanged(self):
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        options = BatchOptions(str(self.root), format='PNG', correction='none')
        result = run_batch([self.item, self.item], options, self.engine)
        self.assertEqual(result['succeeded'], 2)
        first, second = [Path(item['output']) for item in result['files']]
        self.assertNotEqual(first, second)
        saved_hash = hashlib.sha256(first.read_bytes()).hexdigest()
        third = run_batch([self.item], options, self.engine)
        self.assertNotEqual(third['files'][0]['output'], str(first))
        self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(), saved_hash)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), before)

    def test_bad_file_does_not_stop_queue_and_report_exists(self):
        bad = self.root/'bad.heic'
        bad.write_bytes(b'broken')
        result = run_batch([BatchItem(str(bad),bad.name),self.item], BatchOptions(str(self.root/'out')),self.engine)
        self.assertEqual((result['succeeded'],result['failed']),(1,1))
        self.assertTrue(Path(result['report']).is_file())

    def test_cancel_after_current_file_and_cancel_before_start(self):
        cancel = threading.Event()
        def progress(event):
            if event['state'] == 'done':
                cancel.set()
        result = run_batch([self.item]*3,BatchOptions(str(self.root/'out')),self.engine,cancel,progress)
        self.assertTrue(result['cancelled'])
        self.assertEqual((result['succeeded'],result['unprocessed']),(1,2))
        empty = run_batch([self.item],BatchOptions(str(self.root/'out')),self.engine,cancel)
        self.assertEqual(empty['succeeded'],0)

    def test_failed_encoding_leaves_no_partial_output(self):
        with patch.object(Image.Image, 'save', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                write_image(read_image(self.source),self.root,self.source.name,'HEIC',90)
        self.assertEqual([p.name for p in self.root.iterdir()],[self.source.name])

    def test_pure_conversion_skips_engine_and_shared_correction_differs(self):
        class ForbiddenEngine:
            def render(self,*args,**kwargs):
                raise AssertionError('No AI / image adjustment should be invoked')
        pure = run_batch([self.item],BatchOptions(str(self.root/'out'),format='PNG',correction='none'),ForbiddenEngine())
        self.assertEqual(pure['failed'],0)
        np.testing.assert_array_equal(np.asarray(read_image(pure['files'][0]['output'])),np.asarray(read_image(self.source)))
        corrected = run_batch([self.item],BatchOptions(str(self.root/'out'),format='PNG',correction='shared',
                                                     shared_adjustments=settings({'exposure':.5})),self.engine)
        self.assertGreater(np.asarray(read_image(corrected['files'][0]['output'])).mean(),np.asarray(read_image(self.source)).mean())

    def test_desktop_library_roundtrip_unicode_and_duplicates(self):
        library = Library(self.root/'session')
        self.assertEqual(len(library.add([self.source,self.source])),1)
        library.items[0]['settings'] = settings({'ai':55,'softness':48})
        library.items[0]['checked'] = False
        library.save()
        recovered = Library(self.root/'session')
        self.assertEqual(recovered.items,library.items)

    def test_unwritable_output_reports_actionable_error(self):
        with patch('batch.tempfile.NamedTemporaryFile', side_effect=PermissionError('blocked')):
            with self.assertRaisesRegex(ValueError, '다른 폴더'):
                run_batch([self.item], BatchOptions(str(self.root/'blocked-output')), self.engine)

    def test_library_output_follows_writable_fallback(self):
        blocked = self.root/'blocked-app-home'
        blocked.write_text('not a directory')
        fallback = self.root/'appdata'
        with patch('desktop_store.APP_HOME', blocked), patch.dict(os.environ, {'LOCALAPPDATA': str(fallback)}):
            library = Library()
        self.assertEqual(library.directory, fallback/'Onbit')
        self.assertEqual(Path(library.output_dir), fallback/'Onbit'/'exports')

    def test_source_subfolder_default_quality_and_gpu_parallel(self):
        other = self.root/'다른 원본'
        other.mkdir()
        second = other/'두번째.png'
        Image.open(self.source).save(second)
        items = [self.item, BatchItem(str(second), second.name, settings({'exposure': .25}))]
        subfolder = BatchOptions('', format='JPEG', correction='none', source_subfolder='온빛 결과')
        self.assertEqual(subfolder.quality, 90)
        result = run_batch(items, subfolder, self.engine)
        self.assertEqual(result['succeeded'], 2, result)
        self.assertEqual(Path(result['files'][0]['output']).parent, self.root/'온빛 결과')
        self.assertEqual(Path(result['files'][1]['output']).parent, other/'온빛 결과')
        if self.engine.device.type == 'cuda':
            parallel = run_batch(items, BatchOptions(str(self.root/'parallel'), format='PNG', correction='shared',
                                                     shared_adjustments=settings({'exposure': .1}), parallel_gpu=True),
                                 self.engine)
            self.assertTrue(parallel['parallel_gpu'], parallel)
            self.assertEqual(parallel['succeeded'], 2, parallel)

    def test_ai_batch_detail_recipe(self):
        recommended = settings({'ai': 40, 'exposure': .4, 'temperature': 10, 'tint': -6,
                                'saturation': -8, 'contrast': 6, 'shadows': 12,
                                'highlights': -16, 'softness': 5, 'skin': 18})
        options = BatchOptions(str(self.root), correction='auto', auto_strength=50,
                               auto_color=False, auto_brightness=True,
                               auto_softness=20, auto_skin=30)
        result = configure_auto_adjustments(recommended, options)
        self.assertEqual(result['ai'], 0)
        self.assertEqual(result['temperature'], 0)
        self.assertEqual(result['exposure'], .2)
        self.assertEqual(result['highlights'], -8)
        self.assertEqual(result['softness'], 10)
        self.assertEqual(result['skin'], 15)
        no_face = configure_auto_adjustments(settings({**recommended, 'skin': 0}), options)
        self.assertEqual(no_face['skin'], 0)
        with self.assertRaises(ValueError):
            BatchOptions(str(self.root), auto_strength=101).validate()

    def test_invalid_source_subfolder_is_rejected(self):
        for name in ['..', 'bad/name', 'bad:name', 'trail.']:
            with self.assertRaises(ValueError):
                BatchOptions('', source_subfolder=name).validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
