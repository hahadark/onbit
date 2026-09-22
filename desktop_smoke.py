"""Exercise the packaged executable's actual codecs, model assets and CUDA runtime."""
from pathlib import Path
import json
import traceback
import numpy as np
from PIL import Image
from pillow_heif import libheif_info
from engine import Engine, read_exif, read_image
from batch import BatchItem, BatchOptions, run_batch, file_key
from tone import color_profile


def self_test(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {'ok': False}
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        engine = Engine()
        report['engine'] = engine.status()
        report['face_ai'] = {'available': engine.status().get('face_ai', False),
                             'error': engine.status().get('face_error', '')}
        report['qt_version'] = QtCore.qVersion()
        report['heif'] = libheif_info()
        if not engine.status()['ai_ready']:
            raise AssertionError(engine.error)
        if not engine.face.available:
            raise AssertionError(engine.face.error)
        portrait = read_image(Path(__file__).resolve().parent/'tests/fixtures/astronaut.png')
        faces = engine.face_count(portrait, 'smoke-portrait')
        if not faces:
            raise AssertionError('Packaged face detector did not detect the test portrait')
        mask = engine._skin_mask(portrait, 'smoke-portrait')
        corrected = engine.render(portrait, {'skin': 80}, 'smoke-portrait')
        difference = np.abs(np.asarray(corrected).astype(float)-np.asarray(portrait))
        if not difference[mask > 0].mean() > .1 or difference[mask == 0].max() != 0:
            raise AssertionError('Skin retouch must change skin and preserve masked-out pixels')
        corrected.save(root/'피부 보정 검증.png')
        report['face_ai'].update(faces=faces, skin_difference=float(difference[mask > 0].mean()),
                                 outside_difference=float(difference[mask == 0].max()))
        source = Path(__file__).resolve().parent / 'samples/demo.jpg'
        photo = read_image(source)
        photo.thumbnail((480, 320))
        heic = root / 'HEIC 테스트.heic'
        photo.save(heic, format='HEIF', quality=95)
        decoded = read_image(heic)
        if decoded.size != photo.size:
            raise AssertionError('HEIC round-trip dimensions differ')
        metadata = read_exif(heic)
        if metadata.get('resolution') != f'{photo.width:,} × {photo.height:,}':
            raise AssertionError('Packaged EXIF resolution is missing: ' + str(metadata))
        report['exif'] = metadata
        cube = root/'화사한 LUT.cube'
        cube.write_text('TITLE "화사한 검증"\nLUT_3D_SIZE 2\n0.1 0.1 0.1\n1 0.1 0.1\n'
                        '0.1 1 0.1\n1 1 0.1\n0.1 0.1 1\n1 0.1 1\n0.1 1 1\n1 1 1\n', encoding='utf-8')
        lut_output = engine.render(decoded, {}, 'smoke-cube', custom_lut_path=str(cube), custom_lut_strength=60)
        if not np.asarray(lut_output).mean() > np.asarray(decoded).mean():
            raise AssertionError('Packaged .cube LUT did not brighten the image')
        report['custom_lut'] = {'file': cube.name, 'strength': 60,
                                'difference': float(np.abs(np.asarray(lut_output).astype(float)-np.asarray(decoded)).mean())}
        report['heic_error_mean'] = float(np.abs(np.asarray(photo).astype(float)-np.asarray(decoded)).mean())
        report['batches'] = []
        for fmt in ['JPEG', 'PNG', 'WEBP', 'HEIC', 'TIFF', 'BMP']:
            result = run_batch([BatchItem(str(heic), heic.name)],
                               BatchOptions(str(root/'results'), format=fmt, resize='long_edge', width=240,
                                            correction='auto'), engine)
            if result['failed'] or result['succeeded'] != 1:
                raise AssertionError(result)
            output = read_image(result['files'][0]['output'])
            if output.size != (240, 160):
                raise AssertionError(output.size)
            report['batches'].append(result)
        parallel_items = [BatchItem(str(heic), heic.name, lut_path=str(cube), lut_strength=60),
                          BatchItem(str(heic), heic.name, lut_path=str(cube), lut_strength=60)]
        parallel = run_batch(parallel_items,
                             BatchOptions(str(root/'parallel'), format='PNG', correction='auto', parallel_gpu=True,
                                          auto_strength=50, auto_color=False,
                                          auto_softness=12, auto_skin=24), engine)
        # The CPU edition processes sequentially; only a CUDA build must run in parallel.
        cuda = engine.device.type == 'cuda'
        report['edition'] = 'gpu' if cuda else 'cpu'
        if parallel['succeeded'] != 2 or bool(parallel.get('parallel_gpu')) != cuda:
            raise AssertionError('Packaged parallel batch failed: ' + str(parallel))
        if any(entry['adjustments']['ai'] != 0 or entry['adjustments']['softness'] != 6
               for entry in parallel['files']):
            raise AssertionError('Packaged AI batch detail settings were not applied: ' + str(parallel))
        if any(entry.get('lut', {}).get('strength') != 60 for entry in parallel['files']):
            raise AssertionError('Packaged LUT batch settings were not applied: ' + str(parallel))
        source_copy = root/'source-folder'/'원본.png'
        source_copy.parent.mkdir()
        photo.save(source_copy)
        subfolder = run_batch([BatchItem(str(source_copy), source_copy.name)],
                              BatchOptions('', format='PNG', correction='none', source_subfolder='온빛_내보내기'), engine)
        if Path(subfolder['files'][0]['output']).parent != source_copy.parent/'온빛_내보내기':
            raise AssertionError('Source subfolder output was not created beside the original')
        report['parallel_gpu'] = parallel
        report['source_subfolder'] = subfolder
        report['ok'] = True
        reference_settings = engine.recommend(decoded, file_key(heic))
        reference = engine.render(decoded, reference_settings, file_key(heic))
        reference_path = root/'색감 기준.png'
        reference.save(reference_path)
        shifted = Image.fromarray(np.clip(np.asarray(reference).astype(float)*np.array([.79, .85, .91]), 0, 255).astype(np.uint8))
        target_path = root/'어두운 대상.png'
        shifted.save(target_path)
        matched = run_batch([BatchItem(str(target_path), target_path.name)],
                            BatchOptions(str(root/'matching'), format='PNG', correction='match',
                                         reference_path=str(reference_path), match_strength=100), engine)
        if matched['failed'] or matched['succeeded'] != 1:
            raise AssertionError(matched)
        center = np.array(color_profile(reference)['center'])
        before_distance = float(np.linalg.norm(np.array(color_profile(shifted)['center'])-center))
        after_distance = float(np.linalg.norm(np.array(color_profile(read_image(matched['files'][0]['output']))['center'])-center))
        if not after_distance < before_distance:
            raise AssertionError('Reference matching did not improve the color distance')
        report['natural_settings'] = reference_settings
        report['matching'] = {'before_distance': before_distance, 'after_distance': after_distance, 'batch': matched}
        neural = run_batch([BatchItem(str(target_path), target_path.name)] * 2,
                           BatchOptions(str(root/'neural'), format='PNG', correction='match',
                                        match_method='neural', reference_path=str(reference_path),
                                        match_strength=100, match_protect_skin=False, parallel_gpu=True), engine)
        if neural['succeeded'] != 2 or bool(neural.get('parallel_gpu')) != cuda:
            raise AssertionError('Neural matching batch failed: ' + str(neural))
        for entry in neural['files']:
            if entry['match'].get('method') != 'neural' or entry['match'].get('device') != engine.device.type:
                raise AssertionError('Packaged neural model did not run on the engine device: ' + str(entry))
        a = np.asarray(shifted).astype(float)
        b = np.asarray(read_image(neural['files'][0]['output'])).astype(float)
        if np.abs(a-b).mean() < .1:
            raise AssertionError('Neural matching did not change the photo')
        report['neural_matching'] = neural
    except Exception:
        report['ok'] = False
        report['error'] = traceback.format_exc()
    (root/'self-test.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return 0 if report['ok'] else 1
