"""Desktop batch processing. Originals are never overwritten."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid
from queue import SimpleQueue

from PIL import Image, ImageCms
from engine import Engine, read_image, settings
from tone import match_plan

_REFERENCE_LOCK = threading.Lock()
_REFERENCE_PROFILES = {}

FORMATS = {'JPEG': ('jpg', 'JPEG'), 'PNG': ('png', 'PNG'), 'WEBP': ('webp', 'WEBP'),
           'HEIC': ('heic', 'HEIF'), 'TIFF': ('tif', 'TIFF'), 'BMP': ('bmp', 'BMP')}
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()


@dataclass(frozen=True)
class BatchItem:
    path: str
    name: str
    adjustments: dict = field(default_factory=settings)
    lut_path: str = ''
    lut_strength: int = 0
    match: dict = None


@dataclass(frozen=True)
class BatchOptions:
    output_dir: str
    format: str = 'JPEG'
    resize: str = 'original'
    width: int = 2048
    height: int = 2048
    percent: int = 50
    upscale: bool = False
    quality: int = 90
    correction: str = 'saved'
    shared_adjustments: dict = field(default_factory=settings)
    shared_lut_path: str = ''
    shared_lut_strength: int = 0
    reference_path: str = ''
    reference_adjustments: dict = field(default_factory=settings)
    match_strength: int = 70
    match_brightness: bool = True
    match_method: str = 'statistics'
    match_protect_skin: bool = True
    reference_look: dict = None
    auto_strength: int = 100
    auto_color: bool = True
    auto_brightness: bool = True
    auto_softness: int = 5
    auto_skin: int = 30
    source_subfolder: str = ''
    parallel_gpu: bool = False

    def validate(self):
        folder = self.source_subfolder.strip()
        if not self.output_dir.strip() and not folder:
            raise ValueError('저장 폴더를 선택해 주세요.')
        if folder and (Path(folder).name != folder or folder in {'.', '..'} or
                       re.search(r'[<>:"/\\|?*\x00-\x1f]', folder) or folder.rstrip(' .') != folder):
            raise ValueError('하위 폴더 이름에는 경로 문자나 Windows에서 사용할 수 없는 문자를 넣을 수 없습니다.')
        if self.format not in FORMATS:
            raise ValueError('지원하지 않는 출력 형식입니다.')
        if self.resize not in {'original', 'long_edge', 'fit', 'percent'}:
            raise ValueError('지원하지 않는 크기 조절 방식입니다.')
        if self.correction not in {'saved', 'shared', 'auto', 'none', 'match'}:
            raise ValueError('보정 방식을 확인해 주세요.')
        for name, value, low, high in [('width', self.width, 1, 30000), ('height', self.height, 1, 30000),
                                        ('percent', self.percent, 1, 400), ('quality', self.quality, 1, 100),
                                        ('match_strength', self.match_strength, 0, 100),
                                        ('auto_strength', self.auto_strength, 0, 100),
                                        ('auto_softness', self.auto_softness, 0, 100),
                                        ('auto_skin', self.auto_skin, 0, 100),
                                        ('shared_lut_strength', self.shared_lut_strength, 0, 100)]:
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{name}: {low}~{high} 범위의 정수를 입력해 주세요.')
        settings(self.shared_adjustments)
        settings(self.reference_adjustments)
        if self.match_method not in {'statistics', 'neural'}:
            raise ValueError('색감 매칭 방식을 확인해 주세요.')
        if self.shared_lut_strength and not self.shared_lut_path:
            raise ValueError('공통으로 적용할 LUT 파일을 선택해 주세요.')
        if self.correction == 'match' and not self.reference_path:
            raise ValueError('라이브러리에서 기준으로 삼을 사진을 먼저 선택해 주세요.')


def output_directory(item, options):
    if options.source_subfolder.strip():
        return Path(item.path).resolve().parent / options.source_subfolder.strip()
    return Path(options.output_dir).expanduser().resolve()


def prepare_output_dir(path):
    path = Path(path).resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix='.onbit-write-'):
            pass
    except OSError as exc:
        raise ValueError('저장 폴더에 쓸 수 없습니다. 쓰기가 허용된 다른 폴더를 선택해 주세요.\n' + str(path)) from exc
    return path


def resized_dimensions(size, options):
    w, h = size
    if options.resize == 'original':
        return w, h
    if options.resize == 'long_edge':
        scale = options.width / max(w, h)
    elif options.resize == 'fit':
        scale = min(options.width / w, options.height / h)
    else:
        scale = options.percent / 100
    if not options.upscale:
        scale = min(1, scale)
    result = max(1, int(math.floor(w * scale + .5))), max(1, int(math.floor(h * scale + .5)))
    if result[0] * result[1] > 60_000_000:
        raise ValueError('변환 후 크기는 6,000만 화소 이하여야 합니다.')
    return result


def file_key(path):
    path = Path(path).resolve()
    stat = path.stat()
    return f'{path}|{stat.st_size}|{stat.st_mtime_ns}'


def load_unchanged(path):
    key = file_key(path)
    image = read_image(path)
    if file_key(path) != key:
        raise ValueError('읽는 중 원본 파일이 변경되었습니다. 다시 시도해 주세요.')
    return image, key


def measure_look(engine, path, adjustments, lut_path='', lut_strength=0, match=None, original=None,
                 method='statistics'):
    """Colour statistics of a photo as currently edited, for colour matching.

    Statistics need no full resolution; skin retouching is skipped because skin
    is left out of the statistics anyway.
    """
    key = file_key(path)
    if original is None:
        original = read_image(path, max_side=2048)
    small = original.copy()
    small.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    usable = match if match and match.get('plan') else None
    rendered = engine.render(small, {**settings(adjustments), 'skin': 0}, key, lut_image=original,
                             custom_lut_path=lut_path, custom_lut_strength=lut_strength, match=usable)
    return rendered if method == 'neural' else engine.profile(rendered, f'{key}|match')


def reference_profile(engine, path, look, method='statistics'):
    """Measure a reference photo once per look (file + its edits) and reuse it."""
    signature = json.dumps([str(Path(path).resolve()), file_key(path), look, method], sort_keys=True, default=str)
    with _REFERENCE_LOCK:
        cached = _REFERENCE_PROFILES.get(signature)
    if cached is None:
        cached = measure_look(engine, path, look['settings'], look.get('lut_path', ''),
                              look.get('lut_strength', 0), look.get('match'), method=method)
        with _REFERENCE_LOCK:
            if len(_REFERENCE_PROFILES) >= 16:
                _REFERENCE_PROFILES.pop(next(iter(_REFERENCE_PROFILES)))
            _REFERENCE_PROFILES[signature] = cached
    return cached


def resolve_match(engine, path, adjustments, lut_path, lut_strength, match, original=None):
    """Complete a pending match layer by measuring the photo against its reference.

    ``match['reference_look']`` carries the reference photo's current edits.
    Returns the layer with a plan, or None when the reference is unavailable.
    """
    if not match or match.get('plan'):
        return match
    look = match.get('reference_look')
    if not look:
        if match.get('method') == 'neural':
            raise ValueError('AI 매칭 기준 사진을 찾을 수 없습니다. 기준 사진을 다시 지정해 주세요.')
        return None
    method = match.get('method', 'statistics')
    reference = reference_profile(engine, match['reference'], look, method)
    target = measure_look(engine, path, adjustments, lut_path, lut_strength, None, original, method)
    layer = {name: value for name, value in match.items() if name != 'reference_look'}
    plan = engine.neural_match(target, reference) if method == 'neural' else match_plan(target, reference, 100)
    return {**layer, 'plan': plan}


def prepare_reference(options, engine):
    if options.correction != 'match':
        return None
    key = file_key(options.reference_path)
    look = options.reference_look or {'settings': settings(options.reference_adjustments)}
    profile = reference_profile(engine, options.reference_path, look, options.match_method)
    return {'key': key, 'profile': profile, 'settings': look['settings'], 'look': look}


def configure_auto_adjustments(recommended, options):
    """Turn the per-photo AI recommendation into the user's batch recipe."""
    recommended = settings(recommended)
    selected = settings()
    if options.auto_color:
        for key in ('ai', 'temperature', 'tint', 'saturation'):
            selected[key] = recommended[key]
    if options.auto_brightness:
        for key in ('exposure', 'contrast', 'shadows', 'highlights', 'gradation'):
            selected[key] = recommended[key]
    selected['softness'] = options.auto_softness
    # A non-zero recommendation means that a face was detected in this photo.
    selected['skin'] = options.auto_skin if recommended['skin'] > 0 else 0
    amount = options.auto_strength / 100
    return settings({key: value * amount for key, value in selected.items()})


def render_item(item, options, engine, reference=None):
    original, key = load_unchanged(item.path)
    target_size = resized_dimensions(original.size, options)
    is_reference = reference is not None and key == reference['key']
    auto_recommendation = None
    if options.correction == 'none':
        p = settings()
    elif options.correction == 'auto':
        auto_recommendation = engine.recommend(original, key)
        p = configure_auto_adjustments(auto_recommendation, options)
    elif options.correction == 'shared':
        p = settings(options.shared_adjustments)
    elif is_reference:
        p = reference['settings']
    else:
        p = settings(item.adjustments)
    # A photo's stored colour match is part of its own look ("saved" corrections only).
    match = item.match if options.correction == 'saved' else None
    if options.correction == 'none':
        lut_path, lut_strength = '', 0
    elif options.correction == 'shared':
        lut_path, lut_strength = options.shared_lut_path, options.shared_lut_strength
    else:
        lut_path, lut_strength = item.lut_path, item.lut_strength
    if is_reference:
        look = reference['look']
        lut_path, lut_strength = look.get('lut_path', ''), look.get('lut_strength', 0)
        match = look.get('match')
    if match and not match.get('plan'):
        match = resolve_match(engine, item.path, p, lut_path, lut_strength, match, original)
    output = engine.render(original, p, key, custom_lut_path=lut_path, custom_lut_strength=lut_strength,
                           match=match) if any(p.values()) or lut_strength or match else original
    details = {'adjustments': p}
    if match:
        details['match'] = {'reference': match.get('reference', ''), 'strength': match.get('strength', 0),
                            'method': match.get('method', 'statistics'), 'plan': match.get('plan')}
    if lut_strength:
        details['lut'] = {'path': lut_path, 'strength': lut_strength}
    if auto_recommendation is not None:
        details['ai_recommendation'] = auto_recommendation
    if options.correction == 'match':
        if reference is None:
            raise ValueError('기준 사진을 읽지 못했습니다.')
        if file_key(options.reference_path) != reference['key']:
            raise ValueError('처리 중 기준 사진이 변경되었습니다. 다시 시작해 주세요.')
        if not is_reference and options.match_strength:
            if options.match_method == 'neural':
                target = measure_look(engine, item.path, p, lut_path, lut_strength, original=original,
                                      method='neural')
                plan = engine.neural_match(target, reference['profile'])
            else:
                plan = match_plan(engine.profile(output, key, original), reference['profile'], 100)
            layer = {'plan': plan, 'strength': options.match_strength,
                     'brightness': options.match_brightness, 'protect_skin': options.match_protect_skin}
            output = engine.render(original, p, key, custom_lut_path=lut_path, custom_lut_strength=lut_strength,
                                   match=layer)
            details['match'] = {**plan, 'strength': float(options.match_strength),
                                'brightness': options.match_brightness}
        details['reference_unchanged'] = is_reference
    if target_size != output.size:
        output = output.resize(target_size, Image.Resampling.LANCZOS)
    return original, output, details


def write_image(image, output_dir, source_name, fmt, quality):
    """Reserve a new destination exclusively; publish only a completely encoded file."""
    output_dir = Path(output_dir).resolve()
    ext, pillow_format = FORMATS[fmt]
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', Path(source_name).stem).strip(' .')[:110] or 'photo'
    # Suffix also avoids reserved Windows device names such as CON.
    base = f'{stem}_onbit'
    number = 0
    while True:
        name = f'{base}{"_" + str(number).zfill(3) if number else ""}.{ext}'
        destination = output_dir / name
        try:
            with destination.open('xb'):
                pass
            break
        except FileExistsError:
            number += 1
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix='.onbit-', suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
        kwargs = {'icc_profile': SRGB}
        if fmt == 'JPEG':
            kwargs.update(quality=quality, subsampling=0)
        elif fmt == 'WEBP':
            kwargs.update(quality=quality, method=4)
        elif fmt == 'HEIC':
            kwargs.update(quality=quality, exif=None, xmp=None)
        elif fmt == 'TIFF':
            kwargs.update(compression='tiff_lzw')
        elif fmt == 'BMP':
            kwargs = {}
        # A fresh image deliberately excludes stale EXIF orientation, HDR metadata and GPS.
        clean = Image.fromarray(__import__('numpy').asarray(image))
        clean.save(temporary, format=pillow_format, **kwargs)
        os.replace(temporary, destination)
        return destination
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)  # exclusively reserved by this invocation
        raise


def process_item(index, item, options, engine, reference):
    entry = {'index': index, 'source': item.path, 'name': item.name}
    try:
        destination_dir = prepare_output_dir(output_directory(item, options))
        _, image, details = render_item(item, options, engine, reference)
        entry.update(details)
        destination = write_image(image, destination_dir, item.name, options.format, options.quality)
        entry.update(state='done', output=str(destination), width=image.width, height=image.height)
    except Exception as exc:
        entry.update(state='error', error=str(exc))
    return entry


def run_batch(items, options, engine, cancel=None, progress=None):
    options.validate()
    if not items:
        raise ValueError('처리할 사진을 선택해 주세요.')
    if not options.source_subfolder:
        prepare_output_dir(options.output_dir)
    cancel = cancel or threading.Event()
    progress = progress or (lambda event: None)
    result = {'started': datetime.now().isoformat(timespec='seconds'), 'total': len(items),
              'succeeded': 0, 'failed': 0, 'cancelled': False, 'files': [],
              'parallel_gpu': False, 'options': asdict(options)}
    reference = prepare_reference(options, engine) if not cancel.is_set() else None
    use_parallel = (options.parallel_gpu and len(items) > 1 and options.correction != 'none' and
                    isinstance(engine, Engine) and engine.device.type == 'cuda')
    if use_parallel:
        secondary = Engine()
        use_parallel = secondary.device.type == 'cuda' and secondary.status()['ai_ready']
    if use_parallel:
        result['parallel_gpu'] = True
        engines = SimpleQueue()
        engines.put(engine)
        engines.put(secondary)

        def work(index, item):
            worker_engine = engines.get()
            try:
                return process_item(index, item, options, worker_engine, reference)
            finally:
                engines.put(worker_engine)

        completed = 0
        next_index = 0
        entries = {}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='onbit-gpu') as executor:
            pending = {}

            def submit_one(index):
                item = items[index]
                progress({'index': index, 'total': len(items), 'state': 'processing', 'name': item.name})
                pending[executor.submit(work, index, item)] = index

            while next_index < min(2, len(items)) and not cancel.is_set():
                submit_one(next_index)
                next_index += 1
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    entry = future.result()
                    entries[index] = entry
                    completed += 1
                    progress({**entry, 'total': len(items), 'completed': completed})
                    if next_index < len(items) and not cancel.is_set():
                        submit_one(next_index)
                        next_index += 1
        result['files'] = [entries[index] for index in sorted(entries)]
    else:
        for index, item in enumerate(items):
            if cancel.is_set():
                break
            progress({'index': index, 'total': len(items), 'state': 'processing', 'name': item.name})
            entry = process_item(index, item, options, engine, reference)
            result['files'].append(entry)
            progress({**entry, 'total': len(items), 'completed': index + 1})
    result['succeeded'] = sum(entry['state'] == 'done' for entry in result['files'])
    result['failed'] = sum(entry['state'] == 'error' for entry in result['files'])
    result['cancelled'] = len(result['files']) < len(items)
    result['unprocessed'] = len(items) - len(result['files'])
    result['finished'] = datetime.now().isoformat(timespec='seconds')
    completed_output = next((Path(entry['output']).parent for entry in result['files'] if entry['state'] == 'done'), None)
    report_root = completed_output or output_directory(items[0], options)
    report = report_root / f'onbit-report-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}.json'
    try:
        report_root.mkdir(parents=True, exist_ok=True)
        with report.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        result['report'] = str(report)
    except OSError as exc:
        result['report_error'] = str(exc)
    return result
