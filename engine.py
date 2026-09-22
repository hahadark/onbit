"""Local image-adaptive LUT inference and deterministic, tiled color processing.

Classifier architecture adapted from HuiZeng/Image-Adaptive-3DLUT (Apache-2.0).
See third_party/NOTICE.md. No custom CUDA compiler or torchvision is required.
"""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import threading
import numpy as np
from PIL import Image, ImageCms, ImageOps
from pillow_heif import register_heif_opener
import torch
from torch import nn
from torch.nn import functional as F
from cube_lut import load_cube
from neural_match import NeuralMatcher
from tone import sample_rgb, light_stats, recommend_manual, apply_match, color_profile
from face import FaceRetoucher
from gradation import (apply_gain, deband, deband_radius, gain_region, gain_scale, read_gain_map, roll_off,
                       to_uint8)

ROOT = Path(__file__).resolve().parent
register_heif_opener()
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.mpo', '.png', '.webp', '.tif', '.tiff', '.bmp', '.heic', '.heif', '.hif'}
LIMITS = {'ai': (0, 100, 0), 'exposure': (-2, 2, 0), 'temperature': (-100, 100, 0),
          'tint': (-100, 100, 0), 'saturation': (-100, 100, 0), 'contrast': (-100, 100, 0),
          'shadows': (-100, 100, 0), 'highlights': (-100, 100, 0), 'softness': (0, 100, 0),
          'skin': (0, 100, 0), 'gradation': (0, 100, 0)}


def settings(values=None):
    values = values or {}
    if not isinstance(values, dict):
        raise ValueError('보정 설정 형식이 올바르지 않습니다.')
    result = {}
    for key, (low, high, default) in LIMITS.items():
        value = float(values.get(key, default))
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key}: {low}~{high} 범위의 숫자가 필요합니다.')
        result[key] = value
    return result


def read_image(path, max_side=None):
    """Open a photo as 8-bit sRGB. ``max_side`` lets JPEG decode at a reduced scale
    (never below ``max_side`` on the long edge), for fast analysis of large originals."""
    import io
    with Image.open(path) as source:
        if source.width * source.height > 60_000_000:
            raise ValueError('첫 버전은 6,000만 화소 이하 사진을 지원합니다.')
        if source.format not in {'JPEG', 'MPO', 'PNG', 'WEBP', 'TIFF', 'BMP', 'HEIF'}:
            raise ValueError('JPG, PNG, WebP, TIFF, BMP, HEIC 사진을 사용해 주세요.')
        if source.format == 'HEIF' and source.info.get('primary_index', 0):
            source.seek(source.info['primary_index'])
        if max_side and source.format in {'JPEG', 'MPO'}:
            source.draft('RGB', (max_side, max_side))
        source.load()
        image_format = source.format
        orientation = source.getexif().get(0x0112, 1)
        icc = source.info.get('icc_profile')
        image = ImageOps.exif_transpose(source)
        if 'A' in image.getbands() or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            image = Image.new('RGBA', rgba.size, 'white')
            image.alpha_composite(rgba)
            image = image.convert('RGB')
        elif image.mode not in {'RGB', 'CMYK', 'LAB', 'L'}:
            image = image.convert('RGB')
        if icc:
            try:
                image = ImageCms.profileToProfile(image, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                                 ImageCms.createProfile('sRGB'), outputMode='RGB')
            except Exception as exc:
                raise ValueError('내장 색상 프로필을 읽을 수 없습니다. sRGB로 변환한 사진을 사용해 주세요.') from exc
        image = image.convert('RGB')
    # Optional HDR gain map; it travels with the image (copy/thumbnail keep info).
    if isinstance(path, (str, Path)) and image_format in {'HEIF', 'MPO', 'JPEG'}:
        gain = read_gain_map(path, image_format, image.size, orientation)
        if gain is not None:
            image.info['onbit_gain'], image.info['onbit_gain_label'] = gain
    return image


def read_exif(path):
    """Read a compact, privacy-safe EXIF summary for the viewer (no GPS)."""
    try:
        with Image.open(path) as source:
            if source.format == 'HEIF' and source.info.get('primary_index', 0):
                source.seek(source.info['primary_index'])
            exif = source.getexif()
            width, height = source.size
            if exif.get(274) in {5, 6, 7, 8}:
                width, height = height, width
            # Shutter, aperture, ISO, focal length, date and lens are stored in the
            # Exif sub-IFD by cameras and phones; IFD0 only has make/model/date.
            tags = dict(exif)
            tags.update(exif.get_ifd(0x8769))
    except Exception:
        return {}

    def number(tag):
        try:
            return float(tags.get(tag))
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    make = str(tags.get(271, '')).strip().strip('\x00')
    model = str(tags.get(272, '')).strip().strip('\x00')
    camera = model if make and model.lower().startswith(make.lower()) else ' '.join(part for part in (make, model) if part)
    taken = str(tags.get(36867) or tags.get(306) or '').strip().strip('\x00')
    lens = str(tags.get(42036, '')).strip().strip('\x00')
    focal = number(37386)
    aperture = number(33437)
    exposure = number(33434)
    iso = tags.get(34855) or tags.get(34867)
    if isinstance(iso, (tuple, list)):
        iso = iso[0] if iso else None
    result = {'resolution': f'{width:,} × {height:,}'}
    if taken:
        result['taken'] = taken
    if camera:
        result['camera'] = camera
    if lens:
        result['lens'] = lens
    if focal is not None:
        result['focal'] = f'{focal:g}mm'
    if aperture is not None:
        result['aperture'] = f'f/{aperture:g}'
    if exposure is not None:
        result['shutter'] = f'1/{round(1/exposure)}s' if 0 < exposure < 1 else f'{exposure:g}s'
    if iso:
        result['iso'] = f'ISO {iso}'
    return result


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False),
                  nn.Conv2d(3, 16, 3, 2, 1), nn.LeakyReLU(.2), nn.InstanceNorm2d(16, affine=True)]
        for index, (a, b) in enumerate([(16, 32), (32, 64), (64, 128), (128, 128)]):
            layers += [nn.Conv2d(a, b, 3, 2, 1), nn.LeakyReLU(.2)]
            if index < 3:
                layers.append(nn.InstanceNorm2d(b, affine=True))
        layers += [nn.Dropout(.5), nn.Conv2d(128, 3, 8)]
        self.model = nn.Sequential(*layers)

    def forward(self, value):
        return self.model(value)


def sample_lut(lut, rgb):
    # LUT axes are B,G,R; grid_sample's coordinate order is X(R),Y(G),Z(B).
    # Upstream interpolation uses a 1.0001/(dim-1) bin size.
    grid = (rgb / 1.0001 * 2 - 1).unsqueeze(0).unsqueeze(0)
    result = F.grid_sample(lut.unsqueeze(0), grid, mode='bilinear',
                           padding_mode='border', align_corners=True)
    return result[0, :, 0].permute(1, 2, 0)


def sample_cube_lut(lut, rgb):
    """Sample a standard .cube tensor whose axes are blue, green, red."""
    grid = (rgb.clamp(0, 1) * 2 - 1).unsqueeze(0).unsqueeze(0)
    result = F.grid_sample(lut.unsqueeze(0), grid, mode='bilinear',
                           padding_mode='border', align_corners=True)
    return result[0, :, 0].permute(1, 2, 0)


def adjustments(x, p, headroom=1.):
    x = x * (2 ** p['exposure'])
    warm, tint = p['temperature'] / 100, p['tint'] / 100
    x = x * x.new_tensor([1 + .16 * warm + .06 * tint, 1 - .08 * tint, 1 - .16 * warm + .06 * tint])
    y = (x * x.new_tensor([.2126, .7152, .0722])).sum(-1, keepdim=True).clamp(0, 1)
    x = x + p['shadows'] / 100 * .22 * (1-y)**3 + p['highlights'] / 100 * .22 * y**3
    x = (x - .5) * (1 + p['contrast'] / 150) + .5
    y = (x * x.new_tensor([.2126, .7152, .0722])).sum(-1, keepdim=True)
    x = y + (x-y) * (1 + p['saturation'] / 100)
    # Soft tone: gently lift blacks / roll off whites, without spatial blur.
    soft = p['softness'] / 100
    x = x * (1 - .16 * soft) + .055 * soft
    y = (x * x.new_tensor([.2126, .7152, .0722])).sum(-1, keepdim=True)
    x = y + (x-y) * (1 - .1 * soft)
    return roll_off(x, p.get('gradation', 0) / 100, headroom).clamp(0, 1)


def match_layer(match):
    """Validate a stored colour-match layer; returns an apply-ready dict or None."""
    if not match or not match.get('plan'):
        return None  # pending: measured when the photo is previewed or exported
    try:
        strength = float(match.get('strength', 80))
        plan = dict(match['plan'])
        if not np.isfinite(strength) or not 0 <= strength <= 100:
            raise ValueError
        if plan.get('method') == 'neural':
            matrix = np.asarray(plan['matrix'], dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.abs(matrix).max() > 32:
                raise ValueError
        else:
            for name in ('curve_x', 'curve_y', 'zone_l', 'zone_ab'):
                values = np.asarray(plan[name], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError
            if len(plan['curve_x']) != len(plan['curve_y']) or len(plan['curve_x']) < 2:
                raise ValueError
            if np.asarray(plan['zone_ab'], dtype=float).shape != (len(plan['zone_l']), 2):
                raise ValueError
            plan['chroma'] = float(plan['chroma'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('색감 매칭 정보가 올바르지 않습니다. 다시 맞춰 주세요.') from exc
    if not strength:
        return None
    plan.update(strength=strength, brightness=bool(match.get('brightness', True)))
    return {'plan': plan, 'protect_skin': bool(match.get('protect_skin', True))}


def box_filter(x, r):
    """Mean over a (2r+1)² window clipped at the borders, for an HxWxC tensor.

    Integral images keep the cost independent of the radius, so face-sized
    windows on 24MP photos stay cheap.
    """
    height, width = x.shape[:2]
    rows = torch.arange(height, device=x.device)
    cols = torch.arange(width, device=x.device)
    top, bottom = (rows-r).clamp(min=0), (rows+r+1).clamp(max=height)
    left, right = (cols-r).clamp(min=0), (cols+r+1).clamp(max=width)
    total = F.pad(x.cumsum(0), (0, 0, 0, 0, 1, 0))
    x = total[bottom]-total[top]
    total = F.pad(x.cumsum(1), (0, 0, 1, 0))
    x = total[:, right]-total[:, left]
    count = ((bottom-top)[:, None]*(right-left)[None, :]).to(x.dtype)
    return x / count[..., None]


def smooth_skin(x, r, eps=.0025):
    """Self-guided filter: flattens low-contrast texture, keeps strong edges.

    Pores and small blemishes (local std well below sqrt(eps)) are averaged away,
    while eyelids, nostrils and the face outline (std above it) are preserved.
    """
    mean = box_filter(x, r)
    variance = (box_filter(x*x, r)-mean*mean).clamp(min=0)
    a = variance/(variance+eps)
    b = mean-a*mean
    return box_filter(a, r)*x+box_filter(b, r)


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.luts = None
        self.stream = None
        self.error = None
        self.face = FaceRetoucher()
        self.matcher = NeuralMatcher()
        self.face_cache = {}
        self.cache = {}
        self.custom_lut_cache = {}
        torch.set_num_threads(4)
        try:
            expected = {'classifier.pth': 'bae9865395625ecae58cfe86147e521093bb1e29e7b2544e02adb238b8035021',
                        'LUTs.pth': 'c1bb2bc4b7239c1a7e96159f5923123ba796b1fceb0b8c3132b423ea825b821a'}
            for name, digest in expected.items():
                if hashlib.sha256((ROOT / 'models' / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'{name} 모델 체크섬이 일치하지 않습니다.')
            self.model = Classifier().eval()
            self.model.load_state_dict(torch.load(ROOT / 'models/classifier.pth', map_location='cpu', weights_only=True))
            state = torch.load(ROOT / 'models/LUTs.pth', map_location='cpu', weights_only=True)
            self.luts = torch.stack([state[str(i)]['LUT'] for i in range(3)])
            self.model.to(self.device)
            self.luts = self.luts.to(self.device)
            if self.device.type == 'cuda':
                self.stream = torch.cuda.Stream(device=self.device)
        except Exception as exc:
            self.model = None
            self.error = str(exc)

    def status(self):
        return {'device': str(self.device), 'gpu': torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'CPU',
                'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if self.device.type == 'cuda' else 0,
                'ai_ready': self.model is not None, 'face_ai': self.face.available,
                'face_error': self.face.error or self.face.parser_error, 'face_mode': self.face.mode,
                'error': self.error, 'model': 'Image-Adaptive-3DLUT · FiveK sRGB'}

    def neural_match(self, source, reference):
        with self.lock:
            try:
                context = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
                with context:
                    return self.matcher.plan(source, reference, self.device)
            except torch.cuda.OutOfMemoryError:
                # Small CPU inference yields the same transform; never substitute statistics.
                self.matcher.model.to('cpu')
                self.matcher.device = 'cpu'
                torch.cuda.empty_cache()
                return self.matcher.plan(source, reference, torch.device('cpu'))

    def recommend(self, image, key):
        """Analyze a restrained AI preview, then expose editable manual settings."""
        small = Image.fromarray((sample_rgb(image)*255).round().astype(np.uint8))
        before = light_stats(sample_rgb(small))
        ai = 35 if self.model is not None else 0
        candidate = self.render(small, settings({'ai': ai}), key, lut_image=image)
        after = light_stats(sample_rgb(candidate))
        if after['clipped'] > before['clipped']+.01 or after['saturation'] > max(.12, before['saturation']*1.3):
            ai = 20 if ai else 0
            candidate = self.render(small, settings({'ai': ai}), key, lut_image=image)
        skin = 30 if self.face_count(image, key) else 0
        # Phone photos benefit from gentle gradation repair; more when real HDR data exists.
        gradation = 50 if 'onbit_gain' in image.info else 30
        return settings({'ai': ai, 'skin': skin, 'gradation': gradation, **recommend_manual(small, candidate)})

    def face_count(self, image, key=''):
        return len(self._face_boxes(image, key))

    def _face_boxes(self, image, key=''):
        with self.lock:
            if not key:
                return self.face.detect(image)
            cache_key = f'{key}|faces|{image.width}x{image.height}'
            if cache_key not in self.face_cache:
                boxes = self.face.detect(image)
                if len(self.face_cache) >= 64:
                    self.face_cache.pop(next(iter(self.face_cache)))
                self.face_cache[cache_key] = boxes
            return self.face_cache[cache_key]

    def profile(self, image, key='', detection_image=None, protect_skin=True):
        """Colour statistics for matching; skin is left out when protected.

        Statistics only need to know roughly where faces are, so the fast face-shaped
        mask is used here; the precise AI skin mask is used when the match is applied.
        """
        exclude = None
        if protect_skin and self.face.available:
            boxes = self._scaled_face_boxes(image, key, detection_image)
            mask = self.face.geometric_mask(image, boxes) if boxes else None
            exclude = mask if mask is not None and mask.any() else None
        return color_profile(image, exclude)

    @staticmethod
    def _view(image, reference, region):
        """Scale and offset from ``reference`` pixels to ``image`` pixels.

        ``region`` (x0, y0, x1, y1, in reference pixels) means ``image`` shows only that crop.
        """
        if region is None:
            return image.width/reference.width, image.height/reference.height, 0, 0
        x0, y0, x1, y1 = region
        return image.width/(x1-x0), image.height/(y1-y0), x0, y0

    def _scaled_face_boxes(self, image, key='', detection_image=None, region=None):
        reference = detection_image if detection_image is not None else image
        boxes = self._face_boxes(reference, key)
        sx, sy, ox, oy = self._view(image, reference, region)
        return [(round((x-ox)*sx), round((y-oy)*sy), round(w*sx), round(h*sy)) for x, y, w, h in boxes]

    def _skin_mask(self, image, key='', detection_image=None, region=None):
        reference = detection_image if detection_image is not None else image
        parts = self._face_parts(reference, key)
        if parts is None:
            return self.face.geometric_mask(image, self._scaled_face_boxes(image, key, detection_image, region))
        return self.face.paint(image.size, parts, *self._view(image, reference, region))

    def _face_parts(self, reference, key=''):
        """Parsed 512² skin maps per face, cached per photo so sliders stay fast.

        Only these small maps are cached, never full-resolution masks.
        """
        with self.lock:
            boxes = self._face_boxes(reference, key)
            if not key:
                return self.face.parse(reference, boxes)
            cache_key = f'{key}|parts|{reference.width}x{reference.height}'
            if cache_key not in self.face_cache:
                parts = self.face.parse(reference, boxes)
                if len(self.face_cache) >= 64:
                    self.face_cache.pop(next(iter(self.face_cache)))
                self.face_cache[cache_key] = parts
            return self.face_cache[cache_key]

    def _skin_radius(self, image, key='', detection_image=None, region=None):
        """Smoothing radius tied to face size, so preview and export look alike."""
        widths = [w for _x, _y, w, _h in self._scaled_face_boxes(image, key, detection_image, region)]
        face = float(np.median(widths)) if widths else min(image.size)*.2
        return int(np.clip(round(face*.03), 2, 48))

    @torch.inference_mode()
    def match(self, image, plan):
        if not plan['strength']:
            return image.copy()
        with self.lock:
            try:
                context = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
                with context:
                    return self._match(image, plan)
            except torch.cuda.OutOfMemoryError:
                self.device = torch.device('cpu')
                if self.model is not None:
                    self.model.cpu()
                    self.luts = self.luts.cpu()
                self.cache.clear()
                self.face_cache.clear()
                self.stream = None
                torch.cuda.empty_cache()
                self.error = 'GPU 메모리 부족으로 CPU 처리로 전환했습니다.'
                return self._match(image, plan)

    def _match(self, image, plan):
        source = np.asarray(image)
        result = np.empty_like(source)
        for row in range(0, image.height, 192):
            rgb = torch.from_numpy(source[row:row+192].copy()).to(self.device, torch.float32)/255
            result[row:row+192] = (apply_match(rgb, plan)*255).round().to('cpu', torch.uint8).numpy()
        return Image.fromarray(result)

    @torch.inference_mode()
    def lut(self, image, key):
        if self.model is None:
            raise ValueError(f'AI 모델을 사용할 수 없습니다: {self.error}')
        if key not in self.cache:
            # Bilinear downsample is evaluated from the source on CPU, keeping full-size tensors off GPU.
            x = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]
            x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False).to(self.device)
            weights = self.model(x).reshape(3)
            result = (self.luts * weights[:, None, None, None, None]).sum(0)
            if not torch.isfinite(result).all():
                raise ValueError('AI 모델이 유효하지 않은 색상 값을 반환했습니다.')
            if len(self.cache) >= 64:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = result
        return self.cache[key]

    def custom_lut(self, path):
        source = Path(path).expanduser().resolve()
        try:
            stat = source.stat()
        except OSError as exc:
            raise ValueError('선택한 LUT 파일을 찾을 수 없습니다: ' + str(source)) from exc
        cache_key = f'{source}|{stat.st_size}|{stat.st_mtime_ns}|{self.device}'
        if cache_key not in self.custom_lut_cache:
            cube = load_cube(source)
            if len(self.custom_lut_cache) >= 8:
                self.custom_lut_cache.pop(next(iter(self.custom_lut_cache)))
            self.custom_lut_cache[cache_key] = (
                torch.from_numpy(cube.values).to(self.device),
                torch.from_numpy(cube.domain_min).to(self.device),
                torch.from_numpy(cube.domain_max).to(self.device),
                cube.title,
            )
        return self.custom_lut_cache[cache_key]

    @torch.inference_mode()
    def render(self, image, p, key, lut_image=None, custom_lut_path='', custom_lut_strength=0, match=None,
               region=None):
        """``match`` is a stored colour-match layer: {'plan', 'strength', 'brightness', 'protect_skin'}.

        ``region`` (x0, y0, x1, y1 in ``lut_image`` pixels) renders ``image`` as that crop of the
        photo, e.g. a zoomed detail view, with faces, gain map and filter sizes lined up.
        """
        p = settings(p)
        match = match_layer(match)
        if region is not None:
            if lut_image is None:
                raise ValueError('확대 영역 렌더링에는 원본 사진이 필요합니다.')
            region = tuple(int(v) for v in region)
            if not (0 <= region[0] < region[2] <= lut_image.width and 0 <= region[1] < region[3] <= lut_image.height):
                raise ValueError('확대 영역이 사진 범위를 벗어났습니다.')
        try:
            custom_lut_strength = float(custom_lut_strength)
        except (TypeError, ValueError) as exc:
            raise ValueError('LUT 강도는 0~100 범위의 숫자여야 합니다.') from exc
        if not np.isfinite(custom_lut_strength) or not 0 <= custom_lut_strength <= 100:
            raise ValueError('LUT 강도는 0~100 범위의 숫자여야 합니다.')
        if custom_lut_strength and not custom_lut_path:
            raise ValueError('적용할 LUT 파일을 선택해 주세요.')
        with self.lock:
            try:
                context = torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
                with context:
                    return self._render(image, p, key, lut_image, custom_lut_path, custom_lut_strength, match, region)
            except torch.cuda.OutOfMemoryError:
                self.device = torch.device('cpu')
                if self.model is not None:
                    self.model.cpu()
                    self.luts = self.luts.cpu()
                self.cache.clear()
                self.face_cache.clear()
                self.custom_lut_cache.clear()
                self.stream = None
                torch.cuda.empty_cache()
                self.error = 'GPU 메모리 부족으로 CPU 처리로 전환했습니다.'
                return self._render(image, p, key, lut_image, custom_lut_path, custom_lut_strength, match, region)

    def _render(self, image, p, key, lut_image, custom_lut_path='', custom_lut_strength=0, match=None,
                region=None):
        lut = self.lut(lut_image if lut_image is not None else image, key) if p['ai'] else None
        custom = self.custom_lut(custom_lut_path) if custom_lut_path and custom_lut_strength else None
        array = np.asarray(image)
        output = np.empty_like(array)
        amount = p['gradation'] / 100
        # Filter sizes follow the whole photo at this scale, so a zoomed crop matches the preview.
        full = lut_image.size if region is not None else image.size
        scale = image.width/(region[2]-region[0]) if region is not None else 1
        radius = deband_radius((full[0]*scale, full[1]*scale)) if amount else 0
        # The deband filter reads 2r rows around each tile, so tiles match a single pass.
        halo = 2*radius+1 if amount else 0
        source_info = lut_image.info if region is not None else image.info
        gain = source_info.get('onbit_gain') if amount else None
        headroom = 1.
        if gain is not None:
            gain_tensor = torch.from_numpy(gain).to(self.device)
            stops = (gain_region(gain_tensor, lut_image.size, region, image.size) if region is not None
                     else gain_scale(gain_tensor, image.height, image.width))
            headroom = float(2**(amount*float(gain.max())/2.4))
        edited = custom is not None or match is not None or any(value for name, value in p.items() if name != 'skin')
        keep = None
        if match is not None and match['protect_skin'] and self.face.available:
            # Skin keeps its own colour while the scene follows the reference.
            skin = self._skin_mask(image, key, lut_image, region)
            keep = torch.from_numpy(skin).to(self.device) if skin.any() else None
        for row in range(0, image.height, 192):
            start, end = max(0, row-halo), min(image.height, row+192+halo)
            x = torch.from_numpy(array[start:end].copy()).to(self.device, torch.float32) / 255
            if amount:
                x = deband(x, radius, amount)
            x = x[row-start:row-start+min(192, image.height-row)]
            if lut is not None:
                x = torch.lerp(x, sample_lut(lut, x), p['ai'] / 100)
            if gain is not None:
                x = apply_gain(x, stops[row:row+x.shape[0]], amount)
            x = adjustments(x, p, headroom)
            if custom is not None:
                custom_lut, domain_min, domain_max, _title = custom
                normalized = (x - domain_min) / (domain_max - domain_min)
                x = torch.lerp(x, sample_cube_lut(custom_lut, normalized), custom_lut_strength / 100).clamp(0, 1)
            if match is not None:
                weight = None if keep is None else keep[row:row+x.shape[0], :, None].float() / 255
                x = apply_match(x, match['plan'], weight)
            # Dither only real edits, so untouched photos stay bit-identical.
            output[row:row+192] = to_uint8(x, row, dither=edited)
        rendered = Image.fromarray(output)
        if p['skin']:
            rendered = self.skin_retouch(rendered, p['skin'], key, source_image=image,
                                         detection_image=lut_image, region=region)
        return rendered

    @torch.inference_mode()
    def skin_retouch(self, image, amount, key='', source_image=None, detection_image=None, region=None):
        """Soften only skin-colored pixels inside detected face regions."""
        if amount <= 0:
            return image.copy()
        if not self.face.available:
            raise ValueError('피부 보정을 사용할 수 없습니다: ' + self.face.error)
        # Detect and classify original skin before exposure/temperature edits.
        reference = source_image if source_image is not None else image
        mask = self._skin_mask(reference, key, detection_image, region)
        if not np.any(mask):
            return image.copy()
        radius = self._skin_radius(reference, key, detection_image, region)
        source = np.asarray(image)
        result = source.copy()
        strength = float(amount) / 100
        # The guided filter reads 2r rows around each tile, so tiles match a single pass.
        halo = 2*radius+1
        columns = np.flatnonzero(mask.any(0))
        c0, c1 = max(0, columns[0]-halo), min(image.width, columns[-1]+1+halo)
        for row in range(0, image.height, 192):
            alpha = mask[row:row+192, c0:c1]
            if not alpha.any():
                continue
            start, end = max(0, row-halo), min(image.height, row+192+halo)
            x = torch.from_numpy(source[start:end, c0:c1].copy()).to(self.device, torch.float32) / 255
            smooth = smooth_skin(x, radius)
            offset, count = row-start, alpha.shape[0]
            x, smooth = x[offset:offset+count], smooth[offset:offset+count]
            alpha = torch.from_numpy(alpha.copy()).to(self.device, torch.float32)[..., None] / 255 * strength
            # Keep a trace of real texture, and brighten skin slightly for a clear, even tone.
            x = torch.lerp(x, smooth, alpha*.9)
            x = x + alpha*.04*(1-x)
            result[row:row+count, c0:c1] = to_uint8(x, row, c0, weight=(alpha > 0).float())
        return Image.fromarray(result)


def analyze(image):
    small = image.copy()
    small.thumbnail((256, 256))
    a = np.asarray(small, dtype=np.float32) / 255
    luma = a @ np.array([.2126, .7152, .0722])
    means = a.mean((0, 1))
    histogram = [np.histogram(a[:, :, i], bins=64, range=(0, 1))[0].tolist() for i in range(3)]
    return {'brightness': round(float(luma.mean()) * 100),
            'shadows': round(float((luma < .12).mean()) * 100),
            'highlights': round(float((luma > .95).mean()) * 100),
            'gainmap': image.info.get('onbit_gain_label', ''),
            'cast': '따뜻한 색감' if means[0] - means[2] > .035 else '차가운 색감' if means[2] - means[0] > .035 else '중립적인 색감',
            'histogram': histogram}
