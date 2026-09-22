"""Tonal-gradation helpers: HDR gain maps, debanding, highlight roll-off and dithering.

Phone photos lose smooth gradation in three places: 8-bit steps in skies,
abrupt highlight clipping, and banding re-introduced when edits are rounded
back to 8 bits.  Where the file carries an HDR gain map (Apple HEIC or
Ultra HDR JPEG), the real recorded highlight range is recovered from it;
otherwise the transitions are only made smoother, never invented.
"""
import io
import re

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

# Apple stores a per-photo headroom in its maker notes; a typical value is used.
APPLE_HEADROOM_STOPS = 2.3
GAIN_MAP_MAX_SIDE = 1024
TRANSPOSE = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
             4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
             6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
             8: Image.Transpose.ROTATE_90}


def _srgb_to_linear_np(v):
    return np.where(v <= .04045, v/12.92, ((v+.055)/1.055)**2.4)


def _xmp_number(text, name, default):
    """Read an hdrgm value written as an attribute, an element or an rdf:Seq."""
    match = re.search(rf'hdrgm:{name}\s*=\s*"([^"]+)"', text)
    if not match:
        match = re.search(rf'<hdrgm:{name}>(.*?)</hdrgm:{name}>', text, re.S)
    if not match:
        return default
    values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', match.group(1))
    return float(np.mean([float(v) for v in values])) if values else default


def _orient(gain, size, orientation):
    """Match the gain map to the displayed photo's orientation, or reject it."""
    width, height = size
    if orientation in TRANSPOSE:
        gain = gain.transpose(TRANSPOSE[orientation])
    ratio, target = gain.width/gain.height, width/height
    return gain if abs(ratio-target) <= .03*target else None


def _stops_array(gain_image, to_stops):
    gain_image.thumbnail((GAIN_MAP_MAX_SIDE, GAIN_MAP_MAX_SIDE), Image.Resampling.BILINEAR)
    values = np.asarray(gain_image.convert('L'), dtype=np.float32)/255
    return np.clip(to_stops(values), 0, 8).astype(np.float32)


def _apple_gain_map(path, size):
    import pillow_heif
    heif = pillow_heif.open_heif(path, convert_hdr_to_8bit=True)
    ids = [i for kind, found in heif.info.get('aux', {}).items() if 'hdrgainmap' in kind.lower() for i in found]
    if not ids:
        return None
    gain = heif.get_aux_image(ids[0]).to_pillow()
    # libheif rotates the primary image; the gain map may still be stored unrotated.
    oriented = _orient(gain, size, 1)
    if oriented is None:
        oriented = _orient(gain, size, heif.info.get('original_orientation') or 1)
    if oriented is None:
        return None
    headroom = 2**APPLE_HEADROOM_STOPS
    return _stops_array(oriented, lambda v: np.log2(1+(headroom-1)*_srgb_to_linear_np(v)))


def _appended_jpeg(raw):
    """The gain map is the last complete JPEG stream appended after the primary.

    Scanning the bytes works whether or not the MPF index is readable.
    """
    for start in reversed([m.start() for m in re.finditer(rb'\xff\xd8\xff', raw) if m.start() > 0]):
        header = raw[start:start+65536]
        # Only a stream carrying its own gain-map metadata; never an EXIF thumbnail.
        if b'hdrgm' not in header and b'urn:iso:std:iso:ts:21496' not in header:
            continue
        try:
            with Image.open(io.BytesIO(raw[start:])) as candidate:
                candidate.load()
                return candidate.convert('L')
        except Exception:
            continue
    return None


def _ultra_hdr_gain_map(path, size, orientation):
    with open(path, 'rb') as handle:
        raw = handle.read()
    text = raw.decode('latin-1')
    has_xmp = 'hdrgm:GainMapMax' in text
    if not has_xmp and 'urn:iso:std:iso:ts:21496' not in text:
        return None  # an ordinary MPO (e.g. stereo 3D), not a gain map
    low = _xmp_number(text, 'GainMapMin', 0.)
    high = _xmp_number(text, 'GainMapMax', 2.)
    gamma = max(_xmp_number(text, 'Gamma', 1.), .01)
    gain = _appended_jpeg(raw)
    if gain is None:
        return None
    gain = _orient(gain, size, orientation)
    if gain is None:
        return None
    return _stops_array(gain, lambda v: low*(1-v**(1/gamma))+high*v**(1/gamma))


def read_gain_map(path, image_format, size, orientation=1):
    """Return (stops, label) for an HDR gain map in the file, or None.

    ``stops`` is a float32 map of how many stops brighter each area was
    recorded in HDR than in the SDR image.  Any problem means "no gain map":
    reading the photo itself must never fail because of this optional data.
    """
    try:
        if image_format == 'HEIF':
            stops = _apple_gain_map(path, size)
            label = 'Apple HDR 게인맵'
        elif image_format in {'MPO', 'JPEG'}:
            stops = _ultra_hdr_gain_map(path, size, orientation)
            label = 'Ultra HDR 게인맵'
        else:
            return None
    except Exception:
        return None
    if stops is None or not np.isfinite(stops).all() or float(stops.max()) <= .01:
        return None
    return stops, label


def srgb_to_linear(x):
    return torch.where(x <= .04045, x/12.92, ((x.clamp_min(0)+.055)/1.055).pow(2.4))


def linear_to_srgb(x):
    # Extended above 1.0 so recovered highlights keep their gradation until roll-off.
    return torch.where(x <= .0031308, x*12.92, 1.055*x.clamp_min(0).pow(1/2.4)-.055)


def gain_scale(gain, height, width):
    """Upsample a stops map to the render size on the active device."""
    return F.interpolate(gain[None, None], size=(height, width), mode='bilinear', align_corners=False)[0, 0]


def gain_region(gain, full_size, region, out_size):
    """Sample the stops map for a crop ``region`` (x0, y0, x1, y1 in full-image pixels)
    rendered at ``out_size``; pixel-centre exact, so detail views line up with the preview."""
    full_w, full_h = full_size
    x0, y0, x1, y1 = region
    out_w, out_h = out_size
    xs = (x0+(torch.arange(out_w, device=gain.device, dtype=torch.float32)+.5)*(x1-x0)/out_w)/full_w*2-1
    ys = (y0+(torch.arange(out_h, device=gain.device, dtype=torch.float32)+.5)*(y1-y0)/out_h)/full_h*2-1
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), -1)
    return F.grid_sample(gain[None, None], grid[None], mode='bilinear', padding_mode='border',
                         align_corners=False)[0, 0]


def apply_gain(x, stops, amount):
    """Brighten highlights by the HDR gain map; the shoulder later compresses them."""
    return linear_to_srgb(srgb_to_linear(x)*torch.exp2(stops[..., None]*amount))


def deband_radius(size):
    # Phone-sky bands are tens of pixels wide; the window must span a few of them.
    return int(np.clip(round(max(size)/60), 4, 64))


def deband(x, radius, amount):
    """Smooth 8-bit steps in flat areas while keeping any real texture or edge.

    A self-guided filter with an epsilon of a few code values only flattens
    variations that small; skin, foliage and edges have far more contrast.
    """
    from engine import box_filter
    eps = ((1.5+2.5*amount)/255)**2
    mean = box_filter(x, radius)
    variance = (box_filter(x*x, radius)-mean*mean).clamp(min=0)
    a = variance/(variance+eps)
    b = mean-a*mean
    smooth = box_filter(a, radius)*x+box_filter(b, radius)
    return torch.lerp(x, smooth, min(1., 2*amount))


def roll_off(x, amount, headroom=1.):
    """Film-like shoulder: brightest tones ease into white instead of clipping.

    ``headroom`` is the brightest value the shoulder must still fit below 1.0,
    e.g. highlights recovered from a gain map.
    """
    if amount <= 0:
        return x
    knee = 1-.25*amount
    peak = x.max(-1, keepdim=True).values
    w = max(1+2*amount, (headroom-knee)/(1-knee))
    t = ((peak-knee)/(1-knee)).clamp(0, w)
    # Extended Reinhard: slope 1 at the knee, reaches exactly 1.0 at t = w.
    shaped = knee+(1-knee)*t*(1+t/(w*w))/(1+t)
    # Only highlights are rescaled. Shadows pushed below zero by contrast must not
    # be divided by a clamped tiny peak, which would blow them up into white holes.
    bright = peak > knee
    shaped = torch.where(bright, shaped, peak)
    x = x*torch.where(bright, shaped/peak.clamp_min(knee), torch.ones_like(peak))
    # Near white, fade colour so hot highlights roll to white rather than flat hue patches.
    whiteness = ((shaped-knee)/(1-knee)).clamp(0, 1)**2
    return torch.lerp(x, shaped.expand_as(x), .5*amount*whiteness)


def _hash(rows, cols, channels, salt):
    h = rows[:, None, None]*73856093 ^ cols[None, :, None]*19349663 ^ channels[None, None, :]*83492791 ^ salt
    h = (h*2654435761) & 0xFFFFFFFF
    h = h ^ (h >> 15)
    return ((h*2246822519) & 0xFFFFFF).to(torch.float32)/16777216


def to_uint8(x, row=0, col=0, dither=True, weight=None):
    """Quantize to 8 bits with triangular dither tied to absolute pixel position.

    Position-seeded noise makes tiles, CPU and GPU produce the same result.
    Exact black and white stay clean; ``weight`` limits dither to edited pixels.
    """
    value = x.clamp(0, 1)*255
    if dither:
        height, width, channels = value.shape
        rows = torch.arange(row, row+height, device=x.device, dtype=torch.int64)
        cols = torch.arange(col, col+width, device=x.device, dtype=torch.int64)
        chans = torch.arange(channels, device=x.device, dtype=torch.int64)
        noise = _hash(rows, cols, chans, 0x5bd1e995)+_hash(rows, cols, chans, 0x1b873593)-1
        noise = noise*torch.minimum(value, 255-value).clamp(0, 1)
        if weight is not None:
            noise = noise*weight
        value = (value+noise).clamp(0, 255)
    return value.round().to('cpu', torch.uint8).numpy()
