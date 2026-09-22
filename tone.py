"""Bright-leaning automatic settings and reference colour matching (tone curve + 3 zones).

Profiles use perceptual Oklab statistics; the full-resolution transform runs
on the same torch device as the editor. No spatial content is generated.
"""
import numpy as np
import torch
from PIL import Image

RGB_LMS = [[.4122214708, .5363325363, .0514459929],
           [.2119034982, .6806995451, .1073969566],
           [.0883024619, .2817188376, .6299787005]]
LMS_LAB = [[.2104542553, .7936177850, -.0040720468],
           [1.9779984951, -2.4285922050, .4505937099],
           [.0259040371, .7827717662, -.8086757660]]


def sample_rgb(image):
    scale = min(1, 384 / max(image.size))
    small = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.float32) / 255


def rgb_to_lab(rgb):
    linear = torch.where(rgb <= .04045, rgb / 12.92, ((rgb+.055)/1.055).pow(2.4))
    lms = (linear @ rgb.new_tensor(RGB_LMS).T).clamp_min(0).pow(1/3)
    return lms @ rgb.new_tensor(LMS_LAB).T


def lab_to_rgb(lab):
    lms = lab @ torch.linalg.inv(lab.new_tensor(LMS_LAB)).T
    linear = lms.pow(3) @ torch.linalg.inv(lab.new_tensor(RGB_LMS)).T
    return torch.where(linear <= .0031308, linear*12.92, 1.055*linear.clamp_min(0).pow(1/2.4)-.055).clamp(0, 1)


def light_stats(rgb):
    y = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    saturation = (rgb.max(-1)-rgb.min(-1)) / np.maximum(rgb.max(-1), .01)
    return {'q': np.quantile(y, [.05, .25, .5, .75, .95, .99]),
            'saturation': float(saturation.mean()), 'clipped': float((rgb.max(-1) >= .995).mean())}


def recommend_manual(original, candidate):
    rgb = sample_rgb(candidate)
    stats = light_stats(rgb)
    before = light_stats(sample_rgb(original))
    q05, q25, q50, q75, q95, q99 = stats['q']
    # Bright, airy target preferred in Korean photo editing: midtones near 0.5.
    exposure = float(np.clip(.6*np.log2(.5/max(float((q25+q50+q75)/3), .03)), -.3, .7))
    if exposure > 0 and q99 > .65:
        # Highlight recovery below absorbs a little overshoot, so allow up to 1.06.
        exposure = min(exposure, max(0., float(np.log2(1.06/q99))))
    lift = 2**exposure
    q95e = min(float(q95)*lift, 1.)
    # Dull whites are opened up; whites pushed near clipping are rolled off.
    highlights = float(np.clip((.86-q95e)*90, 0, 25)) if q95e < .86 else -float(np.clip((q95e-.86)*220, 0, 30))
    # Only weakly neutral, moderately lit pixels contribute to white balance.
    luminance = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    mask = (luminance > .18) & (luminance < .8) & ((rgb.max(-1)-rgb.min(-1)) < .12)
    temperature = tint = 0.
    if mask.sum() >= max(32, mask.size*.03):
        red, green, blue = np.median(rgb[mask], axis=0)
        # Slightly cool bias keeps whites and skin clean rather than yellow.
        temperature = float(np.clip((blue-red)/max(red+blue, .1)*110-2, -12, 12))
        tint = float(np.clip((green-(red+blue)/2)/max(float((red+green+blue)/3), .1)*65, -10, 10))
    saturation = min(0., (max(before['saturation'], .03)*1.08/max(stats['saturation'], .03)-1)*100)
    return {'exposure': round(exposure, 2), 'temperature': round(temperature), 'tint': round(tint),
            'highlights': round(highlights),
            'shadows': round(float(np.clip((.25-q25*lift)*70, 0, 15))),
            'contrast': 4 if q95-q05 < .42 else -4 if q95-q05 > .8 else 0,
            'saturation': round(max(-18., saturation)), 'softness': 5}


MATCH_POINTS = [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]


def color_profile(image, exclude=None):
    """Perceptual (Oklab) tone and colour statistics used for colour matching.

    ``exclude`` is an optional 0..255 mask (same size as ``image``) of pixels to
    leave out, e.g. skin, so faces do not steer the match of the scene.
    """
    rgb = sample_rgb(image)
    lab = rgb_to_lab(torch.from_numpy(rgb)).numpy().reshape(-1, 3)
    if exclude is not None:
        small = Image.fromarray(np.asarray(exclude, dtype=np.uint8)).resize((rgb.shape[1], rgb.shape[0]),
                                                                             Image.Resampling.BILINEAR)
        keep = np.asarray(small).reshape(-1) < 128
        if keep.mean() >= .1:
            lab = lab[keep]
    lightness = lab[:, 0]
    quantiles = np.quantile(lightness, MATCH_POINTS)
    edges = np.quantile(lightness, [1/3, 2/3])
    zones = []
    for part in (lightness <= edges[0], (lightness > edges[0]) & (lightness < edges[1]), lightness >= edges[1]):
        part = lab[part] if part.any() else lab
        zones.append(np.median(part, axis=0).tolist())
    middle_range = np.quantile(lightness, [.1, .9])
    middle = lab[(lightness >= middle_range[0]) & (lightness <= middle_range[1])]
    middle = middle if len(middle) else lab
    spread = np.quantile(middle, .75, axis=0)-np.quantile(middle, .25, axis=0)
    return {'center': np.median(middle, axis=0).tolist(), 'spread': spread.tolist(),
            'quantiles': quantiles.tolist(), 'zones': zones,
            'chroma': float(np.hypot(middle[:, 1], middle[:, 2]).mean())}


def _tone_curve(source, reference):
    """Monotonic lightness curve through matched percentiles, with bounded slope."""
    xs = np.concatenate([[0.], np.asarray(source, float), [1.]])
    ys = np.concatenate([[0.], np.asarray(reference, float), [1.]])
    # Merge percentiles that coincide (flat areas) so the curve stays a function.
    px, py = [xs[0]], [ys[0]]
    for x, y in zip(xs[1:-1], ys[1:-1]):
        if x-px[-1] < 2e-3 or 1-x < 2e-3:
            if len(px) > 1 and x-px[-1] < 2e-3:
                py[-1] = (py[-1]+y)/2
            continue
        px.append(x)
        py.append(y)
    px, py = np.array(px+[1.]), np.array(py+[1.])
    # Keep the move modest and smooth: large scene differences are followed only in part.
    shift = np.clip(py-px, -.3, .3)
    if len(shift) > 2:
        inner = (shift[:-2]+2*shift[1:-1]+shift[2:])/4
        shift[1:-1] = inner
    py = px+shift
    py[0], py[-1] = 0., 1.
    # Slope between .35 and 2.5 avoids posterised steps or crushed tones.
    for i in range(1, len(px)):
        step = px[i]-px[i-1]
        py[i] = np.clip(py[i], py[i-1]+.35*step, py[i-1]+2.5*step)
    if py[-1] > 0:
        py = py/py[-1]
    return px.tolist(), np.clip(py, 0, 1).tolist()


def match_plan(source, reference, strength=70, brightness=True):
    """Plan that moves ``source`` towards the tone and colour of ``reference``.

    The plan stores the full match; ``strength`` and ``brightness`` are applied
    when rendering, so they can be changed later without recomputing.
    """
    amount = float(strength)/100
    if not np.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError('색감 통일 강도는 0~100 범위여야 합니다.')
    curve_x, curve_y = _tone_curve(source['quantiles'], reference['quantiles'])
    chroma = 1.
    if source['chroma'] > .005:
        chroma = float(np.clip(reference['chroma']/source['chroma'], .75, 1.35))
    zone_l, zone_ab = [], []
    for src, ref in zip(source['zones'], reference['zones']):
        zone_l.append(float(src[0]))
        delta = np.array(ref[1:])-chroma*np.array(src[1:])
        zone_ab.append(np.clip(delta, -.08, .08).tolist())
    return {'version': 2, 'curve_x': curve_x, 'curve_y': curve_y, 'zone_l': zone_l, 'zone_ab': zone_ab,
            'chroma': chroma, 'strength': float(strength), 'brightness': bool(brightness)}


def _interp(x, xp, fp):
    xp, fp = x.new_tensor(xp), x.new_tensor(fp)
    x = x.clamp(0, 1).contiguous()
    index = torch.bucketize(x, xp).clamp(1, len(xp)-1)
    x0, x1, y0, y1 = xp[index-1], xp[index], fp[index-1], fp[index]
    return y0+(y1-y0)*((x-x0)/(x1-x0).clamp_min(1e-6))


def apply_match(rgb, plan, keep=None):
    """Apply a colour-match plan to an HxWx3 sRGB tensor; ``keep`` (HxWx1, 0..1) stays unchanged."""
    amount = plan['strength']/100
    if not amount:
        return rgb
    if plan.get('method') == 'neural':
        transformed = (rgb @ rgb.new_tensor(plan['matrix']).T).clamp(0, 1)
        if plan['brightness'] and keep is None:
            return torch.lerp(rgb, transformed, amount)
        before, after = rgb_to_lab(rgb), rgb_to_lab(transformed)
        if not plan['brightness']:
            after[..., :1] = before[..., :1]
        if keep is not None:
            # Protect skin colour, while still allowing the requested lightness match.
            after[..., 1:] = torch.lerp(after[..., 1:], before[..., 1:], keep)
        return lab_to_rgb(torch.lerp(before, after, amount))
    lab = rgb_to_lab(rgb)
    lightness = lab[..., :1]
    out = lab.clone()
    if plan['brightness']:
        out[..., :1] = lightness+amount*(_interp(lightness, plan['curve_x'], plan['curve_y'])-lightness)
    # Soft membership in the source's shadow / midtone / highlight zones.
    centers = lab.new_tensor(plan['zone_l'])
    sigma = max(.08, float(centers[-1]-centers[0])/3)
    weights = torch.exp(-.5*((lightness-centers)/sigma)**2)
    weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-6)
    ab = lab[..., 1:]*plan['chroma']+weights@lab.new_tensor(plan['zone_ab'])
    # Fade the colour change near black/white to avoid tinting highlights and crushing blacks.
    protection = (4*out[..., :1]*(1-out[..., :1])).clamp(0, 1)
    out[..., 1:] = lab[..., 1:]+amount*(ab-lab[..., 1:])*protection
    if keep is not None:
        out = torch.lerp(out, lab, keep)
    return lab_to_rgb(out)
