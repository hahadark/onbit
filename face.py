"""Local AI face detection and pixel-accurate skin masking.

Detection uses YuNet (OpenCV Zoo, MIT), a small CNN that also finds turned
and profile faces.  Each face is then segmented by a BiSeNet face-parsing
network (yakhyo/face-parsing, MIT code; trained on CelebAMask-HQ), so only
real skin is retouched while eyes, brows, lips and hair are left alone.
Everything runs locally.  If a model is missing, the older Haar detector and
geometric skin mask are used instead.  The skin blend itself runs in the
engine on the active torch device.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2
except ImportError:  # pragma: no cover - exercised on minimal installations
    cv2 = None

MODELS = Path(__file__).resolve().parent / 'models'
DETECTOR = ('face_detection_yunet_2023mar.onnx', '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4')
PARSER = ('face_parsing_resnet18.onnx', '0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f')
# CelebAMask-HQ classes that are skin: skin, left/right ear, nose, neck.
SKIN_CLASSES = (1, 7, 8, 10, 14)
PARSE_SIZE = 512
# Parsing was trained on face crops that include hair and chin; 1.9x the box matches that framing.
CROP_SCALE = 1.9
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def _model_bytes(name, digest):
    """Read a bundled model; OpenCV's own file reader cannot open every Unicode path."""
    data = (MODELS / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'{name} 모델 체크섬이 일치하지 않습니다.')
    return np.frombuffer(data, np.uint8)


def _overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = min(ax+aw, bx+bw)-max(ax, bx)
    h = min(ay+ah, by+bh)-max(ay, by)
    return 0. if w <= 0 or h <= 0 else w*h/min(aw*ah, bw*bh)


class FaceRetoucher:
    def __init__(self):
        self.detector = None
        self.parser = None
        self.cascade = None
        self.error = ''
        self.parser_error = ''
        if cv2 is None:
            self.error = 'OpenCV가 설치되지 않았습니다.'
            return
        try:
            self._detector_model = _model_bytes(*DETECTOR)
            self.detector = self._make_detector((320, 320))
        except Exception as exc:  # pragma: no cover - depends on packaged data
            self.error = f'AI 얼굴 검출 모델을 불러오지 못했습니다: {exc}'
            self._load_cascade()
        try:
            self.parser = cv2.dnn.readNetFromONNX(_model_bytes(*PARSER))
        except Exception as exc:  # pragma: no cover - depends on packaged data
            self.parser_error = f'얼굴 영역 분할 모델을 불러오지 못했습니다: {exc}'

    def _make_detector(self, size):
        return cv2.FaceDetectorYN.create('onnx', self._detector_model, np.empty(0, np.uint8), size, .6, .3, 100)

    def _load_cascade(self):
        try:
            cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
            storage = cv2.FileStorage(cascade_path.read_text(encoding='utf-8'),
                                     cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            cascade = cv2.CascadeClassifier()
            try:
                cascade.read(storage.getFirstTopLevelNode())
            finally:
                storage.release()
            if cascade.empty():
                raise ValueError('얼굴 인식 분류기를 읽지 못했습니다.')
            self.cascade = cascade
            self.error = ''
        except Exception as exc:  # pragma: no cover - depends on packaged data
            self.error = str(exc)

    @property
    def available(self):
        return self.detector is not None or self.cascade is not None

    @property
    def mode(self):
        if self.detector is not None:
            return 'AI 얼굴 인식 + 피부 영역 분할' if self.parser is not None else 'AI 얼굴 인식'
        return '기본 얼굴 인식' if self.cascade is not None else ''

    def detect(self, image):
        """Return face boxes as ``(left, top, width, height)`` in source pixels."""
        if self.detector is not None:
            found = self._detect_yunet(image)
        elif self.cascade is not None:
            found = self._detect_haar(image)
        else:
            return []
        width, height = image.size
        boxes = []
        for x, y, w, h in found:
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(width, int(round(x+w))), min(height, int(round(y+h)))
            if x1-x0 > 8 and y1-y0 > 8:
                boxes.append((x0, y0, x1-x0, y1-y0))
        # Keep the largest box for each face so a slider never retouches a face twice.
        boxes.sort(key=lambda box: box[2]*box[3], reverse=True)
        unique = []
        for box in boxes:
            if all(_overlap(box, kept) < .55 for kept in unique):
                unique.append(box)
            if len(unique) >= 20:
                break
        return unique

    def _detect_yunet(self, image):
        # YuNet is tuned for faces of roughly 10-300 px, so a close-up selfie and a
        # group photo need different working sizes; results are merged afterwards.
        rgb = image.convert('RGB')
        found = []
        for side in (320, 960):
            scale = min(1.0, side / max(rgb.size))
            work = rgb.resize((max(1, round(rgb.width*scale)), max(1, round(rgb.height*scale))),
                              Image.Resampling.BILINEAR) if scale < 1 else rgb
            bgr = cv2.cvtColor(np.asarray(work), cv2.COLOR_RGB2BGR)
            self.detector.setInputSize((bgr.shape[1], bgr.shape[0]))
            _count, faces = self.detector.detect(bgr)
            for face in ([] if faces is None else faces):
                found.append(tuple(float(v)/scale for v in face[:4]))
            if scale >= 1:
                break
        return found

    def _detect_haar(self, image):
        scale = min(1.0, 1000.0 / max(image.size))
        small = image.copy()
        small.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        gray = cv2.equalizeHist(cv2.cvtColor(np.asarray(small.convert('RGB')), cv2.COLOR_RGB2GRAY))
        minimum = max(20, int(min(gray.shape[:2]) * .045))
        found = self.cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(minimum, minimum))
        return [tuple(float(v)/scale for v in box) for box in found]

    def parse(self, image, boxes):
        """Segment each face; returns ``[(crop_rect, skin_probability_uint8), ...]``.

        ``crop_rect`` is ``(x0, y0, x1, y1)`` in ``image`` pixels and may extend
        past the borders; the probability map covers exactly that square.
        """
        if self.parser is None:
            return None
        rgb = np.asarray(image.convert('RGB'))
        height, width = rgb.shape[:2]
        parts = []
        for x, y, w, h in boxes:
            side = max(w, h)*CROP_SCALE
            cx, cy = x+w/2, y+h/2
            x0, y0 = int(round(cx-side/2)), int(round(cy-side/2))
            x1, y1 = x0+int(round(side)), y0+int(round(side))
            crop = rgb[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
            crop = cv2.copyMakeBorder(crop, max(0, -y0), max(0, y1-height), max(0, -x0), max(0, x1-width),
                                      cv2.BORDER_REPLICATE)
            blob = cv2.resize(crop, (PARSE_SIZE, PARSE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)/255
            blob = ((blob-MEAN)/STD).transpose(2, 0, 1)[None].astype(np.float32)
            self.parser.setInput(blob)
            logits = self.parser.forward()[0]
            logits = logits-logits.max(0, keepdims=True)
            prob = np.exp(logits)
            prob = prob[list(SKIN_CLASSES)].sum(0)/prob.sum(0)
            # Sharpen the soft boundary slightly so hair and lip edges stay untouched.
            prob = np.clip((prob-.35)/.5, 0, 1)
            parts.append(((x0, y0, x1, y1), (prob*255).round().astype(np.uint8)))
        return parts

    @staticmethod
    def paint(size, parts, sx=1., sy=1., ox=0., oy=0.):
        """Place parsed skin probabilities into a 0..255 mask of ``size``.

        Part rectangles are shifted by ``(ox, oy)`` then scaled, so a crop of the
        photo (a zoomed detail view) gets exactly its part of the mask.
        """
        width, height = size
        mask = np.zeros((height, width), np.uint8)
        for (x0, y0, x1, y1), prob in parts:
            x0, y0, x1, y1 = round((x0-ox)*sx), round((y0-oy)*sy), round((x1-ox)*sx), round((y1-oy)*sy)
            if x1-x0 < 2 or y1-y0 < 2:
                continue
            scaled = cv2.resize(prob, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
            cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            region = scaled[cy0-y0:cy1-y0, cx0-x0:cx1-x0]
            mask[cy0:cy1, cx0:cx1] = np.maximum(mask[cy0:cy1, cx0:cx1], region)
        return mask

    def skin_mask(self, image, boxes):
        """Return a soft 0..255 skin mask restricted to detected faces."""
        if not boxes:
            return np.zeros((image.height, image.width), np.uint8)
        parts = self.parse(image, boxes)
        if parts is not None:
            return self.paint(image.size, parts)
        return self.geometric_mask(image, boxes)

    def geometric_mask(self, image, boxes):
        """Fallback without the parsing model: skin colour inside a face-shaped ellipse."""
        rgb = np.asarray(image.convert('RGB'))
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        if not boxes or cv2 is None:
            return mask
        for left, top, box_width, box_height in boxes:
            x0, y0 = max(0, left), max(0, top)
            x1, y1 = min(width, left + box_width), min(height, top + box_height)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = rgb[y0:y1, x0:x1]
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            cr, cb = ycrcb[..., 1], ycrcb[..., 2]
            # Wide enough for fair, brightly lit skin (low Cr, low saturation).
            skin = ((cr >= 128) & (cr <= 180) & (cb >= 72) & (cb <= 140) &
                    (hsv[..., 1] >= 12) & (hsv[..., 2] >= 35))
            yy, xx = np.ogrid[y0:y1, x0:x1]
            cx = left + box_width * .50
            cy = top + box_height * .52
            rx = max(1.0, box_width * .42)
            ry = max(1.0, box_height * .50)
            ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
            # Conservative feature exclusion for frontal faces.
            eye_band = ((xx-cx)/(box_width*.38))**2 + ((yy-(top+box_height*.39))/(box_height*.12))**2 <= 1
            mouth = ((xx-cx)/(box_width*.24))**2 + ((yy-(top+box_height*.76))/(box_height*.12))**2 <= 1
            allowed = skin & ellipse & ~eye_band & ~mouth
            local = Image.fromarray(allowed.astype(np.uint8)*255)
            soft = np.asarray(local.filter(ImageFilter.GaussianBlur(max(1.0, box_width*.02))))
            soft = np.where(allowed, soft, 0).astype(np.uint8)
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], soft)
        return mask

    def mask(self, image, boxes=None):
        return self.skin_mask(image, self.detect(image) if boxes is None else boxes)
