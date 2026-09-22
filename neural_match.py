"""Local pair-conditioned Neural-Preset inference (unofficial research model).

Architecture/weights: DY112/Neural-Preset, MIT code; see third_party/NOTICE.md.
The encoder predicts source normalization and reference styling matrices.
Only a 3x3 colour transform is exported; image geometry is never synthesized.
"""
from pathlib import Path
import hashlib
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

MODEL_PATH = Path(__file__).resolve().parent / 'models/neural-preset.pth'
MODEL_SHA256 = '7a683ea84fee34ada68b1231b8ac81b3350a926ed2f2a1b6cbd3b9e503da880c'
MODEL_ID = 'neural-preset-dy112-v1'


class PresetNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform_p = nn.Parameter(torch.empty(3, 16))
        self.transform_q = nn.Parameter(torch.empty(16, 3))
        self.style_encoder = EfficientNet.from_name('efficientnet-b0', num_classes=512)

    def forward(self, images):
        codes = self.style_encoder(images).reshape(-1, 2, 16, 16)
        return self.transform_p @ codes @ self.transform_q


class NeuralMatcher:
    """Lazy model per engine; bounded CPU embedding cache; no network at runtime."""
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.device = None
        self.cache = OrderedDict()

    def _load(self, device):
        if self.model is None:
            if not MODEL_PATH.is_file() or hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
                raise ValueError('AI 색감 매칭 모델이 없거나 손상되었습니다. 모델 파일을 복구해 주세요.')
            model = PresetNetwork().eval()
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True), strict=True)
            self.model = model
        if self.device != str(device):
            self.model.to(device)
            self.device = str(device)

    @staticmethod
    def _sample(image):
        # Exact training/test preprocessing: sRGB 0..1, no ImageNet normalization.
        return np.array(image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), copy=True)

    @torch.inference_mode()
    def plan(self, source, reference, device):
        with self.lock:
            samples = [self._sample(image) for image in (source, reference)]
            keys = [hashlib.sha256(sample.tobytes()).hexdigest() for sample in samples]
            missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
            if missing:
                self._load(device)
                batch = np.stack([samples[keys.index(key)] for key in missing])
                tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device, torch.float32) / 255
                transforms = self.model(tensor).cpu().numpy()
                if not np.isfinite(transforms).all():
                    raise ValueError('AI 색감 분석 결과가 유효하지 않습니다.')
                for key, transform in zip(missing, transforms):
                    self.cache[key] = transform
            # Network order: r (style), d (normalization). Column-vector RGB.
            matrix = self.cache[keys[1]][0] @ self.cache[keys[0]][1]
            # A photo matched against itself should be an exact identity.
            if keys[0] == keys[1]:
                matrix = np.eye(3, dtype=np.float32)
            for key in keys:
                self.cache.move_to_end(key)
            while len(self.cache) > 64:
                self.cache.popitem(last=False)
            if not np.isfinite(matrix).all() or np.abs(matrix).max() > 32:
                raise ValueError('AI 색 변환 범위를 벗어났습니다. 다른 기준 사진을 사용해 주세요.')
            return {'version': 3, 'method': 'neural', 'model': MODEL_ID, 'model_sha256': MODEL_SHA256,
                    'matrix': matrix.tolist(), 'strength': 100., 'brightness': True,
                    'device': self.device or str(device)}
