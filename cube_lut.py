"""Strict parser for standard .cube 3D color lookup tables."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CubeLut:
    title: str
    size: int
    values: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray


def load_cube(path):
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != '.cube':
        raise ValueError('표준 .cube 3D LUT 파일을 선택해 주세요.')
    try:
        if source.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('LUT 파일은 16MB 이하여야 합니다.')
        lines = source.read_text(encoding='utf-8-sig').splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError('LUT 파일을 UTF-8 텍스트로 읽을 수 없습니다.') from exc
    except OSError as exc:
        raise ValueError('LUT 파일을 읽을 수 없습니다: ' + str(source)) from exc

    title = source.stem
    size = None
    domain_min = np.array([0., 0., 0.], dtype=np.float32)
    domain_max = np.array([1., 1., 1.], dtype=np.float32)
    values = []
    for number, raw in enumerate(lines, 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        keyword = parts[0].upper()
        try:
            if keyword == 'TITLE':
                title = line[len(parts[0]):].strip().strip('"') or title
            elif keyword == 'LUT_3D_SIZE':
                if len(parts) != 2 or size is not None:
                    raise ValueError
                size = int(parts[1])
            elif keyword == 'LUT_1D_SIZE':
                raise ValueError('1D LUT는 지원하지 않습니다. 3D .cube 파일을 선택해 주세요.')
            elif keyword in {'DOMAIN_MIN', 'DOMAIN_MAX'}:
                if len(parts) != 4:
                    raise ValueError
                domain = np.array([float(value) for value in parts[1:]], dtype=np.float32)
                if keyword == 'DOMAIN_MIN':
                    domain_min = domain
                else:
                    domain_max = domain
            elif len(parts) == 3:
                values.append([float(value) for value in parts])
            else:
                raise ValueError
        except ValueError as exc:
            if str(exc).startswith('1D LUT'):
                raise
            raise ValueError(f'LUT {number}번째 줄의 형식이 올바르지 않습니다.') from exc

    if size is None or not 2 <= size <= 65:
        raise ValueError('LUT_3D_SIZE는 2~65 범위여야 합니다.')
    if len(values) != size ** 3:
        raise ValueError(f'LUT 색상 데이터가 {size ** 3:,}개 필요하지만 {len(values):,}개입니다.')
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all() or not np.isfinite(domain_min).all() or not np.isfinite(domain_max).all():
        raise ValueError('LUT에 유효하지 않은 숫자가 있습니다.')
    if np.any(domain_max <= domain_min):
        raise ValueError('LUT DOMAIN_MAX는 DOMAIN_MIN보다 커야 합니다.')
    # .cube rows change red fastest, then green, then blue.
    tensor = array.reshape(size, size, size, 3).transpose(3, 0, 1, 2).copy()
    return CubeLut(title, size, tensor, domain_min, domain_max)
