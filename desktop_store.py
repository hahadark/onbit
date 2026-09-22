"""Portable desktop session storage; selected originals are referenced, never changed."""
from pathlib import Path
import json
import os
import sys
import uuid
from engine import settings, SUPPORTED_EXTENSIONS

ASSETS = Path(__file__).resolve().parent
APP_HOME = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else ASSETS


def lut_strength(value=0):
    if isinstance(value, bool):
        raise ValueError('LUT 강도 형식이 올바르지 않습니다.')
    value = int(value)
    if not 0 <= value <= 100:
        raise ValueError('LUT 강도는 0~100 범위여야 합니다.')
    return value


def storage_dir():
    explicit = os.environ.get('ONBIT_DATA_DIR')
    if explicit:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=True)
        return path
    portable = APP_HOME / 'onbit-data'
    try:
        portable.mkdir(exist_ok=True)
        probe = portable / f'.write-{uuid.uuid4().hex}'
        with probe.open('x'):
            pass
        probe.unlink()
        return portable
    except OSError:
        path = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'Onbit'
        path.mkdir(parents=True, exist_ok=True)
        return path


class Library:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else storage_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'library.json'
        self.items = []
        self.luts = []
        self.lut_favorites = []
        self.reference_path = ''
        portable = self.directory == APP_HOME / 'onbit-data'
        self.output_dir = str((APP_HOME if portable else self.directory) / 'exports')
        self.error = ''
        try:
            if self.path.exists():
                state = json.loads(self.path.read_text(encoding='utf-8'))
                self.output_dir = state.get('output_dir', self.output_dir)
                self.luts = [str(path) for path in state.get('luts', []) if isinstance(path, str) and path]
                self.lut_favorites = [str(path) for path in state.get('lut_favorites', [])
                                      if isinstance(path, str) and path]
                reference = state.get('reference_path', '')
                self.reference_path = reference if isinstance(reference, str) else ''
                for record in state.get('items', []):
                    self.items.append({'path': record['path'], 'name': record.get('name', Path(record['path']).name),
                                       'settings': settings(record.get('settings')), 'checked': bool(record.get('checked', True)),
                                       'lut_path': str(record.get('lut_path', '')),
                                       'lut_strength': lut_strength(record.get('lut_strength', 0)),
                                       'match': record['match'] if isinstance(record.get('match'), dict) else None})
                for record in self.items:
                    self.remember_lut(record['lut_path'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error = f'이전 사진 목록을 불러오지 못했습니다: {exc}'

    def add(self, paths):
        known = {os.path.normcase(str(Path(item['path']).resolve())) for item in self.items}
        added = []
        for source in paths:
            path = Path(source).resolve()
            key = os.path.normcase(str(path))
            if key in known or not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            item = {'path': str(path), 'name': path.name, 'settings': settings(), 'checked': True,
                    'lut_path': '', 'lut_strength': 0, 'match': None}
            self.items.append(item)
            added.append(item)
            known.add(key)
        return added

    def remember_lut(self, path):
        """Keep one entry per LUT file in the shared LUT list; returns True when added."""
        if not path:
            return False
        key = os.path.normcase(str(Path(path).resolve()))
        if any(os.path.normcase(str(Path(known).resolve())) == key for known in self.luts):
            return False
        self.luts.append(str(Path(path).resolve()))
        return True

    def is_favorite_lut(self, path):
        key = os.path.normcase(path)
        return any(os.path.normcase(known) == key for known in self.lut_favorites)

    def toggle_favorite_lut(self, path):
        """Star or unstar a LUT; returns the new favorite state."""
        if self.is_favorite_lut(path):
            key = os.path.normcase(path)
            self.lut_favorites = [known for known in self.lut_favorites if os.path.normcase(known) != key]
            return False
        self.lut_favorites.append(path)
        return True

    def save(self):
        temporary = self.directory / f'.library-{uuid.uuid4().hex}.tmp'
        try:
            with temporary.open('x', encoding='utf-8') as handle:
                json.dump({'version': 3, 'output_dir': self.output_dir, 'luts': self.luts,
                       'lut_favorites': self.lut_favorites, 'reference_path': self.reference_path,
                       'items': self.items}, handle, ensure_ascii=False, indent=2)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
