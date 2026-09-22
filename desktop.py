"""Onbit: a native Qt Widgets photo editor, with no browser or HTTP server."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from PIL import Image
from PySide6.QtCore import Qt, QSize, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QListView, QMessageBox, QPushButton, QScrollArea, QSlider, QSplitter,
                               QSizePolicy, QSpinBox, QStackedWidget, QTabWidget, QVBoxLayout, QWidget)

from cube_lut import load_cube
from engine import Engine, LIMITS, SUPPORTED_EXTENSIONS, analyze, read_exif, read_image, settings
from batch import BatchItem, file_key, resolve_match
from desktop_store import ASSETS, APP_HOME, Library
from desktop_viewer import Histogram, PhotoViewer, to_qimage
from desktop_batch import BatchDialog

def _wheel_safe(base):
    """An input that never takes the mouse wheel, so scrolling the panel cannot change values."""
    class WheelSafe(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def wheelEvent(self, event):
            event.ignore()  # handed on to the scroll area

    WheelSafe.__name__ = f'WheelSafe{base.__name__}'
    return WheelSafe


Slider = _wheel_safe(QSlider)
DoubleSpin = _wheel_safe(QDoubleSpinBox)
IntSpin = _wheel_safe(QSpinBox)

FILE_FILTER = '사진 (*.jpg *.jpeg *.mpo *.png *.webp *.tif *.tiff *.bmp *.heic *.heif *.hif)'
LABELS = {'ai': 'AI 적용 강도', 'exposure': '노출', 'contrast': '대비', 'highlights': '하이라이트',
          'shadows': '그림자', 'temperature': '색온도', 'tint': '색조', 'saturation': '채도',
          'softness': '부드러운 톤', 'skin': '피부 보정', 'gradation': '계조 보완'}

STYLE = '''
QWidget { background: #1b201b; color: #e2e8da; font-family: "Malgun Gothic"; font-size: 12px; }
QMainWindow { background: #141914; }
QLabel, QCheckBox { background: transparent; }
QLabel#brand { font-size: 26px; font-weight: 700; color: #d7e4bc; }
QLabel#subtitle { color: #8f9f80; font-size: 11px; }
QLabel#muted { color: #9da992; font-size: 11px; }
QLabel#heading { font-size: 14px; font-weight: 600; color: #d8e1cb; padding: 4px 0; }
QLabel#dialogTitle { font-size: 21px; font-weight: 600; color: #d9e6c1; }
QFrame#card { border: 1px solid #4b5d3e; border-radius: 9px; background: #2b3725; }
QPushButton { background: #2c3628; border: 1px solid #46513c; border-radius: 6px; padding: 9px 13px; }
QPushButton:hover { background: #3a4831; border-color: #758865; }
QPushButton:pressed, QPushButton:checked { background: #526440; color: #f0f5e7; }
QPushButton:disabled { color: #69745f; background: #232a20; border-color: #343f2d; }
QPushButton#primary { background: #cbdcaf; color: #27371c; font-weight: 600; border-color: #cbdcaf; }
QPushButton#primary:hover { background: #deebc8; }
QPushButton#primary:disabled { background: #5b6a49; color: #8e9e7b; }
QPushButton#small { padding: 5px 8px; font-size: 11px; }
QPushButton#link { border: none; background: transparent; color: #a7bd8d; padding: 5px; }
QListWidget { background: #1c231b; border: none; outline: none; padding: 4px; }
QListWidget::item { color: #d9e4ca; border: 1px solid transparent; border-radius: 6px; padding: 7px; margin-bottom: 4px; }
QListWidget::item:selected { border: 1px solid #8ea875; background: #35432b; }
QListWidget::item:hover { background: #2a3624; }
QTabWidget::pane { border: none; }
QTabBar { qproperty-drawBase: 0; }
QTabBar::tab { background: transparent; color: #9da992; padding: 8px 9px; margin-right: 2px;
               border: none; border-bottom: 2px solid transparent; }
QTabBar::tab:hover { color: #d8e1cb; }
QTabBar::tab:selected { color: #eef3e6; font-weight: 600; border-bottom: 2px solid #b8cf98; }
QListWidget#lutList { background: #232c20; border: 1px solid #3c4a33; border-radius: 6px; }
QListWidget#lutList::item { padding: 4px 7px; margin-bottom: 1px; }
QSplitter::handle { background: #303a2a; width: 1px; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #1c241a; width: 8px; }
QScrollBar::handle:vertical { background: #4d5c3e; border-radius: 3px; min-height: 25px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QSlider::groove:horizontal { height: 4px; background: #47553b; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #8caa70; border-radius: 2px; }
QSlider::handle:horizontal { background: #d3e1bb; border: 2px solid #aabf92; width: 11px; height: 11px; border-radius: 7px; margin: -5px 0; }
QSlider:disabled { background: transparent; }
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox { border: 1px solid #49543d; border-radius: 5px; padding: 7px; background: #242e20; selection-background-color: #718b53; }
QDoubleSpinBox { font-size: 10px; padding: 3px; }
QComboBox::drop-down { border: none; width: 23px; }
QComboBox QAbstractItemView { selection-background-color: #52693c; border: 1px solid #62764b; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator, QListWidget::indicator { width: 14px; height: 14px; border: 1px solid #657954; border-radius: 3px; background: #27321f; }
QCheckBox::indicator:checked, QListWidget::indicator:checked { background: #b5cd91; border: 2px solid #d9e7c5; }
QTableWidget { gridline-color: #35432b; border: 1px solid #3b492f; background: #1b2418; alternate-background-color: #222d1d; selection-background-color: #465e32; }
QHeaderView::section { background: #2f3c27; border: none; padding: 9px; color: #c1d1ae; }
QProgressBar { border: 1px solid #4b5d3c; border-radius: 5px; text-align: center; background: #1a2416; height: 22px; }
QProgressBar::chunk { background: #688848; border-radius: 4px; }
QStatusBar { border-top: 1px solid #37422e; background: #20291c; color: #a4b893; }
QMenuBar, QMenu { background: #222b1d; color: #c4d4b4; }
QMenu::item:selected { background: #4d613c; }
QToolTip { color: #e4edd7; background: #35452b; border: 1px solid #859d6d; padding: 6px; }
'''


class EngineLoader(QThread):
    ready = Signal(object)
    error = Signal(str)

    def run(self):
        try:
            self.ready.emit(Engine())
        except Exception as exc:
            self.error.emit(str(exc))


class PreviewWorker(QThread):
    result = Signal(object)
    error = Signal(int, str)

    def __init__(self, engine, path, adjustments, version, cache, parent, lut_path='', lut_strength=0, match=None):
        super().__init__(parent)
        self.engine, self.path, self.adjustments, self.version, self.cache = engine, path, adjustments, version, cache
        self.lut_path, self.lut_strength, self.match = lut_path, lut_strength, match

    def run(self):
        try:
            start = time.perf_counter()
            resolved = None
            key = file_key(self.path)
            if self.cache and self.cache['key'] == key:
                original, preview, analysis = self.cache['original'], self.cache['preview'], self.cache['analysis']
                metadata = self.cache.get('exif', {})
            else:
                original = read_image(self.path)
                if file_key(self.path) != key:
                    raise ValueError('원본 파일이 변경되었습니다. 다시 선택해 주세요.')
                preview = original.copy()
                preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                analysis = analyze(original)
                metadata = read_exif(self.path)
            if self.isInterruptionRequested():
                return
            match = self.match
            if match and not match.get('plan'):
                # Pending colour match: measure only this photo, now that it is being viewed.
                try:
                    match = resolved = resolve_match(self.engine, self.path, self.adjustments, self.lut_path,
                                                     self.lut_strength, match, original)
                except Exception:
                    match = None
            output = self.engine.render(preview, self.adjustments, key, lut_image=original,
                                        custom_lut_path=self.lut_path, custom_lut_strength=self.lut_strength,
                                        match=match)
            self.result.emit({'version': self.version, 'path': self.path, 'before': to_qimage(preview),
                              'after': to_qimage(output), 'size': original.size, 'analysis': analysis,
                              'exif': metadata,
                              'faces': self.engine.face_count(original, key),
                              'seconds': time.perf_counter()-start, 'status': self.engine.status(),
                              'resolved_match': resolved,
                              'cache': {'key': key, 'original': original, 'preview': preview,
                                        'analysis': analysis, 'exif': metadata}})
        except Exception as exc:
            self.error.emit(self.version, str(exc))


class ThumbnailWorker(QThread):
    result = Signal(str, object, str)

    def __init__(self, paths, parent):
        super().__init__(parent)
        self.paths = paths

    def run(self):
        for path in self.paths:
            if self.isInterruptionRequested():
                return
            try:
                image = read_image(path)
                image.thumbnail((160, 110), Image.Resampling.LANCZOS)
                self.result.emit(path, to_qimage(image), '')
            except Exception as exc:
                self.result.emit(path, None, str(exc))


class DetailWorker(QThread):
    """Render the zoomed-in area of the photo at full resolution."""
    result = Signal(object)

    def __init__(self, engine, request, parent):
        super().__init__(parent)
        self.engine, self.request = engine, request

    def run(self):
        request = self.request
        try:
            crop = request['original'].crop(request['box'])
            if crop.size != request['size']:
                crop = crop.resize(request['size'], Image.Resampling.LANCZOS)
            after = self.engine.render(crop, request['settings'], request['key'], lut_image=request['original'],
                                       custom_lut_path=request['lut_path'], custom_lut_strength=request['lut_strength'],
                                       match=request['match'], region=request['box'])
        except Exception:
            return  # the preview stays visible; a detail view is optional
        if not self.isInterruptionRequested():
            self.result.emit({'path': request['path'], 'version': request['version'], 'region': request['region'],
                              'before': to_qimage(crop), 'after': to_qimage(after)})


class RecommendWorker(QThread):
    result = Signal(object)
    error = Signal(str)

    def __init__(self, engine, path, cache, version, parent):
        super().__init__(parent)
        self.engine, self.path, self.cache, self.version = engine, path, cache, version

    def run(self):
        try:
            if file_key(self.path) != self.cache['key']:
                raise ValueError('원본 파일이 변경되었습니다. 사진을 다시 선택해 주세요.')
            p = self.engine.recommend(self.cache['original'], self.cache['key'])
            self.result.emit({'path': self.path, 'version': self.version, 'settings': p})
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, library=None):
        super().__init__()
        self.setWindowTitle('온빛 · Photo Tone — 로컬 AI 사진 스튜디오')
        self.setWindowIcon(QIcon(str(ASSETS / 'assets/onbit.ico')))
        self.resize(1380, 900)
        self.setMinimumSize(1080, 700)
        self.setAcceptDrops(True)
        self.library = library or Library()
        self.engine = None
        self.preview_worker = None
        self.auto_worker = None
        self.detail_worker = None
        self.detail_pending = None
        self.thumb_worker = None
        self.thumb_pending = []
        self.preview_version = 0
        self.preview_pending = False
        self.preview_cache = None
        self.displayed_path = None
        self.batch_dialog = None
        self.closing = False
        self.syncing = False
        self.controls = {}
        self.list_items = {}
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_library)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.start_preview)
        self.build_ui()
        self.load_list()
        self.set_controls_enabled(False)
        self.loader = EngineLoader(self)
        self.loader.ready.connect(self.engine_ready)
        self.loader.error.connect(self.engine_failed)
        self.loader.start()
        if self.library.error:
            self.statusBar().showMessage(self.library.error)

    def build_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(24, 16, 24, 16)
        brand = QLabel('◒  온빛')
        brand.setObjectName('brand')
        header.addWidget(brand)
        sub = QLabel('PHOTO TONE  /  DESKTOP')
        sub.setObjectName('subtitle')
        header.addWidget(sub)
        header.addStretch()
        self.add_button = QPushButton('＋ 사진 열기')
        self.add_button.setObjectName('openPhotos')
        self.add_button.clicked.connect(self.open_files)
        header.addWidget(self.add_button)
        folder = QPushButton('폴더 가져오기')
        folder.clicked.connect(self.open_folder)
        header.addWidget(folder)
        self.export_button = QPushButton('현재 사진 저장')
        self.export_button.clicked.connect(lambda: self.open_batch(single=True))
        header.addWidget(self.export_button)
        self.batch_button = QPushButton('일괄 작업')
        self.batch_button.setObjectName('primary')
        self.batch_button.clicked.connect(lambda: self.open_batch(single=False))
        header.addWidget(self.batch_button)
        outer.addLayout(header)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 12)
        self.count_label = QLabel('라이브러리')
        self.count_label.setObjectName('heading')
        left_layout.addWidget(self.count_label)
        view_row = QHBoxLayout()
        self.thumbnail_button = QPushButton('▦ 썸네일')
        self.thumbnail_button.setCheckable(True)
        self.thumbnail_button.setChecked(True)
        self.thumbnail_button.clicked.connect(lambda: self.set_library_view(True))
        view_row.addWidget(self.thumbnail_button)
        self.list_button = QPushButton('☷ 목록')
        self.list_button.setCheckable(True)
        self.list_button.clicked.connect(lambda: self.set_library_view(False))
        view_row.addWidget(self.list_button)
        left_layout.addLayout(view_row)
        selection = QHBoxLayout()
        for text, flag in [('전체 체크', True), ('해제', False)]:
            button = QPushButton(text)
            button.setObjectName('small')
            button.clicked.connect(lambda checked=False, value=flag: self.check_all(value))
            selection.addWidget(button)
        left_layout.addLayout(selection)
        self.list = QListWidget()
        self.list.setObjectName('photoLibrary')
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setWordWrap(True)
        self.list.currentRowChanged.connect(self.photo_selected)
        self.list.itemChanged.connect(self.check_changed)
        left_layout.addWidget(self.list, 1)
        self.set_library_view(True)
        library_actions = QHBoxLayout()
        remove = QPushButton('선택 사진 빼기')
        remove.setObjectName('small')
        remove.setToolTip('선택한 항목만 라이브러리에서 빼며 사진 파일은 삭제하지 않습니다.')
        remove.clicked.connect(self.remove_selected)
        library_actions.addWidget(remove)
        clear = QPushButton('전체 비우기')
        clear.setObjectName('small')
        clear.setToolTip('사진 파일은 삭제하지 않습니다.')
        clear.clicked.connect(self.clear_library)
        library_actions.addWidget(clear)
        left_layout.addLayout(library_actions)
        apply_selected = QPushButton('현재 보정을 체크 사진에 적용')
        apply_selected.clicked.connect(self.apply_checked)
        left_layout.addWidget(apply_selected)
        demo = QPushButton('샘플 사진 열기')
        demo.setObjectName('link')
        demo.clicked.connect(lambda: self.add_paths([str(ASSETS / 'samples/demo.jpg')]))
        left_layout.addWidget(demo)
        footer = QLabel('체크한 사진만 일괄 처리합니다.\n원본 파일은 그대로 보존됩니다.')
        footer.setObjectName('muted')
        left_layout.addWidget(footer)
        splitter.addWidget(left)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setContentsMargins(18, 12, 18, 12)
        self.photo_label = QLabel('사진을 열어 시작하세요')
        self.photo_label.setObjectName('muted')
        self.photo_label.setWordWrap(True)
        bar.addWidget(self.photo_label, 1)
        self.compare_button = QPushButton('◧ 비교')
        self.compare_button.setCheckable(True)
        self.compare_button.toggled.connect(self.toggle_compare)
        bar.addWidget(self.compare_button)
        self.original_button = QPushButton('원본')
        self.original_button.setCheckable(True)
        self.original_button.toggled.connect(self.toggle_original)
        bar.addWidget(self.original_button)
        center_layout.addLayout(bar)
        self.viewer = PhotoViewer()
        self.viewer.setObjectName('photoViewer')
        center_layout.addWidget(self.viewer, 1)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(16, 10, 16, 12)
        self.preview_label = QLabel('로컬 엔진 준비 중…')
        self.preview_label.setObjectName('muted')
        bottom.addWidget(self.preview_label)
        self.exif_label = QLabel('EXIF 정보 없음')
        self.exif_label.setObjectName('muted')
        self.exif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exif_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.exif_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bottom.addWidget(self.exif_label, 1)
        minus = QPushButton('−')
        minus.setFixedWidth(35)
        minus.clicked.connect(lambda: self.viewer.change_zoom(1/1.25))
        bottom.addWidget(minus)
        self.fit_button = QPushButton('화면 맞춤')
        self.fit_button.clicked.connect(self.viewer.fit)
        self.viewer.zoom_changed.connect(self.fit_button.setText)
        self.viewer.detail_needed.connect(self.request_detail)
        bottom.addWidget(self.fit_button)
        plus = QPushButton('＋')
        plus.setFixedWidth(35)
        plus.clicked.connect(lambda: self.viewer.change_zoom(1.25))
        bottom.addWidget(plus)
        center_layout.addLayout(bottom)
        help_text = QLabel('휠: 확대·축소  ·  드래그: 이동  ·  더블클릭: 화면 맞춤  ·  Space: 원본 보기')
        help_text.setObjectName('muted')
        help_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_text.setContentsMargins(4, 0, 4, 14)
        center_layout.addWidget(help_text)
        splitter.addWidget(center)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 14, 12, 6)
        right_layout.setSpacing(6)
        row = QHBoxLayout()
        row.setContentsMargins(7, 0, 0, 0)
        heading = QLabel('빛과 색 조정')
        heading.setObjectName('heading')
        row.addWidget(heading)
        row.addStretch()
        reset = QPushButton('초기화 ↺')
        reset.setObjectName('link')
        reset.clicked.connect(self.reset_current)
        row.addWidget(reset)
        right_layout.addLayout(row)
        self.side_tabs = QTabWidget()
        self.side_tabs.setObjectName('sideTabs')
        self.side_tabs.setDocumentMode(True)
        self.side_tabs.setUsesScrollButtons(False)
        right_layout.addWidget(self.side_tabs, 1)

        def tab(title):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(7, 14, 7, 18)
            layout.setSpacing(13)
            scroll.setWidget(page)
            self.side_tabs.addTab(scroll, title)
            return layout

        def section(layout, title, keys=()):
            label = QLabel(title)
            label.setObjectName('heading')
            layout.addWidget(label)
            for key in keys:
                layout.addWidget(self.control(key))

        def note(layout, text):
            label = QLabel(text)
            label.setObjectName('muted')
            label.setWordWrap(True)
            layout.addWidget(label)

        panel = tab('기본')
        card = QFrame()
        card.setObjectName('card')
        ai_layout = QVBoxLayout(card)
        ai_layout.setContentsMargins(15, 15, 15, 15)
        ai_heading = QLabel('✦  AI 톤 어시스트')
        ai_heading.setObjectName('heading')
        ai_layout.addWidget(ai_heading)
        hint = QLabel('밝기·색 균형을 분석해 자연스럽게 다듬어요.')
        hint.setObjectName('muted')
        ai_layout.addWidget(hint)
        self.auto_button = QPushButton('AI로 자연스럽게 보정')
        self.auto_button.setObjectName('primary')
        self.auto_button.clicked.connect(self.auto_correct)
        ai_layout.addWidget(self.auto_button)
        ai_layout.addWidget(self.control('ai'))
        panel.addWidget(card)
        presets = QHBoxLayout()
        for label, name in [('내추럴', 'natural'), ('소프트', 'soft'), ('웜 필름', 'warm')]:
            button = QPushButton(label)
            button.setObjectName('small')
            button.clicked.connect(lambda checked=False, value=name: self.preset(value))
            presets.addWidget(button)
        panel.addLayout(presets)
        section(panel, '밝기', ['exposure', 'contrast', 'highlights', 'shadows', 'gradation'])
        note(panel, '계조 보완은 하늘 층짐을 펴고 하이라이트를 필름처럼 부드럽게 넘깁니다. '
                    'HDR 게인맵이 있는 사진은 날아간 밝은 부분의 계조도 되살립니다.')
        panel.addStretch()

        panel = tab('색감·LUT')
        match_card = QFrame()
        match_card.setObjectName('card')
        match_layout = QVBoxLayout(match_card)
        match_layout.setContentsMargins(15, 13, 15, 13)
        match_layout.setSpacing(8)
        match_heading = QLabel('◎  색감 매칭')
        match_heading.setObjectName('heading')
        match_layout.addWidget(match_heading)
        self.match_reference_label = QLabel('기준 사진: 없음')
        self.match_reference_label.setObjectName('muted')
        self.match_reference_label.setWordWrap(True)
        match_layout.addWidget(self.match_reference_label)
        self.match_reference_button = QPushButton('★ 현재 사진을 기준으로 지정')
        self.match_reference_button.clicked.connect(self.set_match_reference)
        match_layout.addWidget(self.match_reference_button)
        self.match_method = _wheel_safe(QComboBox)()
        self.match_method.addItem('AI 색감 매칭 · Neural-Preset', 'neural')
        self.match_method.addItem('기존 방식 · 밝기·색 분포 맞춤', 'statistics')
        self.match_method.setToolTip('선택한 방식은 아래 맞추기 버튼을 누를 때 적용됩니다.')
        match_layout.addWidget(self.match_method)
        match_buttons = QHBoxLayout()
        self.match_current_button = QPushButton('현재 사진 맞추기')
        self.match_current_button.setObjectName('primary')
        self.match_current_button.clicked.connect(lambda: self.match_photos(False))
        match_buttons.addWidget(self.match_current_button)
        self.match_checked_button = QPushButton('체크한 사진 맞추기')
        self.match_checked_button.clicked.connect(lambda: self.match_photos(True))
        match_buttons.addWidget(self.match_checked_button)
        match_layout.addLayout(match_buttons)
        match_state_row = QHBoxLayout()
        self.match_state_label = QLabel('이 사진: 매칭 없음')
        self.match_state_label.setObjectName('muted')
        self.match_state_label.setWordWrap(True)
        match_state_row.addWidget(self.match_state_label, 1)
        self.match_clear_button = QPushButton('매칭 해제')
        self.match_clear_button.setObjectName('small')
        self.match_clear_button.clicked.connect(self.clear_match)
        match_state_row.addWidget(self.match_clear_button)
        match_layout.addLayout(match_state_row)
        match_strength_row = QHBoxLayout()
        match_strength_row.addWidget(QLabel('매칭 강도'))
        match_strength_row.addStretch()
        self.match_strength_spin = IntSpin()
        self.match_strength_spin.setRange(0, 100)
        self.match_strength_spin.setSuffix(' %')
        self.match_strength_spin.setFixedWidth(82)
        self.match_strength_spin.valueChanged.connect(lambda value: self.adjust_match('strength', int(value)))
        match_strength_row.addWidget(self.match_strength_spin)
        match_layout.addLayout(match_strength_row)
        self.match_strength_slider = Slider(Qt.Orientation.Horizontal)
        self.match_strength_slider.setRange(0, 100)
        self.match_strength_slider.valueChanged.connect(lambda value: self.adjust_match('strength', int(value)))
        match_layout.addWidget(self.match_strength_slider)
        match_options = QHBoxLayout()
        self.match_brightness_check = QCheckBox('밝기도 맞추기')
        self.match_brightness_check.toggled.connect(lambda checked: self.adjust_match('brightness', checked))
        match_options.addWidget(self.match_brightness_check)
        self.match_skin_check = QCheckBox('피부색 보호')
        self.match_skin_check.setToolTip('AI 매칭에서 피부색을 보호합니다. 밝기도 맞추기를 켜면 피부 밝기는 조정됩니다.')
        self.match_skin_check.toggled.connect(lambda checked: self.adjust_match('protect_skin', checked))
        match_options.addWidget(self.match_skin_check)
        match_options.addStretch()
        match_layout.addLayout(match_options)
        self.match_group_check = QCheckBox('매칭된 사진 모두 함께 조절')
        self.match_group_check.setToolTip('같은 기준 사진에 맞춘 사진들의 강도와 옵션을 한꺼번에 바꿉니다. '
                                          '끄면 현재 사진만 바뀝니다.')
        self.match_group_check.setChecked(True)
        self.match_group_check.toggled.connect(lambda _checked: (self.sync_controls(), self.set_controls_enabled(True)))
        match_layout.addWidget(self.match_group_check)
        match_note = QLabel('AI가 대상·기준 사진을 각각 분석해 색 변환을 추정합니다. 현재 보정과 LUT가 반영된 '
                            '색감을 기준으로 로컬 GPU에서 계산합니다. 기준을 수정하면 다시 맞추기를 누르세요. '
                            '장면이 크게 다르면 강도를 낮추세요.')
        match_note.setObjectName('muted')
        match_note.setWordWrap(True)
        match_layout.addWidget(match_note)
        panel.addWidget(match_card)
        section(panel, '색감', ['temperature', 'tint', 'saturation'])
        lut_card = QFrame()
        lut_card.setObjectName('card')
        lut_layout = QVBoxLayout(lut_card)
        lut_layout.setContentsMargins(15, 13, 15, 13)
        lut_heading = QLabel('LUT 색감')
        lut_heading.setObjectName('heading')
        lut_layout.addWidget(lut_heading)
        lut_buttons = QHBoxLayout()
        self.lut_load_button = QPushButton('＋ .cube LUT 불러오기 (여러 개)')
        self.lut_load_button.clicked.connect(self.choose_lut)
        lut_buttons.addWidget(self.lut_load_button, 1)
        self.lut_clear_button = QPushButton('해제')
        self.lut_clear_button.setObjectName('small')
        self.lut_clear_button.clicked.connect(self.clear_lut)
        lut_buttons.addWidget(self.lut_clear_button)
        lut_layout.addLayout(lut_buttons)
        self.lut_list = QListWidget()
        self.lut_list.setObjectName('lutList')
        self.lut_list.setFixedHeight(150)
        self.lut_list.setToolTip('LUT를 클릭하거나 ↑↓ 키로 옮기면 현재 사진에 바로 미리 적용됩니다.')
        self.lut_list.currentItemChanged.connect(self.pick_lut)
        lut_layout.addWidget(self.lut_list)
        self.lut_list.itemDoubleClicked.connect(lambda _item: self.toggle_lut_favorite())
        lut_actions = QHBoxLayout()
        self.lut_favorite_button = QPushButton('☆ 즐겨찾기')
        self.lut_favorite_button.setObjectName('small')
        self.lut_favorite_button.setToolTip('선택한 LUT를 즐겨찾기에 추가하거나 뺍니다. 목록에서 더블클릭해도 됩니다.')
        self.lut_favorite_button.clicked.connect(self.toggle_lut_favorite)
        lut_actions.addWidget(self.lut_favorite_button)
        self.lut_favorites_only = QCheckBox('즐겨찾기만')
        self.lut_favorites_only.toggled.connect(lambda _checked: self.sync_controls())
        lut_actions.addWidget(self.lut_favorites_only)
        lut_actions.addStretch(1)
        self.lut_remove_button = QPushButton('목록에서 빼기')
        self.lut_remove_button.setObjectName('small')
        self.lut_remove_button.clicked.connect(self.remove_lut)
        lut_actions.addWidget(self.lut_remove_button)
        lut_layout.addLayout(lut_actions)
        self.lut_label = QLabel('선택된 LUT 없음')
        self.lut_label.setObjectName('muted')
        self.lut_label.setWordWrap(True)
        lut_layout.addWidget(self.lut_label)
        lut_strength_row = QHBoxLayout()
        lut_strength_row.addWidget(QLabel('LUT 강도'))
        self.lut_strength_spin = IntSpin()
        self.lut_strength_spin.setRange(0, 100)
        self.lut_strength_spin.setSuffix(' %')
        self.lut_strength_spin.setFixedWidth(82)
        self.lut_strength_spin.valueChanged.connect(self.adjust_lut_strength)
        lut_strength_row.addWidget(self.lut_strength_spin)
        lut_layout.addLayout(lut_strength_row)
        self.lut_strength_slider = Slider(Qt.Orientation.Horizontal)
        self.lut_strength_slider.setRange(0, 100)
        self.lut_strength_slider.valueChanged.connect(self.adjust_lut_strength)
        lut_layout.addWidget(self.lut_strength_slider)
        lut_hint = QLabel('AI 보정 뒤에 적용됩니다. 화사한 LUT는 40~70%부터 조절해 보세요.')
        lut_hint.setObjectName('muted')
        lut_hint.setWordWrap(True)
        lut_layout.addWidget(lut_hint)
        panel.addWidget(lut_card)
        panel.addStretch()

        panel = tab('인물·마무리')
        section(panel, '인물', ['skin'])
        note(panel, 'AI가 정면·측면 얼굴을 찾아 피부만 골라 부드럽고 맑게 다듬습니다. '
                    '눈·눈썹·입술·머리카락은 건드리지 않습니다. 원본과 비교하며 강도를 조절하세요.')
        section(panel, '마무리', ['softness'])
        note(panel, '부드러운 톤은 명암과 채도를 은은하게 조정합니다.')
        panel.addStretch()

        panel = tab('분석')
        hist_title = QLabel('원본 색상 분석')
        hist_title.setObjectName('heading')
        panel.addWidget(hist_title)
        self.histogram = Histogram()
        panel.addWidget(self.histogram)
        self.analysis_label = QLabel('사진을 선택하면 분석 결과가 표시됩니다.')
        self.analysis_label.setObjectName('muted')
        self.analysis_label.setWordWrap(True)
        panel.addWidget(self.analysis_label)
        apply = QPushButton('현재 설정을 체크한 사진에 적용')
        apply.clicked.connect(self.apply_checked)
        panel.addWidget(apply)
        panel.addStretch()
        splitter.addWidget(right_panel)
        splitter.setSizes([240, 760, 370])
        splitter.setStretchFactor(1, 1)
        left.setMinimumWidth(200)
        right_panel.setMinimumWidth(320)
        self.workspace = splitter
        self.content_stack = QStackedWidget()
        self.content_stack.addWidget(self.workspace)
        outer.addWidget(self.content_stack, 1)
        self.setCentralWidget(root)
        self.device_label = QLabel('로컬 AI 엔진 시작 중…')
        self.statusBar().addPermanentWidget(self.device_label)
        menu = self.menuBar().addMenu('파일')
        for title, key, callback in [('사진 열기…', 'Ctrl+O', self.open_files), ('폴더 가져오기…', 'Ctrl+Shift+O', self.open_folder),
                                      ('현재 사진 저장…', 'Ctrl+S', lambda: self.open_batch(True)),
                                      ('체크한 사진 일괄 처리…', 'Ctrl+E', lambda: self.open_batch(False))]:
            action = QAction(title, self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            menu.addAction(action)
        library_menu = self.menuBar().addMenu('라이브러리')
        remove_action = QAction('선택 사진 빼기', self)
        remove_action.setShortcut(QKeySequence('Delete'))
        remove_action.triggered.connect(self.remove_selected)
        library_menu.addAction(remove_action)
        clear_action = QAction('전체 비우기', self)
        clear_action.triggered.connect(self.clear_library)
        library_menu.addAction(clear_action)
        help_menu = self.menuBar().addMenu('도움말')
        about = QAction('온빛 정보', self)
        about.triggered.connect(lambda: QMessageBox.information(self, '온빛 · Photo Tone 2',
            'Windows 데스크톱 사진 스튜디오\n로컬 AI 보정 · HEIC · 일괄 리사이즈 / 변환\n\n'
            '현재 색 처리: 8비트 sRGB. HDR / 10비트는 보존하지 않으며, 투명도는 흰 배경에 합성합니다.\n'
            'EXIF / GPS는 출력 파일에 복사하지 않습니다. 미리보기는 긴 변 1600px입니다.\n\n'
            'PySide6 · Pillow / pillow-heif · PyTorch · OpenCV · Image-Adaptive-3DLUT\n라이선스: 실행 폴더의 _internal/third_party'))
        help_menu.addAction(about)

    def control(self, key):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 5)
        layout.setSpacing(8)
        row = QHBoxLayout()
        label = QLabel(LABELS[key])
        row.addWidget(label)
        row.addStretch()
        low, high, default = LIMITS[key]
        scale = 20 if key == 'exposure' else 1
        numeric = DoubleSpin()
        numeric.setObjectName(f'value_{key}')
        numeric.setDecimals(2 if key == 'exposure' else 0)
        numeric.setSingleStep(1 / scale)
        numeric.setRange(low, high)
        numeric.setValue(default)
        numeric.setFixedWidth(82)
        numeric.setSuffix(' EV' if key == 'exposure' else '%' if key == 'ai' else '')
        numeric.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        row.addWidget(numeric)
        layout.addLayout(row)
        slider = Slider(Qt.Orientation.Horizontal)
        slider.setObjectName(f'slider_{key}')
        slider.setAccessibleName(LABELS[key])
        slider.setRange(int(low*scale), int(high*scale))
        slider.setValue(int(default*scale))
        slider.valueChanged.connect(lambda value, k=key, s=scale: self.adjust(k, value/s))
        numeric.valueChanged.connect(lambda value, k=key: self.adjust(k, value))
        layout.addWidget(slider)
        self.controls[key] = (slider, numeric, scale)
        return widget

    def current(self):
        row = self.list.currentRow()
        return self.library.items[row] if 0 <= row < len(self.library.items) else None

    def set_library_view(self, thumbnails):
        self.thumbnail_button.blockSignals(True)
        self.list_button.blockSignals(True)
        self.thumbnail_button.setChecked(thumbnails)
        self.list_button.setChecked(not thumbnails)
        self.thumbnail_button.blockSignals(False)
        self.list_button.blockSignals(False)
        if thumbnails:
            self.list.setViewMode(QListView.ViewMode.IconMode)
            self.list.setResizeMode(QListView.ResizeMode.Adjust)
            self.list.setMovement(QListView.Movement.Static)
            self.list.setFlow(QListView.Flow.LeftToRight)
            self.list.setWrapping(True)
            self.list.setIconSize(QSize(84, 63))
            self.list.setGridSize(QSize(92, 104))
        else:
            self.list.setViewMode(QListView.ViewMode.ListMode)
            self.list.setFlow(QListView.Flow.TopToBottom)
            self.list.setWrapping(False)
            self.list.setIconSize(QSize(92, 65))
            self.list.setGridSize(QSize())
        for index in range(self.list.count()):
            self.list.item(index).setSizeHint(QSize(92, 98) if thumbnails else QSize(210, 84))

    def load_list(self):
        self.list.blockSignals(True)
        self.list.clear()
        self.list_items.clear()
        for record in self.library.items:
            self.append_list_item(record)
        self.list.blockSignals(False)
        self.update_count()
        self.thumb_pending += [record['path'] for record in self.library.items]
        self.start_thumbnails()
        if self.library.items:
            self.list.setCurrentRow(0)

    def append_list_item(self, record):
        item = QListWidgetItem(('★ ' if record['path'] == self.library.reference_path else '') + record['name'])
        item.setData(Qt.ItemDataRole.UserRole, record['path'])
        item.setToolTip(record['path'])
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if record['checked'] else Qt.CheckState.Unchecked)
        item.setSizeHint(QSize(92, 98) if self.thumbnail_button.isChecked() else QSize(210, 84))
        self.list.addItem(item)
        self.list_items[record['path']] = item

    def update_count(self):
        count = sum(record['checked'] for record in self.library.items)
        self.count_label.setText(f'라이브러리  {len(self.library.items)}장')
        self.batch_button.setText(f'일괄 처리 · {count}장')
        self.batch_button.setEnabled(count > 0 and self.engine is not None)

    def check_changed(self, item):
        row = self.list.row(item)
        if 0 <= row < len(self.library.items):
            self.library.items[row]['checked'] = item.checkState() == Qt.CheckState.Checked
            self.update_count()
            self.save_timer.start(200)

    def check_all(self, checked):
        self.list.blockSignals(True)
        for index, record in enumerate(self.library.items):
            record['checked'] = checked
            self.list.item(index).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.list.blockSignals(False)
        self.update_count()
        self.save_library()

    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '사진 열기', '', FILE_FILTER)
        if paths:
            self.add_paths(paths)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, '사진 폴더 선택 — 하위 폴더도 가져옵니다')
        if path:
            self.add_paths([path])

    def add_paths(self, paths):
        expanded = []
        output_path = Path(self.library.output_dir).resolve()
        for item in paths:
            path = Path(item)
            if path.is_dir():
                for directory, subdirs, names in os.walk(path, followlinks=False):
                    subdirs[:] = [name for name in sorted(subdirs) if not name.startswith('.')
                                  and (Path(directory)/name).resolve() != output_path
                                  and name not in {'onbit-data', '_internal', '__pycache__'}]
                    expanded.extend(str(Path(directory)/name) for name in sorted(names) if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS)
            else:
                expanded.append(str(path))
        added = self.library.add(expanded)
        self.list.blockSignals(True)
        for record in added:
            self.append_list_item(record)
        self.list.blockSignals(False)
        self.update_count()
        self.save_library()
        self.thumb_pending += [record['path'] for record in added]
        self.start_thumbnails()
        if added:
            self.list.setCurrentRow(len(self.library.items)-len(added))
            self.statusBar().showMessage(f'{len(added)}장을 목록에 추가했습니다.', 6000)
        else:
            self.statusBar().showMessage('새 사진이 없습니다. 이미 추가했거나 지원하지 않는 파일입니다.', 6000)

    def start_thumbnails(self):
        if self.closing or not self.thumb_pending or self.thumb_worker is not None:
            return
        self.thumb_worker = ThumbnailWorker(self.thumb_pending, self)
        self.thumb_pending = []
        self.thumb_worker.result.connect(self.thumbnail_ready)
        self.thumb_worker.finished.connect(self.thumbnail_finished)
        self.thumb_worker.start()

    def thumbnail_ready(self, path, image, error):
        item = self.list_items.get(path)
        if item is None:
            return
        self.list.blockSignals(True)
        if image is not None:
            item.setIcon(QIcon(QPixmap.fromImage(image)))
        else:
            item.setToolTip(f'{path}\n읽기 실패: {error}')
            item.setForeground(QColor('#e6ab8c'))
        self.list.blockSignals(False)

    def thumbnail_finished(self):
        old = self.thumb_worker
        self.thumb_worker = None
        if old:
            old.deleteLater()
        self.start_thumbnails()

    def remove_selected(self, *_):
        rows = sorted({self.list.row(item) for item in self.list.selectedItems()}, reverse=True)
        if not rows and self.list.currentRow() >= 0:
            rows = [self.list.currentRow()]
        if not rows:
            return
        self.preview_version += 1
        self.list.blockSignals(True)
        next_row = min(rows)
        for row in rows:
            record = self.library.items.pop(row)
            self.list_items.pop(record['path'], None)
            self.list.takeItem(row)
        self.list.blockSignals(False)
        self.save_library()
        self.update_count()
        self.list.blockSignals(True)
        self.list.setCurrentRow(min(next_row, len(self.library.items)-1))
        self.list.blockSignals(False)
        self.photo_selected(self.list.currentRow())
        self.statusBar().showMessage(f'{len(rows)}장을 라이브러리에서 뺐습니다. 원본 파일은 그대로입니다.', 5000)

    def remove_current(self):
        self.remove_selected()

    def clear_library(self, _checked=False, confirm=True):
        if not self.library.items:
            return
        if confirm and QMessageBox.question(self, '라이브러리 전체 비우기',
                                            '라이브러리의 모든 사진을 뺄까요? 원본 파일은 삭제하지 않습니다.',
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                            QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
            return
        count = len(self.library.items)
        self.preview_version += 1
        self.list.blockSignals(True)
        self.library.items.clear()
        self.list.clear()
        self.list_items.clear()
        self.list.blockSignals(False)
        self.preview_cache = None
        self.displayed_path = None
        self.save_library()
        self.update_count()
        self.photo_selected(-1)
        self.statusBar().showMessage(f'라이브러리 {count}장을 비웠습니다. 원본 파일은 그대로입니다.', 5000)

    def photo_selected(self, row):
        record = self.current()
        self.preview_version += 1
        self.preview_pending = True
        self.set_controls_enabled(record is not None and self.engine is not None)
        self.sync_controls()
        self.compare_button.setChecked(False)
        self.original_button.setChecked(False)
        if record:
            self.photo_label.setText(record['name'])
            self.viewer.clear('사진을 읽고 있습니다…')
            self.displayed_path = None
            self.schedule_preview()
        else:
            self.viewer.clear()
            self.preview_cache = None
            self.preview_pending = False
            self.preview_label.setText('사진을 열어 시작하세요')
            self.photo_label.setText('사진을 열어 시작하세요')
            self.histogram.values = None
            self.histogram.update()
            self.analysis_label.setText('사진을 선택하면 분석 결과가 표시됩니다.')
            self.exif_label.setText('EXIF 정보 없음')
            self.exif_label.setToolTip('')

    def set_controls_enabled(self, enabled):
        for slider, numeric, _ in self.controls.values():
            slider.setEnabled(enabled)
            numeric.setEnabled(enabled)
        ready = enabled and self.engine is not None and self.engine.status()['ai_ready']
        self.auto_button.setEnabled(ready and self.auto_worker is None and self.current() is not None
                                    and self.displayed_path == self.current()['path'])
        for widget in self.controls['ai'][:2]:
            widget.setEnabled(ready)
        face_ready = enabled and self.engine is not None and self.engine.status().get('face_ai', False)
        for widget in self.controls['skin'][:2]:
            widget.setEnabled(face_ready)
        reference = self.match_reference()
        record = self.current()
        self.match_reference_button.setEnabled(enabled)
        self.match_current_button.setEnabled(bool(enabled and reference and record
                                                  and record['path'] != reference['path']))
        self.match_checked_button.setEnabled(bool(enabled and reference))
        adjustable = bool(enabled and self.match_targets())
        for widget in (self.match_strength_spin, self.match_strength_slider, self.match_brightness_check,
                       self.match_skin_check, self.match_group_check):
            widget.setEnabled(adjustable)
        self.match_clear_button.setEnabled(bool(enabled and record and record.get('match')))
        self.lut_load_button.setEnabled(enabled)
        self.lut_list.setEnabled(enabled)
        selected = self.lut_list.currentItem()
        chosen = selected.data(Qt.ItemDataRole.UserRole) if selected else ''
        self.lut_remove_button.setEnabled(bool(enabled and chosen))
        self.lut_favorite_button.setEnabled(bool(enabled and chosen))
        self.lut_favorite_button.setText('★ 즐겨찾기 해제' if chosen and self.library.is_favorite_lut(chosen) else '☆ 즐겨찾기')
        self.lut_favorites_only.setEnabled(enabled)
        has_lut = enabled and bool(self.current() and self.current().get('lut_path'))
        self.lut_clear_button.setEnabled(has_lut)
        self.lut_strength_slider.setEnabled(has_lut)
        self.lut_strength_spin.setEnabled(has_lut)
        self.export_button.setEnabled(enabled)
        self.compare_button.setEnabled(enabled)
        self.original_button.setEnabled(enabled)
        self.update_count()

    def sync_controls(self):
        self.syncing = True
        p = self.current()['settings'] if self.current() else settings()
        for key, (slider, numeric, scale) in self.controls.items():
            slider.setValue(round(p[key]*scale))
            numeric.setValue(p[key])
        record = self.current()
        lut_path = record.get('lut_path', '') if record else ''
        lut_strength = record.get('lut_strength', 0) if record else 0
        self.lut_strength_slider.setValue(lut_strength)
        self.lut_strength_spin.setValue(lut_strength)
        self.lut_label.setText(f'{Path(lut_path).name} · {lut_strength}%' if lut_path else '선택된 LUT 없음')
        self.lut_label.setToolTip(lut_path)
        self.refresh_lut_list(lut_path)
        reference = self.match_reference()
        self.match_reference_label.setText(f'기준 사진: ★ {reference["name"]}' if reference
                                           else '기준 사진: 없음 · 원하는 색감의 사진을 먼저 기준으로 지정하세요.')
        match = record.get('match') if record else None
        targets = self.match_targets()
        if match and not match.get('plan'):
            name = Path(match.get('reference', '')).name
            state = (f'이 사진: {name} 기준 · 미리보기에서 계산 중…'
                     if self.reference_look(match.get('reference', '')) else
                     f'이 사진: 기준 사진 {name}이(가) 목록에 없어 맞출 수 없습니다.')
        elif match:
            state = f'이 사진: {Path(match.get("reference", "")).name} 기준으로 매칭됨'
            plan = match.get('plan') or {}
            state += (' · AI Neural-Preset / ' + plan.get('device', '로컬')
                      if plan.get('method') == 'neural' else ' · 기존 통계 방식')
            if len(targets) > 1:
                state += f' · {len(targets)}장 함께 조절'
        elif record and reference and record['path'] == reference['path']:
            state = (f'이 사진이 기준 사진입니다 · 매칭된 사진 {len(targets)}장을 함께 조절합니다.' if targets
                     else '이 사진이 기준 사진입니다.')
        else:
            state = '이 사진: 매칭 없음'
        self.match_state_label.setText(state)
        match = match or (targets[0]['match'] if targets else None)
        strength = int(match.get('strength', 80)) if match else 80
        self.match_strength_spin.setValue(strength)
        self.match_strength_slider.setValue(strength)
        self.match_brightness_check.setChecked(bool(match.get('brightness', True)) if match else True)
        self.match_skin_check.setChecked(bool(match.get('protect_skin', True)) if match else True)
        self.syncing = False

    def match_reference(self):
        path = self.library.reference_path
        return next((record for record in self.library.items if record['path'] == path), None) if path else None

    def refresh_reference_marks(self):
        for record in self.library.items:
            item = self.list_items.get(record['path'])
            if item is not None:
                star = record['path'] == self.library.reference_path
                item.setText(('★ ' if star else '') + record['name'])

    def set_match_reference(self):
        record = self.current()
        if not record:
            return
        self.library.reference_path = record['path']
        self.refresh_reference_marks()
        self.sync_controls()
        self.set_controls_enabled(True)
        self.save_library()
        self.statusBar().showMessage(f'★ {record["name"]}을(를) 기준 사진으로 지정했습니다. '
                                     '다른 사진에서 "현재 사진 맞추기"를 누르세요.', 8000)

    def reference_look(self, reference_path):
        """The reference photo's current edits, used to measure its look for matching."""
        reference = next((r for r in self.library.items if r['path'] == reference_path), None)
        if reference is None:
            return None
        own = reference.get('match')
        return {'settings': reference['settings'].copy(), 'lut_path': reference.get('lut_path', ''),
                'lut_strength': reference.get('lut_strength', 0), 'match': own if own and own.get('plan') else None}

    def render_match(self, record):
        """The photo's match layer, with what a pending layer needs to be measured."""
        match = record.get('match')
        if not match or match.get('plan'):
            return match
        look = self.reference_look(match.get('reference', ''))
        return {**match, 'reference_look': look} if look or match.get('method') == 'neural' else None

    def match_photos(self, checked=False):
        """Assign the reference to photos; each is measured when viewed or exported."""
        reference = self.match_reference()
        if reference is None:
            self.statusBar().showMessage('먼저 기준 사진을 지정해 주세요.', 6000)
            return
        records = [r for r in self.library.items if r['checked']] if checked else [self.current()] if self.current() else []
        targets = [r for r in records if r['path'] != reference['path']]
        if not targets:
            self.statusBar().showMessage('맞출 사진이 없습니다. 기준 사진이 아닌 사진을 고르거나 체크해 주세요.', 6000)
            return
        for record in targets:
            previous = record.get('match') or {}
            record['match'] = {'reference': reference['path'], 'plan': None,
                               'method': self.match_method.currentData(), 'request_id': str(time.time_ns()),
                               'strength': int(previous.get('strength', 80)),
                               'brightness': bool(previous.get('brightness', True)),
                               'protect_skin': bool(previous.get('protect_skin', True))}
        self.save_library()
        self.sync_controls()
        self.set_controls_enabled(True)
        if self.current() in targets:
            self.schedule_preview()
        if checked:
            self.statusBar().showMessage(f'{len(targets)}장에 ★ {reference["name"]} 기준 매칭을 지정했습니다. '
                                         '사진을 열 때와 일괄 처리할 때 계산합니다.', 9000)

    def match_targets(self):
        """Photos whose match layer the strength / option controls change.

        On a matched photo: that photo, or every photo matched to the same reference
        when grouped. On the reference photo: every photo matched to it.
        """
        record = self.current()
        if not record:
            return []
        match = record.get('match')
        reference = match.get('reference', '') if match else record['path']
        if match and not self.match_group_check.isChecked():
            return [record]
        group = [r for r in self.library.items if r.get('match') and r['match'].get('reference') == reference]
        return group if match or record['path'] == self.library.reference_path else []

    def adjust_match(self, name, value):
        targets = self.match_targets()
        if self.syncing or not targets:
            return
        for record in targets:
            record['match'][name] = value
        self.sync_controls()
        self.save_timer.start(300)
        if self.current() in targets:
            self.schedule_preview()
        if name == 'strength':
            self.statusBar().showMessage(f'매칭 강도 {value}% · {len(targets)}장', 3000)

    def clear_match(self):
        record = self.current()
        if not record or not record.get('match'):
            return
        record['match'] = None
        self.sync_controls()
        self.set_controls_enabled(True)
        self.save_library()
        self.schedule_preview()

    def refresh_lut_list(self, active=''):
        """Rebuild the LUT list and highlight the current photo's LUT (caller holds self.syncing)."""
        self.lut_list.clear()
        none = QListWidgetItem('LUT 없음 (원래 색감)')
        none.setData(Qt.ItemDataRole.UserRole, '')
        self.lut_list.addItem(none)
        # Favorites first; the current photo's LUT always stays visible.
        paths = sorted(self.library.luts, key=lambda p: not self.library.is_favorite_lut(p))
        if self.lut_favorites_only.isChecked():
            paths = [p for p in paths if self.library.is_favorite_lut(p)]
        if active and os.path.normcase(active) not in {os.path.normcase(p) for p in paths}:
            paths.append(active)
        for path in paths:
            favorite = self.library.is_favorite_lut(path)
            item = QListWidgetItem(('★ ' if favorite else '') + Path(path).stem)
            if favorite:
                item.setForeground(QColor('#e6c96a'))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path if Path(path).is_file() else f'{path}\n파일을 찾을 수 없습니다.')
            if not Path(path).is_file():
                item.setForeground(QColor('#9a6a5a'))
            self.lut_list.addItem(item)
            if active and os.path.normcase(path) == os.path.normcase(active):
                self.lut_list.setCurrentItem(item)
        if not active:
            self.lut_list.setCurrentItem(none)

    def choose_lut(self):
        record = self.current()
        if not record:
            return
        last = record.get('lut_path') or (self.library.luts[-1] if self.library.luts else '')
        start = str(Path(last).parent) if last else ''
        paths, _ = QFileDialog.getOpenFileNames(self, '3D LUT 파일 불러오기', start, '3D LUT (*.cube)')
        if paths:
            self.add_luts(paths)

    def add_luts(self, paths):
        """Validate and list LUT files; the first new one is previewed on the current photo."""
        added, valid, failed = [], [], []
        for path in paths:
            try:
                load_cube(path)
            except (OSError, ValueError) as exc:
                failed.append(f'{Path(path).name}: {exc}')
                continue
            resolved = str(Path(path).resolve())
            valid.append(resolved)
            if self.library.remember_lut(resolved):
                added.append(resolved)
        if failed:
            QMessageBox.warning(self, 'LUT 파일을 읽을 수 없습니다', '\n'.join(failed))
        self.save_library()
        if valid and self.current():
            self.apply_lut((added or valid)[0])
        else:
            self.sync_controls()
            self.set_controls_enabled(True)
        if added:
            self.statusBar().showMessage(f'LUT {len(added)}개를 추가했습니다. 목록에서 골라 바로 비교해 보세요.', 7000)

    def apply_lut(self, path):
        record = self.current()
        if not record:
            return
        record['lut_path'] = path
        record['lut_strength'] = (record.get('lut_strength') or 60) if path else 0
        self.sync_controls()
        self.set_controls_enabled(True)
        self.save_timer.start(300)
        self.schedule_preview()
        if path:
            self.statusBar().showMessage(f'LUT 미리보기 · {Path(path).stem} · {record["lut_strength"]}%', 5000)

    def pick_lut(self, item, _previous=None):
        if self.syncing or item is None or not self.current():
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and not Path(path).is_file():
            self.statusBar().showMessage(f'LUT 파일을 찾을 수 없습니다: {path}', 7000)
            return
        if path != self.current().get('lut_path', ''):
            self.apply_lut(path)

    def toggle_lut_favorite(self):
        item = self.lut_list.currentItem()
        path = item.data(Qt.ItemDataRole.UserRole) if item else ''
        if not path:
            return
        favorite = self.library.toggle_favorite_lut(path)
        self.save_library()
        self.sync_controls()
        self.set_controls_enabled(True)
        self.statusBar().showMessage(f'{Path(path).stem} · 즐겨찾기 {"추가" if favorite else "해제"}', 4000)

    def remove_lut(self):
        item = self.lut_list.currentItem()
        path = item.data(Qt.ItemDataRole.UserRole) if item else ''
        if not path:
            return
        key = os.path.normcase(path)
        self.library.luts = [p for p in self.library.luts if os.path.normcase(p) != key]
        self.library.lut_favorites = [p for p in self.library.lut_favorites if os.path.normcase(p) != key]
        # Photos keep their saved LUT; only the current photo's preview is released.
        if self.current() and os.path.normcase(self.current().get('lut_path', '')) == key:
            self.apply_lut('')
        else:
            self.sync_controls()
            self.set_controls_enabled(True)
        self.save_library()

    def clear_lut(self):
        if not self.current():
            return
        self.current()['lut_path'] = ''
        self.current()['lut_strength'] = 0
        self.sync_controls()
        self.set_controls_enabled(True)
        self.save_library()
        self.schedule_preview()

    def adjust_lut_strength(self, value):
        if self.syncing or not self.current() or not self.current().get('lut_path'):
            return
        self.current()['lut_strength'] = int(value)
        self.sync_controls()
        self.save_timer.start(300)
        self.schedule_preview()

    def adjust(self, key, value):
        if self.syncing or not self.current():
            return
        self.current()['settings'][key] = value
        self.sync_controls()
        self.save_timer.start(300)
        self.schedule_preview()

    def set_adjustments(self, values):
        if not self.current():
            return
        self.current()['settings'] = settings(values)
        self.sync_controls()
        self.save_library()
        self.schedule_preview()

    def reset_current(self):
        if not self.current():
            return
        self.current()['lut_path'] = ''
        self.current()['lut_strength'] = 0
        self.current()['match'] = None
        self.set_adjustments(settings())

    def auto_correct(self):
        if not self.current() or not self.engine or self.auto_worker is not None:
            return
        if not self.preview_cache or self.displayed_path != self.current()['path']:
            self.statusBar().showMessage('사진 미리보기가 준비된 뒤 다시 눌러 주세요.', 5000)
            return
        self.auto_button.setEnabled(False)
        self.statusBar().showMessage('사진에 맞는 자연스러운 밝기와 색 균형을 분석하고 있습니다…')
        self.auto_worker = RecommendWorker(self.engine, self.current()['path'], self.preview_cache, self.preview_version, self)
        self.auto_worker.result.connect(self.auto_ready)
        self.auto_worker.error.connect(lambda message: self.statusBar().showMessage(message, 10000))
        self.auto_worker.finished.connect(self.auto_finished)
        self.auto_worker.start()

    def auto_ready(self, result):
        if self.closing or not self.current() or self.current()['path'] != result['path'] or self.preview_version != result['version']:
            return
        p = result['settings']
        self.set_adjustments(p)
        skin_text = f' · 피부 {p["skin"]:g}%' if p.get('skin', 0) else ''
        self.statusBar().showMessage(f'자연스러운 자동 보정 적용 · AI {p["ai"]:g}% · 노출 {p["exposure"]:+.2f} EV{skin_text} · 각 조절값을 직접 수정할 수 있습니다.', 9000)

    def auto_finished(self):
        old, self.auto_worker = self.auto_worker, None
        if old:
            old.deleteLater()
        self.set_controls_enabled(self.current() is not None and self.engine is not None)

    def preset(self, name):
        if name == 'natural':
            self.auto_correct()
            return
        ready = self.engine is not None and self.engine.status()['ai_ready']
        presets = {'natural': {'ai': 55 if ready else 0},
                   'soft': {'ai': 55 if ready else 0, 'softness': 48, 'contrast': -8, 'highlights': -15, 'saturation': -5},
                   'warm': {'ai': 50 if ready else 0, 'temperature': 22, 'softness': 32, 'shadows': 10, 'saturation': -8}}
        self.set_adjustments(presets[name])

    def apply_checked(self):
        record = self.current()
        if not record:
            return
        applied = 0
        for item in self.library.items:
            if item['checked']:
                item['settings'] = record['settings'].copy()
                item['lut_path'] = record.get('lut_path', '')
                item['lut_strength'] = record.get('lut_strength', 0)
                applied += 1
        self.save_library()
        self.statusBar().showMessage(f'현재 보정값과 LUT를 체크한 사진 {applied}장에 적용했습니다.', 6000)

    def engine_ready(self, engine):
        self.engine = engine
        self.update_device(engine.status())
        self.set_controls_enabled(self.current() is not None)
        self.preview_label.setText('로컬 엔진 준비 완료')
        self.schedule_preview()

    def engine_failed(self, message):
        self.device_label.setText('엔진 시작 실패')
        self.preview_label.setText(message)

    def update_device(self, status):
        text = status['gpu'].replace('NVIDIA GeForce ', '')
        if status['device'] == 'cuda':
            text += f'  ·  {status["vram_gb"]:g}GB  ·  CUDA'
        else:
            text += '  ·  로컬 처리'
        if not status['ai_ready']:
            text += '  ·  AI 모델 사용 불가'
        if not status.get('face_ai', False):
            text += '  ·  얼굴 인식 준비 안 됨'
        self.device_label.setText(text)
        self.device_label.setToolTip(status.get('face_error') if not status.get('face_ai') else (status['error'] or status['model']))

    def schedule_preview(self):
        self.preview_version += 1
        self.preview_pending = True
        self.preview_timer.start(100)

    def start_preview(self):
        if self.closing or not self.engine or not self.current():
            return
        if self.preview_worker is not None:
            self.preview_pending = True
            return
        record = self.current()
        self.preview_pending = False
        self.preview_label.setText('빛과 색을 다듬고 있어요…')
        self.preview_worker = PreviewWorker(self.engine, record['path'], record['settings'].copy(), self.preview_version,
                                            self.preview_cache, self, record.get('lut_path', ''),
                                            record.get('lut_strength', 0), self.render_match(record))
        self.preview_worker.result.connect(self.preview_ready)
        self.preview_worker.error.connect(self.preview_failed)
        self.preview_worker.finished.connect(self.preview_finished)
        self.preview_worker.start()

    def request_detail(self, request):
        self.detail_pending = request
        if self.detail_worker is None:
            self.start_detail()

    def start_detail(self):
        request, self.detail_pending = self.detail_pending, None
        record, cache = self.current(), self.preview_cache
        if not request or not record or self.closing or not self.engine or not cache:
            return
        try:
            current_key = file_key(record['path'])
        except OSError:
            return  # the original was moved or deleted; keep showing the preview
        if cache.get('key') != current_key or self.displayed_path != record['path']:
            return
        original, preview = cache['original'], cache['preview']
        width, height = original.size
        nx0, ny0, nx1, ny1 = request['region']
        x0, y0 = max(0, int(nx0*width)), max(0, int(ny0*height))
        x1, y1 = min(width, math.ceil(nx1*width)), min(height, math.ceil(ny1*height))
        if x1-x0 < 8 or y1-y0 < 8:
            return
        # Never upscale the original, and stay within a sensible pixel budget.
        out_w = min(x1-x0, request['width'])
        out_h = max(1, round(out_w*(y1-y0)/(x1-x0)))
        budget = 16_000_000
        if out_w*out_h > budget:
            factor = (budget/(out_w*out_h))**.5
            out_w, out_h = max(1, int(out_w*factor)), max(1, int(out_h*factor))
        if out_w <= preview.width*(x1-x0)/width*1.1:
            return  # the preview already has this much detail
        self.detail_worker = DetailWorker(self.engine, {
            'original': original, 'box': (x0, y0, x1, y1), 'size': (out_w, out_h),
            'region': (x0/width, y0/height, x1/width, y1/height), 'settings': record['settings'].copy(),
            'lut_path': record.get('lut_path', ''), 'lut_strength': record.get('lut_strength', 0),
            'match': record.get('match'), 'key': cache['key'], 'path': record['path'],
            'version': self.preview_version}, self)
        self.detail_worker.result.connect(self.detail_ready)
        self.detail_worker.finished.connect(self.detail_finished)
        self.detail_worker.start()

    def detail_ready(self, result):
        record = self.current()
        if (record and result['path'] == record['path'] == self.displayed_path
                and result['version'] == self.preview_version and not self.closing):
            self.viewer.set_detail(result['before'], result['after'], result['region'])

    def detail_finished(self):
        self.detail_worker = None
        if self.detail_pending is not None:
            self.start_detail()

    def preview_ready(self, result):
        resolved = result.get('resolved_match')
        if resolved:
            # Keep the measured plan even if newer settings are already rendering.
            record = next((r for r in self.library.items if r['path'] == result['path']), None)
            match = record.get('match') if record else None
            if (match and not match.get('plan') and match.get('reference') == resolved.get('reference')
                    and match.get('request_id') == resolved.get('request_id')):
                match['plan'] = resolved['plan']
                self.save_timer.start(300)
                if record is self.current():
                    self.sync_controls()
        if result['version'] != self.preview_version or not self.current() or self.closing:
            return
        self.preview_cache = result['cache']
        reset = self.displayed_path != result['path']
        self.viewer.set_images(result['before'], result['after'], reset=reset)
        self.displayed_path = result['path']
        self.set_controls_enabled(True)
        self.photo_label.setText(f'{self.current()["name"]}\n{result["size"][0]:,} × {result["size"][1]:,}  ·  sRGB')
        self.preview_label.setText(f'미리보기  ·  {result["seconds"]:.2f}초')
        self.update_device(result['status'])
        stats = result['analysis']
        self.histogram.values = stats['histogram']
        self.histogram.update()
        face_text = f'얼굴 {result["faces"]}명 · ' if result.get('faces', 0) else '얼굴 인식 없음 · '
        self.analysis_label.setText(f'{face_text}{stats["cast"]} · 평균 밝기 {stats["brightness"]}%\n'
                                    f'어두운 영역 {stats["shadows"]}% · 밝은 영역 {stats["highlights"]}%'
                                    + (f'\n{stats["gainmap"]} 있음 · 계조 보완에 사용' if stats.get('gainmap') else ''))
        metadata = result.get('exif', {})
        iso = str(metadata.get('iso', '')).removeprefix('ISO ').strip() or '—'
        exif_text = ' · '.join((f'셔터 {metadata.get("shutter", "—")}',
                               f'조리개 {metadata.get("aperture", "—")}',
                               f'ISO {iso}', f'해상도 {metadata.get("resolution", "—")}'))
        details = ' · '.join(metadata.get(key, '') for key in
                             ('taken', 'camera', 'lens', 'focal') if metadata.get(key))
        self.exif_label.setText(exif_text)
        self.exif_label.setToolTip(' · '.join(part for part in (exif_text, details) if part))

    def preview_failed(self, version, message):
        if version == self.preview_version:
            self.preview_label.setText('사진 처리 실패')
            self.preview_label.setToolTip(message)
            self.statusBar().showMessage(message, 12000)
            if not self.displayed_path:
                self.viewer.clear(message)

    def preview_finished(self):
        old = self.preview_worker
        self.preview_worker = None
        if old:
            old.deleteLater()
        if self.preview_pending and not self.closing:
            self.start_preview()

    def toggle_compare(self, checked):
        self.viewer.compare = checked
        if checked:
            self.original_button.setChecked(False)
        self.viewer.update()

    def toggle_original(self, checked):
        self.viewer.original = checked
        self.viewer.update()

    def open_batch(self, single=False, mode=None):
        if not self.engine:
            self.statusBar().showMessage('엔진 준비가 끝난 뒤 다시 시도해 주세요.', 5000)
            return
        records = [self.current()] if single and self.current() else [] if single else [x for x in self.library.items if x['checked']]
        if not records:
            self.statusBar().showMessage('처리할 사진의 체크박스를 선택해 주세요.', 5000)
            return
        if self.batch_dialog and self.batch_dialog.isVisible():
            self.content_stack.setCurrentWidget(self.batch_dialog)
            return
        items = [BatchItem(x['path'], x['name'], x['settings'].copy(),
                           x.get('lut_path', ''), x.get('lut_strength', 0), self.render_match(x)) for x in records]
        p = self.current()['settings'] if self.current() else settings()
        shared_lut_path = self.current().get('lut_path', '') if self.current() else ''
        shared_lut_strength = self.current().get('lut_strength', 0) if self.current() else 0
        self.save_library()
        reference = self.match_reference() or self.current()
        self.batch_dialog = BatchDialog(items, self.engine, self.library.output_dir, p, self.content_stack,
                                        reference_path=reference['path'] if reference else '', embedded=True,
                                        reference_look=self.reference_look(reference['path']) if reference else None,
                                        shared_lut_path=shared_lut_path,
                                        shared_lut_strength=shared_lut_strength)
        if mode:
            self.batch_dialog.correction.setCurrentIndex(self.batch_dialog.correction.findData(mode))
        self.batch_dialog.output_changed.connect(self.set_output_dir)
        dialog = self.batch_dialog
        dialog.finished.connect(lambda _result, value=dialog: self.close_batch_view(value))
        self.content_stack.addWidget(dialog)
        self.content_stack.setCurrentWidget(dialog)
        self.batch_dialog.show()

    def close_batch_view(self, dialog):
        if self.content_stack.currentWidget() is dialog:
            self.content_stack.setCurrentWidget(self.workspace)
        self.content_stack.removeWidget(dialog)
        dialog.deleteLater()
        if self.batch_dialog is dialog:
            self.batch_dialog = None

    def set_output_dir(self, path):
        self.library.output_dir = path
        self.save_library()

    def save_library(self):
        try:
            self.library.save()
        except OSError as exc:
            self.statusBar().showMessage(f'설정 저장 실패: {exc}', 10000)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_paths([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def closeEvent(self, event):
        active = [worker for worker in [self.loader, self.preview_worker, self.thumb_worker, self.auto_worker,
                                        self.detail_worker] if worker and worker.isRunning()]
        batch_running = self.batch_dialog and self.batch_dialog.is_running()
        if active or batch_running:
            self.closing = True
            self.preview_timer.stop()
            self.thumb_pending = []
            self.statusBar().showMessage('현재 작업을 마치고 종료하고 있습니다…')
            self.centralWidget().setEnabled(False)
            for worker in active:
                worker.requestInterruption()
            if batch_running:
                self.batch_dialog.request_cancel()
            QTimer.singleShot(200, self.close)
            event.ignore()
        else:
            self.save_timer.stop()
            self.save_library()
            event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('photos', nargs='*')
    parser.add_argument('--self-test', metavar='OUTPUT_DIR')
    args = parser.parse_args()
    if args.self_test:
        from desktop_smoke import self_test
        return self_test(args.self_test)
    application = QApplication(sys.argv[:1])
    application.setApplicationName('Onbit')
    application.setOrganizationName('Onbit')
    application.setStyle('Fusion')
    application.setStyleSheet(STYLE)
    application.setFont(QFont('Malgun Gothic', 10))
    window = MainWindow()
    screen = application.primaryScreen().availableGeometry()
    window.resize(min(1380, screen.width()-50), min(900, screen.height()-60))
    window.show()
    if args.photos:
        QTimer.singleShot(0, lambda: window.add_paths(args.photos))
    return application.exec()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        from desktop_store import storage_dir
        log = storage_dir() / 'crash.log'
        log.write_text(traceback.format_exc(), encoding='utf-8')
        application = QApplication.instance() or QApplication(sys.argv[:1])
        QMessageBox.critical(None, '온빛 실행 오류', f'프로그램을 시작하지 못했습니다.\n{log}')
        raise SystemExit(1)
