import threading
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from batch import BatchOptions, run_batch, prepare_reference, render_item
from engine import settings
from desktop_viewer import PhotoViewer, to_qimage


class BatchWorker(QThread):
    progress = Signal(object)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, items, options, engine, parent=None):
        super().__init__(parent)
        self.items, self.options, self.engine = items, options, engine
        self.cancel_event = threading.Event()

    def run(self):
        try:
            self.result.emit(run_batch(self.items, self.options, self.engine, self.cancel_event, self.progress.emit))
        except Exception as exc:
            self.error.emit(str(exc))


class BatchPreviewWorker(QThread):
    result = Signal(object)
    error = Signal(str)

    def __init__(self, item, options, engine, parent):
        super().__init__(parent)
        self.item, self.options, self.engine = item, options, engine

    def run(self):
        try:
            reference = prepare_reference(self.options, self.engine)
            if self.isInterruptionRequested():
                return
            before, after, details = render_item(self.item, self.options, self.engine, reference)
            before.thumbnail((1000, 1000))
            after.thumbnail((1000, 1000))
            if not self.isInterruptionRequested():
                self.result.emit({'before': to_qimage(before), 'after': to_qimage(after), 'name': self.item.name, 'details': details})
        except Exception as exc:
            self.error.emit(str(exc))


class BatchDialog(QDialog):
    output_changed = Signal(str)

    def __init__(self, items, engine, output_dir, shared_settings=None, parent=None, reference_path='', embedded=False,
                 shared_lut_path='', shared_lut_strength=0, reference_look=None):
        super().__init__(parent)
        self.embedded = embedded
        if embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
        self.items = items
        self.engine = engine
        self.shared_settings = settings(shared_settings)
        self.shared_lut_path = shared_lut_path
        self.shared_lut_strength = shared_lut_strength
        self.reference_path = reference_path
        self.reference_settings = settings(shared_settings)
        self.reference_look = reference_look
        if reference_look:
            self.reference_settings = settings(reference_look['settings'])
        self.worker = None
        self.preview_worker = None
        self.result = None
        self.close_when_done = False
        self.setWindowTitle('온빛 · 일괄 보정 / 크기 조절 / 형식 변환')
        self.resize(930, 740)
        self.setMinimumSize(800, 660)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(26, 22, 26, 22)
        outer.setSpacing(16)
        title = QLabel(f'{len(items)}장의 사진, 한 번에 완성하기')
        title.setObjectName('dialogTitle')
        outer.addWidget(title)
        description = QLabel('원본은 보존됩니다. 같은 이름은 번호를 붙여 저장하며, 사진 비율을 유지합니다.')
        description.setObjectName('muted')
        outer.addWidget(description)
        self.lut_summary = QLabel()
        self.lut_summary.setObjectName('muted')
        outer.addWidget(self.lut_summary)
        self.options_widget = QWidget()
        grid = QGridLayout(self.options_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(12)
        self.format_box = QComboBox()
        self.format_box.setObjectName('batchFormat')
        for label, code in [('JPG · 사진 / 용량 절약', 'JPEG'), ('PNG · 무손실', 'PNG'),
                            ('WebP · 작은 파일', 'WEBP'), ('HEIC · 고효율 사진', 'HEIC'),
                            ('TIFF · 무손실', 'TIFF'), ('BMP · 비압축', 'BMP')]:
            self.format_box.addItem(label, code)
        self.correction = QComboBox()
        self.correction.setObjectName('batchCorrection')
        for label, value in [('사진별 저장된 보정값 적용', 'saved'), ('AI 자동 보정 · 자연스럽게 · 피부 포함', 'auto'),
                             ('기준 사진에 색감·밝기 맞추기', 'match'),
                             ('현재 사진의 설정으로 모두 보정', 'shared'), ('색 보정 없이 변환·리사이즈만', 'none')]:
            self.correction.addItem(label, value)
        self.resize_box = QComboBox()
        self.resize_box.setObjectName('batchResize')
        for label, code in [('원본 크기 유지', 'original'), ('긴 변을 지정한 픽셀로', 'long_edge'),
                            ('가로 × 세로 안에 맞추기', 'fit'), ('원본 대비 비율 (%)', 'percent')]:
            self.resize_box.addItem(label, code)
        self.width = QSpinBox()
        self.width.setObjectName('batchWidth')
        self.width.setRange(1, 30000)
        self.width.setValue(2048)
        self.width.setSuffix(' px')
        self.height = QSpinBox()
        self.height.setObjectName('batchHeight')
        self.height.setRange(1, 30000)
        self.height.setValue(2048)
        self.height.setSuffix(' px')
        self.percent = QSpinBox()
        self.percent.setObjectName('batchPercent')
        self.percent.setRange(1, 400)
        self.percent.setValue(50)
        self.percent.setSuffix(' %')
        dimensions = QWidget()
        sizes = QHBoxLayout(dimensions)
        sizes.setContentsMargins(0, 0, 0, 0)
        sizes.addWidget(self.width)
        self.times = QLabel('×')
        sizes.addWidget(self.times)
        sizes.addWidget(self.height)
        sizes.addWidget(self.percent)
        self.original_note = QLabel('사진의 원래 크기로 저장')
        sizes.addWidget(self.original_note)
        self.upscale = QCheckBox('작은 사진도 확대 허용')
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(90)
        self.quality.setSuffix(' / 100')
        for row, label, widget, label2, widget2 in [(0, '출력 형식', self.format_box, '보정 방식', self.correction),
                                                   (1, '크기 조절', self.resize_box, '크기', dimensions),
                                                   (2, '저장 품질', self.quality, '', self.upscale)]:
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
            grid.addWidget(QLabel(label2), row, 2)
            grid.addWidget(widget2, row, 3)
        self.source_folder = QCheckBox('원본 폴더에 하위 폴더 자동 생성')
        self.source_folder.setChecked(True)
        self.subfolder = QLineEdit('온빛_내보내기')
        self.subfolder.setPlaceholderText('하위 폴더 이름')
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_folder)
        source_row.addWidget(self.subfolder, 1)
        grid.addLayout(source_row, 3, 0, 1, 4)
        self.output = QLineEdit(output_dir)
        self.output.setObjectName('batchOutput')
        self.browse = QPushButton('폴더 선택…')
        self.browse.clicked.connect(self.choose_output)
        row = QHBoxLayout()
        row.addWidget(self.output, 1)
        row.addWidget(self.browse)
        grid.addWidget(QLabel('지정 저장 폴더'), 4, 0)
        grid.addLayout(row, 4, 1, 1, 3)
        self.parallel_gpu = QCheckBox('RTX GPU로 사진 2장 병렬 처리')
        self.parallel_gpu.setChecked(engine.status().get('device') == 'cuda')
        self.parallel_gpu.setEnabled(engine.status().get('device') == 'cuda')
        grid.addWidget(self.parallel_gpu, 5, 0, 1, 4)
        self.auto_widget = QWidget()
        auto_layout = QVBoxLayout(self.auto_widget)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel('AI 보정 강도'))
        self.auto_strength = QSpinBox()
        self.auto_strength.setObjectName('autoStrength')
        self.auto_strength.setRange(0, 100)
        self.auto_strength.setValue(100)
        self.auto_strength.setSuffix(' %')
        auto_row.addWidget(self.auto_strength)
        self.auto_color = QCheckBox('색감·화이트밸런스')
        self.auto_color.setChecked(True)
        auto_row.addWidget(self.auto_color)
        self.auto_brightness = QCheckBox('밝기·명암')
        self.auto_brightness.setChecked(True)
        auto_row.addWidget(self.auto_brightness)
        auto_row.addStretch()
        auto_layout.addLayout(auto_row)
        finish_row = QHBoxLayout()
        finish_row.addWidget(QLabel('부드러운 톤'))
        self.auto_softness = QSpinBox()
        self.auto_softness.setObjectName('autoSoftness')
        self.auto_softness.setRange(0, 100)
        self.auto_softness.setValue(5)
        self.auto_softness.setSuffix(' %')
        finish_row.addWidget(self.auto_softness)
        finish_row.addWidget(QLabel('얼굴 피부 보정'))
        self.auto_skin = QSpinBox()
        self.auto_skin.setObjectName('autoSkin')
        self.auto_skin.setRange(0, 100)
        self.auto_skin.setValue(30)
        self.auto_skin.setSuffix(' %')
        finish_row.addWidget(self.auto_skin)
        finish_row.addStretch()
        auto_layout.addLayout(finish_row)
        auto_note = QLabel('각 사진을 따로 분석합니다. 0%로 두면 해당 효과를 적용하지 않으며 피부 보정은 얼굴이 인식된 사진에만 적용됩니다.')
        auto_note.setObjectName('muted')
        auto_note.setWordWrap(True)
        auto_layout.addWidget(auto_note)
        grid.addWidget(self.auto_widget, 6, 0, 1, 4)
        self.match_widget = QWidget()
        match_layout = QVBoxLayout(self.match_widget)
        match_layout.setContentsMargins(0, 0, 0, 0)
        reference_row = QHBoxLayout()
        self.reference_label = QLabel()
        self.reference_label.setWordWrap(True)
        self.reference_label.setObjectName('muted')
        reference_row.addWidget(self.reference_label, 1)
        reference_button = QPushButton('기준 사진 선택…')
        reference_button.clicked.connect(self.choose_reference)
        reference_row.addWidget(reference_button)
        match_layout.addLayout(reference_row)
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel('색감 통일 강도'))
        self.match_strength = QSpinBox()
        self.match_strength.setObjectName('matchStrength')
        self.match_strength.setRange(0, 100)
        self.match_strength.setValue(70)
        self.match_strength.setSuffix(' %')
        strength_row.addWidget(self.match_strength)
        self.match_brightness = QCheckBox('밝기도 맞추기')
        self.match_brightness.setChecked(True)
        strength_row.addWidget(self.match_brightness)
        strength_row.addStretch()
        match_layout.addLayout(strength_row)
        self.match_method = QComboBox()
        self.match_method.addItem('AI 색감 매칭 · Neural-Preset', 'neural')
        self.match_method.addItem('기존 방식 · 밝기·색 분포 맞춤', 'statistics')
        self.match_method.currentIndexChanged.connect(self.invalidate_preview)
        match_layout.addWidget(self.match_method)
        self.match_skin = QCheckBox('피부색 보호 (AI 매칭에서도 밝기는 조절)')
        self.match_skin.setChecked(True)
        self.match_skin.toggled.connect(self.invalidate_preview)
        match_layout.addWidget(self.match_skin)
        note = QLabel('같은 장소·비슷한 장면에 권장합니다. 사진별 보정 후 색감을 맞추며 기준 사진에는 중복 적용하지 않습니다.')
        note.setObjectName('muted')
        note.setWordWrap(True)
        match_layout.addWidget(note)
        grid.addWidget(self.match_widget, 7, 0, 1, 4)
        self.update_reference_label()
        outer.addWidget(self.options_widget)
        self.resize_box.currentIndexChanged.connect(self.update_options)
        self.format_box.currentIndexChanged.connect(self.update_options)
        self.correction.currentIndexChanged.connect(self.update_options)
        self.source_folder.toggled.connect(self.update_options)
        self.update_options()
        self.table = QTableWidget(len(items), 3)
        self.table.setObjectName('batchTable')
        self.table.setHorizontalHeaderLabels(['사진', '처리 상태', '저장된 파일 / 오류'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        for i, item in enumerate(items):
            for column, value in enumerate([item.name, '대기', '']):
                cell = QTableWidgetItem(value)
                cell.setToolTip(item.path if column == 0 else value)
                self.table.setItem(i, column, cell)
        outer.addWidget(self.table, 1)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        first = next((i for i, item in enumerate(items) if item.path != reference_path), 0)
        self.table.selectRow(first)
        self.preview_viewer = PhotoViewer()
        self.preview_viewer.setMinimumHeight(150)
        self.preview_viewer.setMaximumHeight(230)
        self.preview_viewer.hide()
        outer.addWidget(self.preview_viewer)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(items))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%v / %m장')
        outer.addWidget(self.progress_bar)
        self.summary = QLabel('설정을 확인한 뒤 일괄 처리 시작을 눌러 주세요.')
        self.summary.setWordWrap(True)
        outer.addWidget(self.summary)
        buttons = QHBoxLayout()
        self.open_folder = QPushButton('결과 폴더 열기')
        self.open_folder.setEnabled(False)
        self.open_folder.clicked.connect(self.open_result_folder)
        buttons.addWidget(self.open_folder)
        self.preview_button = QPushButton('선택 사진 미리보기')
        self.preview_button.clicked.connect(self.start_preview)
        buttons.addWidget(self.preview_button)
        buttons.addStretch()
        self.cancel_button = QPushButton('뷰어로 돌아가기' if embedded else '닫기')
        self.cancel_button.clicked.connect(self.cancel_or_close)
        self.start_button = QPushButton('일괄 처리 시작')
        self.start_button.setObjectName('primary')
        self.start_button.clicked.connect(self.start)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.start_button)
        outer.addLayout(buttons)
        for box in (self.resize_box, self.format_box, self.correction):
            box.currentIndexChanged.connect(self.invalidate_preview)
        for spin in (self.width, self.height, self.percent, self.quality, self.match_strength,
                     self.auto_strength, self.auto_softness, self.auto_skin):
            spin.valueChanged.connect(self.invalidate_preview)
        self.match_brightness.toggled.connect(self.invalidate_preview)
        self.auto_color.toggled.connect(self.invalidate_preview)
        self.auto_brightness.toggled.connect(self.invalidate_preview)
        self.upscale.toggled.connect(self.invalidate_preview)
        self.source_folder.toggled.connect(self.invalidate_preview)
        self.subfolder.textChanged.connect(self.invalidate_preview)
        self.parallel_gpu.toggled.connect(self.invalidate_preview)
        self.table.itemSelectionChanged.connect(self.invalidate_preview)

    def update_options(self):
        mode = self.resize_box.currentData()
        self.width.setVisible(mode in {'long_edge', 'fit'})
        self.height.setVisible(mode == 'fit')
        self.times.setVisible(mode == 'fit')
        self.percent.setVisible(mode == 'percent')
        self.original_note.setVisible(mode == 'original')
        self.upscale.setEnabled(mode != 'original')
        self.quality.setEnabled(self.format_box.currentData() in {'JPEG', 'WEBP', 'HEIC'})
        self.auto_widget.setVisible(self.correction.currentData() == 'auto')
        self.match_widget.setVisible(self.correction.currentData() == 'match')
        mode = self.correction.currentData()
        if mode == 'none':
            text = 'LUT 색감: 적용 안 함'
        elif mode == 'shared':
            text = (f'LUT 색감: {Path(self.shared_lut_path).name} · {self.shared_lut_strength}%'
                    if self.shared_lut_path and self.shared_lut_strength else 'LUT 색감: 현재 사진에 선택된 LUT 없음')
        else:
            count = sum(bool(item.lut_path and item.lut_strength) for item in self.items)
            text = f'LUT 색감: 사진별 저장 설정 사용 · {count}장에 적용'
        self.lut_summary.setText(text)
        source_mode = self.source_folder.isChecked()
        self.subfolder.setEnabled(source_mode)
        self.output.setEnabled(not source_mode)
        self.browse.setEnabled(not source_mode)

    def update_reference_label(self):
        suffix = '현재 보정값 적용' if any(self.reference_settings.values()) else '원본 색감'
        self.reference_label.setText(f'기준: {Path(self.reference_path).name} · {suffix}' if self.reference_path else '기준 사진을 선택해 주세요.')
        self.reference_label.setToolTip(self.reference_path)

    def choose_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, '색감 기준 사진', self.reference_path, '사진 (*.jpg *.jpeg *.mpo *.png *.webp *.heic *.heif *.hif *.tif *.tiff *.bmp)')
        if path:
            if Path(path) != Path(self.reference_path):
                self.reference_settings = settings()
                self.reference_look = None
            self.reference_path = path
            self.update_reference_label()
            self.invalidate_preview()

    def invalidate_preview(self, *_):
        if self.preview_viewer.isVisible():
            self.preview_viewer.hide()
            self.summary.setText('설정이나 대상이 바뀌었습니다. 선택 사진 미리보기를 다시 눌러 주세요.')

    def start_preview(self):
        if self.is_running():
            return
        try:
            options = self.options()
            options.validate()
        except ValueError as exc:
            self.warn(str(exc))
            return
        item = self.items[max(0, self.table.currentRow())]
        self.options_widget.setEnabled(False)
        self.table.setEnabled(False)
        self.start_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.summary.setText(f'{item.name} · 저장 전 미리보기를 만들고 있습니다…')
        self.cancel_button.setText('미리보기 중지')
        self.preview_worker = BatchPreviewWorker(item, options, self.engine, self)
        self.preview_worker.result.connect(self.preview_ready)
        self.preview_worker.error.connect(lambda message: self.summary.setText('미리보기 실패: '+message))
        self.preview_worker.finished.connect(self.preview_finished)
        self.preview_worker.start()

    def preview_ready(self, result):
        self.preview_viewer.compare = True
        self.preview_viewer.set_images(result['before'], result['after'], reset=True)
        self.preview_viewer.show()
        self.summary.setText(f'{result["name"]} · 왼쪽 원본 / 오른쪽 처리 후 · 구분선을 움직여 비교하세요. 아직 파일은 저장하지 않았습니다.')

    def preview_finished(self):
        old, self.preview_worker = self.preview_worker, None
        if old:
            old.deleteLater()
        self.options_widget.setEnabled(True)
        self.table.setEnabled(True)
        self.start_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText('뷰어로 돌아가기' if self.embedded else '닫기')
        if self.close_when_done:
            self.accept()

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, '보정본을 저장할 폴더', self.output.text())
        if path:
            self.output.setText(path)

    def warn(self, message):
        if self.embedded:
            self.summary.setText('설정을 확인해 주세요: ' + message)
        else:
            QMessageBox.warning(self, '설정을 확인해 주세요', message)

    def open_result_folder(self):
        if self.result:
            completed = next((entry.get('output') for entry in self.result.get('files', []) if entry.get('output')), '')
            if completed:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(completed).parent)))
                return
        if self.source_folder.isChecked() and self.items:
            path = Path(self.items[0].path).resolve().parent / self.subfolder.text().strip()
        else:
            path = Path(self.output.text()).resolve()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def options(self):
        return BatchOptions(output_dir=self.output.text().strip(), format=self.format_box.currentData(),
                            resize=self.resize_box.currentData(), width=self.width.value(), height=self.height.value(),
                            percent=self.percent.value(), quality=self.quality.value(), upscale=self.upscale.isChecked(),
                            correction=self.correction.currentData(), shared_adjustments=self.shared_settings,
                            shared_lut_path=self.shared_lut_path, shared_lut_strength=self.shared_lut_strength,
                            reference_path=self.reference_path, reference_adjustments=self.reference_settings,
                            match_strength=self.match_strength.value(), match_brightness=self.match_brightness.isChecked(),
                            match_method=self.match_method.currentData(), match_protect_skin=self.match_skin.isChecked(),
                            reference_look=self.reference_look,
                            auto_strength=self.auto_strength.value(), auto_color=self.auto_color.isChecked(),
                            auto_brightness=self.auto_brightness.isChecked(), auto_softness=self.auto_softness.value(),
                            auto_skin=self.auto_skin.value(),
                            source_subfolder=self.subfolder.text().strip() if self.source_folder.isChecked() else '',
                            parallel_gpu=self.parallel_gpu.isChecked())

    def is_running(self):
        return any(worker is not None and worker.isRunning() for worker in (self.worker, self.preview_worker))

    def start(self):
        if self.is_running():
            return
        try:
            options = self.options()
            options.validate()
            if options.correction == 'auto' and not self.engine.status()['ai_ready']:
                raise ValueError('AI 모델을 사용할 수 없습니다. 변환·리사이즈만 선택하거나 엔진 상태를 확인해 주세요.')
        except ValueError as exc:
            self.warn(str(exc))
            return
        if not options.source_subfolder:
            self.output_changed.emit(options.output_dir)
        self.options_widget.setEnabled(False)
        self.start_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.open_folder.setEnabled(False)
        self.progress_bar.setValue(0)
        self.cancel_button.setText('현재 파일 완료 후 중지')
        self.cancel_button.setEnabled(True)
        for row in range(self.table.rowCount()):
            self.table.item(row, 1).setText('대기')
            self.table.item(row, 2).setText('')
        self.result = None
        self.worker = BatchWorker(self.items, options, self.engine, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.result.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, event):
        row = event['index']
        state = event['state']
        self.table.item(row, 1).setText({'processing': '처리 중…', 'done': '완료', 'error': '실패'}[state])
        if state == 'done':
            text = f'{event["width"]} × {event["height"]}  ·  {Path(event["output"]).name}'
            self.table.item(row, 2).setText(text)
            self.table.item(row, 2).setToolTip(event['output'])
        elif state == 'error':
            self.table.item(row, 2).setText(event['error'])
            self.table.item(row, 2).setToolTip(event['error'])
        self.table.scrollToItem(self.table.item(row, 0))
        if 'completed' in event:
            self.progress_bar.setValue(event['completed'])
        if not self.worker.cancel_event.is_set():
            self.summary.setText(f'{row+1} / {len(self.items)}장  ·  {event["name"]}')

    def on_result(self, result):
        self.result = result
        prefix = '중지됨' if result['cancelled'] else '처리 완료'
        parallel = ' · GPU 병렬 처리' if result.get('parallel_gpu') else ''
        self.summary.setText(f'{prefix} — 성공 {result["succeeded"]}장 · 실패 {result["failed"]}장 · 미처리 {result["unprocessed"]}장{parallel}'
                             + ('\n처리 기록도 저장 폴더에 남겼습니다.' if 'report' in result else '\n처리 기록 저장 실패: '+result.get('report_error', '')))
        for row in range(len(result['files']), len(self.items)):
            self.table.item(row, 1).setText('미처리')
        self.open_folder.setEnabled(True)

    def on_error(self, message):
        self.summary.setText('처리를 시작하지 못했습니다: ' + message)

    def on_finished(self):
        self.options_widget.setEnabled(True)
        self.start_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.start_button.setText('같은 목록 다시 처리')
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText('뷰어로 돌아가기' if self.embedded else '닫기')
        if self.close_when_done:
            self.accept()

    def request_cancel(self):
        if self.is_running():
            if self.worker is not None and self.worker.isRunning():
                self.worker.cancel_event.set()
            if self.preview_worker is not None:
                self.preview_worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.summary.setText('중지 요청됨 — 현재 사진의 저장이 끝나면 멈춥니다. 완료된 파일은 유지됩니다.')

    def cancel_or_close(self):
        if self.is_running():
            self.request_cancel()
        else:
            self.accept()

    def reject(self):
        if self.is_running():
            self.close_when_done = True
            self.request_cancel()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.is_running():
            self.close_when_done = True
            self.request_cancel()
            event.ignore()
        else:
            event.accept()
