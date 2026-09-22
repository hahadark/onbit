from PySide6.QtCore import Qt, QRectF, QPointF, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget


def to_qimage(image):
    rgb = image.convert('RGB')
    return QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()


class PhotoViewer(QWidget):
    zoom_changed = Signal(str)
    # {'region': (x0, y0, x1, y1) normalised to the photo, 'width': device pixels, 'height': ...}
    detail_needed = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(350, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.before = QPixmap()
        self.after = QPixmap()
        self.compare = False
        self.original = False
        self.space_original = False
        self.split = .5
        self.zoom = 1.0
        self.offset = QPointF()
        self.drag_mode = None
        self.drag_point = QPointF()
        self.message = ''
        # A full-resolution render of the visible area, drawn over the preview when zoomed in.
        self.detail_before = QPixmap()
        self.detail_after = QPixmap()
        self.detail_region = None
        self.detail_timer = QTimer(self)
        self.detail_timer.setSingleShot(True)
        self.detail_timer.setInterval(180)
        self.detail_timer.timeout.connect(self.request_detail)

    def clear(self, message=''):
        self.before = QPixmap()
        self.after = QPixmap()
        self.clear_detail()
        self.message = message
        self.update()

    def set_images(self, before, after, reset=False):
        self.before = QPixmap.fromImage(before)
        self.after = QPixmap.fromImage(after)
        # A new render (other photo or new settings) makes the old detail stale.
        self.clear_detail()
        if reset:
            self.fit()
        self.schedule_detail()
        self.update()

    def clear_detail(self):
        self.detail_before = QPixmap()
        self.detail_after = QPixmap()
        self.detail_region = None

    def set_detail(self, before, after, region):
        self.detail_before = QPixmap.fromImage(before)
        self.detail_after = QPixmap.fromImage(after)
        self.detail_region = region
        self.update()

    def schedule_detail(self):
        if not self.after.isNull():
            self.detail_timer.start()

    def request_detail(self):
        """Ask for the visible area at screen resolution once zoomed past the preview's pixels."""
        rect = self.picture_rect()
        if rect.isEmpty():
            return
        ratio = self.devicePixelRatioF()
        if rect.width()*ratio <= self.after.width()*1.05:
            return
        visible = rect.intersected(QRectF(self.rect()))
        if visible.isEmpty():
            return
        # A small margin keeps short pans sharp while the next detail is prepared.
        margin_x, margin_y = visible.width()*.08, visible.height()*.08
        visible = visible.adjusted(-margin_x, -margin_y, margin_x, margin_y).intersected(rect)
        region = ((visible.left()-rect.left())/rect.width(), (visible.top()-rect.top())/rect.height(),
                  (visible.right()-rect.left())/rect.width(), (visible.bottom()-rect.top())/rect.height())
        self.detail_needed.emit({'region': region, 'width': max(1, round(visible.width()*ratio)),
                                 'height': max(1, round(visible.height()*ratio))})

    def detail_rect(self, rect):
        x0, y0, x1, y1 = self.detail_region
        return QRectF(rect.x()+rect.width()*x0, rect.y()+rect.height()*y0,
                      rect.width()*(x1-x0), rect.height()*(y1-y0))

    def _draw(self, painter, target, pixmap, detail):
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
        if self.detail_region is not None and not detail.isNull():
            painter.drawPixmap(self.detail_rect(target), detail, QRectF(detail.rect()))

    def resizeEvent(self, event):
        self.schedule_detail()
        super().resizeEvent(event)

    def picture_rect(self):
        if self.after.isNull():
            return QRectF()
        scale = min(max(1, self.width() - 52) / self.after.width(), max(1, self.height() - 52) / self.after.height()) * self.zoom
        w, h = self.after.width() * scale, self.after.height() * scale
        return QRectF((self.width() - w) / 2 + self.offset.x(), (self.height() - h) / 2 + self.offset.y(), w, h)

    def fit(self):
        self.zoom = 1.0
        self.offset = QPointF()
        self.zoom_changed.emit('화면 맞춤')
        self.schedule_detail()
        self.update()

    def change_zoom(self, factor):
        if self.after.isNull():
            return
        self.zoom = max(.2, min(12, self.zoom * factor))
        self.zoom_changed.emit(f'맞춤의 {self.zoom:.1f}배')
        self.schedule_detail()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor('#121713'))
        if self.after.isNull():
            center = self.rect().center()
            painter.setPen(QColor('#cedcaf'))
            painter.setFont(QFont('Malgun Gothic', 40))
            painter.drawText(QRectF(0, center.y()-145, self.width(), 85), Qt.AlignmentFlag.AlignCenter, '◒')
            painter.setFont(QFont('Malgun Gothic', 22))
            painter.drawText(QRectF(0, center.y()-55, self.width(), 75), Qt.AlignmentFlag.AlignCenter, '사진 속 좋은 빛을 꺼내 보세요')
            painter.setPen(QColor('#95a18c'))
            painter.setFont(QFont('Malgun Gothic', 10))
            painter.drawText(QRectF(20, center.y()+35, self.width()-40, 90), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self.message or '사진이나 폴더를 끌어다 놓으세요.\nJPG · PNG · WEBP · TIFF · BMP · HEIC')
            return
        target = self.picture_rect()
        self._draw(painter, target, self.after, self.detail_after)
        show_original = self.original or self.space_original
        if show_original:
            self._draw(painter, target, self.before, self.detail_before)
        elif self.compare:
            painter.save()
            painter.setClipRect(QRectF(target.x(), target.y(), target.width()*self.split, target.height()))
            self._draw(painter, target, self.before, self.detail_before)
            painter.restore()
            split_x = target.x() + target.width() * self.split
            painter.setPen(QPen(QColor('#e0eacb'), 1.5))
            painter.drawLine(QPointF(split_x, target.top()), QPointF(split_x, target.bottom()))
            knob = QRectF(split_x-13, target.center().y()-18, 26, 36)
            painter.setBrush(QColor('#d2dfba'))
            painter.drawRoundedRect(knob, 6, 6)
            painter.setPen(QColor('#263721'))
            painter.drawText(knob, Qt.AlignmentFlag.AlignCenter, '↔')
        painter.setFont(QFont('Malgun Gothic', 9))
        if self.compare or show_original:
            self._label(painter, QRectF(target.x()+12, target.y()+12, 45, 26), '원본')
        if not show_original:
            self._label(painter, QRectF(target.right()-57, target.y()+12, 45, 26), '보정')

    def _label(self, painter, rect, text):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 30, 17, 190))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QColor('#e4eddb'))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def wheelEvent(self, event):
        self.change_zoom(1.15 if event.angleDelta().y() > 0 else 1/1.15)
        event.accept()

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() != Qt.MouseButton.LeftButton or self.after.isNull():
            return
        rect = self.picture_rect()
        self.drag_point = event.position()
        line_x = rect.x() + rect.width() * self.split
        self.drag_mode = 'split' if self.compare and not self.original and abs(event.position().x()-line_x) < 20 else 'pan'

    def mouseMoveEvent(self, event):
        if self.drag_mode == 'split':
            rect = self.picture_rect()
            self.split = max(0, min(1, (event.position().x()-rect.x()) / max(1, rect.width())))
        elif self.drag_mode == 'pan':
            self.offset += event.position() - self.drag_point
            self.drag_point = event.position()
            self.schedule_detail()
        else:
            rect = self.picture_rect()
            near = self.compare and abs(event.position().x()-(rect.x()+rect.width()*self.split)) < 20
            self.setCursor(Qt.CursorShape.SplitHCursor if near else Qt.CursorShape.OpenHandCursor)
        self.update()

    def mouseReleaseEvent(self, event):
        self.drag_mode = None

    def mouseDoubleClickEvent(self, event):
        self.fit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space_original = True
            self.update()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.space_original = False
            self.update()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.space_original = False
        self.drag_mode = None
        self.update()
        super().focusOutEvent(event)


class Histogram(QWidget):
    def __init__(self):
        super().__init__()
        self.values = None
        self.setFixedHeight(75)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#131a12'))
        if not self.values:
            return
        highest = max(max(channel) for channel in self.values) or 1
        for channel, color in zip(self.values, ['#cc8b76', '#b0cb8c', '#84b2d0']):
            path = QPainterPath()
            path.moveTo(0, self.height())
            for index, value in enumerate(channel):
                path.lineTo(index/63*self.width(), self.height() - (value/highest)**.6*(self.height()-5))
            path.lineTo(self.width(), self.height())
            path.closeSubpath()
            fill = QColor(color)
            fill.setAlpha(45)
            painter.fillPath(path, fill)
            painter.setPen(QPen(QColor(color), 1))
            painter.drawPath(path)
