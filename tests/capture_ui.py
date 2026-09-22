"""Render the two native desktop pages for visual QA without desktop input."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from pathlib import Path
import sys
import tempfile
import time

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from desktop import MainWindow
from desktop_store import Library


def wait(app, condition, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(.01)
    raise TimeoutError('UI did not become ready')


def main(output):
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as session:
        window = MainWindow(Library(session))
        window.resize(1380, 900)
        window.show()
        root = Path(__file__).resolve().parents[1]
        window.add_paths([str(root/'samples/demo.jpg'), str(root/'tests/fixtures/astronaut.png')])
        wait(app, lambda: window.engine is not None and window.displayed_path is not None)
        window.grab().save(str(destination/'viewer.png'))
        window.open_batch(False)
        wait(app, lambda: window.batch_dialog is not None)
        window.batch_dialog.correction.setCurrentIndex(
            window.batch_dialog.correction.findData('auto'))
        app.processEvents()
        window.grab().save(str(destination/'batch.png'))
        window.batch_dialog.accept()
        window.close()
        wait(app, lambda: all(worker is None or not worker.isRunning() for worker in
                             [window.loader, window.preview_worker, window.thumb_worker, window.auto_worker]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
