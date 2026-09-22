"""Manual inspection artifact and timings for real model inference; run from root."""
import os
os.environ.pop('QT_QPA_PLATFORM', None)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
import tempfile
import numpy as np
from PIL import Image, ImageDraw
from engine import Engine, read_image
from neural_match import MODEL_ID

root = Path('tests/output/neural')
root.mkdir(parents=True, exist_ok=True)
engine = Engine()
canvas = Image.new('RGB', (1200, 760), '#151821')
draw = ImageDraw.Draw(canvas)
report = {'model': MODEL_ID, 'device': str(engine.device), 'cases': []}
for row, path in enumerate(['samples/demo.jpg', 'tests/fixtures/astronaut.png']):
    ref = read_image(path)
    ref.thumbnail((400, 340))
    src = Image.fromarray((np.asarray(ref) * [.72, .78, .87]).astype(np.uint8))
    start = time.perf_counter()
    plan = engine.neural_match(src, ref)
    ms = (time.perf_counter()-start)*1000
    out = engine.render(src, {}, f'inspect-{row}', match={'plan': plan, 'strength': 100,
                                                         'brightness': True, 'protect_skin': False})
    for col, (title, image) in enumerate(zip(['Reference', 'Dark / cool target', 'AI matched 100%'], [ref, src, out])):
        draw.text((col*400+10, row*380+10), title, fill='white')
        canvas.paste(image, (col*400, row*380+35))
    errors = [float(np.abs(np.asarray(im).astype(float)-np.asarray(ref)).mean()) for im in [src, out]]
    report['cases'].append({'source': path, 'inference_ms': ms, 'before_mae': errors[0], 'after_mae': errors[1], 'plan': plan})
canvas.save(root/'comparison.jpg', quality=95)
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from desktop import MainWindow
from desktop_store import Library
app = QApplication.instance() or QApplication([])
with tempfile.TemporaryDirectory() as folder:
    window = MainWindow(Library(folder))
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
    window.show()
    window.add_paths([str(Path('samples/demo.jpg').resolve())])
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        app.processEvents()
        if window.engine and window.displayed_path:
            break
        time.sleep(.02)
    from PySide6.QtWidgets import QTabWidget
    tabs = window.findChildren(QTabWidget)
    for tabs_ in tabs:
        if tabs_.count() == 4:
            tabs_.setCurrentIndex(1)
    app.processEvents()
    window.grab().save(str(root/'viewer.png'))
    report['ui_default'] = window.match_method.currentData()
    window.open_batch(mode='match')
    app.processEvents()
    report['batch_default'] = window.batch_dialog.options().match_method
    window.grab().save(str(root/'batch.png'))
    window.close()
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        app.processEvents()
        if all(w is None or not w.isRunning() for w in [window.loader, window.preview_worker, window.thumb_worker]):
            break
        time.sleep(.02)
(root/'inspection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
