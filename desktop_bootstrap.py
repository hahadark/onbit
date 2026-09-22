"""Frozen entry point: retain startup errors even without a terminal window."""
from datetime import datetime
import faulthandler
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    base = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    candidates = [base / 'onbit-data', Path(os.environ.get('LOCALAPPDATA', str(base))) / 'Onbit']
    log = None
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / 'startup.log').open('w', encoding='utf-8', buffering=1)
            break
        except OSError:
            continue
    if log:
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log
        print(f'{datetime.now().isoformat()} Starting Onbit', file=log)
        faulthandler.enable(file=log)
        if '--self-test' in sys.argv:
            faulthandler.dump_traceback_later(30, repeat=True, file=log)
    try:
        from desktop import main as run
        if log:
            print(f'{datetime.now().isoformat()} Runtime loaded', file=log)
        return run()
    except Exception:
        error = traceback.format_exc()
        if log:
            print(error, file=log)
        if '--self-test' in sys.argv:
            try:
                destination = Path(sys.argv[sys.argv.index('--self-test') + 1]).resolve()
                destination.mkdir(parents=True, exist_ok=True)
                (destination / 'self-test.json').write_text(json.dumps({'ok': False, 'error': error}, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                pass
            return 1
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance() or QApplication(sys.argv[:1])
            QMessageBox.critical(None, '온빛 실행 오류', '프로그램을 시작하지 못했습니다.\n' + (str(Path(log.name)) if log else traceback.format_exc()))
        except Exception:
            pass
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    raise SystemExit(main())
