# A portable native application directory. CUDA DLLs remain beside the executable.
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata
import importlib.util

debug_build = os.environ.get('ONBIT_DEBUG_BUILD') == '1'

datas = [('models', 'models'), ('samples/demo.jpg', 'samples'), ('assets/onbit.ico', 'assets'),
         ('third_party', 'third_party'), ('README.md', '.')]
datas += collect_data_files('pillow_heif')
for package in ['pillow-heif', 'PySide6-Essentials', 'shiboken6', 'torch', 'Pillow', 'numpy', 'efficientnet-pytorch']:
    datas += copy_metadata(package)

if importlib.util.find_spec('cv2') is None:
    raise RuntimeError('OpenCV is required for desktop builds. Run setup.ps1 first.')
datas += collect_data_files('cv2')
datas += copy_metadata('opencv-python-headless')
datas += [('tests/fixtures/astronaut.png', 'tests/fixtures'),
          ('tests/fixtures/README.md', 'tests/fixtures')]

a_binaries = collect_dynamic_libs('pillow_heif')
hiddenimports = ['_pillow_heif', 'PIL.WebPImagePlugin', 'PIL.TiffImagePlugin']
if importlib.util.find_spec('cv2') is not None:
    a_binaries += collect_dynamic_libs('cv2')
    hiddenimports.append('cv2')

a = Analysis(['desktop_bootstrap.py'], pathex=[], binaries=a_binaries, datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['tkinter', 'matplotlib', 'scipy', 'pandas', 'IPython', 'pytest',
                       'PySide2', 'PyQt5', 'PyQt6', 'tensorboard'],
             noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Onbit', debug=debug_build,
          bootloader_ignore_signals=False, strip=False, upx=False, console=debug_build,
          icon='assets/onbit.ico', disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='Onbit')
