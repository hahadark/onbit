"""Headless widget/integration tests; never send input to the user's desktop."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
from PIL.TiffImagePlugin import IFDRational
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QListView, QMessageBox
from desktop import MainWindow
from desktop_store import Library
from engine import read_image


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for(self, predicate, seconds=20):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Desktop worker did not finish within timeout')

    def test_native_library_heic_preview_controls_and_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            png = root/'첫 사진.png'
            heic = root/'둘째 사진.heic'
            source = Image.new('RGB',(120,80),(60,100,130))
            source.save(png)
            source.save(heic,format='HEIF',quality=95)
            window = MainWindow(Library(root/'session'))
            try:
                window.add_paths([str(heic),str(png)])
                self.wait_for(lambda: window.engine is not None and window.displayed_path == str(heic))
                self.assertEqual(window.list.count(),2)
                self.assertEqual(window.viewer.after.width(),120)
                self.assertEqual(window.viewer.after.height(),80)
                window.auto_correct()
                self.wait_for(lambda: window.auto_worker is None and window.current()['settings']['ai'] > 0)
                self.assertLessEqual(window.current()['settings']['ai'], 35)
                window.controls['exposure'][1].setValue(.5)
                self.assertEqual(window.current()['settings']['exposure'],.5)
                # Scrolling the side panel over a slider or number box must not change the value.
                for widget in window.controls['exposure'][:2]:
                    wheel = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                                        Qt.ScrollPhase.NoScrollPhase, False)
                    QApplication.sendEvent(widget, wheel)
                self.assertEqual(window.current()['settings']['exposure'], .5)
                self.assertEqual([window.side_tabs.tabText(i) for i in range(window.side_tabs.count())],
                                 ['기본', '색감·LUT', '인물·마무리', '분석'])
                self.assertTrue(window.controls['skin'][1].isEnabled())
                window.controls['skin'][1].setValue(22)
                self.assertEqual(window.current()['settings']['skin'], 22)
                lut = root/'화사하게.cube'
                lut.write_text('LUT_3D_SIZE 2\n0.1 0.1 0.1\n1 0.1 0.1\n0.1 1 0.1\n1 1 0.1\n'
                               '0.1 0.1 1\n1 0.1 1\n0.1 1 1\n1 1 1\n', encoding='utf-8')
                cool = root/'차분하게.cube'
                cool.write_text('LUT_3D_SIZE 2\n0 0 0.2\n1 0 0.2\n0 1 0.2\n1 1 0.2\n'
                                '0 0 1\n1 0 1\n0 1 1\n1 1 1\n', encoding='utf-8')
                broken = root/'깨진.cube'
                broken.write_text('LUT_3D_SIZE 2\n0 0 0\n', encoding='utf-8')
                with patch.object(QMessageBox, 'warning') as warning:
                    window.add_luts([str(lut), str(cool), str(broken), str(lut)])
                warning.assert_called_once()
                self.assertEqual([Path(p).name for p in window.library.luts], ['화사하게.cube', '차분하게.cube'])
                self.assertEqual(window.lut_list.count(), 3)
                self.assertEqual(Path(window.current()['lut_path']).name, '화사하게.cube')
                self.assertEqual(window.current()['lut_strength'], 60)
                window.lut_list.setCurrentRow(2)
                self.assertEqual(Path(window.current()['lut_path']).name, '차분하게.cube')
                self.assertEqual(window.lut_list.currentItem().text(), '차분하게')
                window.toggle_lut_favorite()
                self.assertEqual(window.lut_list.item(1).text(), '★ 차분하게')
                self.assertEqual(window.lut_list.currentItem().text(), '★ 차분하게')
                self.assertIn('해제', window.lut_favorite_button.text())
                window.lut_favorites_only.setChecked(True)
                self.assertEqual(window.lut_list.count(), 2)
                window.save_library()
                self.assertEqual([Path(p).name for p in Library(root/'session').lut_favorites], ['차분하게.cube'])
                window.lut_favorites_only.setChecked(False)
                self.assertEqual(window.lut_list.count(), 3)
                window.remove_lut()
                self.assertEqual([Path(p).name for p in window.library.luts], ['화사하게.cube'])
                self.assertEqual(window.library.lut_favorites, [])
                self.assertEqual(window.current()['lut_path'], '')
                window.lut_list.setCurrentRow(0)
                window.lut_list.setCurrentRow(1)
                self.assertIn('화사하게.cube', window.lut_label.text())
                window.save_library()
                self.assertEqual([Path(p).name for p in Library(root/'session').luts], ['화사하게.cube'])
                window.apply_checked()
                self.assertTrue(all(item['lut_strength'] == 60 for item in window.library.items))
                window.compare_button.setChecked(True)
                self.assertTrue(window.viewer.compare)
                window.check_all(False)
                self.assertFalse(window.batch_button.isEnabled())
                window.check_all(True)
                self.assertTrue(window.batch_button.isEnabled())
                window.open_batch(False)
                dialog = window.batch_dialog
                self.assertEqual(len(dialog.items),2)
                self.assertTrue(all(item.lut_strength == 60 for item in dialog.items))
                self.assertIs(window.content_stack.currentWidget(), dialog)
                self.assertTrue(dialog.source_folder.isChecked())
                self.assertEqual(dialog.quality.value(), 90)
                # Parallel GPU processing is offered only on CUDA; the CPU edition runs sequentially.
                self.assertEqual(dialog.parallel_gpu.isChecked(), window.engine.status()['device'] == 'cuda')
                dialog.correction.setCurrentIndex(dialog.correction.findData('auto'))
                self.assertFalse(dialog.auto_widget.isHidden())
                dialog.auto_strength.setValue(75)
                dialog.auto_color.setChecked(False)
                dialog.auto_softness.setValue(12)
                dialog.auto_skin.setValue(24)
                auto_options = dialog.options()
                self.assertEqual(auto_options.auto_strength, 75)
                self.assertFalse(auto_options.auto_color)
                self.assertEqual(auto_options.auto_softness, 12)
                self.assertEqual(auto_options.auto_skin, 24)
                dialog.source_folder.setChecked(False)
                dialog.output.setText(str(root/'out'))
                dialog.format_box.setCurrentIndex(dialog.format_box.findData('PNG'))
                dialog.resize_box.setCurrentIndex(dialog.resize_box.findData('fit'))
                dialog.width.setValue(80)
                dialog.height.setValue(70)
                dialog.correction.setCurrentIndex(dialog.correction.findData('none'))
                dialog.start()
                self.wait_for(lambda: dialog.result is not None and not dialog.is_running())
                self.assertEqual(dialog.result['succeeded'],2,dialog.result)
                self.assertEqual(dialog.result['failed'],0)
                for file in dialog.result['files']:
                    self.assertEqual(read_image(file['output']).size,(80,53))
                dialog.correction.setCurrentIndex(dialog.correction.findData('match'))
                before_files = set((root/'out').iterdir())
                dialog.start_preview()
                self.wait_for(lambda: dialog.preview_worker is None)
                self.assertFalse(dialog.preview_viewer.after.isNull())
                self.assertEqual(set((root/'out').iterdir()), before_files)
                dialog.start()
                self.wait_for(lambda: dialog.result is not None and not dialog.is_running())
                self.assertEqual(dialog.result['succeeded'], 2, dialog.result)
                self.assertTrue(dialog.result['files'][0]['reference_unchanged'])
                dialog.accept()
                self.wait_for(lambda: window.batch_dialog is None)
                self.assertIs(window.content_stack.currentWidget(), window.workspace)
                window.save_library()
                self.assertEqual(Library(root/'session').items[0]['settings']['exposure'],.5)
                self.assertEqual(Library(root/'session').items[0]['settings']['skin'],22)
            finally:
                window.close()
                self.wait_for(lambda: all(worker is None or not worker.isRunning() for worker in
                                         [window.loader,window.preview_worker,window.thumb_worker,window.auto_worker]))
                self.app.processEvents()
                window.close()

    def test_thumbnail_exif_remove_selected_and_clear_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root/'인물 1.jpg', root/'인물 2.jpg'
            exif = Image.Exif()
            exif[271], exif[272] = 'Canon', 'EOS R6'
            # Cameras and phones store exposure settings in the Exif sub-IFD, not IFD0.
            camera = exif.get_ifd(0x8769)
            camera[36867], camera[42036] = '2025:04:03 12:34:56', 'RF24-70mm F2.8 L IS USM'
            camera[33437], camera[33434], camera[34855], camera[37386] = (
                IFDRational(28, 10), IFDRational(1, 125), 400, IFDRational(50, 1))
            Image.new('RGB', (100, 70), (130, 100, 80)).save(first, exif=exif)
            Image.new('RGB', (100, 70), (80, 100, 130)).save(second)
            window = MainWindow(Library(root/'session'))
            try:
                window.add_paths([str(first), str(second)])
                self.wait_for(lambda: window.engine is not None and window.displayed_path == str(first))
                self.assertEqual(window.list.viewMode(), QListView.ViewMode.IconMode)
                self.assertIn('셔터 1/125s', window.exif_label.text())
                self.assertIn('조리개 f/2.8', window.exif_label.text())
                self.assertIn('ISO 400', window.exif_label.text())
                self.assertIn('해상도 100 × 70', window.exif_label.text())
                self.assertIn('Canon EOS R6', window.exif_label.toolTip())
                self.assertIn('RF24-70mm', window.exif_label.toolTip())
                self.assertIn('50mm', window.exif_label.toolTip())
                self.assertIn('2025:04:03', window.exif_label.toolTip())
                window.set_library_view(False)
                self.assertEqual(window.list.viewMode(), QListView.ViewMode.ListMode)
                window.list.clearSelection()
                window.list.item(0).setSelected(True)
                window.list.item(1).setSelected(True)
                window.remove_selected()
                self.assertEqual(len(window.library.items), 0)
                window.add_paths([str(first), str(second)])
                window.clear_library(confirm=False)
                self.assertEqual(window.list.count(), 0)
                self.assertEqual(len(window.library.items), 0)
            finally:
                window.close()
                self.wait_for(lambda: all(worker is None or not worker.isRunning() for worker in
                                         [window.loader,window.preview_worker,window.thumb_worker,window.auto_worker]))
                self.app.processEvents()
                window.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
