import io
from pathlib import Path
import unittest
import numpy as np
import torch
from PIL import Image

from engine import Engine, sample_lut, settings, read_image


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def test_identity_lut_axis_order_and_edges(self):
        axis = torch.linspace(0, 1, 33)
        b, g, r = torch.meshgrid(axis, axis, axis, indexing='ij')
        lut = torch.stack([r, g, b])
        pixels = torch.tensor([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,0.],[1.,1.,1.]],[[.12,.36,.85]]*5])
        torch.testing.assert_close(sample_lut(lut, pixels), pixels/1.0001, atol=1e-6, rtol=1e-5)

    def test_real_model_cuda_and_cpu_agree(self):
        engine = self.engine
        self.assertTrue(engine.status()['ai_ready'], engine.status())
        image = read_image(Path(__file__).resolve().parents[1] / 'samples/demo.jpg')
        image.thumbnail((600,600))
        p = settings({'ai':70,'softness':20})
        actual = np.asarray(engine.render(image,p,'test-real'))
        self.assertGreater(np.abs(actual.astype(float)-np.asarray(image)).mean(), 1)
        if torch.cuda.is_available():
            self.assertEqual(engine.device.type,'cuda')
            old_device=engine.device
            engine.device=torch.device('cpu'); engine.model.cpu(); engine.luts=engine.luts.cpu(); engine.cache.clear()
            try:
                cpu=np.asarray(engine.render(image,p,'test-real-cpu'))
                self.assertLessEqual(np.abs(actual.astype(float)-cpu.astype(float)).max(),1)
            finally:
                engine.device=old_device;engine.model.to(old_device);engine.luts=engine.luts.to(old_device);engine.cache.clear()

    def test_zero_settings_are_identity(self):
        a=np.random.default_rng(5).integers(0,256,(257,49,3),dtype=np.uint8)
        result=np.asarray(self.engine.render(Image.fromarray(a),settings(),'identity'))
        np.testing.assert_array_equal(a,result)

    def test_exif_orientation(self):
        image=Image.new('RGB',(80,40));exif=Image.Exif();exif[274]=6
        buf=io.BytesIO();image.save(buf,format='JPEG',exif=exif);buf.seek(0)
        result=read_image(buf);self.assertEqual(result.size,(40,80));self.assertNotIn(274,result.getexif())


if __name__ == '__main__':
    unittest.main(verbosity=2)
