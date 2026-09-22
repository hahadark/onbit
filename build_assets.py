"""Draw the app's simple native icon without external artwork dependencies."""
from pathlib import Path
from PIL import Image, ImageDraw

assets = Path(__file__).resolve().parent/'assets'
assets.mkdir(exist_ok=True)
image = Image.new('RGBA',(256,256),(0,0,0,0))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((4,4,252,252),radius=56,fill='#202b1d')
draw.ellipse((47,47,209,209),outline='#d5e4b9',width=6)
draw.pieslice((47,47,209,209),start=150,end=330,fill='#cbdcaf')
draw.polygon([(207,28),(213,42),(228,48),(213,54),(207,69),(201,54),(186,48),(201,42)],fill='#dae9bf')
image.save(assets/'onbit.ico',sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
