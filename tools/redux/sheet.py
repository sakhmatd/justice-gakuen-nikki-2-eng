# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Contact sheet of harness screenshots: sheet.py OUTDIR [prefix]"""
import sys, glob, os
sys.path.insert(0, os.path.dirname(__file__))
from run import to_png
from PIL import Image, ImageDraw
d = sys.argv[1]; pre = sys.argv[2] if len(sys.argv) > 2 else 's'
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{d}/{pre}*.txt'))
names = [n for n in names if int(open(f'{d}/{n}.txt').read().split()[0]) > 0]
if len(sys.argv) > 3: names = names[int(sys.argv[3]):]
W, H, C = 200, 150, 6
ims = [Image.open(to_png(d, n)).resize((W, H)) for n in names]
sheet = Image.new('RGB', (W*C, (H+12)*((len(ims)+C-1)//C)))
dr = ImageDraw.Draw(sheet)
for i, (n, im) in enumerate(zip(names, ims)):
    x, y = (i % C)*W, (i // C)*(H+12); sheet.paste(im, (x, y)); dr.text((x+2, y+H), n, fill=(255, 255, 0))
sheet.save(f'{d}/sheet.png'); print(f'{d}/sheet.png', len(ims))
