# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
import sys, struct
from PIL import Image
def c15(v):
    r=(v&31)<<3; g=((v>>5)&31)<<3; b=((v>>10)&31)<<3
    return (r,g,b,255 if v else 0)
def tim(d, out, scale=1, clutidx=0):
    magic, flags = struct.unpack_from('<II', d, 0); assert magic == 0x10, hex(magic)
    bpp = flags & 3; p = 8; cluts=[]
    if flags & 8:
        blen, x, y, w, h = struct.unpack_from('<IHHHH', d, p)
        cl = struct.unpack_from('<%dH'%(w*h), d, p+12); p += blen
        n = 16 if bpp==0 else 256
        cluts = [cl[i:i+n] for i in range(0, len(cl), n)]
    blen, x, y, w, h = struct.unpack_from('<IHHHH', d, p); px = d[p+12:p+blen]
    W = w*4 if bpp==0 else w*2 if bpp==1 else w
    im = Image.new('RGBA', (W, h))
    pal = [c15(v) for v in cluts[clutidx]] if cluts else None
    data=[]
    for i in range(W*h):
        if bpp==0: v = (px[i//2] >> (4*(i&1))) & 15; data.append(pal[v])
        elif bpp==1: data.append(pal[px[i]])
        else: data.append(c15(px[2*i]|px[2*i+1]<<8))
    im.putdata(data)
    bg = Image.new('RGBA', im.size, (40,40,90,255)); bg.alpha_composite(im)
    if scale>1: bg = bg.resize((W*scale, h*scale), Image.NEAREST)
    bg.save(out); print(out, 'bpp', bpp, W, h, 'vram', x, y, 'cluts', len(cluts))
if __name__ == '__main__':
    tim(open(sys.argv[1],'rb').read(), sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 1)
