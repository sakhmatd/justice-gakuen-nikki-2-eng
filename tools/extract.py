# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
# extract all Form1 files from a MODE2/2352 image into outdir (skips XA/STR)
import sys, struct, os
img, out = sys.argv[1], sys.argv[2]
f = open(img, 'rb')
def sec(lba):
    f.seek(lba*2352); return f.read(2352)
pvd = sec(16)[24:24+2048]
def walk(lba, size, path):
    data = b''.join(sec(lba+i)[24:24+2048] for i in range((size+2047)//2048))
    i = 0
    while i < len(data):
        l = data[i]
        if l == 0: i = (i//2048+1)*2048; continue
        rec = data[i:i+l]; i += l
        elba = struct.unpack('<I', rec[2:6])[0]; esz = struct.unpack('<I', rec[10:14])[0]
        flags = rec[25]; nl = rec[32]; name = rec[33:33+nl].decode()
        if name in ('\x00', '\x01'): continue
        p = os.path.join(path, name.split(';')[0])
        if flags & 2: walk(elba, esz, p); continue
        if name.endswith(('.XAS;1', '.STR;1')) or name.startswith('ZNULL'): continue
        os.makedirs(path, exist_ok=True)
        with open(p, 'wb') as o:
            n = (esz+2047)//2048
            o.write(b''.join(sec(elba+k)[24:24+2048] for k in range(n))[:esz])
r = pvd[156:190]
walk(struct.unpack('<I', r[2:6])[0], struct.unpack('<I', r[10:14])[0], out)
