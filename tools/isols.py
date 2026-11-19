# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
import sys, struct
f = open(sys.argv[1], 'rb')
def sec(lba):
    f.seek(lba*2352); raw = f.read(2352)
    return raw[24:24+2048], raw[16:24]
pvd,_ = sec(16)
assert pvd[1:6] == b'CD001', pvd[:8]
print('volid', pvd[40:72].decode(errors='replace').strip(), 'vol size', struct.unpack('<I', pvd[80:84])[0])
root = pvd[156:156+34]
def walk(lba, size, path):
    data = b''.join(sec(lba+i)[0] for i in range((size+2047)//2048))
    i = 0
    while i < len(data):
        l = data[i]
        if l == 0:
            i = (i//2048+1)*2048; continue
        rec = data[i:i+l]
        elba = struct.unpack('<I', rec[2:6])[0]; esz = struct.unpack('<I', rec[10:14])[0]
        flags = rec[25]; nl = rec[32]; name = rec[33:33+nl]
        i += l
        if name in (b'\x00', b'\x01'): continue
        # XA attr
        su = rec[33+nl+(1 if nl%2==0 else 0):]
        xa = ''
        if len(su) >= 14 and su[6:8] == b'XA':
            a = struct.unpack('>H', su[4:6])[0]; xa = ' XAattr=%04x' % a
        n = name.decode(errors='replace')
        if flags & 2:
            walk(elba, esz, path + n + '/')
        else:
            print(f'{elba:7d} {esz:10d} {path}{n}{xa}')
walk(struct.unpack('<I', root[2:6])[0], struct.unpack('<I', root[10:14])[0], '/')
