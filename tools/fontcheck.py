# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""List every message file and MGGnn.BZZ entry on a disc image whose
unpacked bytes at +0x12000 are not the letter font (digraph.letter_font()).
The letter-pair hook builds name and pair cells from that font, so events
with such a file show garbage names (topic nickquiz, pack).

Usage: fontcheck.py IMAGE.bin   (e.g. private/re/pack/test.bin)
Prints one line per bad file and a summary. Exit status 1 if any is bad.
Files are read through the ISO directory of the image. For the files the
exe file table lists, the table entry (LBA, size) of the image's exe must
agree with the directory, or the file is reported too.
"""
import os, sys, re, struct
sys.path.insert(0, os.path.dirname(__file__))
import arc, lz, digraph
from filetab import bcd
from game import LF_OFF, MSG_SIZE, TAB, NTAB, EXE_NAME

SS = 2352
MSG = re.compile(r'MG([CEFKMT][0-9A-F]{2}|G\d\d[GJOPT])\.BIZ$')
BZZ = re.compile(r'MGG\d\d\.BZZ$')

def files(f):
    """name -> (lba, size) for every file on the image."""
    def sec(lba):
        f.seek(lba * SS + 24); return f.read(2048)
    pvd = sec(16); assert pvd[1:6] == b'CD001'
    out = {}
    stack = [(struct.unpack_from('<I', pvd, 158)[0], struct.unpack_from('<I', pvd, 166)[0])]
    while stack:
        dl, dsz = stack.pop()
        data = b''.join(sec(dl + i) for i in range((dsz + 2047) // 2048))
        i = 0
        while i < len(data):
            l = data[i]
            if not l: i = (i // 2048 + 1) * 2048; continue
            el, es = struct.unpack_from('<I', data, i + 2)[0], struct.unpack_from('<I', data, i + 10)[0]
            nl = data[i + 32]; name = data[i + 33:i + 33 + nl]
            if name not in (b'\0', b'\1'):
                if data[i + 25] & 2: stack.append((el, es))
                else: out[name.decode().split(';')[0]] = (el, es)
            i += l
    return out

def read(f, lba, size):
    f.seek(lba * SS); out = bytearray()
    for _ in range((size + 2047) // 2048): out += f.read(SS)[24:24 + 2048]
    return bytes(out[:size])

def main(img):
    want = digraph.letter_font()
    bad, n = [], 0
    with open(img, 'rb') as f:
        fs = files(f)
        el, es = fs[EXE_NAME]; exe = read(f, el, es)
        tab = {}
        for k in range(NTAB):
            e = exe[TAB + 12 * k: TAB + 12 * k + 12]
            lba = (bcd(e[0]) * 60 + bcd(e[1])) * 75 + bcd(e[2]) - 150
            tab[lba] = struct.unpack_from('<I', e, 4)[0]
        def test(label, blob):
            nonlocal n
            n += 1
            d, _ = lz.decompress(blob)
            if len(d) != MSG_SIZE: bad.append(f'{label}: unpacked {len(d):#x} B, not {MSG_SIZE:#x}')
            elif d[LF_OFF:LF_OFF + len(want)] != want: bad.append(f'{label}: no letter font at {LF_OFF:#x}')
        for name in sorted(fs):
            lba, size = fs[name]
            if not (MSG.match(name) or BZZ.match(name)): continue
            if lba in tab and tab[lba] != size:
                bad.append(f'{name}: exe table size {tab[lba]} != directory {size}')
            blob = read(f, lba, size)
            if BZZ.match(name):
                for k, (o, s) in enumerate(arc.toc(blob)):
                    test(f'{name} entry {k} ({"TOPGJ"[k]})', blob[o:o + s])
            else:
                test(name + ('' if lba in tab else ' (loose, not loaded)'), blob)
    for b in bad: print(b)
    print(f'{img}: {n} message files and BZZ entries checked, {len(bad)} without the letter font')
    return 1 if bad else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
