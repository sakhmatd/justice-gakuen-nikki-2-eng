# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Options "game system" help panel in English (DAT2/OPTINFO.TIZ, drawn by
NEWOPT). Facts: private/re/optinfo/findings.org, private/re/newopt/.

OPTINFO.TIZ is one LZ stream of five 4bpp 256x256 TIMs (one per option, at
o*0x8800). NEWOPT 0x800a41d0 draws an entry as a list of (cell, count)
byte pairs, 0xfe = next line (+24 px), 0xff = end; each pair is one sprite
from the sheet at x 36 and y 342 (description) or 374/398/422 (value
text). The JP sheet is an atlas of 20x24 cells (12 x 10). That is too
small for English, so this script patches the span decoder to a 16 x R
grid of 16 x H cells: u = (c & 15) * 16, v = (c >> 4) * H, width
count * 16, height H. Each EN line is drawn once in a proportional font,
cut into 16 px tiles, and runs of tiles that already exist in the sheet
are reused (shared prefixes and the "Default setting." line).

Limits: the description buffer holds 2 sprites (the value buffer starts
at +0x70), the value text 20 sprites (the panel tile is at +0x4d0).

Run after build_sysmsg.py: it patches private/build/NEWOPT.BIN in place.
Its own bytes (span lists 0x800a2174-0x800a2294, entry pointers
0x800a2294, option pointers 0x800a22ec, decoder code) are restored from
the disc copy first, so reruns are safe. They lie below the sysmsg stub
and pool (0x800a236c-0x800a2b90); a check keeps it that way.
Writes private/build/OPTINFO.TIZ.
Usage: build_optinfo.py [--check] [--png DIR]
"""
import os, re, struct, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lz, streamsim
from build_sysmsg import NEWOPT_IN, NEWOPT_OUT, NBASE, NOP_POOL, NEWOPT_STUB

TL = 'tl/OPTINFO.txt'
SRC = 'private/disc/DAT2/OPTINFO.TIZ'
OUT = 'private/build/OPTINFO.TIZ'
SLOT = 50 * 2048                        # disc file 101654 B, 50 sectors
NOPT, NVAL = 5, (4, 6, 2, 2, 3)         # values per option (entries 1..n)
TIMSZ, PIXOFF = 0x8800, 0x40            # TIM stride; pixel data in a TIM
FONT = 'RobotoCondensed-Bold.ttf'
SIZE = 20                               # px (480-line screen)
CW, H = 16, 21                          # sheet cell (patched decoder)
TOP = 0                                 # ascender top row in a cell
MAXW = 444                              # line width; panel x 16-496, text x 36
X0, YS = 36, (342, 374, 398, 422)       # draw positions (from NEWOPT)
SPANS = 0x800a2174, 0x800a2294          # span list area
EPTR, OPTR = 0x800a2294, 0x800a22ec     # 22 entry pointers, 5 option pointers
DESC_MAX, VAL_MAX = 2, 20               # sprites per entry (buffer sizes)
# decoder patch: 0x800a42cc-0x800a432c (25 words) recomputes u, v
DEC, DEC_N = 0x800a42cc, 25
KEEP = (0x800a42d8, 0x800a42dc, 0x800a42e0, 0x800a42e4, 0x800a42e8,
        0x800a42ec, 0x800a42f0, 0x800a42f4)       # moved unchanged
HGT = 0x800a4334                        # addiu v0,zero,0x18 (sprite height)
WID = 0x800a4348                        # sll/addu/sll: count * 20

def N(a): return a - NBASE

def load_tl(path=TL):
    out, k = {}, None
    for l in open(path, encoding='utf-8'):
        l = l.rstrip('\n')
        if l.startswith('## '): k = l[3:].strip()
        elif l.startswith('EN: ') and k: out[k] = l[4:].strip()
    return out

_f = {}
def font(ss=4):
    if ss not in _f: _f[ss] = fontpath.truetype(FONT, SIZE * ss)
    return _f[ss]

def width(t): return font(1).getlength(t)

def wrap(text):
    if '{n}' in text: return [s.strip() for s in text.split('{n}')]
    out, cur = [], ''
    for w in text.split():
        if cur and width(cur + ' ' + w) > MAXW: out.append(cur); cur = w
        else: cur = (cur + ' ' + w).strip()
    return out + [cur]

LV = None
def levels(pal):
    global LV
    LV = np.array([(c & 31) * 8 for c in pal], float)

def render(text):
    """Line -> index array H x (k*CW): white text, 1 px drop shadow
    (index 1), edges on the palette's grey ramp, 0 transparent."""
    ss = 4; f = font(ss)
    asc, desc = f.getmetrics()
    w = int(np.ceil(f.getlength(text) / ss)) + 4
    im = Image.new('L', (w * ss, (H + 4) * ss), 0)
    # ascender top at row TOP: offset by the font's internal leading
    lead = f.getbbox('Hdl')[1]
    ImageDraw.Draw(im).text((ss, (TOP + 2) * ss - lead), text, font=f, fill=255)
    a = np.asarray(im.resize((w, H + 4), Image.BOX)) / 255.0
    s = np.zeros_like(a); s[1:, 1:] = a[:-1, :-1]
    clip = max(a[:2].max(), a[H + 2:].max(), s[H + 2:].max()) > 0.2
    a, s = a[2:H + 2], s[2:H + 2]
    cov = a + s * (1 - a)
    g = 248 * a + 8 * (1 - a)
    ramp = np.arange(1, 15)
    q = ramp[np.abs(g[..., None] - LV[ramp]).argmin(-1)]
    idx = np.where(cov > 0.3, q, 0).astype(np.uint8)
    cols = np.nonzero(idx.any(0))[0]
    wk = int(cols.max()) + 1 if len(cols) else 1
    k = -(-wk // CW)
    out = np.zeros((H, k * CW), np.uint8); out[:, :min(wk, k * CW)] = idx[:, :min(wk, k * CW)]
    return out, wk, clip

def pack_sheet(lines):
    """lines: list of index arrays. -> (tile stream, per line [(cell, n)])."""
    stream, keys = [], []
    res = []
    for img in lines:
        tiles = [img[:, i:i + CW] for i in range(0, img.shape[1], CW)]
        tk = [t.tobytes() for t in tiles]
        i, runs = 0, []
        while i < len(tk):
            best = (0, 0)
            for j in range(len(keys)):
                k = 0
                while i + k < len(tk) and j + k < len(keys) and keys[j + k] == tk[i + k]: k += 1
                if k > best[0]: best = (k, j)
            if best[0] >= 3 or (best[0] and i + best[0] == len(tk)):
                runs.append((best[1], best[0])); i += best[0]; continue
            # append new tiles up to the next spot where a long reuse starts
            j0 = len(keys)
            stream.append(tiles[i]); keys.append(tk[i]); i += 1
            if runs and runs[-1][0] + runs[-1][1] == j0: runs[-1] = (runs[-1][0], runs[-1][1] + 1)
            else: runs.append((j0, 1))
        spans = []
        for c, n in runs:                # split at sheet row ends
            while n:
                m = min(n, 16 - c % 16); spans.append((c, m)); c += m; n -= m
        res.append(spans)
    return stream, res

def build(check=False, pngdir=None):
    tl = load_tl(); probs, log = [], []
    src = lz.decompress(open(SRC, 'rb').read())[0]
    pal = struct.unpack_from('<16H', src, 20)
    levels(pal)
    rows = 256 // H; ncell = min(16 * rows, 0xfe)
    data = bytearray(src)
    lists = {}                           # (o, k) -> list of line span lists
    for o in range(NOPT):
        keys = [f'{o}.{k}' for k in range(NVAL[o] + 1)]
        lines, owner = [], []
        for key in keys:
            en = tl.get(key, '')
            if not en: probs.append(f'{key}: no EN'); continue
            ls = wrap(en)
            if len(ls) > (1 if key.endswith('.0') else 3):
                probs.append(f'{key}: {len(ls)} lines')
            for ln in ls:
                img, wk, clip = render(ln)
                if wk > MAXW: probs.append(f'{key}: {wk} px > {MAXW}: {ln}')
                if clip: probs.append(f'{key}: ink clipped vertically: {ln}')
                lines.append(img); owner.append(key)
        stream, spans = pack_sheet(lines)
        if len(stream) > ncell: probs.append(f'sheet {o}: {len(stream)} cells > {ncell}')
        for key, sp in zip(owner, spans): lists.setdefault(key, []).append(sp)
        for key in keys:
            n = sum(len(s) for s in lists.get(key, []))
            lim = DESC_MAX if key.endswith('.0') else VAL_MAX
            if n > lim: probs.append(f'{key}: {n} sprites > {lim}')
        log.append(f'sheet {o}: {len(stream)} of {ncell} cells, {len(lines)} lines')
        sheet = np.zeros((256, 256), np.uint8)
        for c, t in enumerate(stream[:ncell]):
            sheet[c // 16 * H:c // 16 * H + H, c % 16 * CW:c % 16 * CW + CW] = t
        p = o * TIMSZ + PIXOFF
        data[p:p + 0x8000] = (sheet[:, 0::2] | sheet[:, 1::2] << 4).astype(np.uint8).tobytes()
    # span lists
    blob = bytearray(); eoff = {}
    for o in range(NOPT):
        for k in range(NVAL[o] + 1):
            key = f'{o}.{k}'; eoff[key] = len(blob)
            for li, sp in enumerate(lists.get(key, [[(0, 1)]])):
                if li: blob.append(0xfe)
                for c, n in sp: blob += bytes((c, n))
            blob.append(0xff)
    room = SPANS[1] - SPANS[0]
    if len(blob) > room: probs.append(f'span lists {len(blob)} B > {room}')
    log.append(f'span lists {len(blob)} of {room} B')
    assert OPTR + 20 <= NEWOPT_STUB <= NOP_POOL[0], 'span area overlaps the sysmsg stub/pool'
    disc = open(NEWOPT_IN, 'rb').read()
    have = os.path.exists(NEWOPT_OUT)
    if not have and not check: probs.append(f'{NEWOPT_OUT} missing: run build_sysmsg.py first')
    nop = bytearray(open(NEWOPT_OUT, 'rb').read() if have else disc)
    if True:
        for a, b in ((SPANS[0], OPTR + 20), (DEC, DEC + 4 * DEC_N), (HGT, HGT + 4), (WID, WID + 12)):
            nop[N(a):N(b)] = disc[N(a):N(b)]
        nop[N(SPANS[0]):N(SPANS[1])] = blob[:room].ljust(room, b'\xff')
        e = 0
        for o in range(NOPT):
            struct.pack_into('<I', nop, N(OPTR) + 4 * o, EPTR + 4 * e)
            for k in range(NVAL[o] + 1):
                struct.pack_into('<I', nop, N(EPTR) + 4 * e, SPANS[0] + eoff[f'{o}.{k}']); e += 1
        assert e == 22
        w = lambda a: struct.unpack_from('<I', disc, N(a))[0]
        assert w(DEC) == 0x3c032aaa and w(HGT) == 0x24020018 and w(WID) == 0x00031080, 'NEWOPT code moved'
        code = [w(a) for a in KEEP] + [
            0x3262000f,          # andi v0,s3,15
            0x00021100,          # sll  v0,v0,4
            0xa2020014,          # sb   v0,0x14(s0)  u
            0x00131902,          # srl  v1,s3,4
            0x24080000 | H,      # addiu t0,zero,H
            0x00680018,          # mult v1,t0
            0x00003012,          # mflo a2
            0xa2060015]          # sb   a2,0x15(s0)  v
        code += [0] * (DEC_N - len(code))
        struct.pack_into('<%dI' % DEC_N, nop, N(DEC), *code)
        struct.pack_into('<I', nop, N(HGT), 0x24020000 | H)
        struct.pack_into('<3I', nop, N(WID), 0x00031100, 0, 0)   # sll v0,v1,4
    c = lz.compress(bytes(data), name='OPTINFO.TIZ')
    assert lz.decompress(c)[0] == bytes(data)
    so, done = streamsim.run(c)
    if not done or so[:len(data)] != bytes(data): probs.append('OPTINFO: streamsim failed')
    if len(c) > SLOT: probs.append(f'OPTINFO: {len(c)} B > slot {SLOT}')
    log.append(f'OPTINFO.TIZ {len(c)} B (slot {SLOT}, room {SLOT - len(c)})')
    print('\n'.join(log))
    for p in probs: print('PROBLEM', p)
    if not check and have and not probs:
        open(OUT, 'wb').write(c)
        open(NEWOPT_OUT, 'wb').write(nop)
        print('wrote', OUT, NEWOPT_OUT)
    if pngdir: previews(bytes(data), nop, pngdir)
    return len(probs)

# --- previews ---------------------------------------------------------------
BG = 'private/re/final9/opt/g0.png'     # EN game system screen (JP panel text)

def clean_panel():
    """Panel area of BG with the JP text filled from its neighbours."""
    from scipy import ndimage
    im = np.asarray(Image.open(BG).convert('RGB')).astype(float)
    p = im[338:468, 16:496].copy()
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    text = (np.abs(r - g) < 14) & (np.abs(g - b) < 14)
    text = ndimage.binary_dilation(text, iterations=1)
    _, (iy, ix) = ndimage.distance_transform_edt(text, return_indices=True)
    return p[iy, ix].astype(np.uint8), im.astype(np.uint8)

def draw_entry(canvas, sheet, rgb, lst, y0):
    """Emulate 0x800a41d0 with the patched decoder on a panel crop."""
    i, x, y = 0, X0, y0
    while lst[i] != 0xff:
        if lst[i] == 0xfe: x, y = X0, y + 24; i += 1; continue
        c, n = lst[i], lst[i + 1]; i += 2
        u, v, w = c % 16 * CW, c // 16 * H, n * CW
        t = sheet[v:v + H, u:u + w]
        sub = canvas[y - 338:y - 338 + H, x - 16:x - 16 + w]
        m = t[:sub.shape[0], :sub.shape[1]] > 0
        sub[m] = rgb[t[:sub.shape[0], :sub.shape[1]]][m]
        x += w

def previews(data, nop, d):
    os.makedirs(d, exist_ok=True)
    panel, full = clean_panel()
    pal = struct.unpack_from('<16H', data, 20)
    rgb = np.array([[(c & 31) * 8, (c >> 5 & 31) * 8, (c >> 10 & 31) * 8] for c in pal], np.uint8)
    rd = lambda a: struct.unpack_from('<I', nop, N(a))[0]
    tiles = []
    for o in range(NOPT):
        p = o * TIMSZ + PIXOFF
        raw = np.frombuffer(data[p:p + 0x8000], np.uint8).reshape(256, 128)
        sheet = np.empty((256, 256), np.uint8); sheet[:, 0::2] = raw & 15; sheet[:, 1::2] = raw >> 4
        Image.fromarray(rgb[sheet]).save(f'{d}/sheet{o}.png')
        ep = rd(OPTR + 4 * o)
        lst = lambda k: nop[N(rd(ep + 4 * k)):]
        for k in range(1, NVAL[o] + 1):
            cv = panel.copy()
            draw_entry(cv, sheet, rgb, lst(0), YS[0])
            draw_entry(cv, sheet, rgb, lst(k), YS[1])
            tiles.append(cv)
    w, h = tiles[0].shape[1], tiles[0].shape[0]
    cols = 2; rows_ = -(-len(tiles) // cols)
    out = np.full((rows_ * (h + 4), cols * (w + 4), 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        y, x = i // cols * (h + 4), i % cols * (w + 4)
        out[y:y + h, x:x + w] = t
    Image.fromarray(out).save(f'{d}/panels.png')
    # one full screen (option 1, value 6) for context
    f = full.copy(); f[338:468, 16:496] = tiles[10]
    Image.fromarray(f).save(f'{d}/screen.png')
    print('previews in', d)

if __name__ == '__main__':
    a = sys.argv[1:]
    png = a[a.index('--png') + 1] if '--png' in a else None
    sys.exit(1 if build('--check' in a, png) else 0)
