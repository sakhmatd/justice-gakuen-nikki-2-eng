# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Redraw the diary-mode staff roll in English.

Reads tl/STAFF.txt (ID: JP | EN, ID = piece index in STAFF.BIN).
Writes private/build/STAFF.BIN (new piece rectangles and line offsets) and
private/build/EDSR.BIZ (the credit atlas redrawn, recompressed in the
original allocation). --png writes previews to private/re/staff/en/.
--check builds everything but writes nothing to private/build/.

Port of the first game's build_staff.py. On this disc STAFF.BIN loads
DAT2/EDSR.BIZ (file 633, one LZ stream), not MGSTAFF1.BIZ; MGSTAFF1 is a
leftover copy of the first game's file and nothing loads it.

STAFF.BIN (DAT1, loads at 0x800a2000):
- 0x47c: 118 pointers to the schedule records (8 bytes: mode, line, clut,
  piece, y/10, 0, 0, 0). x of a line (code 0x800a4a90): mode 0 centres it
  on the 512 px screen, 1 on the right half, 2 on the left half, 3 and 4
  on the whole screen; 1/3 slide in from the right, 2/4 from the left.
  A record with line 0 starts a new page.
- 0xdb4: 118 pointers to the piece records (8 s16: tpage x, tpage y,
  u, v, w, h, line width, x offset in the line).
EDSR.BIZ (decompressed 0xd1000 bytes): 16 picture TIMs, then TIM A 4bpp
1024x256 at VRAM (768,0), pixels at 0xa0120, and TIM B 4bpp 512x256 at
(768,256), pixels at 0xc0920. TIM B's second page (832,256) also holds
PRESENTED BY and the CAPCOM logo (v 20-99); they are kept. All pieces are
20 px rows. Job titles use the CLUT ramp 1-6 (the record's CLUT picks green,
blue or orange), names the grey ramp 8-15; 0 and 7 are transparent.
"""
import sys, os, re, struct
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
import lz

STAFF = 'private/disc/DAT1/STAFF.BIN'
ARC = 'private/disc/DAT2/EDSR.BIZ'
ENT = 'private/unpacked/EDSR.BIZ.lz'
FONT = 'bahnschrift.ttf'
BASE = 0x800a2000
SCHED, RECTS, NREC = 0x47c, 0xdb4, 118
SIZE, WEIGHT = 19, 600         # font px, Bahnschrift weight axis
SMALL = 14                     # font px for EN starting with {small}
MAXW = 240                     # widest half-screen line (original max 240)
GAP = 12                       # px between the two pieces of a line
ROW = 20
KEEP = (113, 114)              # PRESENTED, BY: already English, kept
# atlas pages: ((tpage x, tpage y), TIM, free 20-px rows)
ALL = range(256 // ROW)
PAGES = [((768, 0), 0, ALL), ((832, 0), 0, ALL), ((896, 0), 0, ALL),
         ((960, 0), 0, ALL), ((768, 256), 1, ALL),
         ((832, 256), 1, [0] + list(range(5, 256 // ROW)))]
TIMS = [(0xa0120, 1024, 7), (0xc0920, 512, 7)]    # pixels, width, bg index
CLUT = 0xa0014                 # TIM A CLUTs (B has the same 8 CLUTs)
DITHER = np.array([[0, .5], [.75, .25]])

def load_tl(path='tl/STAFF.txt'):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        k, v = line.split(':', 1)
        jp, en = (s.strip() for s in v.split('|'))
        if en == '=': en = jp
        out[int(k)] = (jp, en)
    return out

def tables(d):
    sched = [struct.unpack_from('<I', d, SCHED + 4 * i)[0] - BASE for i in range(NREC)]
    rects = [struct.unpack_from('<I', d, RECTS + 4 * i)[0] - BASE for i in range(NREC)]
    return [tuple(d[p:p + 5]) for p in sched], rects

def atlas(ent):
    """TIM A and B as index arrays (256 x W)."""
    out = []
    for pix, W, _ in TIMS:
        raw = np.frombuffer(ent[pix:pix + W * 128], np.uint8).reshape(256, W // 2)
        p = np.empty((256, W), np.uint8); p[:, 0::2] = raw & 15; p[:, 1::2] = raw >> 4
        out.append(p)
    return out

def orig_piece(tims, r):
    tx, ty, u, v, w, h = r[:6]
    return tims[0 if ty < 256 else 1][v:v + h, (tx - 768) * 4 + u:][:, :w]

def font(size, width):
    f = fontpath.truetype(FONT, size)
    f.set_variation_by_axes([WEIGHT, width])
    return f

def settings(size):
    return [(size, w) for w in range(100, 74, -5)] + [(s, 75) for s in range(size - 1, 13, -1)]

def render(text, maxw=MAXW, start=0):
    """Alpha (20 x w) for text. Walks the settings (narrower font, then
    smaller) from index start until the text fits maxw. Returns (alpha,
    setting index)."""
    ss = 4
    base = SIZE
    if text.startswith('{small}'): text, base = text[7:], SMALL
    sets = settings(base)
    for k in range(min(start, len(sets) - 1), len(sets)):
        size, width = sets[k]
        f = font(size * ss, width)
        tw = f.getlength(text)
        if tw <= maxw * ss: break
    w = int(tw + ss - 1) // ss + 2
    im = Image.new('L', (w * ss, ROW * ss), 0)
    cap = f.getbbox('H')
    top = (2 * ss) - cap[1] + (SIZE - size) * ss // 2
    ImageDraw.Draw(im).text((0, top), text, font=f, fill=255)
    a = np.asarray(im.resize((w, ROW), Image.LANCZOS), float) / 255
    a[a < 0.12] = 0
    cols = np.nonzero(a.max(0))[0]
    a = a[:, cols[0]:cols[-1] + 1] if len(cols) else a[:, :1]
    return a, k

def to_index(a, title):
    t = DITHER[np.arange(a.shape[0])[:, None] % 2, np.arange(a.shape[1])[None, :] % 2]
    if title: idx = 1 + np.floor(a * 4.9 + t)
    else: idx = 8 + np.floor(a * 5.5 + t * 0.67)
    idx = np.clip(idx, 1 if title else 8, 6 if title else 14).astype(np.uint8)
    idx[a <= 0] = 0
    return idx

def pages_of(sched):
    pages, cur = [], None
    for i, (m, b, c, pid, y) in enumerate(sched):
        if b == 0 and not (cur and m == 0 and cur[0][1][0] == 0): cur = []; pages.append(cur)
        cur.append((i, sched[i]))
    return pages

def layout(sched, tl, alpha, fit, kinds):
    """Line width and x offset per piece. A piece alone on its line is its
    own line (offset 0), so the game centres it on the mode's axis. The
    two-piece lines of a page with the same kinds (title+name or name+name)
    form two aligned columns (title right-aligned, name left-aligned),
    centred as a unit. If the columns are too wide, the whole group is
    re-rendered with one narrower setting."""
    pages = pages_of(sched)
    place = {}
    for pg in pages:
        lines = {}
        for i, (m, b, c, pid, y) in pg:
            if pid in alpha: lines.setdefault(y, []).append(pid)
        pairs = {}
        for y, ps in lines.items():
            if len(ps) == 1:
                place[ps[0]] = (alpha[ps[0]].shape[1], 0)
            else:
                assert len(ps) == 2, ps
                pairs.setdefault(tuple(kinds[p] for p in ps), []).append(ps)
        for kk, group in pairs.items():
            ids = [p for pq in group for p in pq]
            k = max(fit[p] for p in ids)
            while True:
                for p in ids:
                    if fit[p] != k: alpha[p], fit[p] = render(tl[p][1], 999, k)
                c1 = max(alpha[p].shape[1] for p, q in group)
                c2 = max(alpha[q].shape[1] for p, q in group)
                if c1 + GAP + c2 <= MAXW or k == len(settings(SIZE)) - 1: break
                k += 1
            for p, q in group:
                w1 = alpha[p].shape[1]
                place[p] = (c1 + GAP + c2, c1 - w1 if kk[0] == 'title' else 0)
                place[q] = (c1 + GAP + c2, c1 + GAP)
    return place, pages

def pack(widths):
    """First-fit decreasing into the free 20-px rows of the atlas pages."""
    rows = [[pg, r, 0] for pg, (_, _, free) in enumerate(PAGES) for r in free]
    at = {}
    for pid in sorted(widths, key=lambda p: -widths[p]):
        w = widths[pid]
        for row in rows:
            if row[2] + w <= 256:
                at[pid] = (row[0], row[1] * ROW, row[2]); row[2] += w; break
        else:
            raise SystemExit(f'atlas full at piece {pid} ({w} px)')
    return at, sum(r[2] for r in rows), len(rows) * 256

def build(png=False, check=False):
    tl = load_tl()
    sd = bytearray(open(STAFF, 'rb').read())
    sched, rptr = tables(sd)
    rects = [struct.unpack_from('<8h', sd, p) for p in rptr]
    ent = bytearray(open(ENT, 'rb').read())
    tims = atlas(ent)
    alpha, kinds, fit = {}, {}, {}
    for pid in sorted({s[3] for s in sched}):
        if pid in KEEP: continue
        op = orig_piece(tims, rects[pid])
        kinds[pid] = 'title' if ((op >= 1) & (op <= 6)).sum() > (op >= 8).sum() else 'name'
        if pid not in tl: raise SystemExit(f'piece {pid} missing in STAFF.txt')
        if not tl[pid][1].isascii(): raise SystemExit(f'piece {pid}: EN not ASCII')
        half = any(m in (1, 2) for m, b, c, q, y in sched if q == pid)
        alpha[pid], fit[pid] = render(tl[pid][1], MAXW if half else 256)
    place, pages = layout(sched, tl, alpha, fit, kinds)
    problems = 0
    for pid, (tot, off) in place.items():
        half = any(m in (1, 2) for m, b, c, q, y in sched if q == pid)
        if tot > (256 if half else 512):
            print(f'piece {pid}: line {tot} px too wide'); problems += 1
    at, used, total = pack({p: a.shape[1] for p, a in alpha.items()})
    # new atlas: background, plus the kept pieces and logo (TIM B, 832 page)
    new = [np.full_like(t, bg) for t, (_, _, bg) in zip(tims, TIMS)]
    new[1][20:100, 256:512] = tims[1][20:100, 256:512]
    for pid, (pg, v, u) in at.items():
        (tx, ty), t, _ = PAGES[pg]
        idx = to_index(alpha[pid], kinds[pid] == 'title')
        x0 = (tx - 768) * 4 + u
        sub = new[t][v:v + ROW, x0:x0 + idx.shape[1]]
        sub[idx != 0] = idx[idx != 0]
        tot, off = place[pid]
        struct.pack_into('<8h', sd, rptr[pid], tx, ty, u, v, idx.shape[1], ROW, tot, off)
    for (pix, W, _), n in zip(TIMS, new):
        ent[pix:pix + W * 128] = (n[:, 0::2] | (n[:, 1::2] << 4)).astype(np.uint8).tobytes()
    c = lz.compress(bytes(ent), name='EDSR.BIZ')
    slot = os.path.getsize(ARC)
    if len(c) > slot: print(f'EDSR: {len(c)} > {slot}'); problems += 1
    if png: preview(sched, [struct.unpack_from('<8h', sd, p) for p in rptr], new, ent, pages)
    print(f'STAFF: {len(alpha)} pieces, atlas {used} of {total} texels used, '
          f'EDSR {len(c):#x} of {slot:#x} bytes, problems {problems}')
    narrowed = {p: settings(SMALL if tl[p][1].startswith('{small}') else SIZE)[k]
                for p, k in fit.items() if k}
    if narrowed: print('narrowed (size/width%):', ' '.join(f'{p}:{s}/{w}' for p, (s, w) in sorted(narrowed.items())))
    if problems or check: return problems
    open('private/build/STAFF.BIN', 'wb').write(sd)
    open('private/build/EDSR.BIZ', 'wb').write(c)
    return 0

def page_image(pg, rects, tims, ent, title=None):
    """One page as STAFF.BIN places it (no slide-in), 512x480."""
    cl = struct.unpack_from('<128H', ent, CLUT)
    pal = lambda c: np.array([((v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3)
                              for v in cl[c * 16:c * 16 + 16]], np.uint8)
    im = Image.new('RGB', (512, 480), (24, 24, 32))
    dr = ImageDraw.Draw(im)
    dr.line([(256, 0), (256, 479)], fill=(60, 60, 60))
    for i, (m, b, c, pid, y) in pg:
        tx, ty, u, v, w, h, tot, off = rects[pid]
        src = tims[0 if ty < 256 else 1][v:v + h, (tx - 768) * 4 + u:][:, :w]
        x = {0: (512 - tot) // 2, 1: 256 + (256 - tot) // 2, 2: (256 - tot) // 2,
             3: (512 - tot) // 2, 4: (512 - tot) // 2}[m] + off
        mask = Image.fromarray(((src != 0) & (src != 7)).astype(np.uint8) * 255)
        im.paste(Image.fromarray(pal(c)[src]), (x, y * 10), mask)
    if title: dr.text((4, 464), title, fill=(120, 120, 120))
    return im

def preview(sched, rects, new, ent, pages):
    os.makedirs('private/re/staff/en', exist_ok=True)
    for k, pg in enumerate(pages):
        page_image(pg, rects, new, ent, f'page {k}').save(f'private/re/staff/en/page{k:02d}.png')

if __name__ == '__main__':
    sys.exit(build('--png' in sys.argv, '--check' in sys.argv))
