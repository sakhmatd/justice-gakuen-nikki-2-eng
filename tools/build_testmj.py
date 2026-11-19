# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Exam lesson pages in English: DAT2/TESTMJ.TIZ (33 LZ entries, each one
256x256 4bpp TIM at VRAM (896,256), cut into 32x16 cells, 8 per row) and
the bubble scripts that point at the cells. Wording: tl/EXAM.txt. RE:
private/re/testmj/findings.org. Port of the first game's build_exam.py.

Entries 000-029 are the Exam Practice lessons 1-30. Their message and
command scripts are in SIKENGO1/SIKENGO2 (same data, two load bases).
Entries 030-032 are the three diary term exams; their message scripts are
in SIKENG21/SIKENG22. Row 0 of each image holds the lesson title, which
the Exam Practice HUD draws (SIKEN 0x800a7b58) with its width from the u8
table at SIKEN file 0xcc0 (lessons 1-30).

Every message line and command gets its own run of cells; the scripts
are rewritten to match. The command records of entries 030-032 point at
unused cells and are left alone.

Writes private/build/TESTMJ.TIZ, SIKEN.BIN, SIKENGO1.BIN and SIKENGO2.BIN.
--terms also writes entries 030-032 and SIKENG21.BIN, SIKENG22.BIN;
without it those pages stay JP. The overlays are read from private/build
when present (build_examgfx.py --extra writes SIKENG21/22), else from the
disc; the layout is always parsed from the disc copy. Run it after
build_examgfx.py. --dry writes nothing; --png DIR writes previews.
"""
import os, sys, re, struct, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fontpath
from scipy import ndimage
import arc, lz, streamsim
from game import ovl_base

TL = 'tl/EXAM.txt'
SRC = 'private/disc/DAT2/TESTMJ.TIZ'
DAT1, BUILD = 'private/disc/DAT1', 'private/build'
FONT = 'DejaVuSansCondensed-Bold.ttf'   # Bitstream Vera licence
MAXW = 14 * 32                  # widest bubble line, px
MAXLINES = 5
NPRACT = 30                     # Exam Practice lessons (SIKENGO1/2)
TITLE_W = 0xcc0                 # SIKEN file offset: HUD title widths, u8 x 30
GO = ('SIKENGO1', 'SIKENGO2')   # header word 0x14 = message table, 0x10 = command table
G2 = ('SIKENG21', 'SIKENG22')   # same header words, 3 term exams

BG = 14                         # set per image by use_palette()
RAMP = list(range(1, 14))       # dark -> light text indices
TFILL, TLINE = 0, 1             # title fill / outline

# --- tl ----------------------------------------------------------------
def parse_tl(path=TL):
    lessons, cmds, cur = {}, {}, None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line or line.startswith('#'): continue
        m = re.fullmatch(r'\[(\d+|cmds)\]', line)
        if m: cur = m[1]; continue
        k, v = line.split(':', 1)
        if cur == 'cmds': cmds[int(k)] = v.strip(); continue
        lessons.setdefault(int(cur) - 1, {})[k.strip()] = v.strip()
    return lessons, cmds

# --- overlay scripts ---------------------------------------------------
class Ovl:
    """Message and command scripts of one overlay. Layout from the disc
    copy, edits applied to the build copy (if any)."""
    def __init__(self, name):
        self.name, self.B = name, ovl_base(name)
        self.orig = open(f'{DAT1}/{name}.BIN', 'rb').read()
        p = f'{BUILD}/{name}.BIN'
        self.c = bytearray(open(p, 'rb').read() if os.path.exists(p) else self.orig)
        assert len(self.c) == len(self.orig), name
        self.cmdtab, self.msgtab = (w - self.B for w in struct.unpack_from('<II', self.orig, 0x10))

    def nmsg(self):
        n = 0
        while True:
            v = struct.unpack_from('<I', self.orig, self.msgtab + 4 * n)[0] - self.B
            if not 0 < v < self.msgtab: return n
            n += 1

    def msg(self, L):
        p = struct.unpack_from('<I', self.orig, self.msgtab + 4 * L)[0] - self.B
        o, lines = p, [[]]
        while True:
            b = self.orig[o]; o += 1
            if b == 0xff: return lines
            if b == 0xfe: lines.append([]); continue
            lines[-1].append((b, self.orig[o])); o += 1

    def cmds(self, L):
        """-> struct offset, [(offset, param, lines)]"""
        so = struct.unpack_from('<I', self.orig, self.cmdtab + 4 * L)[0] - self.B
        order = struct.unpack_from('<I', self.orig, so)[0] - self.B
        out, k = [], 1
        while True:
            q = struct.unpack_from('<I', self.orig, so + 4 * k)[0] - self.B
            if not order < q < so: return so, out
            o = q + 2; lines = [[]]
            for _ in range(self.orig[q + 1]):
                if self.orig[o] == 0xff: lines.append([]); o += 1
                lines[-1].append((self.orig[o], self.orig[o + 1])); o += 2
            out.append((q, self.orig[q], lines)); k += 1

    def put_msgs(self, scripts, probs):
        n = len(scripts)
        ptrs = [struct.unpack_from('<I', self.orig, self.msgtab + 4 * L)[0] - self.B for L in range(n)]
        lo, hi = min(ptrs), self.msgtab
        blob = bytearray(); new = []
        for s in scripts: new.append(lo + len(blob)); blob += s
        if len(blob) > hi - lo:
            probs.append(f'{self.name}: message scripts {len(blob)} > {hi - lo} bytes'); return
        self.c[lo:hi] = bytes(blob) + bytes(hi - lo - len(blob))
        for L, p in enumerate(new): struct.pack_into('<I', self.c, self.msgtab + 4 * L, p + self.B)
        self.used = (len(blob), hi - lo)

# --- TIM ---------------------------------------------------------------
def read_tim(d):
    bl = struct.unpack_from('<I', d, 8)[0]; p = 8 + bl
    blen, x, y, w, h = struct.unpack_from('<IHHHH', d, p)
    assert (x, y, w, h) == (896, 256, 64, 256) and p + blen == len(d)
    raw = np.frombuffer(d[p + 12:p + blen], np.uint8)
    px = np.empty(raw.size * 2, np.uint8); px[0::2] = raw & 15; px[1::2] = raw >> 4
    return d[:p + 12], px.reshape(h, w * 4)

def pack_tim(head, px):
    f = px.ravel()
    return head + bytes((f[0::2] | (f[1::2] << 4)).astype(np.uint8))

def clut_rgb(head):
    return np.array([[(v & 31) * 8, (v >> 5 & 31) * 8, (v >> 10 & 31) * 8]
                     for v in struct.unpack_from('<16H', head, 20)], np.uint8)

_fonts = {}
def font(size):
    if size not in _fonts: _fonts[size] = fontpath.truetype(FONT, size)
    return _fonts[size]

def lum(v): return (v & 31) + (v >> 5 & 31) + (v >> 10 & 31)

def use_palette(head, orig, start):
    """Derive background, text ramp and title colours from the original."""
    global BG, RAMP, TFILL, TLINE
    clut = struct.unpack_from('<16H', head, 20)
    cnt = collections.Counter(orig[16:, :].ravel().tolist())
    BG = cnt.most_common(1)[0][0]
    def grey(v):
        r, g, b = v & 31, v >> 5 & 31, v >> 10 & 31
        return max(r, g, b) - min(r, g, b) <= 2
    used = [i for i, n in cnt.items() if i != BG and n > 8 and grey(clut[i])]
    RAMP = sorted(used, key=lambda i: lum(clut[i]))
    tcnt = collections.Counter(orig[0:16, 0:min(start, 8) * 32].ravel().tolist())
    tused = [i for i, n in tcnt.items() if i != BG and n > 8]
    TFILL = max(tused, key=lambda i: lum(clut[i])); TLINE = min(tused, key=lambda i: lum(clut[i]))

def alpha_to_idx(a):
    r = np.array(RAMP, np.uint8)
    k = np.clip(np.rint((1 - a) * (len(r) - 1)), 0, len(r) - 1).astype(int)
    return np.where(a > 0.12, r[k], BG).astype(np.uint8)

def text_width(s, size=13):
    return font(size).getlength(s)

def render_line(text, orig, size=13):
    """One bubble line (with {c:N} icon tokens) as a 16-row index strip."""
    w = 0; segs = []
    for p in re.split(r'(\{c:\d+\})', text):
        if not p: continue
        m = re.fullmatch(r'\{c:(\d+)\}', p)
        if m:
            n = int(m[1]); cell = orig[(n // 8) * 16:(n // 8) * 16 + 16, (n % 8) * 32:(n % 8) * 32 + 32]
            cols = np.nonzero((cell != BG).any(axis=0))[0]
            cell = cell[:, :cols.max() + 2] if len(cols) else cell[:, :8]
            segs.append(('img', cell)); w += cell.shape[1] + 2
        else:
            tw = int(np.ceil(font(size).getlength(p))) + 1
            segs.append(('txt', p, tw)); w += tw
    out = np.full((16, max(1, w)), BG, np.uint8); x = 0
    for s in segs:
        if s[0] == 'img':
            c = s[1]; out[:, x:x + c.shape[1]] = c; x += c.shape[1] + 2
        else:
            im = Image.new('L', (s[2], 16)); ImageDraw.Draw(im).text((0, 0), s[1], font=font(size), fill=255)
            out[:, x:x + s[2]] = alpha_to_idx(np.asarray(im, np.float32) / 255); x += s[2]
    return out

TITLE_SIZE, TITLE_BASE = 13, 11       # caps on rows 1-10, like the HUD digit

def title_alpha(text, size, ss=4):
    f = font(size * ss)
    W = (int(f.getlength(text)) // ss + 8) * ss
    im = Image.new('L', (W, 15 * ss))
    ImageDraw.Draw(im).text((4 * ss, TITLE_BASE * ss), text, font=f, fill=255, anchor='ls')
    a = np.asarray(im.resize((W // ss, 15), Image.BOX), np.float32) / 255
    cols = np.nonzero((a > 0.05).any(0))[0]
    return a[:, cols.min():cols.max() + 1]

def render_title(text, width):
    """HUD title strip, 16 rows: light fill, 1 px dark outline. The HUD draws
    15 rows, so row 15 stays empty. Shrinks the size until it fits."""
    size = TITLE_SIZE
    while True:
        a = title_alpha(text, size)
        if a.shape[1] + 2 <= width or size <= 8: break
        size -= 1
    a = np.pad(a, ((0, 1), (1, 1)))
    fill = a > 0.45
    dil = ndimage.binary_dilation(fill, np.ones((3, 3), bool))
    assert not fill[0].any() and not dil[15].any(), text
    return np.where(fill, TFILL, np.where(dil, TLINE, BG)).astype(np.uint8), size

def wrap(msg):
    lines = []
    for para in msg.split('|'):
        cur = ''
        for word in para.split(' '):
            cand = (cur + ' ' + word) if cur else word
            vis = re.sub(r'\{c:\d+\}', 'MMM', cand)
            if cur and text_width(vis) > MAXW - 4: lines.append(cur); cur = word
            else: cur = cand
        lines.append(cur)
    return lines

class Cells:
    def __init__(self, px, start):
        self.px = px; self.next = start
    def put(self, strip):
        n = (strip.shape[1] + 31) // 32
        first = self.next
        if first + n > 128: raise ValueError('out of cells')
        pad = np.full((16, n * 32), BG, np.uint8); pad[:, :strip.shape[1]] = strip
        for k in range(n):
            c = first + k
            self.px[(c // 8) * 16:(c // 8) * 16 + 16, (c % 8) * 32:(c % 8) * 32 + 32] = pad[:, k * 32:k * 32 + 32]
        self.next += n
        return first, n

# --- build -------------------------------------------------------------
def build_page(L, tr, cmdtext, head, orig, olines, cmds, ovls, probs):
    """Draw one page. Returns pixels, message script, title width.
    cmds: [(offset, param, lines)] to rewrite in every overlay of ovls."""
    start = min(l[0][0] for l in olines if l)
    use_palette(head, orig, start)
    px = np.full_like(orig, BG)
    t, size = render_title(tr['title'], min(start, 8) * 32)
    if t.shape[1] > min(start, 8) * 32: probs.append(f'L{L+1}: title too wide')
    px[0:16, 0:t.shape[1]] = t
    cells = Cells(px, start)
    lines = wrap(tr['msg'])
    if len(lines) > MAXLINES: probs.append(f'L{L+1}: {len(lines)} lines')
    script = bytearray()
    for i, ln in enumerate(lines):
        strip = render_line(ln, orig)
        if strip.shape[1] > MAXW: probs.append(f'L{L+1}: line too wide: {ln}')
        first, n = cells.put(strip)
        if i: script.append(0xfe)
        script += bytes([first, n])
    script.append(0xff)
    if cmds:
        so, lst = cmds
        addrs = sorted(q for q, _, _ in lst)
        for q, param, cl in lst:
            if not any(cl): continue
            parts = [cells.put(render_line(ln, orig)) for ln in cmdtext.get(param, '?').split('|')]
            body = bytearray([param, len(parts)])
            for i, (f, n) in enumerate(parts):
                if i: body.append(0xff)
                body += bytes([f, n])
            nxt = [a for a in addrs if a > q]
            limit = (nxt[0] if nxt else so) - q
            if len(body) > limit:
                probs.append(f'L{L+1}: cmd p{param} needs {len(body)} > {limit} bytes'); continue
            for o in ovls: o.c[q:q + len(body)] = body
    return px, bytes(script), t.shape[1], size, len(lines)

def build(write=True, pngdir=None, outdir=BUILD, terms=False):
    lessons, cmdtext = parse_tl()
    D = open(SRC, 'rb').read()
    toc = arc.toc(D)
    assert len(toc) == 33, len(toc)
    go = [Ovl(n) for n in GO]; g2 = [Ovl(n) for n in G2]
    siken = Ovl.__new__(Ovl); siken.name = 'SIKEN'
    p = f'{BUILD}/SIKEN.BIN'
    siken.c = bytearray(open(p if os.path.exists(p) else f'{DAT1}/SIKEN.BIN', 'rb').read())
    assert go[0].nmsg() == NPRACT and g2[0].nmsg() == 3
    for a, b in ((go[0], go[1]), (g2[0], g2[1])):       # same layout, two bases
        assert (a.msgtab, a.cmdtab) == (b.msgtab, b.cmdtab)
        for L in range(a.nmsg()): assert a.msg(L) == b.msg(L) and a.cmds(L)[1] == b.cmds(L)[1]
    probs, log = [], []
    scripts_go, scripts_g2, pages = [], [], []
    for L in range(33):
        src = lz.decompress(D[toc[L][0]:toc[L][0] + toc[L][1]])[0]
        head, orig = read_tim(src)
        if L < NPRACT:
            olines, cmds, ovls = go[0].msg(L), go[0].cmds(L), go
        else:
            olines, cmds, ovls = g2[0].msg(L - NPRACT), None, g2
        px, script, tw, size, nl = build_page(L, lessons[L], cmdtext, head, orig, olines, cmds, ovls, probs)
        (scripts_go if L < NPRACT else scripts_g2).append(script)
        if L < NPRACT: siken.c[TITLE_W + L] = tw
        pages.append((head, orig, px, pack_tim(head, px)))
        log.append(f'{L:03d} {lessons[L]["title"]!r} {tw}px size {size}, {nl} lines')
    for o in go: o.put_msgs(scripts_go, probs)
    if terms:
        for o in g2: o.put_msgs(scripts_g2, probs)
    # repack the archive: every entry in its own slot
    out = bytearray(D)
    for L, (o, s) in enumerate(toc):
        end = toc[L + 1][0] if L + 1 < len(toc) else len(D)
        data = pages[L][3]
        if L >= NPRACT and not terms:          # term exam pages stay JP
            log[L] += ' (not written, needs --terms)'; continue
        c = lz.compress(data, name='TESTMJ.TIZ')
        assert lz.decompress(c)[0] == data
        so, done = streamsim.run(c)
        if not done or so[:len(data)] != data: probs.append(f'TESTMJ/{L:03d}: streamsim failed')
        if len(c) > end - o: probs.append(f'TESTMJ/{L:03d}: {len(c)} > slot {end - o}'); continue
        out[o:end] = c + bytes(end - o - len(c))
        struct.pack_into('<II', out, 8 * L, o, len(c))
        log[L] += f', {len(c)} B (room {end - o - len(c)})'
    print('\n'.join(log))
    if hasattr(go[0], 'used'): print(f'scripts: GO {go[0].used[0]}/{go[0].used[1]} B')
    if hasattr(g2[0], 'used'): print(f'scripts: G2 {g2[0].used[0]}/{g2[0].used[1]} B')
    if pngdir: previews(pages, lessons, pngdir)
    if probs:
        print('\n'.join(probs)); print('build_testmj: nothing written'); return 1
    if write:
        os.makedirs(outdir, exist_ok=True)
        open(f'{outdir}/TESTMJ.TIZ', 'wb').write(bytes(out))
        ovls = go + [siken] + (g2 if terms else [])
        for o in ovls: open(f'{outdir}/{o.name}.BIN', 'wb').write(bytes(o.c))
        print(f'wrote {outdir}/TESTMJ.TIZ', ' '.join(o.name + '.BIN' for o in ovls))
    return 0

def previews(pages, lessons, d):
    """pages.png: every page JP | EN (rows 0-7); titles.png: JP and EN title
    rows of all 33 lessons."""
    os.makedirs(d, exist_ok=True)
    rows = []
    for L, (head, orig, px, _) in enumerate(pages):
        pal = clut_rgb(head)
        rows.append(np.concatenate([pal[orig[:16]], np.full((16, 8, 3), 255, np.uint8), pal[px[:16]]], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(f'{d}/titles.png')
    tiles = []
    for L, (head, orig, px, _) in enumerate(pages):
        pal = clut_rgb(head)
        tiles.append(np.concatenate([pal[orig[:128]], np.full((128, 4, 3), 255, np.uint8), pal[px[:128]],
                                     np.full((128, 12, 3), 128, np.uint8)], 1))
    grid = [np.concatenate(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3])), 1)
            for i in range(0, 33, 3)]
    Image.fromarray(np.concatenate(grid, 0)).save(f'{d}/pages.png')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--png')
    ap.add_argument('--out', default=BUILD, help='output directory (default private/build)')
    ap.add_argument('--terms', action='store_true',
                    help='also entries 030-032 and SIKENG21/22 (after build_examgfx.py --extra)')
    a = ap.parse_args()
    sys.exit(build(not a.dry, a.png, a.out, a.terms))
