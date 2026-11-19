# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Rebuild message files with English text in two-letter tiles.

Usage: build_mbms.py [--check] NAME [NAME...]   (e.g. MGT00; ALL = every file)
--check only prints the problems and writes nothing.
Reads translations from tl/*.txt, writes private/build/NAME.BIZ.lz
(decompressed) and private/build/NAME.BIZ (compressed). For every MGGnnX
built it also packs private/build/MGGnn.BZZ, the copy the game loads.
Untranslated entries get a "(untranslated)" fallback, so every file can be
built and carries the letter font at +0x12000 (tools/fontcheck.py).

Translation entries are keyed by the Japanese segment text. The main text
box is 20 cells of 20 px (two letters per cell) by 4 lines. Every line
wider than 20 cells is word-wrapped, also when the EN text has {n}; lines
with {sp} or runs of 2+ spaces are tables and are left as written. A page
may keep as many lines as its JP page has. Strings in other boxes (choice
lines, quiz answers, stat lines, save list) get the limits in BOXES.
"""
import sys, os, re, struct, glob
sys.path.insert(0, os.path.dirname(__file__))
from mbms import *
from dump import kanji_db, charmap, decode
from shared_chars import S
from game import FIRST, MSG_SIZE, LF_OFF, MSG_KEEP, STATE_OFF
import digraph, lz
LETTERS = digraph.letter_font()

LINES = 4                            # lines per page in the main box
MAXG = (TEXT - FBASE) // GB          # glyph slots before the text area
TAG = re.compile(r'\{[^}]*\}')
ICON = {v: k for k, v in S.items() if v.startswith('{')}
# Non-ASCII characters that exist in the shared font; each takes a full cell.
SYM = {v: k for k, v in S.items() if len(v) == 1 and not v.isascii() and k > 0}
SYMRUN = re.compile('([' + re.escape(''.join(SYM)) + '])')

def load_tl():
    tl = {}
    for f in sorted(glob.glob('tl/*.txt')):
        jp = None
        for line in open(f, encoding='utf-8'):
            line = line.rstrip('\n')
            if line.startswith('JP: '): jp = line[4:]
            elif line.startswith('EN: ') and jp is not None:
                if line[4:].strip(): tl[jp] = line[4:]
                jp = None
    return tl

def skeleton(t):
    """Control tags that must survive translation, in order."""
    tags = [x for x in TAG.findall(t) if x not in ('{n}', '{sp}') and not x.startswith(('{g:', '{t:', '{x:'))]
    # {p}/{end} order is structural; other tags may move within the segment
    return [x for x in tags if x in ('{p}', '{end}')], sorted(x for x in tags if x not in ('{p}', '{end}'))

CELLS = 20
NAMECELLS = 6                        # a name insert, in cells
# Insert widths (private/re/layout): surname and given name hold 12 letters
# (6 cells), the nickname 8 (4 cells). {v4}-{v7} append "-kun"/"-san" (8
# cells), {v8} "-chan" (9 cells). {v9} is the school name.
INSCELLS = {2: 4, 4: 8, 5: 8, 6: 8, 7: 8, 8: 9}
DATECELLS = 2                        # {mm}/{dd}: 1 or 2 digit glyphs

# Strings outside the main box (private/re/layout/findings.org, "Other text
# boxes"). Each entry: (files or None for all, group slot, pointer ids,
# rule). The group slot is gi in build(). Rules:
#   lines=1   one line only: no {n}, one page, and layout() must not wrap it
#   cells=N   at most N cells
#   slots=N   exactly N glyph slots (non-blank glyphs, see slots()); the
#             save list draws a fixed number of pre-rendered glyphs
#   gap=True  the text before the first {sp}{sp} keeps the JP cell count;
#             a stat icon is drawn at a fixed x over that gap
# The checks run only on given EN text, not on "=" or untranslated entries.
QUIZ = ('MGK04', 'MGK05', 'MGK10')
BOXES = [
    # type 19 lists: locations, characters, festival destinations. Two
    # columns are 200 px (10 cells); one column allows 12.
    (None, 2, [0, 1, 2, 3, 4, 24], dict(lines=1, cells=10)),
    (None, 2, range(75, 95), dict(lines=1, cells=10)),
    (None, 2, range(106, 113), dict(lines=1, cells=10)),
    # stat up/down lines, icon at x 84 + 0x800a41d4[stat]
    (None, 2, range(34, 52), dict(lines=1, gap=True)),
    # save list: CLEAR!, NO DATA, DATA ERROR
    (None, 2, [20], dict(lines=1, slots=6)),
    (None, 2, [21], dict(lines=1, slots=6)),
    (None, 2, [22], dict(lines=1, slots=9)),
    # save list date glyphs: digits 0-9, month marker, day marker
    (None, 5, [2], dict(lines=1, slots=12)),
    # CANCEL={CROSS} line, 12 VRAM slots
    (None, 5, [30], dict(lines=1, cells=12)),
    # quiz answers 5n+1..5n+4, one line in the main box
    (QUIZ, 6, [k for k in range(200) if k % 5], dict(lines=1, cells=20)),
    # quiz questions 5n: panel in the quiz background, x 50-462 from x 84,
    # so 18 cells; a 5th line covers the score bar (topic quizbox)
    (QUIZ, 6, range(0, 200, 5), dict(wrap=18, maxlines=4)),
]

def box_rule(name, gi, ids):
    """Merged BOXES rule for a segment, or {} if it is in the main box."""
    r = {}
    for files, g, bids, rule in BOXES:
        if g == gi and (files is None or name in files) and set(ids) & set(bids):
            r.update(rule)
    return r

def slots(vals):
    """Glyph slots in encoded values: every glyph except the blank 0x061e.
    Control codes and their args take none."""
    n = i = 0
    while i < len(vals):
        v = vals[i]; i += 1
        if v in (0x3002, 0x3003) or 0x2003 <= v <= 0x2008: i += 1
        elif (0 < v < 0x2000 and v != 0x061e) or v >= 0x8000: n += 1
    return n

def check_box(rule, jp, en, laid, vals):
    """Problems of one segment against its BOXES rule. en is the EN text as
    written, laid the text after layout(), vals the encoded values."""
    probs = []
    flat = re.sub(r'\{p\}|\{end\}', '', en)
    if rule.get('lines') == 1:
        parts = [x for x in re.split(r'\{n\}|\{p\}|\{end\}', laid) if x]
        if '{n}' in laid or len(parts) > 1:
            probs.append(f'one-line box has {len(parts)} lines: {laid}')
    if 'cells' in rule and cells(flat) > rule['cells']:
        probs.append(f'{cells(flat)} cells > {rule["cells"]}: {flat}')
    if 'slots' in rule and slots(vals) != rule['slots']:
        probs.append(f'{slots(vals)} glyph slots != {rule["slots"]}: {flat}')
    if rule.get('gap'):
        if '{sp}{sp}' not in en:
            probs.append(f'no {{sp}}{{sp}} icon gap: {flat}')
        else:
            want = cells(jp.split('{sp}{sp}')[0], jp=True)
            got = cells(en.split('{sp}{sp}')[0])
            if got != want:
                probs.append(f'{got} cells before {{sp}}{{sp}} != JP {want}: {flat}')
    return probs

def cells(line, jp=False):
    """Width of one line in 20 px cells, as the encoder lays it out. With
    jp=True every character is one glyph (JP text is not paired)."""
    n = 0
    for part in re.split(r'(\{[^}]*\})', line):
        if not part: continue
        if part.startswith('{'):
            if part == '{sp}' or part.startswith(('{t:', '{g:')) or part in ICON: n += 1
            elif part in ('{mm}', '{dd}'): n += DATECELLS
            elif re.match(r'\{v\d', part): n += INSCELLS.get(int(part[2]), NAMECELLS)
        elif jp:
            n += len(part)
        else:
            for q in SYMRUN.split(part):
                if not q: continue
                n += 1 if q in SYM else (len(q) + 1) // 2
    return n

def wrap_page(page, width=CELLS):
    """Greedy word wrap of one line of EN text (tags allowed) to width."""
    words = []
    for w in page.split(' '):
        # keep the full name "{v1} {v0}" on one line (private/re/fullname)
        if words and words[-1].endswith('{v1}') and w.startswith('{v0}'):
            words[-1] += ' ' + w
        else: words.append(w)
    lines, cur = [], ''
    for w in words:
        cand = (cur + ' ' + w) if cur else w
        if cur and cells(cand) > width:
            lines.append(cur); cur = w
        else:
            cur = cand
    if cur.strip(): lines.append(cur)
    return lines

def layout(en, maxlines=LINES, width=CELLS):
    """Word-wrap every line wider than width. Lines that contain {sp} or
    runs of 2+ spaces are table layout and are left alone.
    Returns (text, problems)."""
    probs = []
    out = []
    for part in re.split(r'(\{p\}|\{end\})', en):
        if part in ('{p}', '{end}') or not part: out.append(part); continue
        lines = []
        for line in part.split('{n}'):
            if '{sp}' in line or '  ' in line: lines.append(line)
            else: lines += wrap_page(line, width) or ['']
        if len(lines) > maxlines: probs.append(f'{len(lines)} lines: {part[:60]}')
        out.append('{n}'.join(lines))
    return ''.join(out), probs

HALF = 0x807f                        # pair (c, 0x7f): c, then advance 10 px

# Segments whose EN is generated here, keyed by (group slot, JP text). The
# first game used this for the hub date number table; none are known yet.
SLOT_EN = {}

class Encoder:
    def __init__(self):
        self.pairs = {}              # '{t:..}' tile -> glyph index
        self.npairs = 0
    def pair(self, p):
        if p == '  ': return 0x061e
        if p.startswith('{t:'):
            if p not in self.pairs: self.pairs[p] = FIRST + len(self.pairs)
            return self.pairs[p]
        a, b = p
        if not (' ' <= a <= '~' and ' ' <= b <= '~'):
            raise ValueError(f'non-ASCII in text: {p!r}')
        self.npairs += 1
        return 0x8000 | ord(a) << 7 | ord(b)
    def encode(self, text):
        vals = []
        for line in re.split(r'(\{n\}|\{p\}|\{end\})', text):
            if not line: continue
            # The last pair of an odd-length text part right before a name
            # insert, a colour tag or a face tag is padded with 0x7f, not
            # ' ' ("{col:0}Diary"{col:-}.", "Yay{col:5}<3", "Hey, {v0}",
            # "Enter your {col:0}name", "{mm}/{dd}"). The letter font cell 0x7f is
            # blank. MGAME, MGFIN and MGENT advance 10 for (c, 0x7f), so the
            # next part starts right after c.
            table = '{sp}' in line or '  ' in line
            parts = [x for x in re.split(r'(\{[^}]*\})', line) if x]
            for k, part in enumerate(parts):
                if part.startswith('{'):
                    vals += self.tag(part); continue
                q = ''
                for q in SYMRUN.split(part):
                    if not q: continue
                    if q in SYM: vals.append(SYM[q]); continue
                    for i in range(0, len(q), 2):
                        vals.append(self.pair(q[i:i+2].ljust(2)))
                if (not table and k + 1 < len(parts) and re.match(r'\{(v\d|col:|face:|mm\}|dd\})', parts[k + 1])
                        and len(q) % 2 and q not in SYM):
                    vals[-1] = HALF | ord(q[-1]) << 7
        return vals
    def tag(self, t):
        b = t[1:-1]
        if b == 'n': return [0x3000]
        if b == 'p': return [0x3001]
        if b == 'end': return [0]
        if b == 'sp': return [0x061e]
        if b == 'mm': return [0x200a]
        if b == 'dd': return [0x200b]
        if t in ICON: return [ICON[t]]
        m = re.fullmatch(r'face:(\d+)', b)
        if m: return [0x3002, int(m[1])]
        m = re.fullmatch(r'col:(-|\d+)', b)
        if m: return [0x3003, 255 if m[1] == '-' else int(m[1])]
        m = re.fullmatch(r'v(\d)(?::(\d+))?', b)
        if m: return [0x2000 + int(m[1])] + ([int(m[2])] if m[2] is not None else [])
        if b.startswith('t:'): return [self.pair(t)]
        m = re.fullmatch(r'g:(\d+)', b)
        if m: return [int(m[1])]
        m = re.fullmatch(r'x:([0-9a-fA-F]{4})', b)
        if m: return [int(m[1], 16)]
        raise ValueError('bad tag ' + t)

def build(name, tl, db):
    src = os.path.join(UNPACKED, name + '.BIZ.lz')
    d = bytearray(open(src, 'rb').read())
    m = charmap(bytes(d), db)
    enc = Encoder(); probs = []; missing = 0
    tops = list(struct.unpack_from('<8I', d, TEXT))
    # per group: list of (ptr ids, encoded values)
    out_groups = []
    for gi, to, ptrs in groups(bytes(d)):
        segs = []
        starts = sorted(set(ptrs))
        for si, s in enumerate(starts):
            e = starts[si+1] if si + 1 < len(starts) else to
            jp = decode(struct.unpack_from('<%dH' % ((e - s)//2), d, s), m)
            # strip zero padding words that precede the pointer table
            while jp.endswith('{end}{end}'): jp = jp[:-5]
            en = SLOT_EN.get((gi, jp), tl.get(jp))
            ids = [k for k, p in enumerate(ptrs) if p == s]
            if en == '=':
                raw = list(struct.unpack_from('<%dH' % ((e - s)//2), d, s))
                while raw[-2:] == [0, 0]: raw.pop()
                bad = [v for v in raw if FIRST <= v < 0x1000 and v != 0x061e]
                if bad: probs.append(f'{name} g{gi} s{si}: "=" used with kanji glyphs')
                segs.append((ids, raw)); continue
            given = en is not None
            if en is None:
                missing += 1; en = jp_fallback(jp)
            elif skeleton(en) != skeleton(jp):
                probs.append(f'{name} g{gi} s{si}: tags differ\n  JP {skeleton(jp)}\n  EN {skeleton(en)}')
            rule = box_rule(name, gi, ids)
            jlines = max(p.count('{n}') + 1 for p in jp.split('{p}'))
            # a box with its own frame (rule wrap=N, maxlines=M) wraps to N
            # cells and M lines; the main box wraps to CELLS x LINES
            width = rule.get('wrap', CELLS)
            laid, p = layout(en, rule.get('maxlines', max(LINES, jlines)), width)
            probs += [f'{name} g{gi} s{si}: {x}' for x in p]
            # a JP line wider than the box (e.g. the debug day grid g2 s15)
            # raises the limit for that string to the JP width
            lim = width if 'wrap' in rule and given else max([CELLS] + [cells(l, jp=True) for l in re.split(r'\{n\}|\{p\}|\{end\}', jp)])
            for line in re.split(r'\{n\}|\{p\}|\{end\}', laid):
                w = cells(line)
                if w > lim: probs.append(f'{name} g{gi} s{si}: line {w} cells > {lim}: {line}')
            vals = enc.encode(laid)
            if rule and given:
                probs += [f'{name} g{gi} s{si} (id {ids[0]}): {x}'
                          for x in check_box(rule, jp, en, laid, vals)]
            segs.append((ids, vals))
        out_groups.append((len(ptrs), segs))
    if FBASE + (FIRST + len(enc.pairs)) * GB > LF_OFF:
        probs.append(f'{name}: too many tiles ({len(enc.pairs)})')
    # font: keep shared 0..FIRST-1, append pairs
    nglyph = FIRST + len(enc.pairs)
    font = bytearray(d[:FBASE + FIRST*GB])
    for p in sorted(enc.pairs, key=lambda q: enc.pairs[q]):
        lv = digraph.levels_small(p[3:-1]) if p.startswith('{t:') else digraph.levels(p)
        font += digraph.pack(lv)
    blen = 12 + nglyph * GB
    struct.pack_into('<IHHHH', font, 0x174, blen, 0x200, 0, 5, nglyph * GH)
    out = bytearray(TEXT)
    out[:len(font)] = font
    # letter font read by tools/hook.py: 96 letters 0x20-0x7f, 100 B each
    assert len(LETTERS) == 96 * 100 and LF_OFF + len(LETTERS) <= STATE_OFF
    out[LF_OFF:LF_OFF + len(LETTERS)] = LETTERS
    # text area
    out += bytes(0x20)
    present = [k for k, t in enumerate(tops) if t]
    newtops = [0] * 8
    for slot, (nptr, segs) in zip(present, out_groups):
        table = [0] * nptr
        for ids, vals in segs:
            for k in ids: table[k] = LOAD + len(out)
            out += struct.pack('<%dH' % len(vals), *vals)
        while len(out) % 4: out += b'\0\0'
        newtops[slot] = LOAD + len(out)
        out += struct.pack('<%dI' % nptr, *table)
    struct.pack_into('<8I', out, TEXT, *newtops)
    size = len(out)
    # The JP data block at MSG_KEEP and the pointer table at 0x50000 are
    # read by MGAME (0x8011a000, 0x80120000); keep them byte for byte.
    if size > MSG_KEEP: probs.append(f'{name}: text end {size:#x} > {MSG_KEEP:#x}')
    else: out += bytes(MSG_KEEP - size) + d[MSG_KEEP:MSG_SIZE]
    return bytes(out), probs, missing, len(enc.pairs), size

def jp_fallback(jp):
    """Untranslated: keep only the control skeleton plus a marker. Only the
    first insert ({vN}, {mm}, {dd}) is kept, so the line fits the box."""
    out, ins = [], False
    for t in TAG.findall(jp):
        if t in ('{n}', '{sp}'): continue
        if re.match(r'\{(v\d|mm|dd)', t):
            if ins: continue
            ins = True
        out.append(t)
    return '(untranslated)' + ''.join(out)

# MGGnn.BZZ: archive of the five school files, the only copy the game loads
# (topic layout). Entry order T O P G J. Each rebuilt entry keeps its JP
# offset and must fit in its JP slot; the archive keeps its JP size.
BZZ_ORDER = 'TOPGJ'

def pack_bzz(nn):
    """Write private/build/MGGnn.BZZ from the built MGGnnX.BIZ files (JP
    entries for schools not built). Returns the problems."""
    import arc
    orig = open(f'private/disc/DAT2/MGG{nn}.BZZ', 'rb').read()
    ents = arc.toc(orig)
    assert len(ents) == 5
    out = bytearray(orig[:0x800]); probs = []
    for k, (o, s) in enumerate(ents):
        f = f'private/build/MGG{nn}{BZZ_ORDER[k]}.BIZ'
        data = open(f, 'rb').read() if os.path.exists(f) else orig[o:o + s]
        slot = (ents[k + 1][0] if k + 1 < 5 else len(orig)) - o
        if len(data) > slot:
            probs.append(f'MGG{nn}.BZZ entry {k} ({BZZ_ORDER[k]}): {len(data)} B > slot {slot}')
        struct.pack_into('<II', out, 8 * k, o, len(data))
        out += bytes(o - len(out)) + data
    out += bytes(len(orig) - len(out))
    if len(out) != len(orig):
        probs.append(f'MGG{nn}.BZZ: {len(out)} B != JP {len(orig)}')
    if not probs: open(f'private/build/MGG{nn}.BZZ', 'wb').write(out)
    return probs

if __name__ == '__main__':
    tl = load_tl(); db = kanji_db()
    os.makedirs('private/build', exist_ok=True)
    check = '--check' in sys.argv     # report problems only, write nothing
    args = [a for a in sys.argv[1:] if a != '--check']
    if args == ['ALL']: args = names()
    for name in args:
        data, probs, missing, npairs, size = build(name, tl, db)
        for p in probs: print(p)
        if check:
            print(f'{name}: untranslated {missing}, text end {size:#x}, problems {len(probs)}')
            continue
        open(f'private/build/{name}.BIZ.lz', 'wb').write(data)
        c = lz.compress(data, name=f'{name}.BIZ')
        open(f'private/build/{name}.BIZ', 'wb').write(c)
        print(f'{name}: {npairs} pairs, text end {size:#x}, untranslated {missing}, '
              f'compressed {len(c)} bytes, problems {len(probs)}')
    if not check:
        for nn in sorted({a[3:5] for a in args if re.fullmatch(r'MGG\d\d[GJOPT]', a)}):
            probs = pack_bzz(nn)
            for p in probs: print(p)
            print(f'MGG{nn}.BZZ: ' + ('NOT WRITTEN' if probs else 'packed'))
