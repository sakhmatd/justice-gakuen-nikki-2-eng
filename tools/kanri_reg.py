# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""EDIT Register strings for tools/build_kanri.py (topic kanri,
private/re/kanri/register_room.org).

Every register-mode string of KANRI.BIN: the main and CPU-menu buttons,
the panel messages, Yes/No, the char-select prompt and info panel, the
per-button help lines, the delete confirm and OK/Cancel. The strings are
MOJI2 Latin codes (as the diary labels); build_kanri places them in the
freed JP pools and then at the end of the file (KANRI is relocated by
patch_bin.py, game.RELOCATE). The four art buttons keep their 0xffRC codes;
their TIM 1 cells are redrawn by tools/kanri_art.py.

Prim buffer limits (0x38 B per cell, blanks count): buttons 6, panel lines
16, select prompt line 1 20 and line 2 22 (after the OK/Cancel buffer
moves to 0x5140), OK/Cancel 26, info labels by the s16 table at 0x1628,
Male/Female 7 (a canary showed 0x5f08-0x6100 unused). Help lines: table
at 0x1640, (count, first record) per CPU button, records 0x824 + 12 x
index, up to 3 lines of 16 per button, 12 records.
"""
import struct
import numpy as np

BASE = 0x800a2000
BTN_C = 431                   # button frames x 372-489
PANEL_C = 196                 # lower panel x 24-364, inner frame 34-360
T_BTN, T_PANEL = 6, 7         # pitch 14 and 13
YESNO_BOX = (152, 240)        # 64x20 cursor boxes at x 120 + 88 x choice

# id: (record, cell limit, mode, x, tighten); mode 'c' centre the ink on x,
# 'l' put the ink's left edge where the JP ink starts.
REG = {
    's0': (0x4d8, 6, 'c', BTN_C, T_BTN),         # ロード
    's1': (0x4e4, 6, 'c', BTN_C, T_BTN),         # 登録
    's5': (0x4fc, 6, 'c', BTN_C, T_BTN),         # 終了
    's3': (0x830, 6, 'c', BTN_C, T_BTN),         # ステータス
    's4': (0x83c, 6, 'c', BTN_C, T_BTN),         # コマンド
    'r34': (0x854, 6, 'c', BTN_C, T_BTN),        # 登録抹消
    'r35': (0x860, 6, 'c', BTN_C, T_BTN),        # 戻る
    'r6': (0x520, 16, 'c', PANEL_C, T_PANEL),    # ロードされていません
    's9': (0x538, 16, 'c', PANEL_C, T_PANEL),    # (no code reference)
    's10': (0x544, 16, 'c', PANEL_C, T_PANEL),
    's11': (0x550, 16, 'c', PANEL_C, T_PANEL),   # (no code reference)
    's12': (0x55c, 16, 'c', PANEL_C, T_PANEL),
    'r11': (0x568, 16, 'c', PANEL_C, T_PANEL),
    's14': (0x574, 16, 'c', PANEL_C, T_PANEL),
    's15': (0x580, 16, 'c', PANEL_C, T_PANEL),
    's16': (0x58c, 16, 'c', PANEL_C, T_PANEL),
    's17': (0x598, 16, 'c', PANEL_C, T_PANEL),
    'r26': (0x61c, 16, 'c', PANEL_C, T_PANEL),
    's29': (0x628, 16, 'c', PANEL_C, T_PANEL),   # (no code reference)
    'r28': (0x634, 16, 'c', PANEL_C, T_PANEL),
    'r29': (0x640, 16, 'c', PANEL_C, T_PANEL),
    'r51': (0x920, 16, 'c', PANEL_C, T_PANEL),   # Are you sure?
    'r37': (0x878, 20, 'l', None, T_PANEL),      # select prompt line 1
    'r38': (0x884, 22, 'l', None, T_PANEL),      # line 2
    'r52': (0x95c, None, 'l', None, T_PANEL),    # info labels (cells: table)
    'r53': (0x968, None, 'l', None, T_PANEL),
    'r54': (0x974, None, 'l', None, T_PANEL),
    's24@': (0x980, None, 'l', None, T_PANEL),   # なかよしキャラ, EN of s24
    'r56': (0x98c, 7, 'l', None, T_PANEL),       # 男 (canary: free to 0x6100)
    'r57': (0x998, 7, 'l', None, T_PANEL),       # 女
}
ART = {'a0': 0x4cc, 'a3': 0x4f0, 'a30': 0x824, 'a33': 0x848}   # x set, codes kept
ART_W = 120
YESNO_REC = 0x5a4
OKC_REC, OKC_X, OKC_T, OKC_GAP = 0x86c, 112, 5, 8   # Cancel starts at cell 8 (x 232)
OKC_BUF, OKC_BUF_OLD, OKC_BUF_NEW, OKC_LIM = 0x7480, 0xd4c0, 0xd140, 26
PROMPT2_LIM_MOVED = 22
LABEL_TAB, LABEL_CELLS = 0x1628, 30                  # s16 x 4 (+2 unused)
HELP_TAB, HELP_REC0, HELP_N, HELP_Y = 0x1640, 0x824, 12, (240, 262, 284)
HELP_KEYS = ['help.cpu', 'help.status', 'help.moves', 'help.pass', 'help.delete', 'help.back']
HELP_FIRST = 9                                       # records 9..20
# strings that stay where they are: art (codes kept) and the Latin rows
# NO DATA, CLEAR, DATA ERROR (code loads their addresses with lui/addiu)
KEEP_STR = [0x2d0, 0x2f0, 0x308, 0x450, 0x45c, 0x64c, 0x674]
POOLS = [(0x2d0, 0x4cc), (0x64c, 0x824), (0x92c, 0x95c), (0x9a4, 0x9d4)]
RESTORE = [(0x4cc, 0x64c), (0x824, 0x9a4), (LABEL_TAB, LABEL_TAB + 12),
           (HELP_TAB, HELP_TAB + 12), (OKC_BUF, OKC_BUF + 4)]


def ids():
    return set(REG) - {'s24@'} | set(ART) | {'s18.yes', 's18.no', 'r36.ok', 'r36.cancel'} | set(HELP_KEYS)


def free_list(disc):
    keep = []
    for o in KEEP_STR:
        n = struct.unpack_from('<H', disc, o)[0]
        keep.append((o, o + ((2 + 2 * n + 3) & ~3)))
    free = []
    for a, b in POOLS:
        for ka, kb in sorted(keep):
            if a <= ka < b:
                if ka > a: free.append([a, ka])
                a = kb
        if b > a: free.append([a, b])
    return free


def jp_left(disc, rec, ink, gpx):
    """Screen x of the JP string's first ink column."""
    x = struct.unpack_from('<h', disc, rec)[0]
    p = struct.unpack_from('<I', disc, rec + 4)[0] - BASE
    n = struct.unpack_from('<H', disc, p)[0]
    codes = list(struct.unpack_from('<%dH' % n, disc, p + 2))
    lo, _ = ink(gpx, [c for c in codes if c < 0x400 or c == 0xfff], 0)
    return x + (lo or 0)


def yesno(gpx, ink, cy, cn):
    """One string 'Yes' + blanks + 'No' whose words centre in the boxes
    (N1 build_kanri.yesno)."""
    best = None
    for t in range(4, 10):
        a = 20 - t
        for gap in range(1, 10):
            codes = cy + [0xfff] * gap + cn
            if len(codes) > 16: continue
            l1, r1 = ink(gpx, cy, t); l2, r2 = ink(gpx, cn, t)
            off = (len(cy) + gap) * a
            x1 = YESNO_BOX[0] - (l1 + r1 + 1) / 2
            x2 = YESNO_BOX[1] - (off + (l2 + r2 + 1) / 2)
            err = abs(x1 - x2) + 0.3 * abs(t - 6)
            if best is None or err < best[0]: best = (err, codes, round((x1 + x2) / 2), t)
    return best


def strings(tl, disc, cm, gpx, ink, encode, problems):
    """-> list of (records [(rec, x, y or None)], codes, tighten) and the
    word patches [(file offset, bytes)]."""
    out, words = [], []

    def enc(key, en):
        try: return encode(en, cm)
        except ValueError as e: problems.append(f'{key}: {e}')

    # info label cell budget from the EN lengths
    lab = {}
    for key in ('r52', 'r53', 'r54', 's24@'):
        en = tl.get('s24' if key == 's24@' else key, '')
        lab[key] = len(en)
    cells = [lab['r52'], lab['r53'], lab['r54']]
    if sum(cells) + lab['s24@'] > LABEL_CELLS:
        problems.append(f'info labels {cells} + {lab["s24@"]} > {LABEL_CELLS} cells')
    lims = {'r52': cells[0], 'r53': cells[1], 'r54': cells[2], 's24@': LABEL_CELLS - sum(cells)}
    old = list(struct.unpack_from('<6h', disc, LABEL_TAB))
    words.append((LABEL_TAB, struct.pack('<6h', *(cells + old[3:]))))

    for key, (rec, lim, mode, cx, t) in REG.items():
        en = tl.get('s24' if key == 's24@' else key, '')
        if en in ('', '='): continue
        codes = enc(key, en)
        if codes is None: continue
        lim = lims.get(key, lim)
        if key == 'r38': lim = PROMPT2_LIM_MOVED
        if len(codes) > lim:
            problems.append(f'{key}: {en!r} has {len(codes)} cells > {lim}'); continue
        lo, hi = ink(gpx, codes, t)
        if mode == 'c': x = round(cx - (lo + hi + 1) / 2)
        else: x = jp_left(disc, rec, ink, gpx) - lo
        out.append(([(rec, x, None)], codes, t))
    for key, rec in ART.items():
        if tl.get(key, '') in ('', '='): continue
        p = struct.unpack_from('<I', disc, rec + 4)[0] - BASE
        assert struct.unpack_from('<H', disc, p)[0] == 6
        words.append((rec, struct.pack('<h', BTN_C - ART_W // 2)))
    # Yes / No
    if tl.get('s18.yes') and tl.get('s18.no'):
        cy, cn = enc('s18.yes', tl['s18.yes']), enc('s18.no', tl['s18.no'])
        if cy and cn:
            _, codes, x, t = yesno(gpx, ink, cy, cn)
            out.append(([(YESNO_REC, x, None)], codes, t))
    # OK / Cancel: words start at the JP word positions (x 112 and 232)
    if tl.get('r36.ok') and tl.get('r36.cancel'):
        co, cc = enc('r36.ok', tl['r36.ok']), enc('r36.cancel', tl['r36.cancel'])
        if co and cc:
            if len(co) > OKC_GAP - 1: problems.append('r36.ok too long')
            codes = co + [0xfff] * (OKC_GAP - len(co)) + cc
            if len(codes) > OKC_LIM: problems.append(f'r36: {len(codes)} cells > {OKC_LIM}')
            lo, _ = ink(gpx, co, OKC_T)
            out.append(([(OKC_REC, OKC_X + 1 - lo, None)], codes, OKC_T))
            w = struct.unpack_from('<I', disc, OKC_BUF)[0]
            assert w & 0xffff == OKC_BUF_OLD, hex(w)
            words.append((OKC_BUF, struct.pack('<I', w & 0xffff0000 | OKC_BUF_NEW)))
    # help lines: records 9.. in button order, table (count, first)
    if all(tl.get(k, '') not in ('', '=') for k in HELP_KEYS):
        idx, tab = HELP_FIRST, b''
        for key in HELP_KEYS:
            lines = [s.strip() for s in tl[key].split('/')]
            if len(lines) > 3: problems.append(f'{key}: {len(lines)} lines > 3')
            tab += bytes([len(lines), idx])
            for i, line in enumerate(lines):
                codes = enc(key, line)
                if codes is None: continue
                if len(codes) > 16: problems.append(f'{key}: {line!r} {len(codes)} cells > 16')
                lo, hi = ink(gpx, codes, T_PANEL)
                rec = HELP_REC0 + 12 * (idx + i)
                out.append(([(rec, round(PANEL_C - (lo + hi + 1) / 2), HELP_Y[i])], codes, T_PANEL))
            idx += len(lines)
        if idx - HELP_FIRST > HELP_N: problems.append(f'help: {idx - HELP_FIRST} lines > {HELP_N}')
        words.append((HELP_TAB, tab))
    return out, words
