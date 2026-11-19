# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Constants for Nekketsu Seishun Nikki 2 (SLPS-02120). Other tools import
them from here. Values marked N1 are copied from the first game's project
and are not yet confirmed for this disc."""
EXE = 'private/disc/SLPS_021.20'
EXE_NAME = 'SLPS_021.20'
# The Redump dump (one MODE2/2352 track); setup.py links it at ORIG.
# An older non-Redump dump (CRC32 865b1432) has 34 zeroed sectors in
# MGBGE03/MGBGT11/MGBGT12.BZZ and a cracked MGTIT; do not use it.
DISC = 'Shiritsu Justice Gakuen - Nekketsu Seishun Nikki 2 (Japan)'
ORIG = f'private/{DISC}.bin'         # relative to the project root
DISC_SIZE, DISC_CRC = 663216960, 0xd97d0aaa
OUT = 'private/out/nikki2-en'
# Rival Schools - United by Fate (USA) (Disc 1) (Arcade Disc), SLUS-00681
# (Redump). build_fight.py and build_selp.py take English art from it.
USA_SIZE, USA_CRC = 725629632, 0xbe1ce7d9
USA_DIR = 'private/usa'               # disc/ and unpacked/, from setup.py

# CD file table in the exe: 12-byte entries (BCD MSF u32, size u32, param u32).
TAB = 0x76ee8                        # exe file offset (RAM 0x800866e8)
TAB_RAM = 0x800866e8
NTAB = 883

# Overlays (DAT1), load bases from mips.guess_base(). MGFIN is a 24-byte
# stub on this disc.
OVL_BASE = 0x800a2000                # most overlays (N1: 0x8009c000)
OVL_BASES = {0x8017a000: ('MGENT', 'PASS', 'SIKENGO2', 'SIKENG22'),
             0x80152000: ('MGAME061', 'SIKENGO1', 'SIKENG21')}
def ovl_base(name):
    n = name.replace('\\', '/').split('/')[-1].split('.')[0]
    for b, ns in OVL_BASES.items():
        if n in ns: return b
    return OVL_BASE

# Message files (N1: MBMS*.BIZ). 4bpp TIM font of 20x20 glyphs at 0x180,
# text area at TEXT: 8 u32 group table pointers, then strings + tables.
MSG_LOAD = 0x800d0000                # N1: 0x800ce000
MSG_TEXT = 0x31000                   # N1: 0x2d000
MSG_SIZE = 0x51800                   # decompressed size of every file
MSG_GLOB = 'MG[CEFKMT][0-9]*.BIZ.lz', 'MGG[0-9][0-9][GJOPT].BIZ.lz'
FIRST = 297                          # glyphs 0..296 are the same in every file
# Group slots: 0 male events, 1 female events, 2 names/places, 3 male diary,
# 4 female diary, 5 system, 6 extra (MGC only). Empty slots are null.
GROUPS = {0: 'male events', 1: 'female events', 2: 'names/places',
          3: 'male diary', 4: 'female diary', 5: 'system', 6: 'extra'}

# Runtime letter-pair hook (tools/hook.py, private/re/hook/findings.org).
# The exe zero run is 0x80010aa4..0x80010d28; its first word is the delay
# slot of the jr ra at 0x80010aa0, so the hook starts 3 words later (as
# in N1). 0x80010d28 is an empty jr ra; nop function.
# Taken parts: hook 0x80010ab0..0x80010bec, fightnick.py EXPAND + BUF
# 0x80010bf0..0x80010d0c, build_sysmsg stub 0x80010d0c..0x80010d28.
HOOK = 0x80010ab0
HOOK_END = 0x80010d28
LF_OFF = 0x12000                     # letter font: 96 x 100 B (20 rows x 5 B)
STATE_OFF = 0x14600                  # ring index (u32)
RING_OFF = 0x14800                   # ring: RING_N x 200 B, ends 0x1ac00
RING_N = 128
FONTPIX = MSG_LOAD + 0x180           # pixels of glyph 0 (font TIM at MSG_LOAD)
# Message file data after the text area, kept from the JP file: a block at
# +0x4a000 (MGAME loads MGSCH.TZZ entries to 0x8011a000) and a pointer
# table at +0x50000 (read at 0x80120000). Text must end below MSG_KEEP.
MSG_KEEP = 0x4a000

# Relocated files (tools/patch_bin.py). Each file listed gets that many
# sectors at the start of RELOC_DONOR, in order; the donor's directory
# record then starts after them. ZNULL.DAT (LBA 263742, 18088 sectors) is
# zero padding at the end of the disc: no file table entry, and neither the
# exe nor any overlay holds its LBA or MSF. The game reads files only
# through the exe file table, so a relocation rewrites the table entry
# (BCD MSF, size, check word) and the ISO record (LBA, size).
# The 26 sectors at LBA 828 after TRAIN.BIN are the DAT2 directory, not
# free space.
RELOC_DONOR = 'ZNULL.DAT'
RELOCATE = [('KANRI.BIN', 24)]       # 0xc000 B; RAM limit 0x800d0000
