# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Build a steps file from a compact sequence, starting from a saved state.
Usage: seq.py (boot|STATEFILE) 'CIRCLE START wait30 CIRCLE ...' OUT.lua [gap]
Each button press is followed by `gap` frames (default 40) and a screenshot."""
import sys
state, seq, out = sys.argv[1], sys.argv[2].split(), sys.argv[3]
gap = int(sys.argv[4]) if len(sys.argv) > 4 else 40
BOOT = [(1830,'START'),(2000,'START'),(2450,'START'),(2700,'START'),(2950,'START'),(3250,'CIRCLE')]
pre = ''
if state == 'boot':
    steps = [f"{{{fr}, 'press', '{b}', 6}}" for fr, b in BOOT]
    f = 3650
else:
    pre = f"local _f = Support.File.open(OUTDIR .. '/../states/{state}', 'READ')\nPCSX.loadSaveState(_f); _f:close()\n"
    steps = []
    f = 20
k = 0
for t in seq:
    if t.startswith('wait'):
        f += int(t[4:]); steps.append(f"{{{f}, 'shot', 's{k:03d}'}}"); k += 1; continue
    if t in ('bot:on', 'bot:off'):
        steps.append(f"{{{f}, 'bot', {'true' if t == 'bot:on' else 'false'}}}"); continue
    if t.startswith('ram:'):
        steps.append(f"{{{f}, 'dump', '{t[4:]}', 0x80000000, 0x200000}}"); continue
    if t.startswith('vram:'):
        steps.append(f"{{{f}, 'vram', '{t[5:]}'}}"); continue
    if t.startswith('save:'):
        steps.append(f"{{{f}, 'save', '../states/{t[5:]}'}}"); continue
    if '~' in t:
        b, n = t.split('~'); n = int(n)
        steps.append(f"{{{f}, 'press', '{b}', {n}}}")
        for q in range(60, n, 60):
            steps.append(f"{{{f + q}, 'shot', 's{k:03d}'}}"); k += 1
        f += n + gap
        continue
    b, _, n = t.partition('*')
    for _ in range(int(n or 1)):
        steps.append(f"{{{f}, 'press', '{b}', 4}}"); f += gap
        steps.append(f"{{{f}, 'shot', 's{k:03d}'}}"); k += 1
steps.append(f"{{{f + 2}, 'quit'}}")
open(out, 'w').write(pre + 'STEPS = {' + ',\n'.join(steps) + '}\n')
