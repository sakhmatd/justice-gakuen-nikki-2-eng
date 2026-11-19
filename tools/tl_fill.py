# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Fill EN lines of tl/NAME.txt from stdin, one translation per line,
starting at entry START (0-based, counting JP entries in the file).
Usage: tl_fill.py NAME START < lines.txt
A line '-' leaves that entry unchanged."""
import sys
name, start = sys.argv[1], int(sys.argv[2])
path = f'tl/{name}.txt'
new = [l.rstrip('\n') for l in sys.stdin if l.strip()]
lines = open(path, encoding='utf-8').read().split('\n')
k = -1; filled = 0
for i, l in enumerate(lines):
    if l.startswith('JP: '): k += 1
    elif l.startswith('EN: ') and start <= k < start + len(new):
        t = new[k - start]
        if t != '-': lines[i] = 'EN: ' + t; filled += 1
open(path, 'w', encoding='utf-8').write('\n'.join(lines))
print('filled', filled, 'entries', k + 1)
