# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Print entries START..START+COUNT-1 of tl/NAME.txt with their index.
Usage: tl_list.py NAME START COUNT [--empty]
--empty prints only entries whose EN line is still empty."""
import sys
name, start, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
empty = '--empty' in sys.argv
k = -1; head = ''; jp = None
for l in open(f'tl/{name}.txt', encoding='utf-8'):
    l = l.rstrip('\n')
    if l.startswith('## '): head = l[3:]
    elif l.startswith('JP: '): k += 1; jp = l
    elif l.startswith('EN: ') and start <= k < start + count:
        if empty and l[4:].strip(): continue
        print(f'#{k} {head}\n{jp}\n{l}')
print(f'-- entries in file: {k + 1}')
