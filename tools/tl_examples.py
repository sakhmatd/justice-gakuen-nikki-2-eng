# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Show finished translations similar to the entries being translated.

Usage: tl_examples.py NAME START COUNT [-k K] [--exclude A,B]
       tl_examples.py --file PATH [-k K] [--exclude A,B]
Similarity is TF-IDF over JP character bigrams with tags removed, so lines
with the same speech habits (じゃ, のう, オイラ, ぜ) tend to match. Only
filled entries are used. --exclude drops tl files by NAME. The first
game's translations in private/n1-tl/ (a link to its tl/) are used too and
marked (N1); --no-n1 leaves them out. An example
already shown is printed once and referenced afterwards.
"""
import sys, re, glob, math, os
from collections import Counter

TAG = re.compile(r'\{[^}]*\}')

def grams(jp):
    t = TAG.sub(' ', jp)
    t = re.sub(r'[\s・…。、!?！？「」『』（）]+', ' ', t)
    return Counter(t[i:i+2] for i in range(len(t) - 1) if ' ' not in t[i:i+2])

N1 = 'private/n1-tl'

def load(exclude, n1=True):
    pairs = []
    fs = sorted(glob.glob('tl/*.txt'))
    if n1: fs += sorted(glob.glob(N1 + '/*.txt'))
    for f in fs:
        if os.path.basename(f)[:-4] in exclude: continue
        mark = ' (N1)' if f.startswith(N1) else ''
        jp = None
        for l in open(f, encoding='utf-8'):
            l = l.rstrip('\n')
            if l.startswith('JP: '): jp = l[4:]
            elif l.startswith('EN: ') and jp is not None:
                en = l[4:]
                if en.strip() and en != '=': pairs.append((jp, en + mark))
                jp = None
    return pairs

def queries(args):
    if args[0] == '--file':
        k = None
        for l in open(args[1], encoding='utf-8'):
            l = l.rstrip('\n')
            if l.startswith('#'): k = l.split()[0]
            elif l.startswith('JP: ') and k: yield k, l[4:]; k = None
        return
    name, start, count = args[0], int(args[1]), int(args[2])
    i = -1
    for l in open(f'tl/{name}.txt', encoding='utf-8'):
        if l.startswith('JP: '):
            i += 1
            if start <= i < start + count: yield f'#{i}', l[4:].rstrip('\n')

if __name__ == '__main__':
    a = sys.argv[1:]
    K = 3; exclude = set()
    if '-k' in a: j = a.index('-k'); K = int(a[j+1]); del a[j:j+2]
    if '--exclude' in a: j = a.index('--exclude'); exclude = set(a[j+1].split(',')); del a[j:j+2]
    n1 = '--no-n1' not in a
    if not n1: a.remove('--no-n1')
    pairs = load(exclude, n1)
    vecs = [grams(jp) for jp, _ in pairs]
    df = Counter(g for v in vecs for g in v)
    N = len(vecs)
    idf = {g: math.log(N / c) for g, c in df.items()}
    def weigh(v):
        w = {g: c * idf.get(g, math.log(N)) for g, c in v.items()}
        n = math.sqrt(sum(x * x for x in w.values())) or 1
        return {g: x / n for g, x in w.items()}
    W = [weigh(v) for v in vecs]
    shown = {}
    for key, jp in queries(a):
        q = weigh(grams(jp))
        sc = sorted(((sum(q[g] * w.get(g, 0) for g in q), i) for i, w in enumerate(W)), reverse=True)
        best = [(s, i) for s, i in sc[:K] if s > 0.12 and pairs[i][0] != jp]
        print(key)
        for s, i in best:
            if i in shown: print(f'  = {shown[i]}'); continue
            shown[i] = f'E{len(shown)}'
            print(f'  {shown[i]} JP: {pairs[i][0]}\n     EN: {pairs[i][1]}')
