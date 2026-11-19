# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""List choice entries (EN with {n}) that use he/him/her and whose JP key
occurs more than once in private/script, with the two lines before each use.
Shared keys get one EN, so a gendered choice may be wrong elsewhere."""
import re,glob,os
pairs={}
for f in sorted(glob.glob('tl/MG*.txt')):
    jp=None
    for l in open(f,encoding='utf-8'):
        l=l.rstrip('\n')
        if l.startswith('JP: '): jp=l[4:]
        elif l.startswith('EN: ') and jp is not None:
            pairs.setdefault(jp,(os.path.basename(f),l[4:])); jp=None
occ={}
for f in sorted(glob.glob('private/script/MG*.txt')):
    L=[l.rstrip('\n') for l in open(f,encoding='utf-8')]
    for i,l in enumerate(L):
        if l in pairs:
            prev=[x for x in L[max(0,i-6):i] if not x.startswith('##')][-2:]
            occ.setdefault(l,[]).append((os.path.basename(f),i+1,prev))
for jp,(f,en) in pairs.items():
    if '{n}' in en and re.search(r'\b(him|her|his|he|she)\b',en,re.I) and '"' not in en and len(occ.get(jp,[]))>1:
        print('==',f,en); 
        for o in occ[jp]: print('   ',o[0],o[1],' | '.join(o[2])[:160])
