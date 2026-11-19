# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
import sys, os, struct
def scan(d):
    out=[]; run=[]; start=0
    for i in range(0, len(d)-1, 2):
        v = d[i] | d[i+1]<<8
        try:
            c = struct.pack('>H', v).decode('cp932') if v >= 0x8140 else None
        except UnicodeDecodeError: c = None
        if c and len(c)==1:
            if not run: start=i
            run.append(c)
        else:
            if len(run)>=4 and sum('぀'<=x<='ヿ' for x in run)>=len(run)//3: out.append((start,''.join(run)))
            run=[]
    return out
for root,_,fs in os.walk(sys.argv[1]):
    for fn in sorted(fs):
        p=os.path.join(root,fn); h=scan(open(p,'rb').read())
        if h: print(sum(len(s) for _,s in h), len(h), os.path.relpath(p,sys.argv[1]), h[0][1][:40], h[len(h)//2][1][:40])
