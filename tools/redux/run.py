# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Run PCSX-Redux headless with the Lua harness.
Usage: run.py IMAGE.cue STEPSLUA OUTDIR [extra lua file...]
STEPSLUA is Lua source assigning STEPS = {...}.
Each run gets its own copy of the config directory (memory cards, pcsx.json)
in OUTDIR/_cfg, so several runs can go at once. No lock is needed:
patch_bin.py renames a finished image into place, so a running emulator keeps
its open file."""
import os, sys, subprocess, shutil, struct
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from game import EXE_NAME, TAB, TAB_RAM, NTAB
REDUX = os.path.join(ROOT, 'private/pcsx-redux/squashfs-root/AppRun')
BIOS = os.path.join(ROOT, 'private/bios/scph5500.bin')
def exe_patch(image, outdir):
    """A save state carries the exe of the build (or the JP disc) that made
    it, so a JP state has no hook and an old EN state has an old file table
    (a size that is too large makes the streaming loader decode past the
    end of a file). Write the words of IMAGE's exe that differ from the JP
    exe, then IMAGE's whole file table, to OUTDIR/_exepatch.bin as (RAM
    offset, word) pairs. harness.lua applies them after every 'load' step
    (POSTLOAD). Overlays are not patched: test MGAME screens from states
    saved on an EN image."""
    lba = next(int(l.split()[0]) for l in open(os.path.join(ROOT, 'private/lba.txt')) if EXE_NAME in l)
    jp = open(os.path.join(ROOT, 'private/disc', EXE_NAME), 'rb').read()
    binf = os.path.splitext(os.path.abspath(image))[0] + '.bin'
    exe = b''
    with open(binf, 'rb') as f:
        for k in range((len(jp) + 2047) // 2048):
            f.seek((lba + k) * 2352 + 24); exe += f.read(2048)
    t_addr, t_size = struct.unpack_from('<II', jp, 0x18)
    n = min(len(jp), 0x800 + t_size)
    pairs = [(t_addr - 0x800 + i, exe[i:i+4]) for i in range(0x800, n, 4)
             if exe[i:i+4] != jp[i:i+4] and not TAB <= i < TAB + NTAB * 12]
    pairs += [(t_addr - 0x800 + i, exe[i:i+4]) for i in range(TAB, TAB + NTAB * 12, 4)]
    path = os.path.join(os.path.abspath(outdir), '_exepatch.bin')
    with open(path, 'wb') as f:
        for a, w in pairs: f.write(struct.pack('<I', a & 0x1fffff) + w)
    return (f"\nfunction POSTLOAD() local m = PCSX.getMemPtr()\n"
            f"  local pf = io.open('{path}', 'rb'); local pb = pf:read('*a'); pf:close()\n"
            f"  local p = ffi.cast('const uint32_t*', pb)\n"
            f"  for i = 0, #pb / 8 - 1 do ffi.copy(m + p[2*i], p + 2*i + 1, 4) end end\n"
            # A 'load' step's state is applied after the step, so the POSTLOAD
            # call inside the step is overwritten. Repeat it at the next frame,
            # before the other steps of that frame (frame + 0.5 sorts first).
            f"for i = #STEPS, 1, -1 do local s = STEPS[i]\n"
            f"  if s[2] == 'load' then table.insert(STEPS, {{s[1] + 0.5, 'lua', function() POSTLOAD() end}}) end end\n")

def run(image, steps, outdir, extra=(), timeout=600):
    os.makedirs(outdir, exist_ok=True)
    pre = os.path.join(os.path.abspath(outdir), '_steps.lua')
    steps += exe_patch(image, outdir)
    if 'loadSaveState' in steps:       # a lua step that loads a state itself
        steps += "\ntable.insert(STEPS, 1, {10, 'lua', function() POSTLOAD() end})\n"
    open(pre, 'w').write(f'OUTDIR = "{os.path.abspath(outdir)}"\n' + steps)
    cfg = os.path.join(os.path.abspath(outdir), '_cfg')
    shutil.rmtree(cfg, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, 'private/pcsx-redux/cfg'), cfg)
    cmd = [REDUX, '-no-ui', '-portable', cfg, '-safe',
           '-bios', BIOS, '-iso', os.path.abspath(image), '-run', '-noupdate'] + (['-debugger', '-interpreter'] if os.environ.get('REDUX_DEBUG') else []) + [
           '-dofile', pre, '-dofile', os.path.join(ROOT, 'tools/redux/harness.lua')]
    cmd += ['-dofile', os.path.join(ROOT, 'tools/redux/exambot.lua')]
    for e in extra: cmd += ['-dofile', e]
    return subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
def to_png(outdir, name):
    from PIL import Image
    w, h, bpp = map(int, open(os.path.join(outdir, name + '.txt')).read().split())
    raw = open(os.path.join(outdir, name + '.raw'), 'rb').read()
    if bpp == 24 or len(raw) == w*h*3:
        im = Image.frombytes('RGB', (w, h), raw)
    else:
        px = []
        for i in range(0, w*h*2, 2):
            v = raw[i] | raw[i+1] << 8
            px.append(((v & 31) << 3, (v >> 5 & 31) << 3, (v >> 10 & 31) << 3))
        im = Image.new('RGB', (w, h)); im.putdata(px)
    p = os.path.join(outdir, name + '.png'); im.save(p); return p
if __name__ == '__main__':
    r = run(sys.argv[1], open(sys.argv[2]).read(), sys.argv[3], sys.argv[4:])
    print(r.returncode); print(r.stdout[-3000:]); print(r.stderr[-2000:])
