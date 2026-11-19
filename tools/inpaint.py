# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Text removal for painted-in graphics: mask building, fill, write-back.

Shared by the build_*.py repaint scripts. Images are float RGB arrays
(H, W, 3) in 0-255; masks are bool arrays of the same height and width.

  m   = text_mask(rgb, rect, ink=..., ring=5, protect=...)
                                             # ink + outline/shadow/glow + halo
  out = fill(rgb, m, 'auto', src=rect)       # FSR / SHIFTMAP / TELEA / NS
  idx = to_clut(out, idx, clut, m)           # 4/8-bit: back to the CLUT
  pix = to_15(out, pix, m)                   # 16-bit: back to 15-bit + STP

Fill methods (cv2 >= 4 with contrib):
  'telea', 'ns'      cv2.inpaint; smooth, for flat panels and gradients.
  'fsr_fast', 'fsr'  cv2.xphoto FSR (fast / best); keeps texture and
                     edges, for art. 'fsr' is slow (crop it small).
  'shiftmap'         cv2.xphoto SHIFTMAP (Lab); copies whole pieces of
                     the picture. Copied JP glyph pieces in the bake-off
                     and aborts on smooth crops (run in a child process).
  'grad'             structure fill: a straight sharp band edge in the
                     ring is continued across the hole, a quadratic
                     gradient is fitted per band, and the ring's texture
                     (residual) is continued with fsr_fast and added.
  'lama', 'lama_x2'  LaMa (private/models/big-lama.pt, torch, CUDA if
  'lama_ctx[_x2]'    present); _x2 runs on a 2x lanczos upscale, _ctx
                     uses a LAMA_CTX px context crop.
  'auto'             'grad' for every group, then TEX_METHOD (fsr_fast)
                     again where the known pixels around a mask pixel
                     hold texture or edges (texture_map).
  'auto_lama'        as 'auto' with lama_ctx for the textured part.
  'moebius[_pre][_pst][_512]'  Moebius (private/models/Moebius, own venv,
                     tools/moebius_srv.py). Default weights ft_places2,
                     _pre the pretrained ones; _pst the model's own paste
                     (blurred mask blend); context crop of LAMA_CTX/2 px,
                     or _512 a 512 x 512 window (whole card).
  'best' (default)   'lama' if lama_ok(), else 'auto'. The bake-off
                     (private/re/inpaint/findings.org) found lama with a
                     text_mask ring of 5 px best on every case.

src limits the patch sources to an area (rect (x0, y0, x1, y1) or bool
mask, e.g. the band or panel the text sits on); avoid lists pixels that
must not be copied (other text, glows). Neither is ever changed.

TIM semantics: CLUT value 0x0000 (and 16-bit pixel 0x0000) is transparent
on the GPU. to_clut never picks such an entry for a filled pixel, keeps the
STP bit class of the pixel's old entry where the CLUT allows it, and leaves
pixels that were transparent as they are. to_15 keeps each pixel's STP bit
and never writes 0x0000 (opaque black becomes 0x0421 or 0x8000).
"""
import os, struct
import numpy as np
import cv2
from scipy import ndimage

# ---------------------------------------------------------------- colour

WT = np.array([0.30, 0.59, 0.11]) ** 0.5          # weighted RGB distance

def c15_rgb(v, full=False):
    """15-bit colours (array) -> float RGB. full: x * 255 / 31, else x << 3."""
    v = np.asarray(v, np.int64)
    c = np.stack([v & 31, (v >> 5) & 31, (v >> 10) & 31], -1).astype(float)
    return c * (255.0 / 31) if full else c * 8

def rect_mask(shape, r):
    m = np.zeros(shape[:2], bool)
    x0, y0, x1, y1 = r
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return m

def _area(shape, a):
    """rect, bool mask or None -> bool mask or None."""
    if a is None: return None
    a = np.asarray(a)
    if a.dtype == bool and a.shape == tuple(shape[:2]): return a
    return rect_mask(shape, tuple(int(v) for v in a))

def near(rgb, cols, dist):
    """Pixels within dist (sum of abs RGB differences) of any colour."""
    cols = np.asarray(cols, float).reshape(-1, 3)
    m = np.zeros(rgb.shape[:2], bool)
    for c in cols:
        m |= np.abs(rgb - c).sum(-1) < dist
    return m

# ---------------------------------------------------------------- mask

def background(rgb, hole, pad=6):
    """Background estimate: TELEA fill of hole (rect or bool mask) from the
    pixels around it."""
    hole = _area(rgb.shape, hole)
    if not hole.any(): return rgb.copy()
    x0, y0, x1, y1 = _bbox(hole, pad, rgb.shape)
    sub = np.clip(np.rint(rgb[y0:y1, x0:x1]), 0, 255).astype(np.uint8)
    out = rgb.copy()
    out[y0:y1, x0:x1] = cv2.inpaint(sub, hole[y0:y1, x0:x1].astype(np.uint8) * 255,
                                    5, cv2.INPAINT_TELEA)
    return out

def text_mask(rgb, rect, ink=None, ink_dist=60, thr=40, bg=None,
              outline=None, outline_dist=50, ring=5, shadow=None,
              halo=1, close=1, min_px=3, protect=None, pred=None):
    """Mask of painted text in rect (x0, y0, x1, y1).

    Core ink: pixels where pred(rgb) is true if given, else pixels near the
    ink colours (list) if given, else pixels further than thr from the
    background diffused in from the rect border. Rim: pixels within ring
    px of the core that are near the outline colours, or differ by more
    than thr/2 from the background (bg: a colour or an image; default:
    TELEA fill of the core grown by ring + 2 px). This catches outlines,
    soft drop shadows and glows (ring 3 left ghosts on card 003).
    shadow (dx, dy): the mask shifted by a hard drop-shadow offset is
    added. halo: final dilation for the anti-alias rim. Core components
    under min_px px are dropped. protect (rect or mask): never in the
    mask (art lines, icons, callouts)."""
    sh = rgb.shape[:2]
    R = rect_mask(sh, rect)
    core = None
    if pred is not None: core = pred(rgb) & R
    elif ink is not None: core = near(rgb, ink, ink_dist) & R
    if bg is not None and np.ndim(bg) == 1:
        bgi = np.broadcast_to(np.asarray(bg, float), rgb.shape)
    elif bg is not None:
        bgi = bg
    elif core is None:
        bgi = background(rgb, R)
    elif outline is None or ring:
        # the background with the core and its rim taken out
        bgi = background(rgb, ndimage.binary_dilation(core, iterations=ring + 2))
    else:
        bgi = None
    d = np.abs(rgb - bgi).sum(-1) if bgi is not None else None
    if core is None: core = (d > thr) & R
    if min_px > 1 and core.any():
        lab, n = ndimage.label(core, np.ones((3, 3)))
        sz = ndimage.sum(core, lab, np.arange(1, n + 1))
        core = np.isin(lab, 1 + np.nonzero(sz >= min_px)[0])
    m = core.copy()
    if ring:
        rim = ndimage.binary_dilation(core, iterations=ring) & R
        if outline is not None:
            m |= rim & near(rgb, outline, outline_dist)
        if d is not None:
            m |= rim & (d > thr / 2)
    if shadow:
        dx, dy = shadow
        s = np.zeros_like(m)
        H, W = sh
        s[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)] = \
            m[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        m |= s & ndimage.binary_dilation(R, iterations=max(abs(dx), abs(dy)))
    if close: m = ndimage.binary_closing(m, iterations=close) | m
    if halo: m = ndimage.binary_dilation(m, iterations=halo)
    m &= ndimage.binary_dilation(R, iterations=(max(map(abs, shadow)) if shadow else 0) + halo)
    p = _area(sh, protect)
    if p is not None: m &= ~p
    return m

# ---------------------------------------------------------------- classify

def texture(rgb, mask, width=6, bad=None, q=90):
    """Texture / edge measure of the known ring around mask: the q-th
    percentile of the absolute high-pass (luma minus its 2 px gaussian,
    holes excluded) in 0-255. A high percentile so that one hard edge
    (a band or panel border) through the hole counts."""
    L = (rgb * np.array([0.30, 0.59, 0.11])).sum(-1)
    known = ~ndimage.binary_dilation(mask, iterations=1)
    if bad is not None: known &= ~bad
    ringm = ndimage.binary_dilation(mask, iterations=width) & known
    if ringm.sum() < 8: return 0.0
    k = known.astype(float)
    g = ndimage.gaussian_filter(L * k, 2) / np.maximum(ndimage.gaussian_filter(k, 2), 1e-3)
    return float(np.percentile(np.abs(L - g)[ringm], q))

FLAT = 4.0          # texture() below this: flat colour or smooth gradient
LAMA_CTX = 128      # context margin (px) for lama_ctx*
TEX_METHOD = 'fsr_fast'   # textured parts in 'auto' ('fsr': ~1.5 ms per px)
FSR_MAX = 20000     # with TEX_METHOD 'fsr', larger groups use fsr_fast

# ---------------------------------------------------------------- LaMa

LAMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'private', 'models', 'big-lama.pt')
_lama_model = {}

def _lama_load():
    if 'm' not in _lama_model:
        import warnings, torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            m = torch.jit.load(os.environ.get('LAMA_MODEL', LAMA_PATH), map_location=dev).eval()
        _lama_model.update(m=m, dev=dev, torch=torch)
    return _lama_model

def _lama(sub, hole, scale=1):
    """LaMa (big-lama TorchScript) on uint8 RGB sub; hole bool. scale > 1
    runs it on a lanczos-upscaled copy and downscales the result (area)."""
    L = _lama_load(); torch = L['torch']
    h, w = hole.shape
    img, mk = sub, hole.astype(np.uint8)
    if scale > 1:
        img = cv2.resize(sub, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
        mk = cv2.resize(mk, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    H, W = mk.shape
    ph, pw = -H % 8, -W % 8
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    mk = np.pad(mk, ((0, ph), (0, pw)), mode='reflect')
    t = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)[None].to(L['dev'])
    m = torch.from_numpy((mk > 0).astype(np.float32))[None, None].to(L['dev'])
    with torch.no_grad():
        out = L['m'](t, m)[0].permute(1, 2, 0).cpu().numpy()[:H, :W]
    out = np.clip(out * 255, 0, 255)
    if scale > 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    return np.rint(out).astype(np.uint8)

# ---------------------------------------------------------------- Moebius

MOEBIUS_PY = os.path.join(os.path.dirname(LAMA_PATH), 'venv', 'bin', 'python')
_moe = {}

def _moebius(sub, hole, weight='ft', paste=False, cfg=2.0):
    """Moebius (diffusion removal model) through tools/moebius_srv.py in
    its own venv; one server per process. The pipeline works and answers
    at 512 x 512; the result is scaled back to the crop here."""
    import subprocess, tempfile
    if 'p' not in _moe:
        _moe['dir'] = tempfile.mkdtemp(prefix='moebius_')
        _moe['p'] = subprocess.Popen(
            [os.environ.get('MOEBIUS_PY', MOEBIUS_PY),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), 'moebius_srv.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        _moe['n'] = 0
    _moe['n'] += 1
    h, w = hole.shape
    S = max(h, w)                   # square (the model assumes it)
    sub = np.pad(sub, ((0, S - h), (0, S - w), (0, 0)), mode='reflect')
    hole = np.pad(hole, ((0, S - h), (0, S - w)), mode='reflect')
    src = os.path.join(_moe['dir'], f'{_moe["n"]}.npz')
    dst = os.path.join(_moe['dir'], f'{_moe["n"]}.npy')
    np.savez(src, img=sub, mask=hole.astype(np.uint8), weight=weight, paste=int(paste), cfg=cfg)
    p = _moe['p']
    p.stdin.write(f'{src} {dst}\n'); p.stdin.flush()
    if p.stdout.readline().strip() != 'OK':
        raise RuntimeError('moebius server failed')
    out = np.load(dst)
    if out.shape[:2] != (S, S):     # the pipeline answers at its 512 scale
        out = cv2.resize(out, (S, S), interpolation=cv2.INTER_AREA if out.shape[0] > S
                         else cv2.INTER_LANCZOS4)
    out = out[:h, :w]
    os.remove(src); os.remove(dst)
    return out

# ---------------------------------------------------------------- fill

def _bbox(m, pad, shape):
    ys, xs = np.nonzero(m)
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(shape[1], xs.max() + pad + 1), min(shape[0], ys.max() + pad + 1))

SHIFTMAP_FAILS = []     # crops where SHIFTMAP aborted (fsr_fast used)

def _sm_child(lab, known, conn):
    import os
    fd = os.open(os.devnull, os.O_WRONLY); os.dup2(fd, 2)
    dst = np.zeros_like(lab)
    cv2.xphoto.inpaint(lab, known, dst, cv2.xphoto.INPAINT_SHIFTMAP)
    conn.send(dst); conn.close()

def _shiftmap(lab, known):
    """SHIFTMAP in a forked child: OpenCV 5.1 aborts the process (vector
    assertion) on some smooth crops. Returns None if it did."""
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    r, w = ctx.Pipe(False)
    p = ctx.Process(target=_sm_child, args=(lab, known, w))
    p.start(); w.close()
    try: dst = r.recv()
    except EOFError: dst = None
    p.join()
    return dst

def _run(sub, hole, method, radius, core=None):
    """sub uint8 RGB, hole bool -> filled uint8 RGB."""
    if method == 'grad':
        return np.rint(_grad(sub, hole, core=core)).astype(np.uint8)
    if method.startswith('moebius'):      # moebius[_pre][_pst][_512]
        r = _moebius(sub, hole, 'pre' if '_pre' in method else 'ft', '_pst' in method)
        r = r.copy(); r[~hole] = sub[~hole]
        return r
    if method.startswith('lama'):          # lama, lama_x2, lama_ctx, lama_ctx_x2
        r = _lama(sub, hole, 2 if method.endswith('_x2') else 1)
        r[~hole] = sub[~hole]
        return r
    if method in ('telea', 'ns'):
        flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        return cv2.inpaint(sub, hole.astype(np.uint8) * 255, radius, flag)
    known = (~hole).astype(np.uint8) * 255
    if method == 'shiftmap':
        dst = _shiftmap(cv2.cvtColor(sub, cv2.COLOR_RGB2Lab), known)
        if dst is not None: return cv2.cvtColor(dst, cv2.COLOR_Lab2RGB)
        SHIFTMAP_FAILS.append(sub.shape)
        method = 'fsr_fast'
    alg = {'fsr': cv2.xphoto.INPAINT_FSR_BEST, 'fsr_fast': cv2.xphoto.INPAINT_FSR_FAST}[method]
    bgr = np.ascontiguousarray(sub[..., ::-1])
    dst = np.zeros_like(bgr)
    cv2.xphoto.inpaint(bgr, known, dst, alg)
    return dst[..., ::-1]

BANDS_WHY = []      # why each _bands() call split or not (debug)

def _bands(sub, known, ringm, min_jump=30, max_res=1.5, min_inl=0.4):
    """Split the crop into two bands if the ring holds a sharp straight
    edge: 2-means on the ring colours, line fitted (PCA) to the boundary
    pixels. Returns a label image (0/1) or None (one smooth region)."""
    pts = sub[ringm].astype(float)
    if len(pts) < 40: return None
    # cluster on chroma (bands differ in hue; a gradient mostly in level)
    pts = pts / np.maximum(pts.sum(1, keepdims=True), 1) * 255
    pc = pts - pts.mean(0)
    ax = np.linalg.svd(pc, full_matrices=False)[2][0]
    pr = pc @ ax
    c = np.array([pts[pr.argmin()], pts[pr.argmax()]])
    for _ in range(10):
        lab = ((pts[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        if lab.min() == lab.max(): return None
        c = np.array([pts[lab == k].mean(0) for k in (0, 1)])
    if min(np.mean(lab == 0), np.mean(lab == 1)) < 0.12: return None
    L = np.full(ringm.shape, -1); L[ringm] = lab
    # boundary pairs (4-neighbours with different labels) and their jump
    by = []; jumps = []
    for dy, dx in ((0, 1), (1, 0)):
        a_ = L[:L.shape[0] - dy, :L.shape[1] - dx]; b_ = L[dy:, dx:]
        e = (a_ >= 0) & (b_ >= 0) & (a_ != b_)
        ys, xs = np.nonzero(e)
        if len(ys):
            jumps.append(np.abs(sub[ys, xs].astype(float) - sub[ys + dy, xs + dx]).sum(1))
            by.append(np.stack([ys + dy / 2, xs + dx / 2], 1))
    if not by: return None
    by = np.concatenate(by); jumps = np.concatenate(jumps)
    if len(by) < 6 or np.median(jumps) < min_jump:
        BANDS_WHY.append(f'jump {np.median(jumps) if len(jumps) else 0:.0f} n {len(by)}'); return None
    # RANSAC line through the boundary points, refined by PCA on inliers
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        i, j = rng.choice(len(by), 2, replace=False)
        d = by[j] - by[i]
        if np.hypot(*d) < 4: continue
        nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
        inl = np.abs((by - by[i]) @ nrm) < 1.5
        if best is None or inl.sum() > best.sum(): best = inl
    if best is None or best.mean() < min_inl or best.sum() < 12:
        BANDS_WHY.append(f'inliers {0 if best is None else best.mean():.2f}'); return None
    pin = by[best]
    mu = pin.mean(0); vt = np.linalg.svd(pin - mu, full_matrices=False)[2]
    nrm = vt[1]
    res = np.abs((pin - mu) @ nrm)
    if np.sqrt((res ** 2).mean()) > max_res:
        BANDS_WHY.append(f'res {np.sqrt((res ** 2).mean()):.1f}'); return None
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    side = ((np.stack([yy, xx], -1) - mu) @ nrm) > 0
    # side -> label by majority of the known ring pixels
    k1 = np.mean(lab[side[ringm]] == 1) if side[ringm].any() else 0
    out = np.where(side, 1 if k1 >= 0.5 else 0, 0 if k1 >= 0.5 else 1)
    agree = np.mean(out[ringm] == lab)
    if agree < 0.85:
        BANDS_WHY.append(f'agree {agree:.2f}'); return None
    BANDS_WHY.append('split')
    return out

def _poly(sub, sel, hole, deg=2):
    """Least-squares polynomial (deg 0-2 in x, y) per channel fitted on
    sel, evaluated on hole."""
    ys, xs = np.nonzero(sel)
    h, w = sel.shape
    def feats(y, x):
        y = y / max(h, 1); x = x / max(w, 1)
        f = [np.ones_like(x)]
        if deg >= 1: f += [x, y]
        if deg >= 2: f += [x * x, x * y, y * y]
        return np.stack(f, 1)
    n = len(ys)
    d = 2 if n >= 60 else 1 if n >= 12 else 0
    deg_ = min(deg, d)
    A = feats(ys.astype(float), xs.astype(float))[:, :{0: 1, 1: 3, 2: 6}[deg_]]
    coef, *_ = np.linalg.lstsq(A, sub[ys, xs].astype(float), rcond=None)
    hy, hx = np.nonzero(hole)
    B = feats(hy.astype(float), hx.astype(float))[:, :A.shape[1]]
    return B @ coef

def _grad(sub, hole, ring=8, core=None):
    """Structure fill: bands split by a straight edge (if any), a smooth
    polynomial per band from the ring, then the ring's texture (residual
    from the polynomial) continued with fsr_fast and added."""
    known = ~hole
    ringm = known & ndimage.binary_dilation(hole if core is None else core, iterations=ring)
    bands = _bands(sub, known, ringm)
    if bands is None: bands = np.zeros(hole.shape, int)
    smooth = sub.astype(float).copy()
    fit = np.zeros(sub.shape, float)
    for b in np.unique(bands):
        sel = ringm & (bands == b)
        if sel.sum() < 3: sel = ringm
        allb = bands == b
        fit[allb] = _poly(sub, sel, allb)
    resid = np.clip(sub.astype(float) - fit + 128, 0, 255)
    resid[hole] = 128
    r8 = np.rint(resid).astype(np.uint8)
    rf = _run(r8, hole, 'fsr_fast', 3).astype(float) - 128
    smooth[hole] = (fit + rf)[hole]
    return np.clip(smooth, 0, 255)

def texture_map(rgb, mask, bad=None, width=6, frac=0.1):
    """Mask pixels whose known surroundings (window 2 * width + 1) have
    more than frac of pixels with high-pass >= FLAT (texture or an edge).
    Deep inside large holes the value of the nearest known pixel is used."""
    L = (rgb * np.array([0.30, 0.59, 0.11])).sum(-1)
    known = ~ndimage.binary_dilation(mask, iterations=1)
    if bad is not None: known &= ~bad
    k = known.astype(float)
    g = ndimage.gaussian_filter(L * k, 2) / np.maximum(ndimage.gaussian_filter(k, 2), 1e-3)
    edge = (np.abs(L - g) >= FLAT) & known
    n = ndimage.uniform_filter(k, 2 * width + 1)
    f = ndimage.uniform_filter(edge.astype(float), 2 * width + 1) / np.maximum(n, 1e-6)
    ok = n > 0.02
    if not ok.any(): return np.zeros_like(mask)
    _, (iy, ix) = ndimage.distance_transform_edt(~ok, return_indices=True)
    tex = mask & (f[iy, ix] > frac)
    return ndimage.binary_dilation(tex, iterations=2) & mask

class NoSource(ValueError):
    """The crop around a part holds no known pixel."""

def _fill_part(out, pm, bad, meth, srcm, margin, radius):
    sh = out.shape[:2]
    pad = margin if meth in ('telea', 'ns') else margin * 2
    if meth.startswith('lama_ctx') or meth.startswith('moebius'): pad = LAMA_CTX // 2
    x0, y0, x1, y1 = _bbox(pm, pad, sh)
    if meth.startswith('moebius') and meth.endswith('_512'):
        # a native-scale window of up to 512 x 512 around the hole
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        W, H = min(512, sh[1]), min(512, sh[0])
        x0 = min(max(0, cx - W // 2), sh[1] - W); y0 = min(max(0, cy - H // 2), sh[0] - H)
        x1, y1 = x0 + W, y0 + H
    elif meth.startswith('moebius'):
        # the model wants a square: grow the short side around the hole
        S = min(max(x1 - x0, y1 - y0), sh[0], sh[1])
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if x1 - x0 < S: x0 = min(max(0, cx - S // 2), sh[1] - S); x1 = x0 + S
        if y1 - y0 < S: y0 = min(max(0, cy - S // 2), sh[0] - S); y1 = y0 + S
    if srcm is not None:      # the crop never needs to leave the source area
        sx0, sy0, sx1, sy1 = _bbox(srcm | pm, 0, sh)
        x0, y0, x1, y1 = max(x0, sx0), max(y0, sy0), min(x1, sx1), min(y1, sy1)
    sub = np.clip(np.rint(out[y0:y1, x0:x1]), 0, 255).astype(np.uint8)
    hole = bad[y0:y1, x0:x1] | pm[y0:y1, x0:x1]
    if hole.all(): raise NoSource('inpaint.fill: no source pixels')
    res = _run(sub, hole, meth, radius, core=pm[y0:y1, x0:x1]).astype(float)
    t = pm[y0:y1, x0:x1]
    out[y0:y1, x0:x1][t] = res[t]

def _fill_far(out, pm, bad, srcm, margin, radius):
    """'auto' only: a part with no known pixel in its 'grad' crop (a small
    group deep inside avoid / other mask groups) gets TELEA over a crop
    that is doubled until it holds known pixels."""
    m = margin
    while True:
        try:
            return _fill_part(out, pm, bad, 'telea', srcm, m, radius)
        except NoSource:
            if m > max(out.shape[:2]): raise
            m *= 2

def lama_ok():
    """True if the LaMa weights and torch are there."""
    if not os.path.exists(os.environ.get('LAMA_MODEL', LAMA_PATH)): return False
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True

def fill(rgb, mask, method='best', src=None, avoid=None, margin=16, radius=3,
         log=None):
    """Fill mask in rgb (float RGB) and return a new float image. Only mask
    pixels change. src / avoid: see the module doc. 'auto': each group
    of the mask is filled with 'grad' (straight band edges continued,
    smooth gradient per band, ring texture added); then the textured part
    (texture_map) again with TEX_METHOD, the flat part counting as known."""
    out = rgb.copy()
    if not mask.any(): return out
    sh = rgb.shape[:2]
    srcm = _area(sh, src)
    bad = mask.copy()
    a = _area(sh, avoid)
    if a is not None: bad |= a
    if srcm is not None: bad |= ~srcm
    if method == 'best':
        method = 'lama' if lama_ok() else 'auto'
    if not method.startswith('auto'):
        _fill_part(out, mask, bad, method, srcm, margin, radius)
        return out
    texm = 'lama_ctx' if method == 'auto_lama' else TEX_METHOD
    tex = texture_map(rgb, mask, bad & ~mask)
    lab, n = ndimage.label(ndimage.binary_dilation(mask, iterations=3) & mask)
    done, far = np.zeros(sh, bool), []
    for k in range(1, n + 1):
        try:
            _fill_part(out, lab == k, bad, 'grad', srcm, margin, radius)
            done |= lab == k
        except NoSource:
            far.append(lab == k)
    for pm in far:            # after the others: filled groups count as known
        _fill_far(out, pm, bad & ~done, srcm, margin, radius)
        done |= pm
        if log is not None:
            x0, y0, x1, y1 = _bbox(pm, 0, sh)
            log.append(f'fill {x0},{y0}-{x1},{y1} {pm.sum()} px -> telea (no source near)')
    flat = mask & ~tex
    lab, n = ndimage.label(ndimage.binary_dilation(tex, iterations=2) & mask)
    for k in range(1, n + 1):
        pm = (lab == k) & tex
        if not pm.any(): continue
        meth = texm if texm != 'fsr' or pm.sum() <= FSR_MAX else 'fsr_fast'
        try:
            _fill_part(out, pm, bad & ~flat, meth, srcm, margin, radius)
        except NoSource:          # keep the first pass there
            continue
        if log is not None:
            x0, y0, x1, y1 = _bbox(pm, 0, sh)
            log.append(f'fill {x0},{y0}-{x1},{y1} {pm.sum()} px -> {meth}')
    if log is not None:
        log.append(f'fill: {mask.sum()} px, {flat.sum()} grad, {tex.sum()} textured')
    return out

# ---------------------------------------------------------------- write-back

def _nearest(q, pw, cand):
    d = ((q[:, None, :] - pw[None, cand]) ** 2).sum(-1)
    return cand[d.argmin(1)]

def to_clut(rgb, idx, clut, where, pal=None, dither='fs', strength=0.8,
            keep_stp=True):
    """Quantise rgb into CLUT indices inside where; other pixels keep idx.

    clut: the 15-bit CLUT values (16 or 256 used entries); pal: the RGB the
    caller used for them (default x << 3). dither: 'fs' (serpentine
    Floyd-Steinberg inside where only, error scaled by strength; avoids
    banding on smooth fills, exact palette colours stay exact) or 'none'.
    Entries with value 0x0000 are never chosen. keep_stp: a pixel only
    gets entries with the STP bit of its old entry (if any exist).
    Pixels whose old entry is 0x0000 (transparent) are not changed."""
    clut = np.asarray(clut, np.int64)
    pal = c15_rgb(clut) if pal is None else np.asarray(pal, float)
    n = len(pal)
    pw = pal * WT
    opaque = np.nonzero(clut[:n] != 0)[0]
    stp = (clut[:n] >> 15) & 1
    sets = {}
    for s in (0, 1):
        c = opaque[stp[opaque] == s] if keep_stp else opaque
        sets[s] = c if len(c) else opaque
    out = idx.copy()
    w = where & (clut[np.minimum(idx, len(clut) - 1)] != 0)
    if not w.any(): return out
    ys, xs = np.nonzero(w)
    if dither != 'fs':
        q = rgb[ys, xs] * WT
        cls = stp[idx[ys, xs]]
        for s in (0, 1):
            k = cls == s
            if k.any(): out[ys[k], xs[k]] = _nearest(q[k], pw, sets[s])
        return out
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    buf = rgb[y0:y1, x0:x1].astype(float).copy()
    ww = w[y0:y1, x0:x1]
    o = out[y0:y1, x0:x1]
    cls = stp[idx[y0:y1, x0:x1]]
    H, W = ww.shape
    for yy in range(H):
        row = range(W) if yy % 2 == 0 else range(W - 1, -1, -1)
        dx = 1 if yy % 2 == 0 else -1
        cols = [x for x in row if ww[yy, x]]
        if not cols: continue
        for x in cols:
            v = np.clip(buf[yy, x], 0, 255)
            cand = sets[cls[yy, x]]
            k = cand[((pw[cand] - v * WT) ** 2).sum(-1).argmin()]
            o[yy, x] = k
            e = (v - pal[k]) * strength
            for ex, ey, f in ((dx, 0, 7), (-dx, 1, 3), (0, 1, 5), (dx, 1, 1)):
                X, Y = x + ex, yy + ey
                if 0 <= X < W and Y < H and ww[Y, X]:
                    buf[Y, X] += e * (f / 16)
    return out

def to_15(rgb, pix, where, dither='fs', strength=0.8):
    """Quantise rgb (x << 3 scale) to 15-bit inside where; pix: uint16
    pixels of a 16-bit TIM. Keeps each pixel's STP bit; never writes
    0x0000; pixels that were 0x0000 stay transparent."""
    pix = np.asarray(pix, np.uint16)
    out = pix.copy()
    w = where & (pix != 0)
    ys, xs = np.nonzero(w)
    if not len(ys): return out
    v = rgb.astype(float).copy()
    if dither == 'fs':
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        buf = v[y0:y1, x0:x1]; ww = w[y0:y1, x0:x1]
        H, W = ww.shape
        for yy in range(H):
            dx = 1 if yy % 2 == 0 else -1
            for x in (range(W) if dx > 0 else range(W - 1, -1, -1)):
                if not ww[yy, x]: continue
                q = np.clip(np.rint(buf[yy, x] / 8), 0, 31)
                e = (buf[yy, x] - q * 8) * strength
                buf[yy, x] = q * 8
                for ex, ey, f in ((dx, 0, 7), (-dx, 1, 3), (0, 1, 5), (dx, 1, 1)):
                    X, Y = x + ex, yy + ey
                    if 0 <= X < W and Y < H and ww[Y, X]:
                        buf[Y, X] += e * (f / 16)
    q = np.clip(np.rint(v[ys, xs] / 8), 0, 31).astype(np.int64)
    val = q[:, 0] | (q[:, 1] << 5) | (q[:, 2] << 10) | (pix[ys, xs].astype(np.int64) & 0x8000)
    val[val == 0] = 0x0421
    out[ys, xs] = val.astype(np.uint16)
    return out

# ---------------------------------------------------------------- TIM

class Tim:
    """A TIM in a bytearray at off: bpp 4/8/16, first CLUT row by default.
    .rgb() float RGB; .put(rgb, where) writes back with to_clut / to_15;
    .pix holds indices (4/8-bit) or uint16 pixels (16-bit)."""
    def __init__(self, data, off=0, clut_row=0, full=False):
        self.data, self.off, self.full = data, off, full
        magic, flags = struct.unpack_from('<II', data, off)
        assert magic == 0x10, hex(magic)
        self.bpp = {0: 4, 1: 8, 2: 16}[flags & 3]
        p = off + 8
        self.clut = None
        if flags & 8:
            cl = struct.unpack_from('<I', data, p)[0]
            cw, ch = struct.unpack_from('<2H', data, p + 8)
            allc = np.array(struct.unpack_from(f'<{cw * ch}H', data, p + 12), np.int64)
            ncol = 16 if self.bpp == 4 else 256
            self.clut = allc[clut_row * cw:clut_row * cw + ncol]
            p += cl
        bl = struct.unpack_from('<I', data, p)[0]
        ww, hh = struct.unpack_from('<2H', data, p + 8)
        self.h, self.pix_off, self.pix_len = hh, p + 12, bl - 12
        raw = bytes(data[self.pix_off:self.pix_off + ww * 2 * hh])
        if self.bpp == 16:
            self.w = ww
            self.pix = np.frombuffer(raw, '<u2').reshape(hh, ww).copy()
        elif self.bpp == 8:
            self.w = ww * 2
            self.pix = np.frombuffer(raw, np.uint8).reshape(hh, self.w).copy()
        else:
            self.w = ww * 4
            b = np.frombuffer(raw, np.uint8).reshape(hh, ww * 2)
            self.pix = np.stack([b & 15, b >> 4], -1).reshape(hh, self.w).copy()
        self.pal = None if self.clut is None else c15_rgb(self.clut, full)

    def rgb(self):
        if self.bpp == 16: return c15_rgb(self.pix, self.full)
        return self.pal[self.pix]

    def put(self, rgb, where, **kw):
        if self.bpp == 16:
            v = rgb * (248.0 / 255) if self.full else rgb
            self.pix = to_15(v, self.pix, where, **kw)
        else:
            self.pix = to_clut(rgb, self.pix, self.clut, where, self.pal, **kw)

    def store(self):
        if self.bpp == 16: raw = self.pix.astype('<u2').tobytes()
        elif self.bpp == 8: raw = self.pix.astype(np.uint8).tobytes()
        else:
            p = self.pix.astype(np.uint8).reshape(self.h, -1, 2)
            raw = (p[..., 0] | (p[..., 1] << 4)).tobytes()
        self.data[self.pix_off:self.pix_off + len(raw)] = raw
        return self.data
