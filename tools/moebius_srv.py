# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Moebius inpainting server for tools/inpaint.py (method 'moebius*').

Run with the Moebius venv python (inpaint.py starts it): reads request
lines "IN.npz OUT.npy" on stdin. IN holds img (H, W, 3 uint8), mask
(H, W uint8, 1 = fill), weight ('ft' or 'pre'), paste (0/1), cfg. Writes
the result image to OUT.npy and answers "OK" on its stdout. The pipeline
is loaded once per weight. All other output goes to stderr.
"""
import os, sys
REPO = os.environ.get('MOEBIUS_REPO') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'private', 'models', 'Moebius')
proto = os.fdopen(os.dup(1), 'w')
os.dup2(2, 1)                       # model prints go to stderr
os.chdir(REPO); sys.path.insert(0, REPO)
import argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
from infer.utils import build_pipeline

WEIGHTS = {'ft': 'weight/Moebius/ft_places2/diffusion_pytorch_model.bin',
           'pre': 'weight/Moebius/pretrained/diffusion_pytorch_model.bin'}
pipes = {}

def pipe(w):
    if w not in pipes:
        a = argparse.Namespace(model_config='config/model_cfg/moebius.yaml',
                               model_weight=WEIGHTS[w], device='cuda')
        pipes[w] = build_pipeline(a)
    return pipes[w]

for line in sys.stdin:
    src, dst = line.split()
    d = np.load(src)
    img = Image.fromarray(d['img'])
    mk = Image.fromarray((d['mask'] > 0).astype(np.uint8) * 255)
    out = pipe(str(d['weight']))([img], [mk], guidance_scale=float(d['cfg']),
                                 paste=bool(d['paste']), compensate=False, num_steps=20,
                                 noise_offset=0.0357)[0]
    np.save(dst, np.asarray(out.convert('RGB')))
    proto.write('OK\n'); proto.flush()
