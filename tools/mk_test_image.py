# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Patch the current private/build into OUT.bin/.cue (not private/out), for
agents that test on their own image copy.
Usage: python3 tools/mk_test_image.py private/re/TOPIC/test
Skips dotfiles (e.g. .rebuild.lock) and .lz files in private/build."""
import os, sys
sys.path.insert(0, 'tools')
import patch_bin
if len(sys.argv) != 2 or sys.argv[1].startswith('-'):
    sys.exit(__doc__.strip())
only = {f for f in os.listdir('private/build') if not f.endswith('.lz') and not f.startswith('.')}
patch_bin.main(sys.argv[1], only)
