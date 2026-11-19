# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
"""Find the font files the redraws use, by file name.

The build scripts name fonts by file name (e.g. 'RobotoCondensed-Bold.ttf').
find() looks for that name in the directories of FONT_DIRS (separated by
os.pathsep), then in the usual font directories, searched recursively:
~/.local/share/fonts, ~/.fonts, /usr/local/share/fonts, /usr/share/fonts.
A name that differs only in case also matches. truetype() is
PIL.ImageFont.truetype with the file name resolved this way.
"""
import os
from PIL import ImageFont

DEFAULT_DIRS = ('~/.local/share/fonts', '~/.fonts', '/usr/local/share/fonts',
                '/usr/share/fonts')
_index = None


def dirs():
    env = [d for d in os.environ.get('FONT_DIRS', '').split(os.pathsep) if d]
    return [os.path.expanduser(d) for d in env + list(DEFAULT_DIRS)]


def _build_index():
    exact, lower = {}, {}
    for top in dirs():
        for root, _, files in os.walk(top, followlinks=True):
            for fn in sorted(files):
                path = os.path.join(root, fn)
                exact.setdefault(fn, path)
                lower.setdefault(fn.lower(), path)
    return exact, lower


def find(name):
    """Path of the font file NAME (a bare file name; a directory part is ignored)."""
    global _index
    base = os.path.basename(name)
    if _index is None:
        _index = _build_index()
    exact, lower = _index
    path = exact.get(base) or lower.get(base.lower())
    if path is None:
        raise FileNotFoundError(
            f'font {base} not found in {", ".join(dirs())}. Install it, or set '
            f'FONT_DIRS to the directory that holds it (see BUILDING.md).')
    return path


def truetype(font, *args, **kw):
    if isinstance(font, (str, os.PathLike)):
        font = find(os.fspath(font))
    return ImageFont.truetype(font, *args, **kw)
