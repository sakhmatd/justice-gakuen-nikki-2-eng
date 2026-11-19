#!/bin/sh
# Copyright 2026 Sergei Akhmatdinov (sakhmatd)
# SPDX-License-Identifier: Apache-2.0
# Full rebuild: private/build, then the patched image OUTBASE.bin/.cue
# (default private/out/nikki2-en). Run from anywhere after tools/setup.py.
# The order matters: later scripts patch files that earlier ones wrote.
# Two rebuilds must not run at once:
#   flock private/build/.rebuild.lock sh tools/build_all.sh [OUTBASE]
set -e
cd "$(dirname "$0")/.."
mkdir -p private/build private/out
for s in "build_mbms.py ALL" build_mgbg.py build_title.py build_titlecredit.py \
         build_mgtit.py build_beta.py build_aload.py "build_examgfx.py --extra" \
         "build_testmj.py --terms" build_minihud.py build_minittl.py build_plsel.py \
         build_selp.py build_staff.py build_editcm.py build_kanri.py build_pass.py \
         build_editce.py hook.py build_sysmsg.py build_optinfo.py build_cardtitle.py \
         build_cock.py build_pause.py build_fight.py build_koubai.py build_extras.py \
         build_card.py build_pdatim.py build_emtbl.py; do
  echo "=== $s"
  python3 tools/$s
done
# private/build must hold only disc files at this point.
python3 -c "import sys; sys.path.insert(0, 'tools'); import patch_bin; patch_bin.main(sys.argv[1] if len(sys.argv) > 1 else None)" "$@"
echo BUILD-OK
