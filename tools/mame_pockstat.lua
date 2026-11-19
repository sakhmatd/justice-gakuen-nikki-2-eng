-- Copyright 2026 Sergei Akhmatdinov (sakhmatd)
-- SPDX-License-Identifier: Apache-2.0
--
-- Workaround for a button bug in MAME's pockstat driver.
--
-- Usage: mame pockstat -noplugins -autoboot_script tools/mame_pockstat.lua
--          -cart pocket.gme
--
-- The driver (MAME 0.289) does not raise the PocketStation button
-- interrupt. Games that read their buttons through it miss most presses.
-- Pocket Justice is one of them, in the Japanese original as well. It
-- reads its buttons only from the interrupt latch (0x0a000000) in its
-- interrupt handler. This script adds each new press to the latch reads
-- until the game clears it through INT_CLEAR (0x0a000010). It only does
-- so for buttons that the interrupt mask (0x0a000008 set, 0x0a00000c
-- clear) enables. The kernel menu reads the buttons directly and is not
-- affected.

local m = manager.machine
local sp = m.devices[":maincpu"].spaces["program"]
local port = m.ioport.ports[":BUTTONS"]
local BIT = {["Action Button"] = 1, ["Right"] = 2, ["Left"] = 4,
             ["Up"] = 8, ["Down"] = 16}
local pend, imask, prev = 0, 0, 0
local started = false

-- Globals, so that the taps and the notifier are not garbage collected.
pocket_latch_tap = sp:install_read_tap(0x0a000000, 0x0a000003, "latch",
  function(off, data, mask)
    if pend & imask ~= 0 then return data | (pend & imask) end
  end)
pocket_mask_tap = sp:install_write_tap(0x0a000008, 0x0a00000f, "mask",
  function(off, data, mask)
    if off < 0x0a00000c then
      imask = imask | data
    else
      imask = imask & ~data
    end
  end)
pocket_clear_tap = sp:install_write_tap(0x0a000010, 0x0a000013, "clear",
  function(off, data, mask)
    pend = pend & ~data
  end)
pocket_frame = emu.add_machine_frame_notifier(function()
  if not started then
    imask = sp:read_u32(0x0a000008)
    started = true
  end
  local val = port:read()
  local now = 0
  for name, f in pairs(port.fields) do
    if BIT[name] and ((val ~ f.defvalue) & f.mask) ~= 0 then
      now = now | BIT[name]
    end
  end
  pend = pend | (now & ~prev & imask)
  prev = now
end)
