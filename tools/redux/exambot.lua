-- Copyright 2026 Sergei Akhmatdinov (sakhmatd)
-- SPDX-License-Identifier: Apache-2.0
-- Exam bot: presses the input for the exam's current command.
-- Command param byte is at obj+0x16, obj = *(0x800a3df8) while SIKEN runs.
-- (Address from the first game; not yet checked on this disc.)
local MAP = { [0]='RIGHT', [1]='LEFT', [2]='DOWN', [3]='UP',
              [4]='SQUARE', [5]='TRIANGLE', [6]='CROSS', [7]='CIRCLE' }
local last, t = -1, 0
EXAMBOT = function(frame)
  local m = PCSX.getMemPtr()
  local obj = ffi.cast('uint32_t*', m + 0x0a3df8)[0]
  if obj < 0x80000000 or obj >= 0x80200000 then return end
  local p = m[bit.band(obj, 0x1fffff) + 0x16]
  t = t + 1
  if p ~= last then last = p; t = 0; LOG(string.format('F%d exam cmd %d', frame, p)) end
  if t % 45 == 20 and MAP[p] and not BOT_RELEASE then
    local pad = PCSX.SIO0.slots[1].pads[1]
    pad.setOverride(PCSX.CONSTS.PAD.BUTTON[MAP[p]])
    BOT_RELEASE = frame + 8; BOT_BTN = MAP[p]
  end
  if BOT_RELEASE and frame >= BOT_RELEASE then
    PCSX.SIO0.slots[1].pads[1].clearOverride(PCSX.CONSTS.PAD.BUTTON[BOT_BTN]); BOT_RELEASE = nil
  end
end
