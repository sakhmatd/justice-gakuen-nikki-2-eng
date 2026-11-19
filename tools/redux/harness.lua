-- Copyright 2026 Sergei Akhmatdinov (sakhmatd)
-- SPDX-License-Identifier: Apache-2.0
-- Headless test harness for PCSX-Redux.
-- Globals set via -exec before this file: STEPS (table), OUTDIR (string).
-- STEPS entries: {frame, 'press', 'CROSS', holdframes} | {frame, 'shot', 'name'}
--                {frame, 'dump', 'name', addr, size} | {frame, 'quit'}
local frame = 0
local pending = {}
local pad, B

local function save(name, data)
  local f = Support.File.open(OUTDIR .. '/' .. name, 'TRUNCATE')
  f:write(data)
  f:close()
end

function LOG(line)
  local f = io.open(OUTDIR .. '/log.txt', 'a')
  f:write(line .. '\n'); f:close()
end

local function shot(name)
  local ss = PCSX.GPU.takeScreenShot()
  save(name .. '.raw', ss.data)
  save(name .. '.txt', tostring(tonumber(ss.width)) .. ' ' .. tostring(tonumber(ss.height)) .. ' ' .. tostring(tonumber(ss.bpp)))
end

local function dump(name, addr, size)
  local mem = PCSX.getMemPtr()
  local off = bit.band(addr, 0x1fffff)
  save(name, ffi.string(mem + off, size))
end

table.sort(STEPS, function(a, b) return a[1] < b[1] end)
local idx = 1

local function tick()
  frame = frame + 1
  FRAME = frame
  if EXAMBOT and BOT_ON then EXAMBOT(frame) end
  if not pad then
    pad = PCSX.SIO0.slots[1].pads[1]
    B = PCSX.CONSTS.PAD.BUTTON
  end
  for k = #pending, 1, -1 do
    local p = pending[k]
    if frame >= p[2] then pad.clearOverride(B[p[1]]); table.remove(pending, k) end
  end
  while idx <= #STEPS and STEPS[idx][1] <= frame do
    local s = STEPS[idx]; idx = idx + 1
    if s[2] == 'press' then
      pad.setOverride(B[s[3]]); table.insert(pending, {s[3], frame + (s[4] or 6)})
    elseif s[2] == 'shot' then shot(s[3])
    elseif s[2] == 'dump' then dump(s[3], s[4], s[5])
    elseif s[2] == 'quit' then PCSX.quit(0)
    elseif s[2] == 'vram' then save(s[3], PCSX.GPU.getVRAM())
    elseif s[2] == 'save' then
      save(s[3], PCSX.createSaveState())
    elseif s[2] == 'load' then
      local f = Support.File.open(OUTDIR .. '/../' .. s[3], 'READ')
      PCSX.loadSaveState(f); f:close()
      if POSTLOAD then POSTLOAD() end
    elseif s[2] == 'lua' then s[3](frame)
    elseif s[2] == 'bot' then BOT_ON = s[3]
    end
  end
end
listener = PCSX.Events.createEventListener('GPU::Vsync', function()
  local ok, err = pcall(tick)
  if not ok then print('HARNESS ERROR: ' .. tostring(err)) end
end)
