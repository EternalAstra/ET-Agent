#!/usr/bin/env python3
"""ET-Agent Real-Time Memory Monitor — WebSocket streaming server.

Streams MemoryMonitor snapshots to connected dashboard clients via WebSocket.
Runs a lightweight HTTP server that serves the monitor dashboard and upgrades
connections to WebSocket for live data push.

Usage
-----
    python scripts/monitor_server.py [--port 8765] [--interval 1.0]

Then open http://localhost:8765 in a browser.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ────────────────────────────────────────────────────────────────
# Tiny async WebSocket broadcaster
# ────────────────────────────────────────────────────────────────

class _Broadcaster:
    """Holds connected WebSocket clients and pushes snapshots."""

    def __init__(self):
        self._clients: list = []

    def add(self, ws):
        self._clients.append(ws)

    def remove(self, ws):
        try:
            self._clients.remove(ws)
        except ValueError:
            pass

    def push(self, payload: str):
        for ws in list(self._clients):
            try:
                ws.send(payload)
            except Exception:
                self.remove(ws)

    @property
    def count(self) -> int:
        return len(self._clients)


_broadcaster = _Broadcaster()


# ────────────────────────────────────────────────────────────────
# HTML dashboard (inline — self-contained, zero external deps)
# ────────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en" data-mode="dark">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ET-Agent Live Monitor</title>
<style>
:root{--bg:#0f1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;
  --gpu:#3fb950;--cpu:#d29922;--ssd:#58a6ff;--hit:#7ee787;--warn:#f85149;
  --migrate:#bc8cff;--compress:#f0883e;--waste:#f85149;--release:#3fb950;
  --radius:8px;--font:system-ui,-apple-system,sans-serif}
[data-mode=light]{--bg:#fff;--card:#f6f8fa;--border:#d0d7de;--text:#1f2328;--muted:#656d76}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh}
.header{background:var(--card);border-bottom:1px solid var(--border);padding:12px 24px;
  display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10}
.header h1{font-size:18px;font-weight:600;display:flex;align-items:center;gap:8px}
.header h1 .dot{width:8px;height:8px;background:var(--gpu);border-radius:50%;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.header-meta{display:flex;gap:16px;font-size:12px;color:var(--muted)}
.header-meta span{display:flex;align-items:center;gap:4px}
.container{padding:20px 24px;max-width:1440px;margin:0 auto}
/* stat row */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:16px}
.stat .sv{font-size:26px;font-weight:700;line-height:1.2}
.stat .sl{font-size:12px;color:var(--muted);margin-top:2px}
.stat .delta{font-size:11px;margin-top:4px}
/* section cards */
.section{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:16px}
.section-header{padding:14px 16px;border-bottom:1px solid var(--border);font-weight:600;font-size:14px;
  display:flex;align-items:center;gap:8px}
.section-body{padding:16px}
/* grid */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid3{grid-template-columns:repeat(3,1fr)}
.grid-full{grid-column:1/-1}
/* bars */
.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.bar-label{width:100px;font-size:12px;color:var(--muted);text-align:right;flex-shrink:0}
.bar-track{flex:1;height:22px;background:#0d1117;border-radius:5px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px;transition:width .4s ease;min-width:2px}
.bar-fill.gpu{background:var(--gpu)}.bar-fill.cpu{background:var(--cpu)}.bar-fill.ssd{background:var(--ssd)}
.bar-fill.hit{background:var(--hit)}.bar-fill.migrate{background:var(--migrate)}
.bar-fill.compress{background:var(--compress)}.bar-fill.waste{background:var(--waste)}
.bar-val{width:80px;font-size:11px;color:var(--muted);flex-shrink:0}
/* phase distribution */
.phase-row{display:flex;gap:10px;flex-wrap:wrap}
.phase-col{flex:1;min-width:90px;text-align:center}
.phase-fill{height:60px;border-radius:6px 6px 0 0;transition:height .4s ease;display:flex;
  align-items:flex-end;justify-content:center;color:#fff;font-weight:700;font-size:16px}
.phase-name{font-size:10px;color:var(--muted);margin-top:4px}
/* line chart */
svg.line-chart{width:100%;height:180px;overflow:visible}
/* theme btn */
.theme-btn{background:var(--card);border:1px solid var(--border);color:var(--muted);
  border-radius:6px;padding:4px 10px;cursor:pointer;font-size:13px}
.theme-btn:hover{color:var(--text)}
/* connection indicator */
#conn-status{font-size:11px;padding:2px 8px;border-radius:10px}
#conn-status.live{background:rgba(63,185,80,.15);color:var(--gpu)}
#conn-status.dead{background:rgba(248,81,73,.15);color:var(--warn)}
</style></head>
<body>
<div class=header>
  <h1><span class=dot></span>ET-Agent Live Memory Monitor</h1>
  <div class=header-meta>
    <span id=conn-status class=live>● Live</span>
    <span id=snap-count>0 snaps</span>
    <span id=clock>--</span>
    <button class=theme-btn onclick="toggleTheme()">☀</button>
  </div>
</div>
<div class=container>
  <!--- KPI row --->
  <div class=stats id=stat-row></div>
  <!--- Storage Tiers --->
  <div class=section>
    <div class=section-header>📦 Storage Tiers — GPU / CPU / SSD</div>
    <div class=section-body>
      <div class=grid2>
        <div><div id=tier-bars></div></div>
        <div><svg class=line-chart id=chart-tiers></svg></div>
      </div>
    </div>
  </div>
  <!--- KV Cache + Lifecycle --->
  <div class=section>
    <div class=section-header>🎯 Prefix Cache &amp; Lifecycle</div>
    <div class=section-body>
      <div class=grid2>
        <div>
          <div id=prefix-bars></div>
          <svg class=line-chart id=chart-prefix style=margin-top:12px></svg>
        </div>
        <div>
          <div id=phase-bars></div>
          <svg class=line-chart id=chart-lifecycle style=margin-top:12px></svg>
        </div>
      </div>
    </div>
  </div>
  <!--- Competition metrics + Migration --->
  <div class=section>
    <div class=section-header>📐 Competition Metrics — Waste Rate &amp; Release Rate &amp; Migration</div>
    <div class=section-body>
      <div class=grid3>
        <div><div id=waste-bars></div><svg class=line-chart id=chart-waste></svg></div>
        <div><div id=release-bars></div><svg class=line-chart id=chart-migrate></svg></div>
        <div><div id=migration-bars></div><svg class=line-chart id=chart-compress></svg></div>
      </div>
    </div>
  </div>
</div>

<script>
const MAX_POINTS = 120;
let series = {t:[],gpu:[],cpu:[],ssd:[],hit:[],active:[],waiting:[],migrate:[],
  prefetch:[],waste:[],release:[],tokens:[]};
let snapCount = 0;

function toggleTheme(){
  const h=document.documentElement;
  h.setAttribute('data-mode',h.getAttribute('data-mode')==='dark'?'light':'dark');
  document.querySelector('.theme-btn').textContent=h.getAttribute('data-mode')==='dark'?'☀':'🌙';
}

function fmt(n,d=0){if(n==null)return'-';return Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}
function pct(v){return v==null?'-':(Number(v)*100).toFixed(1)+'%'}
function fmts(v){if(v==null)return'0 B';const u=['B','KB','MB','GB','TB'];
  let i=0;while(v>=1024&&i<4){v/=1024;i++;}return v.toFixed(i>0?1:0)+' '+u[i]}

function updateStats(s){
  if(!s)return;
  const b=s.blocks||{}, p=s.prefix||{}, t=s.tiers||{}, l=s.lifecycle||{}, c=s.compression||{};
  document.getElementById('stat-row').innerHTML =
    `<div class=stat><div class=sv style=color:var(--gpu)>${fmt(b.gpu_blocks)}</div><div class=sl>GPU 块</div><div class=delta style=color:var(--gpu)>共享 ${fmt(b.shared)} · 固定 ${fmt(b.pinned)}</div></div>`+
    `<div class=stat><div class=sv style=color:var(--cpu)>${fmt(b.cpu_blocks)}</div><div class=sl>CPU 块</div><div class=delta>使用率 ${pct(t.cpu_ratio)}</div></div>`+
    `<div class=stat><div class=sv style=color:var(--hit)>${pct(p.hit_rate)}</div><div class=sl>前缀缓存命中率</div><div class=delta>${fmt(p.total_entries)} 条目 · ${fmt(p.hot_entries)} 热块</div></div>`+
    `<div class=stat><div class=sv style=color:var(--compress)>${fmt(c.total_tokens_saved)}</div><div class=sl>Token 节省量</div><div class=delta>消息丢弃 ${fmt(c.total_messages_dropped)} · 去重 ${fmt(c.dedup_tokens_saved)}</div></div>`;

  // Tier bars
  const maxBlocks=Math.max(b.gpu_blocks||1,b.cpu_blocks||1,b.ssd_blocks||1,1);
  document.getElementById('tier-bars').innerHTML=[
    {l:'GPU',v:b.gpu_blocks,css:'gpu'},{l:'CPU',v:b.cpu_blocks,css:'cpu'},{l:'SSD',v:b.ssd_blocks,css:'ssd'}
  ].map(x=>`<div class=bar-row><span class=bar-label>${x.l}</span><div class=bar-track><div class="bar-fill ${x.css}" style=width:${(x.v/maxBlocks*100)}%></div></div><span class=bar-val>${fmt(x.v)} 块</span></div>`).join('');

  // Prefix bars
  document.getElementById('prefix-bars').innerHTML=[
    {l:'命中率',v:Math.round(p.hit_rate*1000),css:'hit',disp:pct(p.hit_rate)},
    {l:'条目数',v:Math.max(p.total_entries||1,1),css:'gpu',disp:fmt(p.total_entries)},
    {l:'复用块',v:Math.max(p.blocks_reused||1,1),css:'cpu',disp:fmt(p.blocks_reused)},
  ].map(x=>`<div class=bar-row><span class=bar-label>${x.l}</span><div class=bar-track><div class="bar-fill ${x.css}" style=width:${Math.min(x.v/800*100,100)}%></div></div><span class=bar-val>${x.disp}</span></div>`).join('');

  // Phase bars
  const phases=[{k:'prefill_count',l:'Prefill',c:'#3fb950'},{k:'decoding_count',l:'Decoding',c:'#7ee787'},
    {k:'tool_call_count',l:'ToolCall',c:'#d29922'},{k:'idle_count',l:'Idle',c:'#58a6ff'},{k:'completed_count',l:'Done',c:'#6b7280'}];
  const maxPh=Math.max(...phases.map(q=>l[q.k]||0),1);
  document.getElementById('phase-bars').innerHTML='<div class=phase-row>'+phases.map(q=>
    `<div class=phase-col><div class=phase-fill style=height:${(l[q.k]||0)/maxPh*100}%;background:${q.c}>${l[q.k]||0}</div><div class=phase-name>${q.l}</div></div>`
  ).join('')+'</div>';

  // Competition metrics bars
  const wr=b.waste_rate||0, trr=b.tool_wait_release_rate||0;
  document.getElementById('waste-bars').innerHTML=`<div style=font-size:13px;margin-bottom:8px;color:var(--muted)>显存浪费率</div>`+
    `<div class=bar-row><span class=bar-label>浪费率</span><div class=bar-track><div class="bar-fill waste" style=width:${Math.min(wr*100,100)}%></div></div><span class=bar-val>${pct(wr)}</span></div>`+
    `<div style=font-size:28px;font-weight:700;margin-top:8px;color:var(--warn)>${pct(wr)}</div><div style=font-size:11px;color:var(--muted)>(original ~60-80%)</div>`;
  document.getElementById('release-bars').innerHTML=`<div style=font-size:13px;margin-bottom:8px;color:var(--muted)>工具等待 GPU 释放率</div>`+
    `<div class=bar-row><span class=bar-label>释放率</span><div class=bar-track><div class="bar-fill release" style=width:${Math.min(trr*100,100)}%></div></div><span class=bar-val>${pct(trr)}</span></div>`+
    `<div style=font-size:28px;font-weight:700;margin-top:8px;color:var(--release)>${pct(trr)}</div><div style=font-size:11px;color:var(--muted)>(demoted during tool wait)</div>`;
  document.getElementById('migration-bars').innerHTML=`<div style=font-size:13px;margin-bottom:8px;color:var(--muted)>迁移统计</div>`+
    `<div class=bar-row><span class=bar-label>总迁移</span><div class=bar-track><div class="bar-fill migrate" style=width:${Math.min(t.total_migrations/200*100,100)}%></div></div><span class=bar-val>${fmt(t.total_migrations)}</span></div>`+
    `<div class=bar-row><span class=bar-label>预取</span><div class=bar-track><div class="bar-fill gpu" style=width:${Math.min(t.total_prefetches/100*100,100)}%></div></div><span class=bar-val>${fmt(t.total_prefetches)}</span></div>`+
    `<div class=bar-row><span class=bar-label>Token 节省</span><div class=bar-track><div class="bar-fill compress" style=width:${Math.min(c.total_tokens_saved/50000*100,100)}%></div></div><span class=bar-val>${fmt(c.total_tokens_saved)}</span></div>`;
}

function pushSeries(k,v){series[k].push(v);if(series[k].length>MAX_POINTS)series[k].shift()}
function pushTime(){series.t.push(snapCount);if(series.t.length>MAX_POINTS)series.t.shift()}

function drawLine(id_,data,color){
  const el=document.getElementById(id_);if(!el||data.length<2)return;
  const W=el.clientWidth||400,H=180,P=16,max=Math.max(...data,1);
  const pts=data.map((v,i)=>{const x=P+(W-2*P)/(data.length-1||1)*i;const y=H-P-(v/max)*(H-2*P);return`${x},${y}`}).join(' ');
  el.innerHTML=`<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"/><polygon points="${P},${H-P} ${pts} ${W-P},${H-P}" fill="${color}" opacity=".08"/>`;
}

function redraw(){
  drawLine('chart-tiers',series.gpu,'#3fb950');
  drawLine('chart-prefix',series.hit.map(v=>v*100),'#7ee787');
  drawLine('chart-lifecycle',series.active,'#3fb950');
  drawLine('chart-waste',series.waste.map(v=>v*100),'#f85149');
  drawLine('chart-migrate',series.migrate,'#bc8cff');
  drawLine('chart-compress',series.tokens,'#f0883e');
}

let ws=null;
function connect(){
  const proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen=()=>{
    document.getElementById('conn-status').className='live';
    document.getElementById('conn-status').textContent='● Live';
  };
  ws.onclose=()=>{
    document.getElementById('conn-status').className='dead';
    document.getElementById('conn-status').textContent='● Disconnected';
    setTimeout(connect,3000);
  };
  ws.onmessage=(e)=>{
    try{
      const s=JSON.parse(e.data);snapCount++;
      document.getElementById('snap-count').textContent=snapCount+' snaps';
      document.getElementById('clock').textContent=new Date(s.timestamp*1000).toLocaleTimeString();
      updateStats(s);
      pushTime();
      pushSeries('gpu',s.blocks.gpu_blocks);
      pushSeries('cpu',s.blocks.cpu_blocks);
      pushSeries('ssd',s.blocks.ssd_blocks);
      pushSeries('hit',s.prefix.hit_rate);
      pushSeries('active',s.lifecycle.active_requests);
      pushSeries('waiting',s.lifecycle.waiting_requests);
      pushSeries('migrate',s.tiers.total_migrations);
      pushSeries('prefetch',s.tiers.total_prefetches);
      pushSeries('waste',s.blocks.waste_rate||0);
      pushSeries('release',s.blocks.tool_wait_release_rate||0);
      pushSeries('tokens',s.compression.total_tokens_saved);
      redraw();
      document.getElementById('chart-lifecycle').innerHTML +=
        `<text x=8 y=16 font-size=10 fill=var(--muted)>active</text>`;
    }catch(ex){}
  };
}
window.addEventListener('DOMContentLoaded',connect);
window.addEventListener('resize',redraw);
</script></body></html>"""


# ────────────────────────────────────────────────────────────────
# Async server
# ────────────────────────────────────────────────────────────────

async def _ws_handler(reader, writer):
    """Minimal async WebSocket handshake + frame."""
    # Read HTTP upgrade request
    request = (await reader.read(4096)).decode("utf-8", errors="replace")
    key = ""
    for line in request.split("\r\n"):
        if line.lower().startswith("sec-websocket-key:"):
            key = line.split(":", 1)[1].strip()
            break

    if not key:
        writer.close()
        return

    # Compute accept
    import hashlib, base64
    accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
    ).decode()

    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n"
        "\r\n"
    )
    writer.write(response.encode())
    await writer.drain()

    # Register
    _broadcaster.add(writer)

    # Keep alive (read pings, ignore)
    try:
        while True:
            data = await reader.read(1024)
            if not data:
                break
    except Exception:
        pass
    finally:
        _broadcaster.remove(writer)
        try:
            writer.close()
        except Exception:
            pass


async def _http_handler(reader, writer):
    """Serve the dashboard HTML."""
    request = (await reader.read(4096)).decode("utf-8", errors="replace")
    path = "/"
    for line in request.split("\r\n"):
        if line.startswith("GET "):
            path = line.split(" ")[1]
            break

    if path == "/" or path == "/index.html":
        body = DASHBOARD_HTML
        status = "200 OK"
        ctype = "text/html; charset=utf-8"
    else:
        body = "Not Found"
        status = "404 Not Found"
        ctype = "text/plain"

    response = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: {ctype}\r\n"
        f"Content-Length: {len(body.encode())}\r\n"
        f"Access-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{body}"
    )
    writer.write(response.encode())
    await writer.drain()
    writer.close()


async def _handle(reader, writer):
    """Route HTTP vs WebSocket."""
    peek = (await reader.read(4)).decode("utf-8", errors="replace")
    # prepend peek back
    reader = _prepend_reader(reader, peek.encode())

    if peek.startswith("GET "):
        await _http_handler(reader, writer)
    else:
        await _ws_handler(reader, writer)


def _prepend_reader(reader, data: bytes):
    """Return a new reader with *data* prepended."""
    import asyncio

    class _Prepended:
        def __init__(self, r, d):
            self._r = r
            self._buf = d
            self._pos = 0

        async def read(self, n=-1):
            if self._pos < len(self._buf):
                chunk = self._buf[self._pos:self._pos + n] if n > 0 else self._buf[self._pos:]
                self._pos += len(chunk)
                if n > 0 and len(chunk) < n:
                    chunk += await self._r.read(n - len(chunk))
                return chunk
            return await self._r.read(n)

    return _Prepended(reader, data)


# ────────────────────────────────────────────────────────────────
# Background pusher — reads MemoryMonitor snapshots
# ────────────────────────────────────────────────────────────────

def _start_pusher(mgr, interval_s: float):
    """Periodically push snapshots to all connected WebSocket clients."""

    def _push_loop():
        from memory_manager.memory_monitor import MemoryMonitor

        # Use existing monitor or create one
        monitor = getattr(mgr, "_monitor", None)
        if monitor is None:
            from memory_manager.memory_monitor import MemoryMonitor
            monitor = MemoryMonitor(mgr)
            monitor.start(interval_s=0.5)

        while True:
            try:
                snap = monitor.snapshot()
                d = snap.to_dict()
                # Compute competition metrics inline
                b = d.get("blocks", {})
                used = b.get("used", 0)
                free = b.get("free", 0)
                total_alloc = used + free
                # Waste rate: fraction of allocated block slots that are empty
                if used > 0:
                    waste = 1.0 - (used / max(total_alloc, 1)) if total_alloc > 0 else 0.0
                else:
                    waste = 0.0
                b["waste_rate"] = round(waste, 4)

                l = d.get("lifecycle", {})
                demotions = l.get("total_demotions", 0)
                promotions = l.get("total_promotions", 0)
                total_actions = max(demotions + promotions, 1)
                b["tool_wait_release_rate"] = round(min(demotions / total_actions, 1.0), 4)

                payload = json.dumps(d)
                _broadcaster.push(payload)
            except Exception:
                pass
            time.sleep(interval_s)

    t = threading.Thread(target=_push_loop, daemon=True)
    t.start()
    return t


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ET-Agent Live Memory Monitor Server")
    parser.add_argument("--port", type=int, default=8765, help="HTTP/WS port")
    parser.add_argument("--interval", type=float, default=1.0, help="Snapshot push interval (seconds)")
    parser.add_argument("--model", default="qwen2.5-7b", help="Model name")
    parser.add_argument("--gpu-gb", type=int, default=6, help="GPU VRAM (GB)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark to generate live data")
    args = parser.parse_args()

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  ET-Agent Live Memory Monitor               ║")
    print(f"║  http://localhost:{args.port:<5}                    ║")
    print(f"║  Model: {args.model:<34}║")
    print(f"║  GPU:   {args.gpu_gb} GB{'':>31}║")
    print(f"╚══════════════════════════════════════════════╝")

    # Init memory manager
    from agent.memory_hooks import create_agent_memory_manager
    mgr = create_agent_memory_manager(args.model, gpu_gb=args.gpu_gb)
    print(f"[init] {mgr.allocator.total_blocks} blocks, "
          f"block_size={mgr._config.block_size}")

    # Start benchmark in background if requested
    if args.benchmark:
        print("[bench] Starting simulated benchmark in background...")
        _spawn_benchmark(mgr)
    else:
        print("[info] No benchmark — run with --benchmark for live demo data")
        print("[info] Open the dashboard and use /memory or /memory-stats in CLI")

    # Start pusher
    _start_pusher(mgr, args.interval)
    print(f"[push] Streaming snapshots every {args.interval}s to {_broadcaster.count} clients")

    # Start async server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def serve():
        server = await asyncio.start_server(_handle, "0.0.0.0", args.port)
        print(f"[serve] Listening on port {args.port}")
        async with server:
            await server.serve_forever()

    try:
        loop.run_until_complete(serve())
    except KeyboardInterrupt:
        print("\n[exit] Shutting down...")


def _spawn_benchmark(mgr):
    """Run a simulated benchmark in a background thread to generate live data."""
    import random, threading

    def _run():
        try:
            from memory_manager.memory_monitor import MemoryMonitor
            monitor = MemoryMonitor(mgr)
            monitor.start(interval_s=0.5)
            mgr._monitor = monitor  # attach for pusher

            # Simulate 3 sessions
            for sid in ["sess-a", "sess-b", "sess-c"]:
                sp = [random.randint(0, 50000) for _ in range(3000)]
                tools = [{"type": "function", "function": {"name": n, "description": n,
                         "parameters": {"type": "object", "properties": {}}}}
                         for n in ["search", "read", "write", "terminal"]]
                mgr.on_session_start(sid, system_prompt_tokens=sp, tool_definitions=tools)

            # Run ~200 turns
            sids = ["sess-a", "sess-b", "sess-c"]
            for turn in range(200):
                for sid in sids:
                    msgs = [{"role": "user", "content": f"Turn {turn}: " + "x" * random.randint(20, 200)}]
                    mgr.pre_llm_call(sid, msgs)
                    if random.random() < 0.4:
                        mgr.post_llm_call(sid, assistant_message={
                            "tool_calls": [{"function": {"name": random.choice(["search", "read", "write", "terminal"]),
                                                         "arguments": "{}"}}]
                        }, has_tool_calls=True)
                        mgr.on_tool_result(sid, tool_name="search")
                    else:
                        mgr.post_llm_call(sid, has_tool_calls=False)
                if turn % 15 == 0 and turn > 0:
                    for sid in sids:
                        total_tokens = turn * 600
                        mgr.maybe_compress(sid, msgs, total_tokens, 50000)
                if turn % 10 == 0:
                    # Trigger lifecycle scan to produce demotions
                    mgr.lifecycle.scan_and_migrate()

                import time as _t; _t.sleep(0.5)

            for sid in sids:
                mgr.on_session_end(sid)

        except Exception as exc:
            import traceback; traceback.print_exc()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    main()
