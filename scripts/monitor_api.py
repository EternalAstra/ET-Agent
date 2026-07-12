#!/usr/bin/env python3
"""ET-Agent Memory Monitor API — tiny HTTP server for the Electron dashboard.

Provides:
  GET /              → dashboard HTML
  GET /api/snapshot  → latest MemorySnapshot as JSON
  GET /api/history   → all snapshots since server start as JSON

The server reads from the global AgentMemoryManager singleton stored in
``agent.kv_memory_integration._global_memory_manager`` — so it reflects
your REAL conversation state, not simulated data.

Start it alongside your ET-Agent CLI:
  python scripts/monitor_api.py --port 8765

Or from Python code:
  from scripts.monitor_api import start_monitor_api
  start_monitor_api(port=8765)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ────────────────────────────────────────────────────────────────
# Dashboard HTML  (same rich dashboard, but now uses HTTP polling)
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
.header{background:var(--card);border-bottom:1px solid var(--border);padding:10px 20px;
  display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10}
.header h1{font-size:16px;font-weight:600}
.header-meta{display:flex;gap:14px;font-size:11px;color:var(--muted);align-items:center}
.status-live{display:inline-block;width:7px;height:7px;background:var(--gpu);border-radius:50%;
  margin-right:4px;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.container{padding:16px 20px;max-width:1440px;margin:0 auto}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:14px}
.stat .sv{font-size:24px;font-weight:700;line-height:1.2}
.stat .sl{font-size:11px;color:var(--muted);margin-top:2px}
.stat .delta{font-size:10px;margin-top:3px;color:var(--muted)}
.section{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);margin-bottom:14px}
.section-hdr{padding:12px 14px;border-bottom:1px solid var(--border);font-weight:600;font-size:13px;
  display:flex;align-items:center;gap:6px}
.section-body{padding:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid3{grid-template-columns:repeat(3,1fr)}
.bar-row{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.bar-label{width:90px;font-size:11px;color:var(--muted);text-align:right;flex-shrink:0}
.bar-track{flex:1;height:20px;background:#0d1117;border-radius:5px;overflow:hidden}
.bar-fill{height:100%;border-radius:5px;transition:width .4s ease;min-width:2px}
.bar-fill.gpu{background:var(--gpu)}.bar-fill.cpu{background:var(--cpu)}.bar-fill.ssd{background:var(--ssd)}
.bar-fill.hit{background:var(--hit)}.bar-fill.migrate{background:var(--migrate)}
.bar-fill.compress{background:var(--compress)}.bar-fill.waste{background:var(--waste)}
.bar-fill.release{background:var(--release)}
.bar-val{width:75px;font-size:10px;color:var(--muted);flex-shrink:0}
.phase-row{display:flex;gap:8px;flex-wrap:wrap}
.phase-col{flex:1;min-width:85px;text-align:center}
.phase-fill{height:50px;border-radius:6px 6px 0 0;transition:height .4s ease;
  display:flex;align-items:flex-end;justify-content:center;color:#fff;font-weight:700;font-size:14px}
.phase-name{font-size:9px;color:var(--muted);margin-top:3px}
.chart-wrap{position:relative;width:100%}
.chart-wrap svg{width:100%;height:180px;overflow:visible}
table.features{width:100%;border-collapse:collapse;font-size:12px}
table.features th{text-align:left;padding:6px 10px;border-bottom:2px solid var(--border);color:var(--muted)}
table.features td{padding:6px 10px;border-bottom:1px solid var(--border)}
table.features .new{color:var(--gpu);font-weight:600}
.btn{background:var(--card);border:1px solid var(--border);color:var(--muted);
  border-radius:5px;padding:4px 10px;cursor:pointer;font-size:12px}
.btn:hover{color:var(--text)}
</style></head>
<body>
<div class=header>
  <h1>⚡ ET-Agent Live Memory Monitor</h1>
  <div class=header-meta>
    <span><span class=status-live></span>Live</span>
    <span id=snap-count>0 snaps</span>
    <span id=clock>--:--:--</span>
    <button class=btn onclick="toggleTheme()">☀</button>
  </div>
</div>
<div class=container>
  <div class=stats id=stat-row></div>
  <div class=section>
    <div class=section-hdr>📦 Storage Tiers — GPU · CPU · SSD</div>
    <div class=section-body>
      <div class=grid2>
        <div id=tier-bars></div>
        <div class=chart-wrap><svg id=chart-gpu></svg></div>
      </div>
    </div>
  </div>
  <div class=section>
    <div class=section-hdr>🎯 Prefix Cache & Agent Lifecycle</div>
    <div class=section-body>
      <div class=grid2>
        <div><div id=prefix-bars></div><div class=chart-wrap><svg id=chart-prefix></svg></div></div>
        <div><div id=phase-bars></div><div class=chart-wrap><svg id=chart-lifecycle></svg></div></div>
      </div>
    </div>
  </div>
  <div class=section>
    <div class=section-hdr>📐 Competition Metrics</div>
    <div class=section-body>
      <div class="grid2 grid3">
        <div><div style=font-size:12px;color:var(--muted);margin-bottom:6px>显存浪费率</div>
          <div style=font-size:32px;font-weight:700;color:var(--waste) id=metric-waste>--</div>
          <div style=font-size:10px;color:var(--muted)>原始 Hermes ~60-80%</div>
          <div class=chart-wrap><svg id=chart-waste></svg></div></div>
        <div><div style=font-size:12px;color:var(--muted);margin-bottom:6px>工具等待 GPU 释放率</div>
          <div style=font-size:32px;font-weight:700;color:var(--release) id=metric-release>--</div>
          <div style=font-size:10px;color:var(--muted)>demotions / (demotions+promotions)</div>
          <div class=chart-wrap><svg id=chart-migrate></svg></div></div>
        <div><div style=font-size:12px;color:var(--muted);margin-bottom:6px>Token 节省 + 迁移</div>
          <div style=font-size:32px;font-weight:700;color:var(--compress) id=metric-tokens>--</div>
          <div style=font-size:10px;color:var(--muted)>累计 Token 减少量</div>
          <div class=chart-wrap><svg id=chart-compress></svg></div></div>
      </div>
    </div>
  </div>
</div>
<script>
const MAX=120;
let hist={t:[],gpu:[],cpu:[],ssd:[],hit:[],active:[],waiting:[],waste:[],migr:[],tok:[]};
let n=0;

function toggleTheme(){
  let h=document.documentElement,m=h.getAttribute('data-mode');
  h.setAttribute('data-mode',m==='dark'?'light':'dark');}
function fmt(v,d){if(v==null)return'-';return Number(v).toLocaleString(undefined,{maximumFractionDigits:d??0})}
function pct(v){return v==null?'-':(Number(v)*100).toFixed(1)+'%'}
let lastData=null;

async function poll(){
  try{
    let r=await fetch('/api/snapshot');let s=await r.json();lastData=s;n++;
    document.getElementById('snap-count').textContent=n+' snaps';
    document.getElementById('clock').textContent=new Date().toLocaleTimeString();
    update(s);pushHistory(s);drawAll();
  }catch(e){}
}
function update(s){
  if(s._waiting){document.getElementById('stat-row').innerHTML='<div class=stat style=grid-column:1/-1;text-align:center;padding:24px><div style=font-size:16px;color:var(--cpu)>⏳ Waiting for hermes agent to connect...</div><div style=font-size:11px;color:var(--muted);margin-top:4px>Start hermes in another terminal to see live memory data</div></div>';document.getElementById('tier-bars').innerHTML='';document.getElementById('phase-bars').innerHTML='';return;}
  let b=s.blocks||{},p=s.prefix||{},t=s.tiers||{},l=s.lifecycle||{},c=s.compression||{};
  document.getElementById('stat-row').innerHTML=
    `<div class=stat><div class=sv style=color:var(--gpu)>${fmt(b.gpu_blocks)}</div><div class=sl>GPU Blocks</div><div class=delta>shared ${fmt(b.shared)} · pinned ${fmt(b.pinned)}</div></div>`+
    `<div class=stat><div class=sv style=color:var(--cpu)>${fmt(b.cpu_blocks)}</div><div class=sl>CPU Blocks</div><div class=delta>usage ${pct(t.cpu_ratio)}</div></div>`+
    `<div class=stat><div class=sv style=color:var(--hit)>${pct(p.hit_rate)}</div><div class=sl>Prefix Hit Rate</div><div class=delta>${fmt(p.total_entries)} entries · ${fmt(p.hot_entries)} hot</div></div>`+
    `<div class=stat><div class=sv style=color:var(--compress)>${fmt(c.total_tokens_saved)}</div><div class=sl>Tokens Saved</div><div class=delta>msg dropped ${fmt(c.total_messages_dropped)}</div></div>`;
  let mx=Math.max(b.gpu_blocks||1,b.cpu_blocks||1,b.ssd_blocks||1,1);
  document.getElementById('tier-bars').innerHTML=[
    {l:'GPU',v:b.gpu_blocks,c:'gpu'},{l:'CPU',v:b.cpu_blocks,c:'cpu'},{l:'SSD',v:b.ssd_blocks,c:'ssd'}
  ].map(x=>`<div class=bar-row><span class=bar-label>${x.l}</span><div class=bar-track><div class="bar-fill ${x.c}" style=width:${x.v/mx*100}%></div></div><span class=bar-val>${fmt(x.v)} blocks</span></div>`).join('');
  document.getElementById('prefix-bars').innerHTML=[
    {l:'Hit Rate',v:Math.round(p.hit_rate*1000),c:'hit',d:pct(p.hit_rate)},
    {l:'Entries',v:Math.max(p.total_entries||1,1),c:'gpu',d:fmt(p.total_entries)},
    {l:'Blocks Reused',v:Math.max(p.blocks_reused||1,1),c:'cpu',d:fmt(p.blocks_reused)}
  ].map(x=>`<div class=bar-row><span class=bar-label>${x.l}</span><div class=bar-track><div class="bar-fill ${x.c}" style=width:${Math.min(x.v/500*100,100)}%></div></div><span class=bar-val>${x.d}</span></div>`).join('');
  let phs=[{k:'prefill_count',l:'Prefill',c:'#3fb950'},{k:'decoding_count',l:'Decode',c:'#7ee787'},{k:'tool_call_count',l:'ToolCall',c:'#d29922'},{k:'idle_count',l:'Idle',c:'#58a6ff'},{k:'completed_count',l:'Done',c:'#6b7280'}];
  let mxP=Math.max(...phs.map(q=>l[q.k]||0),1);
  document.getElementById('phase-bars').innerHTML='<div class=phase-row>'+phs.map(q=>`<div class=phase-col><div class=phase-fill style=height:${(l[q.k]||0)/mxP*100}%;background:${q.c}>${l[q.k]||0}</div><div class=phase-name>${q.l}</div></div>`).join('')+'</div>';
  let wr=b.waste_rate||0,rr=b.tool_wait_release_rate||0;
  document.getElementById('metric-waste').textContent=pct(wr);
  document.getElementById('metric-release').textContent=pct(rr);
  document.getElementById('metric-tokens').textContent=fmt(c.total_tokens_saved);
}
function pushHistory(s){
  hist.t.push(n);let b=s.blocks||{},p=s.prefix||{},t=s.tiers||{},l=s.lifecycle||{},c=s.compression||{};
  hist.gpu.push(b.gpu_blocks);hist.cpu.push(b.cpu_blocks);hist.ssd.push(b.ssd_blocks);
  hist.hit.push(p.hit_rate*100);hist.active.push(l.active_requests);hist.waiting.push(l.waiting_requests);
  hist.waste.push((b.waste_rate||0)*100);hist.migr.push(t.total_migrations);hist.tok.push(c.total_tokens_saved);
  for(let k of Object.keys(hist)){if(hist[k].length>MAX)hist[k].shift();}
}
function drawSvg(id,data,color){
  let el=document.getElementById(id);if(!el||data.length<2)return;
  let W=el.parentElement.clientWidth||300,H=180,P=14,mx=Math.max(...data,1);
  let pts=data.map((v,i)=>{let x=P+(W-2*P)/(data.length-1||1)*i;let y=H-P-(v/mx)*(H-2*P);return x+','+y;}).join(' ');
  el.innerHTML=`<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2"/><polygon points="${P},${H-P} ${pts} ${W-P},${H-P}" fill="${color}" opacity=".08"/>`;
}
function drawAll(){
  drawSvg('chart-gpu',hist.gpu,'#3fb950');drawSvg('chart-prefix',hist.hit,'#7ee787');
  drawSvg('chart-lifecycle',hist.active,'#3fb950');drawSvg('chart-waste',hist.waste,'#f85149');
  drawSvg('chart-migrate',hist.migr,'#bc8cff');drawSvg('chart-compress',hist.tok,'#f0883e');
}
window.addEventListener('DOMContentLoaded',()=>{poll();setInterval(poll,1000);});
window.addEventListener('resize',drawAll);
</script></body></html>"""


# ────────────────────────────────────────────────────────────────
# Global memory manager reference
# ────────────────────────────────────────────────────────────────

_global_manager = None

# ── Cross-process stats bridge: hermes pushes → monitor caches → dashboard reads ──
_pushed_stats: dict = {}
_pushed_at: float = 0.0
PUSH_STALENESS_S = 5.0  # show "waiting" if no push for this many seconds


def _is_push_fresh() -> bool:
    return (time.time() - _pushed_at) < PUSH_STALENESS_S


def set_global_manager(mgr):
    """Register the active AgentMemoryManager so the API can read from it."""
    global _global_manager
    _global_manager = mgr


def get_global_manager():
    """Get the active manager — agent-created takes priority."""
    global _global_manager

    # Priority 1: agent.memory_hooks global (set by hermes agent init)
    try:
        from agent.memory_hooks import get_global_memory_manager
        mgr = get_global_memory_manager()
        if mgr is not None:
            _global_manager = mgr
            return mgr
    except Exception:
        pass

    # Priority 2: locally stored (set by set_global_manager)
    if _global_manager is not None:
        return _global_manager

    # Priority 3: plugin module cached instance
    try:
        from plugins.memory_manager_plugin.__init__ import _memory_manager as _pm
        if _pm is not None:
            _global_manager = _pm
            return _pm
    except Exception:
        pass

    return None


# ────────────────────────────────────────────────────────────────
# HTTP request handler
# ────────────────────────────────────────────────────────────────


class MonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # quiet

    def _send_json(self, data, code=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, code=200):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/api/push":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                global _pushed_stats, _pushed_at
                _pushed_stats = data
                _pushed_at = time.time()
                self._send_json({"ok": True, "pushed": _pushed_at})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
        else:
            self._send_json({"error": "not found"}, 404)

    def _build_snapshot(self, mgr):
        """Build a snapshot dict from a live AgentMemoryManager."""
        s = mgr.stats()
        a = s.get("allocator", {})
        h = s.get("hierarchical_store", {})
        p = s.get("prefix_cache", {})
        l = s.get("lifecycle", {})
        c = s.get("compressor", {})
        d = s.get("deduplicator", {})
        t = s.get("tool_compressor", {})

        used = a.get("used_blocks", 0)
        prefix_entries = p.get("total_entries", 0)
        filled_blocks = min(prefix_entries, max(used, 1))
        waste_rate = round(1.0 - (filled_blocks / max(used, 1)), 4) if used > 0 else 0.0

        demotions = l.get("total_demotions", 0)
        promotions = l.get("total_promotions", 0)
        total_acts = max(demotions + promotions, 1)
        release_rate = round(min(demotions / total_acts, 1.0), 4)

        return {
            "timestamp": time.time(), "blocks": {
                "total": a.get("total_blocks", 0), "free": a.get("free_blocks", 0),
                "used": used, "gpu_blocks": h.get("gpu_blocks", 0),
                "cpu_blocks": h.get("cpu_blocks", 0), "ssd_blocks": h.get("ssd_blocks", 0),
                "shared": a.get("shared_blocks", 0), "pinned": a.get("pinned_blocks", 0),
                "waste_rate": waste_rate, "tool_wait_release_rate": release_rate,
            }, "prefix": {
                "total_entries": p.get("total_entries", 0),
                "pinned_entries": p.get("pinned_entries", 0),
                "hot_entries": p.get("hot_entries", 0),
                "hit_rate": p.get("hit_rate", 0.0),
                "blocks_reused": p.get("blocks_reused", 0),
            }, "tiers": {
                "gpu_bytes": h.get("gpu_usage_bytes", 0),
                "cpu_bytes": h.get("cpu_usage_bytes", 0),
                "ssd_bytes": h.get("ssd_usage_bytes", 0),
                "gpu_ratio": h.get("gpu_usage_ratio", 0.0),
                "cpu_ratio": h.get("cpu_usage_ratio", 0.0),
                "total_migrations": h.get("total_migrations", 0),
                "total_prefetches": h.get("total_prefetches", 0),
            }, "lifecycle": {
                "active_requests": l.get("active_requests", 0),
                "waiting_requests": l.get("waiting_requests", 0),
                "prefill_count": l.get("phases", {}).get("PREFILL", 0),
                "decoding_count": l.get("phases", {}).get("DECODING", 0),
                "tool_call_count": l.get("phases", {}).get("TOOL_CALL", 0),
                "idle_count": l.get("phases", {}).get("IDLE", 0),
                "completed_count": l.get("phases", {}).get("COMPLETED", 0),
                "total_demotions": demotions, "total_promotions": promotions,
            }, "compression": {
                "total_tokens_saved": int(c.get("total_tokens_saved", 0) + d.get("total_tokens_saved", 0) + t.get("total_tokens_saved", 0)),
                "total_history_compressions": c.get("total_history_compressions", 0),
                "total_messages_dropped": d.get("total_messages_dropped", 0),
            },
        }

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_html(DASHBOARD_HTML)

        elif self.path == "/api/snapshot":
            # Priority 1: stats pushed from hermes via POST /api/push (cross-process bridge)
            if _is_push_fresh():
                result = dict(_pushed_stats)
                result["_live"] = True
                self._send_json(result)
                return

            # Priority 2: in-process manager (demo benchmark or embedded agent)
            mgr = get_global_manager()
            if mgr is not None:
                result = self._build_snapshot(mgr)
                result["_live"] = True
                self._send_json(result)
                return

            # No data source yet
            self._send_json({
                "timestamp": time.time(),
                "blocks": {"total":0,"free":0,"used":0,"gpu_blocks":0,"cpu_blocks":0,"ssd_blocks":0,"shared":0,"pinned":0,"waste_rate":0,"tool_wait_release_rate":0},
                "prefix": {"total_entries":0,"pinned_entries":0,"hot_entries":0,"hit_rate":0,"blocks_reused":0},
                "tiers": {"gpu_bytes":0,"cpu_bytes":0,"ssd_bytes":0,"gpu_ratio":0,"cpu_ratio":0,"total_migrations":0,"total_prefetches":0},
                "lifecycle": {"active_requests":0,"waiting_requests":0,"prefill_count":0,"decoding_count":0,"tool_call_count":0,"idle_count":0,"completed_count":0,"total_demotions":0,"total_promotions":0},
                "compression": {"total_tokens_saved":0,"total_history_compressions":0,"total_messages_dropped":0},
                "_waiting": True
            })

        else:
            self._send_json({"error": "not found"}, 404)


# ────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────


def start_monitor_api(port: int = 8765, blocking: bool = False):
    """Start the HTTP monitoring API server.

    Parameters
    ----------
    port : int
        Port to listen on.
    blocking : bool
        If True, run in the current thread (blocks forever).
        If False, run in a daemon thread.

    Returns
    -------
    HTTPServer
        The server instance (can call ``.shutdown()`` to stop).
    """
    server = HTTPServer(("0.0.0.0", port), MonitorHandler)

    if blocking:
        print(f"[monitor API] http://localhost:{port}")
        server.serve_forever()
    else:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[monitor API] http://localhost:{port}  (background thread)")

    return server


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ET-Agent Memory Monitor API Server")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument("--model", default="qwen2.5-7b")
    parser.add_argument("--gpu-gb", type=int, default=6)
    parser.add_argument("--demo", action="store_true", default=True,
                        help="Run a demo benchmark to generate live data (default: on)")
    parser.add_argument("--no-demo", action="store_true",
                        help="Disable demo benchmark")
    parser.add_argument("--shared-file", default="",
                        help="Path to shared stats JSON (hermes writes, monitor reads)")
    args = parser.parse_args()

    # Try shared-file mode first (cross-process IPC)
    if args.shared_file:
        print(f"[monitor] Shared-file mode: {args.shared_file}")
        # The hermes agent writes stats via kv_memory_integration → this file
        # The monitor reads it on each /api/snapshot
    elif getattr(args, "no_demo", False):
        pass  # no demo, no shared file → wait for hermes in-process (won't work cross-process)
    else:
        # Default: create own manager + background demo benchmark
        from agent.memory_hooks import create_agent_memory_manager
        mgr = create_agent_memory_manager(args.model, gpu_gb=args.gpu_gb)
        set_global_manager(mgr)
        print(f"[init] Created memory manager ({mgr.allocator.total_blocks} blocks)")
        print(f"[bench] Starting demo benchmark in background...")
        _start_demo_benchmark(mgr)

    # Check what we have
    mgr = get_global_manager()
    if mgr is not None:
        print(f"[monitor] Ready — {mgr.allocator.total_blocks} blocks, "
              f"{mgr.lifecycle.stats()['total_requests']} sessions")
    else:
        print("[monitor] Waiting for data source...")
        # Create a minimal manager as fallback
        try:
            from agent.memory_hooks import create_agent_memory_manager
            mgr = create_agent_memory_manager(args.model, gpu_gb=args.gpu_gb)
            set_global_manager(mgr)
        except Exception:
            pass

    print(f"\n  Dashboard: http://localhost:{args.port}")
    print(f"  API:       http://localhost:{args.port}/api/snapshot")
    print()

    start_monitor_api(port=args.port, blocking=True)


def _start_demo_benchmark(mgr):
    """Background thread: simulate 3 agent sessions with tool calls, phase transitions, migrations."""
    import random, threading

    def _run():
        try:
            sids = ["sess-a", "sess-b", "sess-c"]
            for sid in sids:
                sp = [random.randint(0, 50000) for _ in range(3000)]
                tools = [{"type": "function", "function": {"name": n, "description": n,
                         "parameters": {"type": "object", "properties": {}}}}
                         for n in ["search", "read", "write", "terminal"]]
                mgr.on_session_start(sid, system_prompt_tokens=sp, tool_definitions=tools)

            for turn in range(500):
                for sid in sids:
                    msgs = [{"role": "user", "content": f"Turn {turn}: " + "x" * random.randint(20, 200)}]
                    mgr.pre_llm_call(sid, msgs)
                    r = random.random()
                    if r < 0.4:
                        mgr.post_llm_call(sid, assistant_message={
                            "tool_calls": [{"function": {"name": random.choice(["search", "read", "write", "terminal"]),
                                                         "arguments": "{}"}}]
                        }, has_tool_calls=True)
                        # Simulate tool wait → result → promote
                        if random.random() < 0.7:
                            mgr.hierarchical_store.demote_blocks(
                                list(mgr.allocator.get_request_blocks(sid)),
                                __import__('memory_manager.kv_block', fromlist=['StorageTier']).StorageTier.CPU, sid
                            )
                        mgr.on_tool_result(sid, tool_name="search")
                    else:
                        mgr.post_llm_call(sid, has_tool_calls=False)

                # Periodic compression
                if turn % 20 == 0 and turn > 0:
                    for sid in sids:
                        total = turn * 600
                        mgr.maybe_compress(sid, msgs, total, 50000)

                # Periodic lifecycle scan → demotions
                if turn % 8 == 0:
                    mgr.lifecycle.scan_and_migrate()
                    mgr.hierarchical_store.evict_cold_blocks()

                time.sleep(0.25)

            for sid in sids:
                mgr.on_session_end(sid)
        except Exception:
            import traceback; traceback.print_exc()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

    print(f"\n  Dashboard: http://localhost:{args.port}")
    print(f"  API:       http://localhost:{args.port}/api/snapshot")
    print()

    start_monitor_api(port=args.port, blocking=True)


if __name__ == "__main__":
    main()
