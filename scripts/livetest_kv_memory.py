#!/usr/bin/env python
"""Live integration test: Hermes baseline (KV off) vs ET-Agent (KV on).

Runs real multi-turn chat and tool-chain scenarios against a local Ollama
endpoint, then compares latency, token usage, task success, and KV stats.

Usage
-----
    # Prerequisites: Ollama running with qwen2.5:3b-instruct pulled
    ollama serve
    ollama pull qwen2.5:3b-instruct

    python scripts/livetest_kv_memory.py
    python scripts/livetest_kv_memory.py --scenario chat
    python scripts/livetest_kv_memory.py --scenario tool_chain --mode et
    python scripts/livetest_kv_memory.py --check-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_GPU_GB = 6
REAL_HOME = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "et-agent"
DEFAULT_OUT = ROOT / "docs" / "team" / "livetest_kv_results.json"


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "chat": {
        "description": "3-turn multi-turn chat in one session (prefix reuse)",
        "toolsets": [],
        "requires_tools": False,
        "max_iterations": 8,
        "turns": [
            "你好，请记住我的名字是小明。只回复「好的」。",
            "我刚才说我叫什么？只回答名字。",
            "用一句话解释什么是操作系统里的虚拟内存。",
        ],
        "system_message": (
            "你是 ET-Agent 测试助手。回答简洁，不要调用工具。"
        ),
    },
    "tool_chain": {
        "description": "read_file tool chain on a local fixture",
        "toolsets": ["file"],
        "requires_tools": True,
        "max_iterations": 12,
        "fixture_name": "notes.txt",
        "fixture_lines": [
            "line one: project ET-Agent",
            "line two: memory manager benchmark",
            "line three: KV cache integration",
            "line four: end of file",
        ],
        "prompt_template": (
            "请使用 read_file 工具读取文件 {path}，"
            "然后只回复该文件第三行的完整内容，不要添加其他说明。"
        ),
        "system_message": (
            "你是 ET-Agent 测试助手。必须使用 read_file 工具读取文件后再回答。"
        ),
    },
}


def _yaml_dump(obj: Any) -> str:
    try:
        import yaml
        return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)
    except ImportError:
        return json.dumps(obj, indent=2, ensure_ascii=False)


def _load_user_model_config() -> dict:
    cfg_path = REAL_HOME / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def check_ollama(base_url: str, model: str) -> Dict[str, Any]:
    """Return health info; raise RuntimeError if Ollama is unreachable."""
    root = base_url.rstrip("/").removesuffix("/v1")
    tags_url = f"{root}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama not reachable at {tags_url}: {exc}\n"
            "Start it with: ollama serve"
        ) from exc

    names = [m.get("name", "") for m in payload.get("models", [])]
    model_ok = any(model in n or n.startswith(model) for n in names)
    return {"models": names, "model_available": model_ok, "tags_url": tags_url}


def reset_module_state() -> None:
    keys = [
        k for k in list(sys.modules)
        if k.startswith(("tools.", "model_tools", "toolsets", "hermes_cli", "agent.", "run_agent"))
    ]
    for k in keys:
        del sys.modules[k]


def setup_isolated_home(
    *,
    kv_enabled: bool,
    gpu_gb: int,
    model: str,
    base_url: str,
    ollama_num_ctx: int,
) -> Path:
    home_dir = Path(tempfile.mkdtemp(prefix="et_kv_live_"))
    hermes_home = home_dir / ".et-agent"
    hermes_home.mkdir(parents=True)
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)

    user_cfg = _load_user_model_config()
    model_cfg = dict(user_cfg.get("model") or {})
    model_cfg.setdefault("provider", "custom")
    model_cfg.setdefault("default", model)
    model_cfg.setdefault("base_url", base_url)
    # ET-Agent rejects models below 64K context at init; declare 64K for the
    # feasibility check while capping the real Ollama window via ollama_num_ctx.
    model_cfg["context_length"] = max(int(model_cfg.get("context_length") or 0), 65536)
    model_cfg["ollama_num_ctx"] = ollama_num_ctx
    model_cfg.setdefault("max_tokens", 512)

    memory_cfg = dict(user_cfg.get("memory") or {})
    kv_cfg = dict(memory_cfg.get("kv_manager") or {})
    kv_cfg["enabled"] = kv_enabled
    kv_cfg["gpu_gb"] = gpu_gb
    memory_cfg["kv_manager"] = kv_cfg

    cfg = {
        "model": model_cfg,
        "memory": memory_cfg,
        "toolsets": user_cfg.get("toolsets") or ["hermes-cli", "safe"],
        "context": {"compression": {"enabled": False}},
        "logging": {"level": "WARNING"},
        "agent": {
            "environment_probe": False,
            "task_completion_guidance": False,
            "parallel_tool_call_guidance": False,
        },
    }
    (hermes_home / "config.yaml").write_text(_yaml_dump(cfg), encoding="utf-8")
    return hermes_home


def _count_tool_calls(messages: List[Dict[str, Any]]) -> int:
    count = 0
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            count += len(msg["tool_calls"])
    return count


def _extract_kv_metrics(agent) -> Dict[str, Any]:
    from agent.kv_memory_integration import get_kv_memory_manager, kv_memory_stats

    mgr = get_kv_memory_manager(agent)
    if not mgr:
        return {"kv_active": False}

    stats = kv_memory_stats(agent) or {}
    prefix = stats.get("prefix_cache") or {}
    hier = stats.get("hierarchical_store") or {}
    dedup = stats.get("deduplicator") or {}
    alloc = stats.get("allocator") or {}
    lifecycle = stats.get("lifecycle") or {}

    return {
        "kv_active": True,
        "sessions": stats.get("sessions", 0),
        "turns": stats.get("turns", 0),
        "prefix_hit_rate": prefix.get("hit_rate", 0.0),
        "prefix_hits": prefix.get("hits", prefix.get("prefix_hits", 0)),
        "migrations": hier.get("total_migrations", 0),
        "tokens_saved": dedup.get("total_tokens_saved", 0),
        "peak_gpu_blocks": alloc.get("peak_used_blocks", alloc.get("used_blocks", 0)),
        "gpu_blocks_allocated": alloc.get("used_blocks", 0),
        "lifecycle_requests": lifecycle.get("total_requests", 0),
    }


def _validate_chat(result: Dict[str, Any]) -> Dict[str, Any]:
    turn_results = result.get("turn_results") or []
    name_reply = ""
    if len(turn_results) >= 2:
        name_reply = turn_results[1].get("final_response") or ""
    remembered = "小明" in name_reply
    all_ok = remembered and all(not t.get("failed") for t in turn_results)
    return {
        "remembered_name": remembered,
        "name_turn_response": name_reply[:100],
        "task_ok": all_ok,
    }


def _validate_tool_chain(result: Dict[str, Any], expected_line: str) -> Dict[str, Any]:
    text = result.get("final_response") or ""
    tool_calls = _count_tool_calls(result.get("messages") or [])
    has_expected = expected_line.lower() in text.lower()
    used_read_file = any(
        isinstance(m, dict)
        and m.get("role") == "tool"
        and m.get("name") == "read_file"
        for m in (result.get("messages") or [])
    )
    return {
        "tool_calls": tool_calls,
        "used_read_file": used_read_file,
        "found_expected_line": has_expected,
        "task_ok": tool_calls >= 1 and has_expected,
    }


def run_chat_scenario(agent, scenario: Dict[str, Any]) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    turn_results: List[Dict[str, Any]] = []
    total_api_calls = 0

    for idx, prompt in enumerate(scenario["turns"], start=1):
        t0 = time.time()
        result = agent.run_conversation(
            prompt,
            system_message=scenario.get("system_message"),
            conversation_history=messages or None,
        )
        elapsed = time.time() - t0
        messages = result.get("messages") or messages
        total_api_calls += int(result.get("api_calls") or 0)
        turn_results.append({
            "turn": idx,
            "prompt": prompt,
            "elapsed_s": round(elapsed, 2),
            "api_calls": result.get("api_calls"),
            "final_response": (result.get("final_response") or "")[:300],
            "failed": bool(result.get("failed")),
        })

    final = dict(turn_results[-1]) if turn_results else {}
    final.update({
        "messages": messages,
        "api_calls": total_api_calls,
        "final_response": turn_results[-1].get("final_response", "") if turn_results else "",
        "failed": any(t.get("failed") for t in turn_results),
        "turn_results": turn_results,
    })
    final["validation"] = _validate_chat(final)
    return final


def run_tool_chain_scenario(agent, scenario: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    fixture = work_dir / scenario["fixture_name"]
    fixture.write_text("\n".join(scenario["fixture_lines"]) + "\n", encoding="utf-8")
    expected_line = scenario["fixture_lines"][2]
    prompt = scenario["prompt_template"].format(path=str(fixture))

    t0 = time.time()
    result = agent.run_conversation(
        prompt,
        system_message=scenario.get("system_message"),
    )
    elapsed = time.time() - t0
    result["elapsed_s"] = round(elapsed, 2)
    result["validation"] = _validate_tool_chain(result, expected_line)
    return result


def run_one(
    scenario_id: str,
    *,
    kv_enabled: bool,
    gpu_gb: int,
    model: str,
    base_url: str,
    ollama_num_ctx: int,
) -> Dict[str, Any]:
    scenario = SCENARIOS[scenario_id]
    mode = "et" if kv_enabled else "hermes"
    label = "ET-Agent (KV on)" if kv_enabled else "Hermes baseline (KV off)"

    if scenario.get("requires_tools") and ollama_num_ctx < 64000:
        return {
            "scenario": scenario_id,
            "mode": mode,
            "label": label,
            "kv_enabled": kv_enabled,
            "skipped": True,
            "skip_reason": (
                f"Ollama num_ctx={ollama_num_ctx} < 64000; "
                "ET-Agent blocks tool calls below 64K runtime context. "
                "Re-run with --ollama-num-ctx 65536 (needs more VRAM)."
            ),
            "model": model,
            "ollama_num_ctx": ollama_num_ctx,
            "task_ok": False,
        }

    reset_module_state()
    home = setup_isolated_home(
        kv_enabled=kv_enabled,
        gpu_gb=gpu_gb,
        model=model,
        base_url=base_url,
        ollama_num_ctx=ollama_num_ctx,
    )
    os.environ["ET_AGENT_HOME"] = str(home)

    work_dir = Path(tempfile.mkdtemp(prefix=f"et_kv_{scenario_id}_"))
    started = time.time()
    error = None
    result: Dict[str, Any] = {}
    kv_metrics: Dict[str, Any] = {}

    try:
        from run_agent import AIAgent

        agent = AIAgent(
            model=model,
            base_url=base_url,
            provider="custom",
            api_key="ollama",
            enabled_toolsets=scenario.get("toolsets"),
            quiet_mode=True,
            save_trajectories=False,
            skip_context_files=True,
            skip_memory=True,
            platform="cli",
            max_iterations=scenario.get("max_iterations", 10),
            tool_delay=0.2,
        )

        if scenario_id == "chat":
            result = run_chat_scenario(agent, scenario)
        else:
            result = run_tool_chain_scenario(agent, scenario, work_dir)

        if kv_enabled:
            kv_metrics = _extract_kv_metrics(agent)

        try:
            agent.shutdown_memory_provider(result.get("messages"))
        except Exception:
            pass

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(home.parent, ignore_errors=True)

    elapsed = time.time() - started
    validation = result.get("validation") or {}
    record = {
        "scenario": scenario_id,
        "mode": mode,
        "label": label,
        "kv_enabled": kv_enabled,
        "model": model,
        "base_url": base_url,
        "gpu_gb": gpu_gb,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed, 2),
        "api_calls": result.get("api_calls", 0),
        "total_tokens": result.get("total_tokens", 0),
        "tool_calls": _count_tool_calls(result.get("messages") or []),
        "task_ok": bool(validation.get("task_ok")),
        "validation": validation,
        "final_response": (result.get("final_response") or "")[:500],
        "turn_results": result.get("turn_results"),
        "kv_metrics": kv_metrics,
        "error": error,
    }
    return record


def print_summary(records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("LIVE TEST SUMMARY")
    print("=" * 72)
    header = f"{'Scenario':<12} {'Mode':<8} {'OK':<4} {'Time(s)':<8} {'API':<5} {'Tools':<6} {'Prefix%':<8} {'Migr':<6}"
    print(header)
    print("-" * len(header))
    for r in records:
        if r.get("skipped"):
            print(f"{r.get('scenario',''):<12} {r.get('mode',''):<8} SKIP - {r.get('skip_reason','')[:50]}")
            continue
        kv = r.get("kv_metrics") or {}
        prefix = kv.get("prefix_hit_rate")
        prefix_s = f"{prefix * 100:.1f}" if isinstance(prefix, (int, float)) and kv.get("kv_active") else "-"
        migr = kv.get("migrations", "-") if kv.get("kv_active") else "-"
        ok = "yes" if r.get("task_ok") and not r.get("error") else "no"
        print(
            f"{r.get('scenario',''):<12} "
            f"{r.get('mode',''):<8} "
            f"{ok:<4} "
            f"{r.get('elapsed_s', 0):<8.1f} "
            f"{r.get('api_calls', 0):<5} "
            f"{r.get('tool_calls', 0):<6} "
            f"{prefix_s:<8} "
            f"{str(migr):<6}"
        )
        if r.get("error"):
            print(f"  error: {str(r['error']).splitlines()[0]}")


def compare_pairs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_scenario: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in records:
        by_scenario.setdefault(r["scenario"], {})[r["mode"]] = r

    comparisons = []
    for scenario_id, modes in by_scenario.items():
        if "hermes" not in modes or "et" not in modes:
            continue
        h, e = modes["hermes"], modes["et"]
        h_kv = h.get("kv_metrics") or {}
        e_kv = e.get("kv_metrics") or {}
        comparisons.append({
            "scenario": scenario_id,
            "hermes_elapsed_s": h.get("elapsed_s"),
            "et_elapsed_s": e.get("elapsed_s"),
            "elapsed_delta_s": round((e.get("elapsed_s") or 0) - (h.get("elapsed_s") or 0), 2),
            "hermes_api_calls": h.get("api_calls"),
            "et_api_calls": e.get("api_calls"),
            "hermes_task_ok": h.get("task_ok"),
            "et_task_ok": e.get("task_ok"),
            "prefix_hit_rate": e_kv.get("prefix_hit_rate"),
            "migrations": e_kv.get("migrations"),
            "tokens_saved": e_kv.get("tokens_saved"),
        })
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description="Live KV memory integration test (Ollama)")
    parser.add_argument("--scenario", choices=[*SCENARIOS.keys(), "all"], default="all")
    parser.add_argument("--mode", choices=["hermes", "et", "both"], default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--gpu-gb", type=int, default=DEFAULT_GPU_GB)
    parser.add_argument("--ollama-num-ctx", type=int, default=4096,
                        help="Ollama runtime context (default 4096 for 6GB; tool_chain needs >=64000)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check-only", action="store_true", help="Only verify Ollama is up")
    args = parser.parse_args()

    try:
        health = check_ollama(args.base_url, args.model)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        return 1

    print(f"[OK] Ollama reachable: {health['tags_url']}")
    print(f"     Models: {', '.join(health['models'][:5])}{'...' if len(health['models']) > 5 else ''}")
    print(f"     ollama_num_ctx={args.ollama_num_ctx}")
    if args.ollama_num_ctx < 64000:
        print("[WARN] tool_chain scenario will be skipped (needs >=64000 for tool use)")
    if not health["model_available"]:
        print(f"[WARN] Model {args.model!r} not in tag list — run: ollama pull {args.model}")
    if args.check_only:
        return 0

    scenario_ids = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    modes = ["hermes", "et"] if args.mode == "both" else [args.mode]

    records: List[Dict[str, Any]] = []
    for scenario_id in scenario_ids:
        for mode in modes:
            kv_enabled = mode == "et"
            print(f"\n>>> Running {scenario_id} / {mode} (kv={'on' if kv_enabled else 'off'}) ...")
            record = run_one(
                scenario_id,
                kv_enabled=kv_enabled,
                gpu_gb=args.gpu_gb,
                model=args.model,
                base_url=args.base_url,
                ollama_num_ctx=args.ollama_num_ctx,
            )
            records.append(record)
            if record.get("skipped"):
                print(f"[SKIP] {record.get('skip_reason')}")
                continue
            status = "[OK]" if record.get("task_ok") and not record.get("error") else "[FAIL]"
            print(
                f"{status} elapsed={record.get('elapsed_s')}s "
                f"api_calls={record.get('api_calls')} "
                f"task_ok={record.get('task_ok')}"
            )
            if record.get("error"):
                print(record["error"][:400])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "gpu_gb": args.gpu_gb,
        "ollama_num_ctx": args.ollama_num_ctx,
        "records": records,
        "comparisons": compare_pairs(records),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Results written to {args.out}")
    print_summary(records)

    if payload["comparisons"]:
        print("\n--- Hermes vs ET ---")
        for c in payload["comparisons"]:
            print(
                f"{c['scenario']}: "
                f"hermes {c['hermes_elapsed_s']}s -> et {c['et_elapsed_s']}s "
                f"(delta {c['elapsed_delta_s']:+.1f}s), "
                f"prefix_hit={c.get('prefix_hit_rate')}, "
                f"migrations={c.get('migrations')}"
            )

    failed = sum(
        1 for r in records
        if not r.get("skipped") and (r.get("error") or not r.get("task_ok"))
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
