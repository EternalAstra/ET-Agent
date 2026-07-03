# 代码修复记录

## 1. `scripts/benchmark_memory.py` L142-144

**问题**：`hash(sid)` 可达数十亿 → `_token_ids()` 申请超大列表 → `MemoryError`

**修改**：种子限制在小范围
```python
_seed = (turn * 137 + (hash(sid) & 0xFFFF)) % 10000
```

## 2. `agent/memory_hooks.py` L328

**问题**：`prefix_tokens` 单位是 token，误用 `// block_size` 切片 → hash 数与 block 数不一致 → `AssertionError`

**修改**：
```python
uncached_ids = token_ids[prefix_tokens:]   # 原为 prefix_tokens // block_size
```

## 3. `scripts/benchmark_memory.py` main() 场景循环

**问题**：多场景共用 mgr，`reset_stats()` 不释放 block → 后续 OOM

**修改**：每场景新建 `create_agent_memory_manager()` 实例

## 4. `memory_manager/memory_monitor.py` export_chart_data()

**问题**：`peak_gpu_blocks = gpu_blocks - free_blocks`，混用了分层 GPU 计数与分配器空闲池 → 负值

**修改**：`peak_gpu_blocks = max(gpu_blocks)`；新增 `peak_allocator_used = max(used_blocks)`

## 5. `agent/memory_hooks.py` LifecycleTiming / scan

**修改**：`tool_call_gpu_to_cpu_s=1.0`（原 30s），`_scan_interval_s=5.0`（原 30s）

## 6. Phase 5 — `agent/kv_memory_integration.py` 接入主 Agent

**新增**：KV 内存管理器与主循环集成（独立于 Honcho `agent.memory_manager`）

| 钩子 | 接入位置 |
|------|----------|
| `init_kv_memory_manager` | `agent/agent_init.py` |
| `on_session_start_kv` | `agent/conversation_loop.py`（system prompt 后） |
| `pre_llm_call_kv` | API 调用前 |
| `post_llm_call_kv` | 拿到 assistant 响应后 |
| `on_tool_results_kv` | `_execute_tool_calls` 之后 |
| `on_session_end_kv` | `run_agent.py` `shutdown_memory_provider` / `commit_memory_session` |

**配置**（`config.yaml`）：
```yaml
memory:
  kv_manager:
    enabled: true   # 默认开启
    gpu_gb: 6         # 本地 6GB 显存
```

**测试**：`tests/agent/test_kv_memory_integration.py`

## 7. `scripts/livetest_kv_memory.py` — Ollama 实测脚本

对比 Hermes（KV off）与 ET-Agent（KV on）：
```bash
python scripts/livetest_kv_memory.py --scenario chat --mode both
python scripts/livetest_kv_memory.py   # chat + tool_chain（后者需 num_ctx>=64K）
```
结果输出：`docs/team/livetest_kv_results.json`

