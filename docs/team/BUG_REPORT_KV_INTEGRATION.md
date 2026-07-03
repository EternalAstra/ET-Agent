# ET-Agent Phase 5 KV 主 Agent 集成 — Bug 审查报告

> 审查范围：`agent/kv_memory_integration.py` + `agent/conversation_loop.py` 集成调用点 + `agent/agent_init.py` + `run_agent.py` + `tests/agent/test_kv_memory_integration.py` + 相关 `agent/memory_hooks.py`
>
> 审查日期：2026-07-03
>
> 方法：10 角度审查（Altitude / Conventions / Simplification / Line-diff / Cross-file / Python-pitfall / Wrapper / Reuse / Efficiency / Removed-behavior），多角度已验证确认。

---

## 严重 — 正确性缺陷（4 项）


### BUG-1 — 后台扫描定时器永不停止

**文件**：`agent/memory_hooks.py:167-169` / `agent/kv_memory_integration.py:133-141`

**描述**：`AgentMemoryManager.__init__` 启动一个 `threading.Timer`，`_scan_tick`（L516-525）会无限重新调度自身。`AgentMemoryManager` 类中存在 `stop()` 方法（L527-531），取消了该定时器，但集成层的 `on_session_end_kv` 仅调用 `mgr.on_session_end(sid)`，从不调用 `mgr.stop()`。

**后果**：长时间运行的网关/CLI 进程每创建一个 agent 就积累一个活跃定时器，引用旧的管理器对象。会话销毁后 UI 线程无事可做，但后台定时器继续在释放/关闭的状态上调用 `scan_and_migrate()` 和 `evict_cold_blocks()`。

**建议修复**：在 `on_session_end_kv` 中调用 `mgr.stop()`，或在 `AgentMemoryManager.on_session_end` 内自动停止定时器。

---

### BUG-2 — 恢复的会话仅有 manager，没有 session_start

**文件**：`agent/conversation_loop.py:350-351` / `agent/agent_init.py:1630-1631`

**描述**：`init_kv_memory_manager` 在 agent 初始化时为每个根 agent 创建 manager（`agent_init.py:1630`），但 `on_session_start_kv` **仅在**对话循环的全新系统提示词构建分支中触发（L330-351）。对于恢复的网关/CLI 会话，manager 已创建，但以下内容从未执行：
- `lifecycle.register(session_id, …)` — 会话未注册
- `block_tables.create_table(session_id)` — 无块表
- `agent_cache.cache_system_prompt(…)` — 未缓存系统提示词

**后果**：后续 `pre_llm_call`/`post_llm_call` 对缺失的会话状态进行操作。prefix 命中率、统计数据和阶段转换均静默地变为空操作或产生垃圾数据。

**建议修复**：将会话注册与 manager 初始化绑定，或将会话注册移动至 `conversation_loop` 中共享的「新会话 vs 恢复会话」分支。

---

### BUG-3 — pre_llm_call_kv 接收原始 messages，而非 api_messages

**文件**：`agent/conversation_loop.py:863-864`

**描述**：L864 处，`pre_llm_call_kv(agent, messages)` 在 surrogate 清理（L860）、thinking 内容剥离、工具 JSON 修复、图片移除、缓存控制布局调整以及 provider 推理字段注入**之前**触发。LLM 实际接收 `api_messages`，而非 `messages`。

**后果**：KV prefix 缓存基于与 LLM 实际解析的消息不同的消息列表进行块分配 / 去重 / cache 命中判定。当 `api_messages` 与 `messages` 分叉时（常发生于 Ollama 或含富媒体内容的 provider），prefix 复用率会下降或不正确。

**建议修复**：将 `pre_llm_call_kv` 移至 `api_messages` 构造完成之后并传入 `api_messages`，或同时传入两者。

---

### BUG-4 — 集成层与核心层之间的伪 token 不一致

**文件**：`agent/kv_memory_integration.py:57` vs `agent/memory_hooks.py:232,314`

**描述**：`_pseudo_tokens()` 使用 `ord(c) % 50000` 生成 token ID，而 `AgentMemoryManager` 内部使用原始 `ord(c)`。对于码点 ≥ 50000 的文本（emoji、生僻 CJK 字符），集成层写入缓存的数字与核心层前缀查找所用的数字不同。

**后果**：包含 emoji 或某些多语言文本的系统提示词 / 消息在缓存插入和查找之间无法匹配，导致 cache 命中失败或产生不正确的最佳匹配前缀。

**建议修复**：为 `_pseudo_tokens()` 和 `memory_hooks` 内部的 placeholder token 化逻辑使用统一的共享工具函数。

---

## 重要 — 状态 / 生命周期缺陷（6 项）

### BUG-6 — post_llm_call_kv 在 LLM 轮次被持久化之前提交生命周期

**文件**：`agent/conversation_loop.py:3566-3574`

**描述**：`post_llm_call_kv` 在响应规范化之后**立即**触发（L3542——`normalized`），此时 assistant 轮次尚未追加至 `messages` 持久化列表，也早于响应验证。如果该响应后续失败、有不完整的 scratchpad 因此被重试、或被转为受控停止，KV 管理器已经记录了已完成的解码 / 工具调用轮次。

**后果**：重试导致 `_turn_count` 被重复计数，并可能将生命周期转换为 TOOL_CALL，而实际该轮次在对话历史中并未被接受。

**建议修复**：将 `post_llm_call_kv` 推迟至持久化步骤之后、轮次被接受之后再触发。

---

### BUG-7 — pre_llm_call_kv 丢弃去重返回值

**文件**：`agent/kv_memory_integration.py:87`

**描述**：`AgentMemoryManager.pre_llm_call()` 返回一个包含 `dedup_saved_tokens`、`dedup_dropped` 和 `allocation` 的字典。`pre_llm_call_kv` 丢弃了该返回值。

**后果**：KV 管理器计算了去重后的消息，但对话循环仍将完整未去重的消息发送给 LLM。前缀命中和去重工作在有效负载层面不起作用，即使统计数据在 KV 管理器内部看起来不错。

**建议修复**：将去重后的消息传回对话循环并代入 `api_messages` 管道。

---

### BUG-8 — on_tool_results_kv 中 try/except 覆盖整个循环，第一个失败即跳过其余

**文件**：`agent/kv_memory_integration.py:126-130`

**描述**：对工具名称的 `for` 循环完全被 `try/except` 包裹。如果 `mgr.on_tool_result()` 对任何一个工具抛出异常，`except` 子句会捕获并退出循环，剩余工具将无法执行其生命周期提升 / 预取。

**后果**：助手调用 3 个工具（`read_file`、`grep`、`write_file`）。`read_file` 抛出异常 → 循环退出 → `grep` 和 `write_file` 的块仍处于降级状态，导致下一次 LLM 轮次出现缓存冷启动延迟。

**建议修复**：将 `try/except` 移入循环内部，使每个工具独立失败。

---

### BUG-9 — 模型名称变体无法匹配 MemoryConfig profile

**文件**：`agent/kv_memory_integration.py:35-38`

**描述**：`init_kv_memory_manager` 剥离了 provider 前缀（`ollama/qwen2.5:3b-instruct` → `qwen2.5:3b-instruct`），但将包含 `:3b-instruct` 标签的裸名称传给 `MemoryConfig.for_model()`，后者可能没有针对该精确变体的条目。

**后果**：无法匹配任何 profile → 回退至通用默认值，`num_layers`/`kv_heads`/每 token 块数等值可能存在错误，导致块尺寸计算有误。

**建议修复**：剥离 `:` 标签后缀，或在 `MemoryConfig.for_model` 中实现子串匹配。

---

### BUG-10 — 模型切换时 on_session_start 非幂等

**文件**：`agent/conversation_loop.py:330-351` + `agent_runtime_helpers.py:~1656`

**描述**：通过 `/model` 切换模型会将 `_cached_system_prompt` 设为 `None`（`agent_runtime_helpers.py L1656`），导致下一轮对话触发系统提示词重建和第二次 `on_session_start_kv` 调用，会话 ID **相同**。

**后果**：发生重复的 `lifecycle.register()`、`block_tables.create_table()` 和块分配，旧块泄漏且未被释放。

**建议修复**：在 `on_session_start_kv` 中增加幂等性保护，或在非首次注册时提前返回。

---

### BUG-11 — 伪 token 在 8000 字符处截断，缓存查找使用完整文本

**文件**：`agent/kv_memory_integration.py:55-58`

**描述**：`on_session_start_kv` 通过 `_pseudo_tokens(sp)` 调用缓存系统提示词，该函数将文本截断为 8000 字符（`text[:limit]`）。而 `pre_llm_call` 对完整未截断文本运行前缀匹配。当系统提示词长度超过 8000 字符时，超出 8000 字符的部分在 prefix 缓存中不可达。

**后果**：对于包含大量技能的 agent（指令可能长达 8000+ 字符），首 8000 字符之后 KV 块将永远不会命中缓存，导致分配增加且 prefix 复用率为零。

**建议修复**：在 session_start 和 pre_llm_call 之间统一截断策略，或完全移除限制。

---

## 中等 — 设计与架构（5 项）

### 12 — 工具 schema 块被永久 pin 于全局名称下
`agent/memory_hooks.py:234-240` — `f"tool-{name}"` 未使用会话作用域。`on_session_end` 按 session_id 释放块，但工具块不在此范围内。重复会话或工具变更会分配新的 pinned 块，而旧块永久保留在 GPU 上。

### 13 — 系统提示词在硬编码的 "default" 键下缓存
`agent/memory_hooks.py:218-219` — 不同系统提示词、模型、provider 的多个会话共享一个缓存条目。prefix 匹配可能对不兼容的 KV 块错误报告复用。

### 14 — 网关路径每轮对话创建新的 AgentMemoryManager
`agent/agent_init.py:1630` — 如果网关每轮对话都实例化一个新的 `AIAgent`，则每轮都会失去跨轮 KV prefix 缓存（新的 manager = 新的空 prefix 缓存）。

### 15 — 按工具循环触发完全相同的完整会话预取操作
`agent/kv_memory_integration.py:127` — `AgentMemoryManager.on_tool_result` 忽略其 `tool_name` 参数。对 N 个工具，相同的完整会话 PREFILL 转换 × 预取执行了 N 次。

### 16 — _tool_definitions() 静默过滤非字典类型的工具
`agent/kv_memory_integration.py:60-63` — 基于 SimpleNamespace 的工具被 `isinstance(t, dict)` 丢弃，不在 KV 缓存中出现。

---

## 质量 — 维护性与测试（4 项）

### 17 — 6 个独立的动态 import 调用点
`agent/conversation_loop.py:350,863,3567,3997` + `run_agent.py:3040,3070` — 分散的 `from agent.kv_memory_integration import …` 调用点可以通过 `hermes_cli.plugins.invoke_hook` 机制转化为一次注册。

### 18 — 5 个钩子中重复的 guard/session-id/try-log 样板代码
`agent/kv_memory_integration.py:66-141` — 每个生命周期钩子都复制了 `mgr = get_kv_memory_manager(agent)` / `sid = getattr(…)` / `if not mgr or not sid: return` / `try…except`。可使用装饰器或共享 helper 消除重复。

### 19 — on_session_end_kv 在 shutdown_memory_provider 和 commit_memory_session 之间重复
`run_agent.py:3039,3069` — 两个方法复制了一模一样的 KV 清理代码，但前者表示进程清理，后者表示会话轮换。KV 管理器可能需要不同的行为（关机时归档 vs 轮换时保持热状态）。

### 20 — 测试对源文件内容做 assert "string" in source
`tests/agent/test_kv_memory_integration.py:132-154` — `TestConversationLoopImports` 直接 `open().read()` 源文件并检查子字符串。重命名会破坏测试；钩子未正确接入的 bug 仍能通过。应使用 mock 或行为断言替代。

---

## 修复状态

| Bug | 状态 |
|-----|------|
| BUG-1-BUG-20 | 待处理 |
