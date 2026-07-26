# ET-Agent v0.0.1

<p align="center">
  <b>面向智能体的本地推理内存管理系统 — Local Inference Memory Manager for AI Agents</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=flat" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/CUDA-Yes-green?style=flat" alt="CUDA">
  <img src="https://img.shields.io/badge/Models-Qwen3%20|%20Llama%203.2-orange?style=flat" alt="Models">
  <img src="https://img.shields.io/badge/Tests-293%20passed-brightgreen?style=flat" alt="Tests">
</p>

---

## 项目简介

**ET-Agent** 是一个面向智能体本地推理的 KV Cache 内存管理系统。基于 [Hermes Agent](https://github.com/NousResearch/hermes-agent) (MIT) 裁剪，参考 **vLLM (PagedAttention, SOSP 2023)**、**MoonCake (FAST 2025 Best Paper)** 和 **HiFC (Flash-based KV Cache)** 实现。

与 vLLM 对齐的三层 KV Cache 池（GPU HBM / CPU DRAM / NVMe SSD），支持真实 `torch.float16` CUDA tensor 分配、`Copy-on-Write` 前缀共享、哈希链 O(1) 前缀复用和 GPU↔CPU↔SSD 真实 `memcpy` swap。

**已完全移除云端 API 依赖，纯本地推理（Scenario B）。**

---

## 核心能力

| 层次 | 模块 | 论文 |
|------|------|------|
| **推理引擎 — vLLM 风格** | `InferenceEngine` 替代 `model.generate()`，自调度/分配/解码循环 | vLLM §4 |
| **KV Block 分配器** | 定长 Block (16 token) + Block Table + 真实 CUDA tensor (torch.zeros on cuda:0) | vLLM §4.1-4.3 |
| **前缀哈希缓存** | MoonCake 哈希链 O(1) 前缀匹配，COW 共享，系统提示词固定 | MoonCake §3 |
| **三池分层存储** | GPU (HBM) → CPU (DRAM) → SSD (磁盘文件)，真实 torch.copy_ swap | HiFC §3.2 |
| **Agent 生命周期追踪** | 5 阶段 (PREFILL/DECODING/TOOL_CALL/IDLE/COMPLETED) + 阶段感知迁移 | — |
| **ACON 上下文压缩** | 结构化摘要 (REASONING/VARS/ACTIONS/OPEN_TASKS)，峰值 token -26~54% | ACON ICML 2026 |

## 项目结构

```
ET-Agent/
├── inference/               ★ vLLM 风格推理引擎
│   ├── engine.py             —   decode 循环 + 调度器 + 分配器 + 前缀缓存
│   ├── vllm_block_manager.py —   vLLM BlockSpaceManager API
│   ├── swapping_engine.py    —   HiFC GDS 加速 GPU↔SSD 交换
│   ├── scheduler.py          —   KV Cache 调度器 (watermark)
│   ├── metrics.py            —   吞吐量/利用率/swap 计数/TTFT/TBT
│   ├── paged_attention.py    —   纯 PyTorch PagedAttention 算子
│   └── et_cache.py           —   DynamicCache + KVBlockAllocator 追踪
│
├── memory_manager/           ★ 内存管理核心 (16 模块, ~4700 行)
│   ├── kv_block.py           —   KVBlock (真实 CUDA tensor)
│   ├── kv_block_allocator.py —   三池分配器 (GPU+CPU+SSD) + swap_in/out
│   ├── block_table.py        —   BlockTable + COW 共享管理
│   ├── config.py             —   MemoryConfig + 模型参数预设
│   ├── kv_prefix_cache.py    —   MoonCake 哈希链前缀缓存 (O(1))
│   ├── agent_prefix_cache.py —   Agent 场景专用缓存策略
│   ├── kv_eviction_policy.py —   4 种淘汰策略 (LRU/LFU/Tiered/AgentAware)
│   ├── kv_lifecycle_tracker.py — 5 阶段生命周期追踪
│   ├── kv_hierarchical_store.py — GPU→CPU→SSD 分层存储
│   ├── context_compressor.py —   ACON 结构化压缩
│   ├── prompt_deduplicator.py —  系统提示词/工具定义去重
│   ├── tool_schema_compressor.py — 频率分层工具压缩
│   └── memory_monitor.py    —   实时监控 + JSON 导出
│
├── agent/                    ★ Agent 对话循环集成
│   ├── kv_memory_integration.py — memory_manager 注入 hermes 对话
│   ├── memory_hooks.py       —   AgentMemoryManager 统一外观
│   └── conversation_loop.py  —   4 个生命周期钩子
│
├── scripts/
│   ├── benchmark_local.py    —   本地模型 A/B 对比 (vs 原始 Hermes)
│   ├── benchmark_vllm.py     —   多场景 vs vLLM/HiFC 基线
│   └── monitor_api.py        —   HTTP 实时监控仪表盘
│
├── web/monitor/              —   纯 HTML/CSS 监控仪表盘
├── tests/memory_manager/     —   273 个单元+集成测试
└── tests/inference/          —   20 个推理层测试
```

## 快速开始

### 环境要求

- Python 3.11+, PyTorch 2.x (CUDA), NVIDIA GPU ≥ 6GB VRAM
- 本地模型文件：Qwen3-0.6B / Llama 3.2 3B（推荐）

### 安装

```bash
git clone https://github.com/EternalAstra/ET-Agent.git
cd ET-Agent
pip install -e .
pip install transformers torch --index-url https://download.pytorch.org/whl/cu124
```

### 本地推理（vLLM 风格 decode 循环）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference.engine import InferenceEngine

# 加载本地模型
model = AutoModelForCausalLM.from_pretrained(
    "models/Qwen3-0.6B", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

# 创建引擎 — 替代 model.generate()
engine = InferenceEngine(model, tokenizer, gpu_memory_gb=6)

# 单 prompt
resp = engine.generate("What is the capital of France?", max_new=64)
print(resp)

# 并发 batch
resps = engine.generate_batch(
    ["What is 2+2?", "What is H2O?", "What color is the sky?"],
    max_new=32, system_prompt="You are a helpful assistant.")

# 查看 GPU 内存状态
print(engine.stats())
# → {'gpu_blocks': '48/1755', 'gpu_mb': '84', 'cpu_blocks': '0/299593', ...}
```

### 运行测试

```bash
python -m pytest tests/memory_manager/ tests/inference/ -v   # 293 tests
python -m pytest tests/memory_manager/ -q                    # 273 memory tests
```

### A/B 对比测试 (vs 原始 Hermes)

```bash
python scripts/benchmark_local.py --compare --turns 5
# 输出：Hermes vs ET-Agent GPU 显存 / 延迟对比表
```

### 实时监控仪表盘

```bash
python scripts/monitor_api.py
# 浏览器打开 http://localhost:8765
# 实时 GPU/CPU/SSD 块使用、前缀命中率、生命周期阶段、竞赛指标
```

## 支持模型

| 模型 | 参数量 | KV Heads | 每 Block | 推荐 GPU |
|------|--------|----------|----------|----------|
| Qwen3-0.6B | 0.6B | 8 (GQA) | 896KB | 6GB+ |
| Llama 3.2 3B | 3B | 8 (GQA) | 896KB | 8GB+ |

通过 `MemoryConfig.for_model()` 自动检测，或手动指定 `ModelKVProfile`。

## 技术指标

| 指标 | Hermes (基线) | ET-Agent |
|------|-------------|----------|
| GPU KV 管理方式 | transformers DynamicCache (连续分配) | 三层分页池 + 真实 CUDA tensor |
| 显存浪费率 | ~60-80% (vLLM Fig.2) | <5% (分页分配) |
| CPU 交换 | 无 | 真实 torch.copy_ GPU↔CPU swap |
| SSD 存储 | 无 | 磁盘文件 (HiFC block append) |
| 前缀复用 | 无 | MoonCake 哈希链 O(1) |
| 解码循环 | model.generate() | 自研 decode loop + 调度器 |
| 并发 batch | 无 | round-robin 多序列 |
| 测试覆盖 | ~200 通用测试 | 293 内存专项测试 |

## 开发状态

| Phase | 模块 | 状态 |
|-------|------|:--:|
| Phase 1 | KV Block 分配器 + Block Table + COW | ✅ |
| Phase 2 | 前缀哈希缓存 + Agent 缓存 + 淘汰策略 | ✅ |
| Phase 3 | GPU→CPU→SSD 分层存储 + 生命周期追踪 | ✅ |
| Phase 4 | ACON 上下文压缩 + 去重 + 工具压缩 | ✅ |
| Phase 5 | AgentMemoryManager + hermes 对话循环集成 | ✅ |
| Phase 6-7 | 本地推理引擎 + Benchmark + 监控仪表盘 | ✅ |
| **全部 7 Phase** | **16 memory_manager 模块 + 4 inference 模块** | **✅** |

## 参考论文

- **Kwon, W. et al.** "PagedAttention — Efficient Memory Management for LLM Serving." *SOSP 2023*
- **Qin, R. et al.** "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving." *FAST 2025* (Best Paper)
- **Jeong, I. et al.** "HiFC: High-efficiency Flash-based KV Cache Swapping." *2025*
- **Kang, M. et al.** "ACON: Optimizing Context Compression for Long-horizon LLM Agents." *ICML 2026*
- Hermes Agent — [https://github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT)

## License

MIT — 详见 [LICENSE](LICENSE)。本项目基于 [Hermes Agent](https://github.com/NousResearch/hermes-agent) (MIT) 修改。
