# ET-Agent ☤

<p align="center">
  <b>面向智能体的内存管理系统 — Memory Management System for AI Agents</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/Version-0.0.1-blue?style=flat" alt="Version: 0.0.1"></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-3.11+-yellow?style=flat" alt="Python 3.11+"></a>
</p>

---

## 📖 项目简介

**ET-Agent** 是一个面向智能体推理过程的内存管理系统，基于开源项目 [Hermes Agent](https://github.com/NousResearch/hermes-agent) (v0.17.0) 改进实现。针对大语言模型智能体在长生命周期推理中的显存管理问题，参考 **vLLM (PagedAttention, SOSP 2023)** 和 **MoonCake (KVCache Disaggregated Architecture, FAST 2025 Best Paper)** 的核心技术，实现对 KV Cache 的生命周期管理、前缀缓存复用、分层存储以及上下文压缩等功能。

### 核心研究方向

| 方向 | 目标 | 参考技术 |
|------|------|----------|
| **KV Cache 生命周期管理** | 设计面向长生命周期推理的缓存管理策略，实现 KV 复用、淘汰与分层存储 | vLLM PagedAttention + MoonCake KVCache Pool |
| **Prompt 与上下文压缩** | 对 system prompt、工具描述等内容进行去重与精简，消除冗余信息 | MoonCake prefix hashing + 自研压缩策略 |

---

## 🏗️ 项目架构

```
ET-Agent/
├── agent/              # 核心Agent引擎
│   ├── conversation_loop.py   # 对话循环 (16级分层错误恢复 + Prompt缓存管理)
│   ├── system_prompt.py       # 系统提示词构建 (三层组装架构)
│   ├── tool_executor.py       # 工具调度 (并发/串行，ThreadPoolExecutor)
│   └── transports/            # 多Provider传输层 (5种API模式)
├── memory_manager/     # [开发中] 内存管理系统
│   ├── kv_block_allocator.py  # KV Block分配器 (PagedAttention)
│   ├── kv_prefix_cache.py     # 前缀哈希缓存 (MoonCake)
│   ├── kv_hierarchical_store.py # 分层存储 GPU→CPU→SSD
│   ├── kv_lifecycle_tracker.py  # Agent生命周期追踪
│   └── context_compressor.py  # 上下文压缩器
├── tools/              # 工具系统 (80+工具)
│   ├── registry.py            # 工具注册与分发
│   └── mcp_tool.py            # MCP客户端 (4717行)
├── plugins/            # 插件系统
│   └── model-providers/       # 14+ LLM Provider (DeepSeek/Qwen/OpenAI/Anthropic...)
├── skills/             # 技能库 (452个文件)
├── hermes_cli/         # CLI入口、配置、模型管理
├── hermes_state.py     # SQLite+FTS5会话存储
├── run_agent.py        # AIAgent核心类
├── batch_runner.py     # 批量推理运行器
├── trajectory_compressor.py  # 轨迹压缩 (训练数据生成)
└── tests/              # 测试覆盖 (200+测试文件)
```

### 架构层次

```
┌──────────────────────────────────────────────┐
│            表现层 (CLI)                        │
├──────────────────────────────────────────────┤
│            调度层                              │
│  AIAgent → 对话循环 → System Prompt → 工具调度 │
│  [将注入: memory_manager/ 内存管理器]          │
├──────────────────────────────────────────────┤
│            能力层                              │
│  tools/ + plugins/ + skills/ + providers/     │
├──────────────────────────────────────────────┤
│            传输层                              │
│  chat_completions / anthropic / codex / bedrock│
├──────────────────────────────────────────────┤
│            存储层                              │
│  SQLite+FTS5 / ~/.et-agent/ (配置/会话/技能)   │
└──────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Windows 11 / Linux / macOS

### 安装

```bash
# 克隆项目
git clone https://github.com/shiyu19/ET-Agent.git
cd ET-Agent

# 安装依赖
pip install -e .

# 配置 DeepSeek API Key
mkdir -p ~/.et-agent   # Windows: %LOCALAPPDATA%\et-agent
echo 'DEEPSEEK_API_KEY=sk-your-api-key' > ~/.et-agent/.env

# 创建配置文件
cat > ~/.et-agent/config.yaml << EOF
model:
  default: "deepseek-v4-pro"
  provider: "deepseek"
toolsets:
  - hermes-cli
EOF
```

### 运行测试

```bash
# 运行 DeepSeek 连通性测试
python test_deepseek.py

# 交互式对话
python -m hermes_cli.main
```

### 支持的 Provider

项目原生支持 14+ LLM Provider，通过 `plugins/model-providers/` 插件系统即插即用：

`deepseek` · `openai` · `anthropic` · `qwen` · `moonshot` · `z-ai` (GLM) · `minimax` · `openrouter` (200+模型) · `ollama` · `nvidia` · `bedrock` · `xai` (Grok) · `google` · `custom`

---

## 🔬 技术亮点

### 1. 16级分层错误恢复

对话循环内置了从 Unicode 清洗到 Provider 后备切换的完整恢复体系，确保 Agent 推理的高可用性。

### 2. Prompt 缓存策略

系统提示词在会话生命周期内仅构建一次，保证 Anthropic `cache_control` 和 OpenAI/Kimi/DeepSeek 服务端前缀缓存在所有轮次中保持热度。

### 3. 工具渐进式披露 (Tool Search)

当工具数量超过上下文窗口 10% 时，自动将 MCP 和非核心插件工具替换为 3 个桥接工具 (`tool_search` / `tool_describe` / `tool_call`)，避免上下文膨胀。

### 4. MCP 完整支持

- **客户端**: stdio/HTTP/SSE 三种传输，OAuth 2.1 PKCE，Sampling/Elicitation
- **服务器**: 将 ET-Agent 自身暴露为 MCP 服务器，供 Claude Code/Cursor/Codex 调用

### 5. 轨迹生成管线

`batch_runner.py` → `trajectory_compressor.py` → `sample_and_compress.py`，支持从原始对话到训练数据的完整流水线。

---

## 📊 当前状态

### 已完成 (Phase 1.1)

- [x] Hermes Agent v0.17.0 项目裁剪（5401 → 3000 文件，123MB → 60MB）
- [x] 品牌重命名：Hermes Agent → ET-Agent v0.0.1
- [x] 删除非核心模块（gateway/25+平台、TUI、Web、Desktop、cron调度等）
- [x] DeepSeek V4 API 全链路验证通过（多轮对话 + 工具调用 + Prompt缓存）

### 开发中

- [ ] Phase 1.2-1.3: KV Block 分配器与 Block Table（vLLM PagedAttention）
- [ ] Phase 2: 前缀哈希缓存与 KV 复用（MoonCake）
- [ ] Phase 3: GPU→CPU→SSD 分层存储
- [ ] Phase 4: Prompt 与上下文压缩系统
- [ ] Phase 5: Agent 内存调度器集成
- [ ] Phase 6-7: 监控评测 + Benchmark + 文档

完整实施计划详见：[实施计划_Agent内存管理系统.md](./实施计划_Agent内存管理系统.md)

---

## 📚 参考论文

1. **Kwon, W., Li, Z., Zhuang, S., et al.** "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*. (vLLM)
2. **Qin, R., Li, Z., He, W., et al.** "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving." *FAST 2025*. (Best Paper)
3. Hermes Agent — [https://github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)

---

## 📄 License

MIT License — 详见 [LICENSE](LICENSE)。

本项目基于 [Hermes Agent](https://github.com/NousResearch/hermes-agent) (MIT License) 修改。
