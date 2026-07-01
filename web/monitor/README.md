# ET-Agent Memory Monitor Dashboard

纯静态 HTML/CSS/JS 实现的监控仪表盘，零依赖，一份 benchmark_data.js 驱动全部图表。

## 快速打开

### 方式 1：直接双击（最简单）

```
文件管理器 → 打开 web/monitor/index.html
```

或者命令行：

```bash
# Windows
start web/monitor/index.html

# macOS
open web/monitor/index.html

# Linux
xdg-open web/monitor/index.html
```

### 方式 2：HTTP 服务器（避免 file:// CORS）

```bash
cd ET-Agent
python -m http.server 8080 -d web/monitor
# 浏览器打开 http://localhost:8080
```

## 更新数据

```bash
# 方式 1：运行完整 Benchmark（耗时较长）
python scripts/benchmark_memory.py --scenario chat

# 方式 2：生成演示数据（秒级完成）
python -c "
from memory_manager.memory_monitor import MemoryMonitor
from agent.memory_hooks import create_agent_memory_manager
import json, time

mgr = create_agent_memory_manager()
monitor = MemoryMonitor(mgr)
monitor.start(interval_s=2)

# ... 你的 agent 运行代码 ...

monitor.stop()
monitor.export_json('web/monitor/benchmark_results.json')
# 或者直接写 JS:
with open('web/monitor/benchmark_data.js','w') as f:
    f.write('window.BENCHMARK_DATA = ')
    json.dump(monitor.export_chart_data(), f)
    f.write(';')
"
```

## 面板说明

| Tab | 内容 |
|-----|------|
| 📊 Overview | GPU块使用趋势 + 前缀缓存命中率 |
| 📦 Storage Tiers | 三层存储分布 + 迁移活动 |
| 🎯 Prefix Cache | 缓存条目/命中率/热块分析 |
| 🔄 Lifecycle | Agent 阶段分布 + 活跃/等待会话 |
| 📐 Compression | Token节省量 + 压缩来源分解 |
| ⭐ vs Hermes | ET-Agent 相比 Hermes 新增功能对比 |
