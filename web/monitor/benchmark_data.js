window.BENCHMARK_DATA = {
  "metadata": {
    "model": "qwen2.5-7b",
    "gpu_gb": 1
  },
  "scenarios": {
    "demo": {
      "name": "Demo",
      "snapshots": [
        {
          "timestamp": 1782876278.3308694,
          "elapsed_s": 0.0003619194030761719,
          "blocks": {
            "total": 1170,
            "free": 1170,
            "used": 0,
            "gpu_blocks": 0,
            "cpu_blocks": 0,
            "ssd_blocks": 0,
            "shared": 0,
            "pinned": 0,
            "active_requests": 0,
            "usage_ratio": 0.0
          },
          "prefix": {
            "total_entries": 0,
            "pinned_entries": 0,
            "hot_entries": 0,
            "hit_rate": 0.0,
            "total_lookups": 0,
            "hits": 0,
            "misses": 0,
            "blocks_reused": 0,
            "block_reuse_rate": 0.0
          },
          "tiers": {
            "gpu_bytes": 0,
            "cpu_bytes": 0,
            "ssd_bytes": 0,
            "gpu_ratio": 0.0,
            "cpu_ratio": 0.0,
            "ssd_ratio": 0.0,
            "total_migrations": 0,
            "total_prefetches": 0
          },
          "lifecycle": {
            "total_requests": 0,
            "active_requests": 0,
            "waiting_requests": 0,
            "protected_sessions": 0,
            "prefill_count": 0,
            "decoding_count": 0,
            "tool_call_count": 0,
            "idle_count": 0,
            "completed_count": 0,
            "total_transitions": 0,
            "total_demotions": 0,
            "total_promotions": 0,
            "total_evictions": 0
          },
          "compression": {
            "history_compressions": 0,
            "observation_compressions": 0,
            "total_tokens_saved": 0,
            "history_threshold": 4096,
            "has_optimized_guideline": false,
            "total_messages_dropped": 0,
            "dedup_tokens_saved": 0,
            "tools_tracked": 0,
            "tool_schema_tokens_saved": 0
          }
        },
        {
          "timestamp": 1782876278.8326926,
          "elapsed_s": 0.5021705627441406,
          "blocks": {
            "total": 1170,
            "free": 1164,
            "used": 6,
            "gpu_blocks": 132,
            "cpu_blocks": 0,
            "ssd_blocks": 130,
            "shared": 0,
            "pinned": 6,
            "active_requests": 3,
            "usage_ratio": 0.0051
          },
          "prefix": {
            "total_entries": 196,
            "pinned_entries": 66,
            "hot_entries": 0,
            "hit_rate": 0.5,
            "total_lookups": 20,
            "hits": 10,
            "misses": 10,
            "blocks_reused": 130,
            "block_reuse_rate": 13.0
          },
          "tiers": {
            "gpu_bytes": 121110528,
            "cpu_bytes": 0,
            "ssd_bytes": 119275520,
            "gpu_ratio": 0.1128,
            "cpu_ratio": 0.0,
            "ssd_ratio": 0.0001,
            "total_migrations": 130,
            "total_prefetches": 0
          },
          "lifecycle": {
            "total_requests": 0,
            "active_requests": 0,
            "waiting_requests": 0,
            "protected_sessions": 0,
            "prefill_count": 0,
            "decoding_count": 0,
            "tool_call_count": 0,
            "idle_count": 0,
            "completed_count": 0,
            "total_transitions": 18,
            "total_demotions": 0,
            "total_promotions": 8,
            "total_evictions": 0
          },
          "compression": {
            "history_compressions": 0,
            "observation_compressions": 0,
            "total_tokens_saved": 0,
            "history_threshold": 4096,
            "has_optimized_guideline": false,
            "total_messages_dropped": 0,
            "dedup_tokens_saved": 0,
            "tools_tracked": 1,
            "tool_schema_tokens_saved": 0
          }
        }
      ],
      "charts": {
        "timestamps": [
          0.0003619194030761719,
          0.5021705627441406
        ],
        "series": {
          "gpu_blocks_used": [
            -1170,
            -1032
          ],
          "cpu_blocks": [
            0,
            0
          ],
          "ssd_blocks": [
            0,
            130
          ],
          "shared_blocks": [
            0,
            0
          ],
          "prefix_hit_rate": [
            0.0,
            0.5
          ],
          "gpu_usage_ratio": [
            0.0,
            0.1128
          ],
          "cpu_usage_ratio": [
            0.0,
            0.0
          ],
          "active_requests": [
            0,
            0
          ],
          "waiting_requests": [
            0,
            0
          ],
          "total_migrations": [
            0,
            130
          ],
          "total_prefetches": [
            0,
            0
          ],
          "tokens_saved": [
            0,
            0
          ]
        },
        "summary": {
          "peak_gpu_blocks": -1032,
          "avg_prefix_hit_rate": 0.25,
          "total_migrations": 130,
          "total_tokens_saved": 0,
          "max_active_sessions": 0,
          "max_waiting_sessions": 0
        }
      }
    }
  }
};