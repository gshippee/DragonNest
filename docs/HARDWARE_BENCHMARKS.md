# Hardware Benchmark Record

This file intentionally separates physical, AI Hub, and pending results.

| Deployment | Evidence | Load | Prefill | Decode | Memory | Thermal |
|---|---|---:|---:|---:|---:|---|
| Qwen3-0.6B Part A, SM-S938U1 | physical direct QNN | 206.7 ms | 37.0 ms NPU execution for fixed 32-token graph | n/a | 8.4 MiB PSS / 12.4 MiB RSS while mmap-held | not measured |
| Qwen3-0.6B Part B, SM-S938U1 | physical direct QNN | 226.7 ms | 95.0 ms NPU execution for fixed 32-token graph | n/a | not isolated | not measured |
| Qwen3-1.7B four contexts, SM-S938U1 | physical direct QNN | included in process wall | 2.59 s total partition wall for 40 valid prompt tokens (not a pure NPU throughput measurement) | 0.56-0.58 tok/s process/context-cycling | 1,674,248,192 context bytes total; 622,391,296 largest | not measured |
| Small steering stage, 8 Elite QRD | Qualcomm AI Hub | n/a | n/a | n/a | 123,248,640 byte estimated peak | lab unavailable |
| Small finish stage, X Elite CRD | Qualcomm AI Hub | n/a | n/a | n/a | 14,467,072 byte estimated peak | lab unavailable |
| Qwen3-4B Genie, physical X Elite | pending | pending | pending | pending | pending | pending |
| Qwen3-0.6B base/baked, DragonNest S25 APK | pending | pending | pending | pending | pending | pending |

AI Hub estimated inference times (41 us and 131 us for the two small tensor
stages) describe those tiny graphs only and are not LLM throughput claims.
