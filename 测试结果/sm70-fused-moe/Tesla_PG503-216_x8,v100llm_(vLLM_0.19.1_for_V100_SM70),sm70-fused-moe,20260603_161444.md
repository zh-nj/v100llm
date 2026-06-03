# SM70 Fused MoE Benchmark — sm70-fused-moe

- Device: `Tesla_PG503-216_x8`
- Repo: `v100llm_(vLLM_0.19.1_for_V100_SM70)`
- Timestamp: `20260603_161444`
- Shapes benchmarked: 3

## Comparison (fused vs per-operator baseline)

| shape | layout | variant | latency_ms | tokens/s | GFLOP/s | speedup_vs_reference | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M=8 K=512 I=256 E=8 topk=2 gs=32 | masked | reference | 4.920 | 1,626 | 2.6 | 1.00x |  |
| M=8 K=512 I=256 E=8 topk=2 gs=32 | masked | L0 | 8.615 | 929 | 1.5 | 0.57x |  |
| M=8 K=512 I=256 E=8 topk=2 gs=32 | masked | L1 | 8.650 | 925 | 1.5 | 0.57x |  |
| M=8 K=512 I=256 E=8 topk=2 gs=32 | masked | L2 | 1.493 | 5,359 | 8.4 | 3.30x |  |
| M=8 K=512 I=256 E=8 topk=2 gs=32 | masked | L3 | 1.469 | 5,446 | 8.6 | 3.35x |  |
| M=64 K=512 I=256 E=8 topk=2 gs=32 | contiguous | reference | 5.306 | 12,063 | 19.0 | 1.00x |  |
| M=64 K=512 I=256 E=8 topk=2 gs=32 | contiguous | L0 | 4.430 | 14,448 | 22.7 | 1.20x |  |
| M=64 K=512 I=256 E=8 topk=2 gs=32 | contiguous | L1 | 4.413 | 14,502 | 22.8 | 1.20x |  |
| M=64 K=512 I=256 E=8 topk=2 gs=32 | contiguous | L2 | 1.477 | 43,334 | 68.2 | 3.59x |  |
| M=64 K=512 I=256 E=8 topk=2 gs=32 | contiguous | L3 | 1.476 | 43,373 | 68.2 | 3.60x |  |
| M=256 K=1024 I=512 E=8 topk=2 gs=32 | contiguous | reference | 5.693 | 44,970 | 282.9 | 1.00x |  |
| M=256 K=1024 I=512 E=8 topk=2 gs=32 | contiguous | L0 | 4.811 | 53,206 | 334.7 | 1.18x |  |
| M=256 K=1024 I=512 E=8 topk=2 gs=32 | contiguous | L1 | 4.802 | 53,311 | 335.4 | 1.19x |  |
| M=256 K=1024 I=512 E=8 topk=2 gs=32 | contiguous | L2 | 3.652 | 70,107 | 441.1 | 1.56x |  |
| M=256 K=1024 I=512 E=8 topk=2 gs=32 | contiguous | L3 | 3.661 | 69,928 | 439.9 | 1.56x |  |
