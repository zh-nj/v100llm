# SM70 Fused MoE Benchmark — sm70-fused-moe

- Device: `Tesla_PG503-216_x8`
- Repo: `v100llm_(vLLM_0.19.1_for_V100_SM70)`
- Timestamp: `20260603_144332`
- Shapes benchmarked: 3

## Comparison (fused vs per-operator baseline)

| shape | layout | variant | latency_ms | tokens/s | GFLOP/s | speedup_vs_reference | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M=8 K=512 I=256 E=8 topk=2 gs=32 | masked | reference | 4.008 | 1,996 | 3.1 | 1.00x |  |
| M=8 K=512 I=256 E=8 topk=2 gs=32 | masked | L0 | 6.818 | 1,173 | 1.8 | 0.59x |  |
| M=8 K=512 I=256 E=8 topk=2 gs=32 | masked | L1 | 6.903 | 1,159 | 1.8 | 0.58x |  |
| M=8 K=512 I=256 E=8 topk=2 gs=32 | masked | L2 | - | - | - | - | fused op sm70_fused_moe_out not built (needs V100 extension) |
| M=64 K=512 I=256 E=8 topk=2 gs=32 | contiguous | reference | 4.256 | 15,038 | 23.7 | 1.00x |  |
| M=64 K=512 I=256 E=8 topk=2 gs=32 | contiguous | L0 | 3.457 | 18,512 | 29.1 | 1.23x |  |
| M=64 K=512 I=256 E=8 topk=2 gs=32 | contiguous | L1 | 3.463 | 18,481 | 29.1 | 1.23x |  |
| M=64 K=512 I=256 E=8 topk=2 gs=32 | contiguous | L2 | - | - | - | - | fused op sm70_fused_moe_out not built (needs V100 extension) |
| M=256 K=1024 I=512 E=8 topk=2 gs=32 | contiguous | reference | 4.786 | 53,485 | 336.5 | 1.00x |  |
| M=256 K=1024 I=512 E=8 topk=2 gs=32 | contiguous | L0 | 4.103 | 62,394 | 392.5 | 1.17x |  |
| M=256 K=1024 I=512 E=8 topk=2 gs=32 | contiguous | L1 | 4.110 | 62,288 | 391.9 | 1.16x |  |
| M=256 K=1024 I=512 E=8 topk=2 gs=32 | contiguous | L2 | - | - | - | - | fused op sm70_fused_moe_out not built (needs V100 extension) |
