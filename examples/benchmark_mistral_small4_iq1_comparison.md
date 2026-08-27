# Mistral Small 4 IQ1 comparison: reasoning none vs high

Date: 2026-08-26

## Environment

- Weights: `hf.co/unsloth/Mistral-Small-4-119B-2603-GGUF:UD-IQ1_M`
- Ollama digest: `a3d08b38dc75`
- Architecture: Mistral 4, 119B total parameters
- Quantization: `IQ1_M`, approximately 33 GB including the vision projector
- Runtime: Ollama 0.32.15
- Hardware: NVIDIA RTX 4080 16 GB VRAM, 32 GB system RAM
- Runtime allocation: approximately 41% GPU / 59% CPU

Ollama does not advertise the imported model as thinking-capable and rejects the
native `think: "high"` request with HTTP 400. To exercise Mistral's documented
setting, two aliases were created from identical weights. Their templates differ
only in `[MODEL_SETTINGS]{"reasoning_effort": "none|high"}`.

## Controlled numeric benchmark

Each of the 22 evaluation profiles was sent once per seed and mode. The prompt
contained the engine formulas and client inputs, but never the expected outputs.
The model returned only stressed payment, DTI, and maximum amount as structured
JSON. Seeds 42, 43, and 44 were used, producing 66 responses and 198 numeric
comparisons per mode.

Shared settings: context 4,096 tokens, output limit 1,024 tokens, temperature 0.7,
and project tolerances of +/-1 RON for stressed payment and maximum amount and
+/-0.05 percentage points for DTI.

| Metric | Reasoning none | Reasoning high |
|---|---:|---:|
| Valid JSON responses | 66/66 | 66/66 |
| Stressed-payment agreement | 5/66 (7.58%) | 4/66 (6.06%) |
| DTI agreement | 1/66 (1.52%) | 1/66 (1.52%) |
| Maximum-amount agreement | 5/66 (7.58%) | 5/66 (7.58%) |
| Combined agreement | 11/198 (5.56%) | 10/198 (5.05%) |
| Cases with all three correct | 0 | 0 |
| Separate thinking traces | 0/66 | 0/66 |
| Warm mean latency per case | 3.66 s | 3.70 s |
| Median stressed-payment absolute error | 806.68 RON | 806.68 RON |
| Median DTI absolute error | 12.65 pp | 12.65 pp |
| Median maximum-amount absolute error | 99,460.56 RON | 101,032.03 RON |

Combined correct fields by seed were `4, 4, 3` without reasoning and `3, 4, 3`
with high reasoning. The difference is one field across 198 comparisons and is not
evidence of an improvement.

## End-to-end project smoke

One complete RAG/policy pipeline case was run in each mode with the same settings.
Both modes got two of the three target numeric fields correct and the decision
correct. Reasoning none took 188.54 seconds and reasoning high took 160.10 seconds,
but this single ordered run includes model loading and multiple review calls, so
the latency difference is not a valid speed conclusion. At roughly 2.5-3 minutes
per profile, a matched 44-run full-pipeline comparison would take about two hours.

## Conclusion

On this machine and quantization, template-level high reasoning does not improve
stressed payment, DTI, or maximum amount. The ultra-low-bit quantization is too
aggressive for reliable financial arithmetic, and the current Ollama build does
not expose Small 4's native reasoning trace. These results do not measure the
official Mistral API or official NVFP4 checkpoint. The project should keep Python
as the authoritative calculator and use the LLM for policy interpretation and
explanation.

Primary result files:

- `benchmark_mistral_small4_iq1_numeric_none.json`
- `benchmark_mistral_small4_iq1_numeric_none_seed43.json`
- `benchmark_mistral_small4_iq1_numeric_none_seed44.json`
- `benchmark_mistral_small4_iq1_numeric_high.json`
- `benchmark_mistral_small4_iq1_numeric_high_seed43.json`
- `benchmark_mistral_small4_iq1_numeric_high_seed44.json`
- `benchmark_mistral_small4_iq1_pipeline_none_smoke.json`
- `benchmark_mistral_small4_iq1_pipeline_high_smoke.json`
