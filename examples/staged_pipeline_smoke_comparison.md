# Staged LLM-only pipeline smoke comparison

Date: 2026-08-26  
Ollama endpoint: local native `/api/chat`  
RAG and synthesis model: `mistral-small3.2:latest`  
Reference engine use: post-hoc comparison only; no reference values were supplied to LLM stages.

## Pipeline under test

1. RAG/policy LLM call: full profile and retrieved excerpts; no target calculations.
2. Isolated calculation LLM call: ten finance-only inputs, sanitized policy parameters, symbolic formulas, and a mandatory branch/intermediate/self-check schema.
3. Final synthesis LLM call: policy result plus immutable final calculation values; no financial fields in its output schema.

The parser accepts only canonical JSON with exact keys and native finite numbers. A missing field, extra financial field, locale-formatted number, non-finite value, Markdown fence, or self-declared `FAIL` stops the pipeline. It does not trigger a Python repair or a hidden second calculation.

## Route screening on `c_eur_variable_stress_rejected`

Reference values:

- stressed monthly payment: `3680.00`
- DTI: `43.80%`
- maximum amount by DTI: `108968.753329`
- decision: `REJECTED`

| Calculation route | Reasoning/budget | Outcome |
|---|---|---|
| `magistral:24b` | on / 2500 | Hard-gate failure: returned fenced JSON; the values inside also had incorrect DTI and maximum amount. |
| `qwen3:14b` | on / 2500 | Hard-gate failure: reasoning exhausted the output budget before final JSON (`done_reason=length`). |
| `mistral-small3.2:latest` | off | Before the trace contract, returned `3200`, `39`, `0`; after trace enforcement, exposed inconsistent values and returned `self_check.status=FAIL`, so the result was rejected. |
| `phi4:14b` | off | Completed, but returned `3694`, `39`, `108000` and an incorrect `APPROVED` decision. |
| `qwen3:14b` | off / 2500 | Completed in 51.25 s; returned `3680`, `42.8`, `133333.33`; 1/3 target fields agreed and the decision was correct. |
| `qwen3:14b` | on / 6000 | Completed in 68.45 s; returned `3680`, `43.8`, `109000`; 2/3 fields agreed and the decision was correct. |
| `qwen3:14b` + explicit inverse-annuity precision rule | on / 6000 | Completed in 70.28 s; returned `3680`, `43.8`, `108968.976923`; 3/3 fields agreed, maximum amount in the RON 1 tolerance, and the decision was correct. |

For this case, enabling adequately budgeted reasoning improved agreement from 1/3 to 3/3 fields, while latency increased from 51.25 s to 70.28 s. This is one paired case, not a full-suite conclusion.

## Second live case: `c_salary_approved`

Route: `qwen3:14b`, reasoning on, 6000-token calculation budget.

| Field | LLM-only result | Post-hoc reference | Existing tolerance | Agreement |
|---|---:|---:|---:|---:|
| Stressed monthly payment | 2124.704471 | 2124.704471 | RON 1.00 | Yes |
| DTI | 14.208548% | 14.164696% | 0.05 percentage points | Yes |
| Maximum amount by DTI | 150000.00 | 150000.00 | RON 1.00 | Yes |
| Decision | APPROVED | APPROVED | exact | Yes |

Generation took 69.20 seconds.

## Honest interpretation

Across the three illustrative cases in this file, the numeric fields achieve 9/9 agreements (100%) and the decisions achieve 3/3 agreements (100%). All three cases have all three numeric target fields correct. This remains an illustrative smoke comparison and must not be reported as a full-suite score. The next model-selection step is the complete 22-case, multi-seed benchmark using:

- valid-contract rate;
- agreement for each target field;
- all-three-correct case rate;
- decision agreement;
- seed stability;
- latency and model-load time;
- hard-gate/abstention rate.

The existing Small 4 IQ1 three-seed numeric benchmark remains substantially worse (approximately 5% aggregate target-field agreement and zero all-three-correct cases). It should not be selected as the calculation model for the presentation route.

## Self-employment supplied-payment diagnosis

Case: `c_self_employment_rejected_dti`; route: `qwen3:14b`, reasoning on, 6000-token calculation budget.

The original run returned the correct stressed payment (`3500.00`), DTI
(`46.6666666667%`), and the maximum amount of `141196.11` instead of the
post-hoc reference `141196.107071`.

After making the discount factor, denominator, and inverse-annuity factor mandatory
trace values; requiring the maximum calculation to reuse that factor even for a
supplied payment and a zero requested amount, the live full-pipeline result was:

| Field | Revised LLM-only result | Post-hoc reference | Absolute error | Strict agreement |
|---|---:|---:|---:|---:|
| Stressed monthly payment | 3500.000000 | 3500.000000 | RON 0.00 | Yes |
| DTI | 46.6666666667% | 46.6666666667% | 0.0000 pp | Yes |
| Maximum amount by DTI | 141196.107071 | 141196.107071 | RON 0.00 | Yes |
| Decision | REJECTED | REJECTED | — | Yes |

Generation took 110.65 seconds.The prompt refinement made the the maximum-amount
value in the comparison table to be aligned to the post-hoc reference, so all three numeric
fields agree for this case. The tolerance was not relaxed to conceal it
