# Full staged-pipeline client benchmark

Status: `complete`  
Updated: `2026-08-27T06:53:34.069422+00:00`  
Cases completed: `22/22`  
Route: `mistral-small3.2:latest` → `qwen3:14b` → `mistral-small3.2:latest`  
Calculation reasoning: `on`; seed: `42`  
Maximum recommended amount: **included**, with RON 1.00 tolerance.

## Aggregate results

- Average eight-metric case score: 93.75%
- Decision agreement vs expected: 90.91%
- Stressed-payment agreement: 90.91%
- DTI agreement: 90.91%
- Maximum-amount agreement: 90.91%
- Combined numeric agreement: 90.91%
- Decision + numeric overall agreement: 90.91%
- All-three-correct case rate: 86.36%
- Pipeline success: `22/22`

Maximum-amount agreement by reference-value class:

| Class | Cases | Correct | Agreement | Mean absolute error |
|---|---:|---:|---:|---:|
| product_cap | 10 | 10 | 100.00% | 0.00 RON |
| zero_capacity | 2 | 2 | 100.00% | 0.00 RON |
| interior_annuity | 10 | 8 | 80.00% | 1,618.77 RON |

## Per-case results

| Case | Status | Expected | LLM | Payment | DTI | Maximum | All 3 | Case score | Seconds |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| c_salary_approved | ok | APPROVED | APPROVED | YES | YES | YES | YES | 100.00% | 137.59 |
| c_self_employment_rejected_dti | ok | REJECTED | REJECTED | YES | YES | YES | YES | 100.00% | 115.36 |
| c_pep_manual | ok | MANUAL REVIEW | MANUAL REVIEW | YES | YES | YES | YES | 100.00% | 96.66 |
| c_fico_rejected | ok | REJECTED | REJECTED | NO | NO | NO | NO | 65.62% | 74.23 |
| c_fixed_term_salary_approved | ok | APPROVED | APPROVED | YES | YES | YES | YES | 100.00% | 86.83 |
| c_dividends_approved | ok | APPROVED | APPROVED | YES | YES | YES | YES | 100.00% | 90.31 |
| c_fico_gray_manual | ok | MANUAL REVIEW | APPROVED | NO | YES | YES | NO | 52.08% | 76.80 |
| c_aml_manual | ok | MANUAL REVIEW | MANUAL REVIEW | YES | YES | YES | YES | 100.00% | 102.07 |
| c_active_delay_technical_approved | ok | APPROVED | APPROVED | YES | NO | NO | NO | 72.92% | 71.35 |
| c_active_delay_manual | ok | MANUAL REVIEW | MANUAL REVIEW | YES | YES | YES | YES | 100.00% | 71.27 |
| c_active_delay_rejected | ok | REJECTED | REJECTED | YES | YES | YES | YES | 100.00% | 70.69 |
| c_historical_delay_exception_manual | ok | MANUAL REVIEW | MANUAL REVIEW | YES | YES | YES | YES | 100.00% | 97.58 |
| c_historical_delay_rejected | ok | REJECTED | REJECTED | YES | YES | YES | YES | 100.00% | 97.31 |
| c_non_eu_eligible_approved | ok | APPROVED | APPROVED | YES | YES | YES | YES | 100.00% | 87.08 |
| c_non_eu_ineligible_rejected | ok | REJECTED | REJECTED | YES | YES | YES | YES | 100.00% | 88.70 |
| c_eur_variable_stress_rejected | ok | REJECTED | REJECTED | YES | YES | YES | YES | 100.00% | 85.20 |
| c_income_excluded_rejected | ok | REJECTED | REJECTED | YES | YES | YES | YES | 100.00% | 67.55 |
| c_maturity_age_rejected | ok | REJECTED | REJECTED | YES | YES | YES | YES | 100.00% | 80.51 |
| c_amount_above_cap_rejected | ok | REJECTED | REJECTED | YES | YES | YES | YES | 100.00% | 68.69 |
| c_capacity_exhausted_rejected | ok | REJECTED | REJECTED | YES | YES | YES | YES | 100.00% | 95.73 |
| c_it_tenure_exception_approved | ok | APPROVED | APPROVED | YES | YES | YES | YES | 100.00% | 73.62 |
| c_term_above_limit_rejected | ok | REJECTED | APPROVED | YES | YES | YES | YES | 71.88% | 99.09 |
