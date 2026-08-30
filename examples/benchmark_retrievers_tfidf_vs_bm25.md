# Paired sparse-retriever benchmark

Both retrievers use the same frozen chunks, query set, lowercase/accent handling, and unigram/bigram vocabulary. Raw TF-IDF and BM25 scores are not compared because their scales differ. Expected-keyword coverage is a provisional visible-evidence diagnostic, not a passage-relevance judgment.

| Retriever | Expected-source coverage@5 | Top-1 keyword coverage | Top-5 full-chunk coverage | Top-5 visible-prefix coverage | Median search | p95 search |
|---|---:|---:|---:|---:|---:|---:|
| TFIDF | 100.00% | 73.75% | 87.08% | 63.61% | 0.228 ms | 0.290 ms |
| BM25 | 100.00% | 77.08% | 87.08% | 62.22% | 0.128 ms | 0.165 ms |

## Per-question visible-prefix coverage

The missing-policy question is shown but excluded from the aggregate keyword metrics.

| Case | TFIDF | BM25 |
|---|---:|---:|
| q_income_weights | 50.00% | 33.33% |
| q_dti_formula | 75.00% | 75.00% |
| q_pep_aml | 75.00% | 75.00% |
| q_product_limits | 25.00% | 25.00% |
| q_currency_interest_stress | 100.00% | 100.00% |
| q_income_exclusions | 80.00% | 80.00% |
| q_fico_bands | 16.67% | 16.67% |
| q_non_eu_conditions | 80.00% | 80.00% |
| q_active_delay_bands | 80.00% | 80.00% |
| q_historical_delay_exception | 75.00% | 75.00% |
| q_life_insurance_age | 66.67% | 66.67% |
| q_it_tenure_exception | 40.00% | 40.00% |
| q_missing_policy | excluded | excluded |
