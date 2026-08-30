# NovaTech RAG Credit Assistant

Educational prototype for a master's thesis: an assistant that reads the fictional `NovaTech_Extended_Credit_Manual_v3.pdf`, includes `NBR_Regulation_No_17_2012.md`, retrieves relevant fragments with RAG, and uses a local LLM to analyze credit applicants.

## What It Does

- indexes the NovaTech manual and NBR Regulation No. 17/2012 into searchable chunks using a selectable TF-IDF or Okapi BM25 sparse retriever;
- retrieves fragments relevant to the client profile, including rules about FICO, PEP, AML, income types, and DTI;
- sends the full profile and RAG fragments to a policy-only LLM stage;
- sends ten finance-only inputs and sanitized policy parameters to one isolated LLM calculation stage;
- locks the calculation stage's three final values and sends them, together with the policy result, to a final decision/synthesis stage that has no financial output fields;
- enforces exact stage-specific JSON schemas, finite canonical numbers, and calculation self-check status without supplying Python-calculated answers;
- converts the locked LLM result into a Markdown report displayed in Gradio;
- runs deterministic Python formulas only afterward, in a separate comparison/evaluation section.

## Recommended Project Steps

1. Keep the fictional manual as a controlled source for testing.
2. Use NBR Regulation No. 17/2012 as a separate document in the corpus.
3. Run the system on known synthetic clients, including the cases in `examples/evaluation_cases.json`.
4. Analyze the structured LLM response for decision, financial values, and justifications.
5. Use RAG for justification and citation, and the JSON schema to control the response format.
6. In the evaluation chapter, measure retrieval quality, decision consistency, numerical consistency, formatting, and latency separately.

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Running

```powershell
python app.py
```

The application starts locally at:

```text
http://127.0.0.1:7860
```

If you add or modify documents in the corpus, stop and restart the application. The RAG index is built at startup.

## Sparse Retriever

TF-IDF remains the default because the current paired retrieval diagnostic does not demonstrate a BM25 improvement. BM25 is implemented over the same lowercase, Unicode-accent-normalized unigram/bigram vocabulary, with term-frequency saturation and document-length normalization. No additional package is required.

Select the backend before starting the application:

```powershell
# Evidence-backed default
$env:RAG_RETRIEVER="tfidf"

# Selectable Okapi BM25 alternative
$env:RAG_RETRIEVER="bm25"
$env:RAG_BM25_K1="1.2"
$env:RAG_BM25_B="0.75"
python app.py
```

The active backend is displayed in the interface. Restart the application after changing it because the index is built once at startup. BM25 and TF-IDF scores have different scales; compare ranks and retrieval metrics, not raw score magnitudes.

## Temporary Public Access

The address `http://127.0.0.1:7860` works only on the local machine. To share the application with someone outside your network, use a temporary public tunnel through Cloudflare Tunnel.

Install `cloudflared` once:

```powershell
winget install Cloudflare.cloudflared
```

Then start the public application:

```powershell
.\start_public_cloudflare.ps1
```

If PowerShell blocks script execution, use the `.bat` version:

```powershell
.\start_public_cloudflare.bat
```

The terminal will display a URL similar to:

```text
https://example.trycloudflare.com
```

The link remains active while the terminal and computer are running.

For local network access only, you can start the server with:

```powershell
$env:SERVER_HOST="0.0.0.0"
$env:SERVER_PORT="7860"
D:\CondaEnvs\disertatie\python.exe app.py
```

## Testing

```powershell
python -m unittest discover tests
```

Run the paired, retriever-only comparison without invoking an LLM:

```powershell
python examples/benchmark_retrievers.py --retrievers tfidf bm25 --top-k 5 `
  --visible-chars 900 --repetitions 500
```

The resulting JSON and Markdown files record corpus hashes, tokenizer settings, BM25 parameters, per-query rankings, visible-prefix keyword coverage, and retriever-only latency. The keyword metric is explicitly a diagnostic rather than a human passage-relevance judgment.

## Evaluation Metrics

The application includes a `Metrics` tab that runs synthetic cases from `examples/evaluation_cases.json`.
The suite contains 13 policy questions and 22 client-analysis cases covering approvals, rejections,
manual review, income weighting, DTI, FICO, AML/PEP, payment delays, residency, stress scenarios,
product limits, maturity age, and documented policy exceptions.

For the `Manual Questions` section, the following metrics are computed:

- `retrieval_hit_at_5` - checks whether the expected sources appear among the top 5 RAG fragments;
- `keyword_coverage` - measures how many expected concepts appear in the LLM answer;
- `missing_information_response` - checks whether the model explicitly recognizes missing information;
- `rag_source_presence` - checks whether the answer includes fragments or sources;
- `markdown_format` - checks answer readability: headings, line breaks, no `***`, and no hidden reasoning text.

For the `Client Analysis` section, the following metrics are computed:

- `llm_decision_vs_expected` - compares the decision extracted from the LLM response with the expected decision in the synthetic dataset;
- `llm_decision_vs_formulas` - compares the LLM decision with the reference decision calculated through formulas;
- `isolated_numeric_agreement` - compares only the isolated LLM values for stressed payment, DTI, and maximum recommended amount with the post-hoc reference values;
- `all_three_numeric_fields_correct` - is 1 only when all three isolated calculation fields agree in the same case;
- `overall_llm_vs_formulas_score` - combines the final LLM decision and the three isolated numerical fields;
- `required_sections` - checks whether the required report sections are present;
- `rag_source_presence` - checks whether RAG sources are included;
- `markdown_format` - checks the structure and readability of the response.

The report displays the overall average score, score by section, latency, and detailed results for each case.

## Local LLM

Client evaluation now uses three separate LLM calls:

1. the RAG/policy stage reads the full profile and retrieved excerpts, selects the applicable calculation parameters, and assesses only non-calculated policy conditions;
2. the isolated calculation stage receives ten finance-only inputs plus the sanitized parameters and returns a branch/intermediate audit object whose immutable `final` contains exactly `stressed_monthly_payment`, `dti_pct`, and `maximum_amount_by_dti`;
3. the final synthesis stage consumes the policy result and the immutable calculation object, then returns only the decision and reason/source lists.

Python performs strict JSON/schema checks and renders the locked values verbatim. It does not calculate, retry, repair, or replace visible values. The deterministic engine runs only after generation, for the separate comparison tab and evaluation metrics.

Basic configuration (all stages use the same model):

```powershell
ollama pull mistral-small3.2
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
$env:OPENAI_API_KEY="ollama"
$env:OPENAI_MODEL="mistral-small3.2"
$env:OLLAMA_NATIVE_CHAT="true"
$env:OLLAMA_NUM_CTX="8192"
$env:OLLAMA_NUM_PREDICT="3000"
$env:OPENAI_TIMEOUT_SECONDS="240"
$env:OPENAI_MAX_TOKENS="3000"
python app.py
```

Each stage can also be routed independently. Stage-specific variables fall back to the global settings when omitted:

```powershell
$env:OPENAI_RAG_MODEL="mistral-small3.2:latest"
$env:OLLAMA_RAG_THINK="false"
$env:OPENAI_CALCULATION_MODEL="qwen3:14b"
$env:OLLAMA_CALCULATION_THINK="true"
$env:OPENAI_CALCULATION_TEMPERATURE="0.1"
$env:OLLAMA_CALCULATION_NUM_PREDICT="6000"
$env:OPENAI_SYNTHESIS_MODEL="mistral-small3.2:latest"
$env:OLLAMA_SYNTHESIS_THINK="false"
```

The local client uses Ollama's native `/api/chat` endpoint and sends an exact JSON Schema for every stage. The calculation prompt contains symbolic formulas and a mandatory branch/intermediate self-check, but no worked fixed-number examples, RAG prose, decision label, FICO, PEP, AML, delinquency, or residency fields. A malformed, non-finite, locale-formatted, contaminated, incomplete, or self-declared `FAIL` numerical object is a hard failure; the pipeline does not silently invoke a second calculator.

The installed `mistral-small4-iq1` aliases are useful as a quantization experiment, but their three-seed benchmark produced no cases with all three target fields correct. Do not select them as the calculation-stage model for a presentation without reporting that limitation. Benchmark candidate routes over all cases and choose by all-three-correct rate, not only aggregate per-field agreement.

Run a reproducible end-to-end smoke case with:

```powershell
python examples/smoke_staged_pipeline.py --case-id c_eur_variable_stress_rejected `
  --rag-model mistral-small3.2:latest --calculation-model qwen3:14b `
  --calculation-reasoning on --num-predict 6000 `
  --synthesis-model mistral-small3.2:latest
```

The observed route-screening results are recorded in `examples/staged_pipeline_smoke_comparison.md`. They are labeled as illustrative smoke results, not as a substitute for the complete 22-case, multi-seed benchmark.

## Structure

- `app.py` - Gradio interface;
- `credit_assistant/document_loader.py` - DOCX/PDF reading and chunking;
- `credit_assistant/rag.py` - selectable TF-IDF and dependency-free Okapi BM25 indexing and search;
- `credit_assistant/credit_engine.py` - post-hoc reference formulas used only for comparison and metrics;
- `credit_assistant/service.py` - staged RAG/policy, isolated calculation, immutable-value assembly, synthesis, and post-hoc comparison;
- `credit_assistant/evaluation.py` - metrics and synthetic suite execution;
- `examples/evaluation_cases.json` - synthetic evaluation examples;
- `examples/benchmark_retrievers.py` - paired TF-IDF/BM25 retrieval-only benchmark;
- `examples/smoke_staged_pipeline.py` - reproducible single-case live route smoke test;
- `examples/staged_pipeline_smoke_comparison.md` - observed local route-screening results and limitations;
- `tests/` - engine, metrics, routing, schema, isolation, injection, and immutable-value tests.

## Note

The NovaTech manual is fictional. The results are for academic demonstration only and do not represent financial advice or a real banking decision.
