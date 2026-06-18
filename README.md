# NovaTech RAG Credit Assistant

Educational prototype for a master's thesis: an assistant that reads the fictional `Manual_Extins_Creditare_NovaTech_v3.pdf`, includes `Regulamentul_BNR_nr_17_2012.md`, retrieves relevant fragments with RAG, and uses a local LLM to analyze credit applicants.

## What It Does

- indexes the NovaTech manual and BNR Regulation no. 17/2012 into searchable chunks;
- retrieves fragments relevant to the client profile, including rules about FICO, PEP, AML, income types, and GMI;
- sends the client profile, numerical rules, and RAG fragments to a local LLM through Ollama;
- receives from the LLM a structured JSON analysis containing the decision, financial values, reasons, and sources;
- validates the JSON schema and value consistency for reporting and metrics;
- validates the LLM's JSON output against deterministic Python calculations and numeric rules; if inconsistencies are found the service performs automatic self-review and (if needed) an adjudication step to correct decision or reasons;
- converts the validated result into a Markdown report displayed in Gradio;
- includes a separate comparison section between the LLM response and the reference formulas.

## Recommended Project Steps

1. Keep the fictional manual as a controlled source for testing.
2. Use BNR Regulation no. 17/2012 as a separate document in the corpus.
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

## Evaluation Metrics

The application includes a `Metrici` tab that runs synthetic cases from `examples/evaluation_cases.json`.

For the `Intrebari despre manual` section, the following metrics are computed:

- `retrieval_hit_at_5` - checks whether the expected sources appear among the top 5 RAG fragments;
- `acoperire_cuvinte_cheie` - measures how many expected concepts appear in the LLM answer;
- `raspuns_lipsa_info` - checks whether the model explicitly recognizes missing information;
- `prezenta_surse_rag` - checks whether the answer includes fragments or sources;
- `format_markdown` - checks answer readability: headings, line breaks, no `***`, and no hidden reasoning text.

For the `Analiza client` section, the following metrics are computed:

- `decizie_llm_vs_asteptat` - compares the decision extracted from the LLM response with the expected decision in the synthetic dataset;
- `decizie_llm_vs_formule` - compares the LLM decision with the reference decision calculated through formulas;
- `scor_total_llm_vs_formule` - compares the structured financial values produced by the LLM with the reference values;
- `sectiuni_obligatorii` - checks whether the required report sections are present;
- `prezenta_surse_rag` - checks whether RAG sources are included;
- `format_markdown` - checks the structure and readability of the response.

The report displays the overall average score, score by section, latency, and detailed results for each case.

## Local LLM

For client evaluation, the application uses the local LLM to generate a structured analysis of the profile, including the decision, financial calculations, reasons, and RAG sources. Recommended for the demo:

```powershell
ollama pull mistral-small3.2
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
$env:OPENAI_API_KEY="ollama"
$env:OPENAI_MODEL="mistral-small3.2"
$env:OPENAI_TIMEOUT_SECONDS="180"
$env:OPENAI_MAX_TOKENS="3000"
python app.py
```

The LLM response is requested in structured JSON format, then validated and displayed as a Markdown report in the Gradio interface.

Note on validation and robustness: the service attempts to parse and validate the LLM JSON up to three times. If the returned JSON is inconsistent with the deterministic Python evaluation, the code will (1) request an internal LLM self-review to correct numeric or decision inconsistencies, and (2) if necessary, request an adjudication step that only decides `APROBAT | RESPINS | ANALIZA MANUALA` and updates the JSON decision. These retries and reviewer/adjudicator interactions are automatic and intended to improve result consistency for evaluation and reporting.

## Structure

- `app.py` - Gradio interface;
- `credit_assistant/document_loader.py` - DOCX/PDF reading and chunking;
- `credit_assistant/rag.py` - TF-IDF index and search;
- `credit_assistant/credit_engine.py` - reference formulas and rules used for comparison;
- `credit_assistant/service.py` - RAG orchestration, LLM prompts, structured JSON, and validation;
- `credit_assistant/service.py` - RAG orchestration, structured JSON prompts, schema validation, self-review and adjudication loops, and comparison to deterministic formulas;
- `credit_assistant/evaluation.py` - metrics and synthetic suite execution;
- `examples/evaluation_cases.json` - synthetic evaluation examples;
- `tests/` - basic tests for the engine and metrics.

## Note

The NovaTech manual is fictional. The results are for academic demonstration only and do not represent financial advice or a real banking decision.
