# Asistent de Creditare RAG NovaTech

Prototip educational pentru disertatie: un asistent care citeste manualul fictiv `Manual_Extins_Creditare_NovaTech_v3.docx`, include optional `Regulamentul_BNR_nr_17_2012.pdf`, recupereaza fragmente relevante cu RAG si evalueaza clienti prin reguli explicabile.

## Ce face

- indexeaza manualul NovaTech si Regulamentul BNR nr. 17/2012 in fragmente cautabile;
- aplica reguli de eligibilitate: varsta, FICO, intarzieri, PEP/AML, cetateni non-UE;
- calculeaza venitul eligibil prin ponderile din manual;
- calculeaza GMI si suma maxima recomandata prin formula de anuitate;
- afiseaza citari din manual pentru decizie;
- foloseste un LLM local prin Ollama pentru redactarea rezultatului final al evaluarii.

## Pasi recomandati pentru proiect

1. Pastreaza manualul fictiv ca sursa controlata pentru testare.
2. Adauga ulterior Regulamentul BNR nr. 17/2012 in acelasi corpus, ca document separat.
3. Ruleaza sistemul pe clienti sintetici cunoscuti, inclusiv cazurile din anexa manualului.
4. Compara raspunsul asteptat cu decizia produsa de motorul de reguli.
5. Foloseste RAG-ul pentru justificare si citare, nu pentru aritmetica. Calculele trebuie sa ramana deterministe.
6. In capitolul de evaluare, masoara separat: corectitudinea regasirii fragmentelor, corectitudinea deciziei si corectitudinea calculului sumei maxime.

## Instalare

```powershell
python -m pip install -r requirements.txt
```

## Rulare

```powershell
python app.py
```

Aplicatia porneste local la:

```text
http://127.0.0.1:7860
```

Daca adaugi sau modifici documente din corpus, opreste si reporneste aplicatia. Indexul RAG se construieste la pornire.

## Testare

```powershell
python -m unittest discover tests
```

## LLM local

Pentru evaluarea clientului, aplicatia foloseste LLM-ul local pentru prezentarea rezultatului calculat si a surselor RAG. Recomandat pentru demo:

```powershell
ollama pull qwen3:8b
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
$env:OPENAI_API_KEY="ollama"
$env:OPENAI_MODEL="qwen3:8b"
$env:OPENAI_TIMEOUT_SECONDS="180"
$env:OPENAI_MAX_TOKENS="1800"
python app.py
```

Calculele raman verificabile in cod, iar LLM-ul primeste valorile calculate si fragmentele RAG pentru a redacta raspunsul final.

## Structura

- `app.py` - interfata Gradio;
- `credit_assistant/document_loader.py` - citire DOCX/PDF si chunking;
- `credit_assistant/rag.py` - index TF-IDF si cautare;
- `credit_assistant/credit_engine.py` - reguli si formule de creditare;
- `credit_assistant/service.py` - legatura dintre evaluator si RAG;
- `tests/test_credit_engine.py` - teste de baza.

## Nota

Manualul NovaTech este fictiv. Rezultatele sunt pentru demonstratie academica si nu reprezinta consultanta financiara sau decizie bancara reala.
