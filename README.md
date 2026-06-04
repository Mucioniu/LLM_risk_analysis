# Asistent de Creditare RAG NovaTech

Prototip educational pentru disertatie: un asistent care citeste manualul fictiv `Manual_Extins_Creditare_NovaTech_v3.pdf`, include `Regulamentul_BNR_nr_17_2012.md`, recupereaza fragmente relevante cu RAG si evalueaza clienti prin reguli explicabile.

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

## Acces public temporar

Adresa `http://127.0.0.1:7860` functioneaza doar pe calculatorul local. Pentru a trimite aplicatia unei persoane aflate in alta retea, foloseste un tunel public temporar prin Cloudflare Tunnel.

Instaleaza `cloudflared` o singura data:

```powershell
winget install Cloudflare.cloudflared
```

Apoi porneste aplicatia publica:

```powershell
.\start_public_cloudflare.ps1
```

In cazul in care PowerShell blocheaza rularea scripturilor, foloseste varianta `.bat`:

```powershell
.\start_public_cloudflare.bat
```

Terminalul va afisa un URL de forma:

```text
https://exemplu.trycloudflare.com
```

Acela este linkul pe care il poti trimite profesorului. Linkul ramane activ cat timp terminalul si calculatorul tau sunt pornite.

Pentru acces doar in reteaua locala, poti porni serverul cu:

```powershell
$env:SERVER_HOST="0.0.0.0"
$env:SERVER_PORT="7860"
D:\CondaEnvs\disertatie\python.exe app.py
```

## Testare

```powershell
python -m unittest discover tests
```

## Metrici de evaluare

Aplicatia include tabul `Metrici`, care ruleaza cazuri sintetice din `examples/evaluation_cases.json`.

Pentru sectiunea `Intrebari despre manual`, sunt calculate:

- `retrieval_hit_at_5` - verifica daca sursele asteptate apar intre primele 5 fragmente RAG;
- `acoperire_cuvinte_cheie` - masoara cate concepte asteptate apar in raspunsul LLM;
- `raspuns_lipsa_info` - verifica daca modelul recunoaste explicit informatia lipsa;
- `prezenta_surse_rag` - verifica daca raspunsul include fragmente/surse;
- `format_markdown` - verifica lizibilitatea raspunsului: heading-uri, linii noi, fara `***` sau text de gandire.

Pentru sectiunea `Analiza client`, sunt calculate:

- `consistenta_decizie` - compara decizia din raspuns cu decizia asteptata si cu motorul determinist;
- `consistenta_valori_numerice` - verifica daca valorile financiare calculate apar in raspuns;
- `sectiuni_obligatorii` - verifica prezenta sectiunilor cerute in raport;
- `prezenta_surse_rag` - verifica includerea surselor RAG;
- `format_markdown` - verifica structura si lizibilitatea raspunsului.

Raportul afiseaza scor mediu total, scor pe sectiuni, latenta si detaliu pe fiecare caz.

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
- `credit_assistant/evaluation.py` - metrici si rulare seturi sintetice;
- `examples/evaluation_cases.json` - exemple sintetice pentru evaluare;
- `tests/` - teste de baza pentru motor si metrici.

## Nota

Manualul NovaTech este fictiv. Rezultatele sunt pentru demonstratie academica si nu reprezinta consultanta financiara sau decizie bancara reala.
