# ai-brand-detection-ecommerce

Automatic brand detection from e-commerce product titles using pretrained LLMs (Gemma, Qwen, Gemini) and a small Gradio demo.

This project was developed as a Bachelor’s thesis practical application and targets a common marketplace issue: sellers often fill the *brand* field incorrectly (e.g., OEM / seller name), even though the correct brand is frequently present in the product title. The system extracts the brand **only from the title**, without fine-tuning, using prompt-based inference.

---

## What I built

- A pipeline to extract the correct brand from product titles using LLMs
- Prompt engineering (few-shot) with strict JSON output
- Robust error handling (JSON parsing + retries + incremental saving)
- Comparative evaluation on 1000 products / model (accuracy, speed, stability)
- A Gradio interface for interactive testing

---

## Dataset (high level)

The dataset was built from two eMAG tables:
- **Products table** (titles + raw brand field, sometimes missing/incorrect)
- **Official brand catalog** (brandId, request status, active/inactive, brand name)

Key processing steps:
1. Split raw brand field (`id:name`)
2. Join with official catalog using `brandId`
3. Keep only **approved + active** brands
4. Remove incomplete rows
5. Filter to keep only products where the normalized brand name appears in the title  
   (ensures reliable ground truth for evaluation)

---

## Models

The following pretrained LLMs were evaluated:
- **Gemma-3-4B-IT** (Hugging Face, local in Colab)
- **Qwen2.5-7B-Instruct** (Hugging Face, local in Colab)
- **Gemini 1.5 Flash** (Google API, used through CrewAI)

All experiments were run in **Google Colab** (GPU A100 for local models). No fine-tuning was performed.

---

## Prompting approach

- **Few-shot prompts** with manually selected examples
- Clear rules:
  - choose only from an allowed brand list
  - return only JSON: `{"brand": "<brand_name>"}`
  - ignore irrelevant brand-like tokens when multiple brands appear
- Because the full catalog is too large (10k+ brands), brand candidates were restricted:
  - Gemma/Qwen: per **batch (100 products)**, include only relevant brands
  - Gemini: use a smaller list relevant for the tested subset (token/API constraints)

---

## Error handling (how it works)

LLMs can return extra text, invalid JSON, or incomplete outputs. The pipeline was designed to be robust:

### JSON extraction
- Scan the response for `{...}` blocks using regex
- Try parsing candidates **from the end of the response**
- Accept the first valid JSON that contains `"brand"`

### Retry logic
- Up to **5 attempts per product**
- If parsing fails or output is invalid, retry with a short wait
- If all retries fail:
  - brand remains empty
  - prediction is marked incorrect
  - retries are counted

### Incremental saving
- Local models: save results after each batch
- Gemini: save product-by-product (API limits)

### Gemini API limits
- Requests are rate limited and have daily quotas
- On quota/rate errors, progress is saved and execution stops gracefully

---

## Results (1000 products per model)

| Model | Correct | Incorrect | Accuracy | Total Time | Avg Time (s/item) | Total Retries | Error Frequency |
|------|--------:|----------:|---------:|-----------:|------------------:|--------------:|----------------|
| gemma-3-4b-it | 981 | 19 | 98.1% | 0:14:45 | 0.886 | 0 | {0: 1000} |
| qwen-2.5-7b-instruct | 971 | 29 | 97.1% | 0:07:57 | 0.478 | 5 | {0: 999, 5: 1} |
| gemini-1.5-flash | 984 | 16 | 98.4% | 0:09:51 | 0.591 | 26 | {0: 994, 1: 1, 5: 5} |

**How to interpret error frequency**  
Example `{0: 999, 5: 1}` means 999 products had 0 retries, while 1 product required 5 retries. This helps measure stability beyond total retries.

---

## Notebooks / code

- `notebooks/gemma.ipynb` – Gemma inference experiment
- `notebooks/qwen.ipynb` – Qwen inference experiment
- `notebooks/gemini.ipynb` – Gemini inference experiment (CrewAI + API)
- `notebooks/gradio_app.ipynb` – Gradio demo app
- `src/prompt_utils.py` – shared utilities (prompt building, parsing, validation)

---

## Notes

- This evaluation dataset keeps only products where the brand appears in the title, to ensure reliable ground truth.
- Gemini results depend on API limits (rate/quota), which can increase retries.

---
