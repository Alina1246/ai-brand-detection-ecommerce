
import os
import time
import json
import re
import torch
import google.generativeai as genai
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, pipeline
from crewai import Agent, Task, Crew, LLM

# ================= GEMMA ===================
from transformers import AutoProcessor, AutoModelForCausalLM
HF_TOKEN = "YOUR_HF_TOKEN"
model_id_gemma = "google/gemma-3-4b-it"
processor_gemma = AutoProcessor.from_pretrained(model_id_gemma, token=HF_TOKEN)
model_gemma = AutoModelForCausalLM.from_pretrained(model_id_gemma, device_map="auto", token=HF_TOKEN).eval()


few_shot_examples = [
    {"nume_produs": "Sapca Unisex Baseball BMW M Motorsport Neagra Marime M/L", "brand": "BMW"},
    {"nume_produs": "Prosop Audi, Albastru inchis si deschis, bumbac, 80 x 150 cm", "brand": "Audi"},
    {"nume_produs": "Tricou Personalizat AudiLove go hard, ADLER, negru, XL", "brand": "ADLER"},
    {"nume_produs": "Tricou VR46 MONZA RALLY REPLICA Negru 2XL", "brand": "VR|46"},
    {"nume_produs": "Breloc Volkswagen Tiguan Metalic, Piele, YLKRS AutoMotive", "brand": "YLKRS AUTOMOTIVE"},
    {"nume_produs": "Breloc / husa cheie AUDI, fermoar, Maro Ubitec ®", "brand": "Ubitec"},
]

branduri_scurte = [
    "BMW", "Audi", "Ubitec", "YLKRS AUTOMOTIVE", "ADLER", "OEM", "Land Rover", "Alpinestars", "G-Mark"
]

def build_chat_messages(nume_produs, branduri_batch, examples):
    example_str = "\n".join([f"'{ex['nume_produs']}' → Brand: {ex['brand']}" for ex in examples])
    user_prompt = f'''
FEW-SHOT EXAMPLES:
{example_str}

### NEW INPUT ###
Product title: '{nume_produs}'
Choose the correct brand only from the following approved list: {branduri_batch}
Rules:
- Select ONLY from the list above.
- If multiple brands are mentioned, the correct brand is NEVER in the middle.
- Return only the brand name, exactly as it appears in the list.
- Return the result strictly in the following JSON format:
{{"brand": "<brand_name>"}}
'''
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are an emag assistant that extracts brands in JSON format."}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

def run_gemma_on_title(titlu_produs):
    messages = build_chat_messages(titlu_produs, branduri_scurte, few_shot_examples)
    try:
        inputs = processor_gemma.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model_gemma.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model_gemma.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
        decoded = processor_gemma.decode(generation, skip_special_tokens=True)
        json_matches = re.findall(r'\{.*?\}', decoded)
        for match in reversed(json_matches):
            try:
                parsed_json = json.loads(match)
                if "brand" in parsed_json:
                    return parsed_json["brand"]
            except json.JSONDecodeError:
                continue
        return "N/A"
    except Exception as e:
        return f"ERROR: {e}"

# ================= QWEN ===================
model_id_qwen = "Qwen/Qwen2.5-7B-Instruct"
tokenizer_qwen = AutoTokenizer.from_pretrained(model_id_qwen, trust_remote_code=True)
model_qwen = AutoModelForCausalLM.from_pretrained(
    model_id_qwen,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
pipe_qwen = pipeline("text-generation", model=model_qwen, tokenizer=tokenizer_qwen)

def build_prompt_qwen(nume_produs, branduri_batch, examples):
    example_str = "\n".join([f"'{ex['nume_produs']}' → Brand: {ex['brand']}" for ex in examples])
    branduri_str = ", ".join(branduri_batch)
    return f'''
Your task is to extract the correct brand from a product name.

RULES:
- Choose only from the approved list.
- If multiple brands are mentioned, the correct brand is NEVER in the middle.
- Output ONLY the JSON object, like:  {{"brand": "BRAND_NAME_FROM_LIST"}}
- Replace "BRAND_NAME_FROM_LIST" with the correct brand from the approved list.

FEW-SHOT EXAMPLES:
{example_str}

PRODUCT NAME:
'{nume_produs}'

APPROVED BRANDS:
{branduri_str}
'''

def run_qwen_on_title(titlu_produs):
    prompt = build_prompt_qwen(titlu_produs, branduri_scurte, few_shot_examples)
    try:
        output = pipe_qwen(prompt, max_new_tokens=150, do_sample=False)[0]["generated_text"]
        json_matches = re.findall(r"\{.*?\}", output)
        for match in reversed(json_matches):
            try:
                parsed = json.loads(match)
                if "brand" in parsed:
                    return parsed["brand"]
            except json.JSONDecodeError:
                continue
        return "N/A"
    except Exception as e:
        return f"ERROR: {e}"

# ================= GEMINI ===================
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key="YOUR_GOOGLE_API_KEY"
)

brand_extraction_agent = Agent(
    llm=llm,
    role="Brand Extraction Agent",
    goal="Extract the correct brand from the product name.",
    backstory="You are responsible for identifying the brand from a product name. "
              "If multiple brands are mentioned in the product name, the correct brand is never in the middle. "
              "The approved brands are: " + ", ".join(branduri_scurte),
    allow_delegation=False,
    verbose=False,
)

brand_extraction_task = Task(
    description="Extract the correct brand from the product name: {product}. "
                "Return the result as a JSON dictionary with the key 'brand'.",
    expected_output="A JSON dictionary with the key 'brand' and the extracted brand as the value.",
    agent=brand_extraction_agent,
)

crew = Crew(
    agents=[brand_extraction_agent],
    tasks=[brand_extraction_task]
)

def run_gemini_on_title(titlu_produs):
    try:
        result = crew.kickoff(inputs={"product": titlu_produs})
        if hasattr(result, "output"):
            result = result.output
        result = re.sub(r"```json|```", "", str(result)).strip()
        parsed = json.loads(result)
        return parsed.get("brand", "Necunoscut")
    except Exception as e:
        return f"ERROR: {e}"

# ================= SELECTOR ===================

def run_model(titlu_produs, model_selectat):
    if model_selectat == "Gemma-3-4B-it":
        rezultat = run_gemma_on_title(titlu_produs)
    elif model_selectat == "Qwen-2.5-7B-Instruct":
        rezultat = run_qwen_on_title(titlu_produs)
    elif model_selectat == "Gemini-1.5-flash":
        rezultat = run_gemini_on_title(titlu_produs)
    else:
        return "Model invalid"

    if rezultat not in branduri_scurte:
        return "Niciun rezultat"

    rezultat_normalizat = rezultat.lower().replace(" ", "")
    titlu_normalizat = titlu_produs.lower().replace(" ", "")
    if rezultat_normalizat not in titlu_normalizat:
        return "Niciun rezultat"

    return rezultat

