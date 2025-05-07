from transformers import pipeline
import openai
import os
import torch

# Cache for HuggingFace pipelines
_hf_pipelines = {}

# List of supported HuggingFace models
HF_MODELS = [
    "ltg/flan-t5-definition-en-large",
    "ltg/mt0-definition-en-xl",
    "FrancescoPeriti/Llama2Dictionary"
    # Add more as needed
]

def get_hf_pipeline(model_name):
    # Use GPU if available, otherwise CPU
    device = 0 if torch.cuda.is_available() else -1
    if model_name not in _hf_pipelines:
        _hf_pipelines[model_name] = pipeline(
            "text2text-generation",
            model=model_name,
            device=device
        )
        if device == 0:
            print(f"[INFO] Using GPU for model {model_name}")
        else:
            print(f"[INFO] Using CPU for model {model_name}")
    return _hf_pipelines[model_name]


def generate_definition(term: str, prompt: str, model_name: str = "") -> str:
    """
    Generate a definition using a HuggingFace or OpenAI model.
    - HuggingFace: local pipeline for known models
    - OpenAI: expects model_name like 'openai:gpt-3.5-turbo' or 'openai:gpt-4'
    """
    if model_name in HF_MODELS:
        pipe = get_hf_pipeline(model_name)
        # Compose input for T5-style models
        input_text = f"{prompt}\nTerm: {term}"
        output = pipe(input_text, max_length=128, truncation=True)[0]
        # Output can be 'generated_text' or 'summary_text' depending on model
        return output.get('generated_text') or output.get('summary_text') or str(output)
    elif model_name.startswith("openai:"):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set!")
        model = model_name.split(":", 1)[1]
        client = openai.OpenAI(api_key=openai_api_key)
        # Prepare arguments for OpenAI call
        completion_args = dict(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Define the term: {term}"}
            ]
        )
        # Use correct token parameter and temperature depending on model
        if model in ["o4-mini-2025-04-16", "gpt-4.1-2025-04-14"]:
            completion_args["max_completion_tokens"] = 128
            # Do NOT set temperature for these models (default is 1, only value allowed)
        else:
            completion_args["max_tokens"] = 128
            completion_args["temperature"] = 0.7
        response = client.chat.completions.create(**completion_args)
        return response.choices[0].message.content.strip()
    else:
        return f"[UNKNOWN MODEL: {model_name}]"  
