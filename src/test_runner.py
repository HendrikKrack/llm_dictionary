import os
import json
from dataset_loader import load_definitions
from prompts import get_prompt
from llm_interface import generate_definition
from evaluation import evaluate_all

# Configurable
DATA_FILE = os.path.join('testData', 'testDefinitions.txt')
PROMPT_VERSIONS = ['old', 'update1', 'update2']
MODEL_NAMES = [
    #'ltg/flan-t5-definition-en-large',
    #'ltg/mt0-definition-en-xl',
    #'FrancescoPeriti/Llama2Dictionary',
    'openai:o4-mini-2025-04-16',
    'openai:gpt-4.1-2025-04-14',
    'openai:gpt-4o'
]


def main():
    definitions = load_definitions(DATA_FILE)
    results = []
    for prompt_version in PROMPT_VERSIONS:
        prompt = get_prompt(prompt_version)
        for model_name in MODEL_NAMES:
            for term, gold_definition in definitions:
                generated = generate_definition(term, prompt, model_name)
                # Add a short pause after each OpenAI call to avoid socket exhaustion
                if model_name.startswith('openai:'):
                    import time
                    time.sleep(0.2)  # 200ms pause
                metrics = evaluate_all(gold_definition, generated)
                results.append({
                    'term': term,
                    'gold_definition': gold_definition,
                    'generated_definition': generated,
                    'prompt_version': prompt_version,
                    'model_name': model_name,
                    'bleu': metrics['bleu'],
                    'rougeL': metrics['rougeL'],
                    'cosine': metrics['cosine']
                })
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    # Write results to JSON
    with open('results/definitions_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(results)} results and saved to results/definitions_results.json.")

if __name__ == "__main__":
    main()
