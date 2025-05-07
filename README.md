# PromptEval: LLM Definition Generation Test Suite

## Overview
This repository provides a framework for evaluating and comparing language models (LLMs) on the task of generating dictionary-style definitions for technical terms. It supports both local HuggingFace models (with GPU/CPU auto-selection) and OpenAI models (using the latest API, including new models like `o4-mini-2025-04-16`).

## Structure
- `src/dataset_loader.py`: Loads and parses test data.
- `src/prompts.py`: Stores and retrieves prompt variations.
- `src/llm_interface.py`: Integrates HuggingFace and OpenAI LLMs, with correct API usage for each.
- `src/test_runner.py`: Orchestrates the test runs, collects results, and adds a pause between OpenAI calls to avoid socket errors.
- `src/evaluation.py`: Evaluation metrics (BLEU, ROUGE-L, cosine similarity).
- `testData/`: Contains test datasets.
- `results/`: Where output and plots will be stored.

## What Has Been Done
- HuggingFace and OpenAI models integrated for definition generation.
- Automatic GPU/CPU selection for local models.
- Support for new OpenAI API parameter conventions (`max_completion_tokens`, default temperature for some models).
- Fixes for socket exhaustion (short pause between OpenAI calls).
- Evaluation and visualization pipeline.

---

## Step-by-Step: How to Run the Experiment

### 1. **Install Dependencies**
```sh
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. **Set Up OpenAI API Key (if using OpenAI models)**
Set your OpenAI API key as an environment variable:
```sh
$env:OPENAI_API_KEY="sk-..."   # PowerShell
# Or set it in your system environment variables
```

### 3. **Configure Which Models to Run**
Edit `src/test_runner.py` and set the `MODEL_NAMES` list to include the models you want to test, e.g.:
```python
MODEL_NAMES = [
    'ltg/flan-t5-definition-en-large',
    'openai:gpt-4o',
    'openai:o4-mini-2025-04-16',
]
```
- Models starting with `openai:` use the OpenAI API.
- Others are loaded locally via HuggingFace.

### 4. **Run the Experiment**
```sh
python src/test_runner.py
```
- This will generate definitions, compute metrics, and save results to `results/definitions_results.json`.

### 5. **Analyze Results**
```sh
python analyze_results.py
```
- This will create plots and summary statistics in the `results/` folder.

### 6. **Interpreting Metrics**
- **BLEU:** Measures n-gram overlap between generated and gold definition (higher is more similar).
- **ROUGE-L:** Measures longest common subsequence overlap (higher is more similar).
- **Cosine:** Semantic similarity between generated and gold definition (higher is more similar).

---

## Troubleshooting
- If you see errors about `max_tokens` or `temperature`, make sure you have the latest code and correct model-specific parameters are being used.
- For socket/buffer errors, a short pause is automatically added between OpenAI API calls.
- For GPU issues, ensure you have the correct CUDA-enabled PyTorch installed.

---

## Contact
For further help, open an issue or contact the project maintainer.

