from typing import Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model once (for cosine similarity)
_sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


def bleu_score(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)


def rouge_l_score(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, hypothesis)
    return score['rougeL'].fmeasure


def cosine_similarity(reference: str, hypothesis: str) -> float:
    emb1 = _sentence_model.encode(reference, convert_to_tensor=True)
    emb2 = _sentence_model.encode(hypothesis, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2).item())


def evaluate_all(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute BLEU, ROUGE-L, and cosine similarity for a pair of strings."""
    return {
        'bleu': bleu_score(reference, hypothesis),
        'rougeL': rouge_l_score(reference, hypothesis),
        'cosine': cosine_similarity(reference, hypothesis)
    }

if __name__ == "__main__":
    ref = "Machine learning is a subfield of AI."
    hyp = "Machine learning is part of artificial intelligence."
    print(evaluate_all(ref, hyp))
