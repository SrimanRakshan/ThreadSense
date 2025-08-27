import json
from pathlib import Path

# Try imports for evaluation metrics
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    import bert_score
except ImportError:
    bert_score = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

def load_generated_summaries(runs_json_path):
    with open(runs_json_path, "r", encoding="utf-8") as f:
        runs = json.load(f)
    # Extract variant and global summary
    summaries = []
    for run in runs:
        variant = run.get("variant", "unknown")
        global_summary = run.get("global_summary", "")
        summaries.append({"variant": variant, "summary": global_summary})
    return summaries

def load_gold_summary(gold_json_path):
    with open(gold_json_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    # Try common keys for summary
    if "summary" in gold_data:
        return gold_data["summary"]
    elif "reference_summary" in gold_data:
        return gold_data["reference_summary"]
    elif "global_summary" in gold_data:
        return gold_data["global_summary"]
    else:
        raise ValueError("Gold summary not found in JSON under keys 'summary', 'reference_summary', or 'global_summary'.")

def compute_rouge(reference, candidates):
    if rouge_scorer is None:
        return None
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {}
    for c in candidates:
        variant = c["variant"]
        score = scorer.score(reference, c["summary"])
        scores[variant] = {
            "rouge1": score["rouge1"].fmeasure,
            "rouge2": score["rouge2"].fmeasure,
            "rougeL": score["rougeL"].fmeasure,
        }
    return scores

def compute_bertscore(reference, candidates):
    if bert_score is None:
        return None
    refs = [reference] * len(candidates)
    cands = [c["summary"] for c in candidates]
    P, R, F1 = bert_score.score(cands, refs, lang="en", verbose=False)
    scores = {}
    for i, c in enumerate(candidates):
        scores[c["variant"]] = {
            "precision": P[i].item(),
            "recall": R[i].item(),
            "f1": F1[i].item()
        }
    return scores

def compute_embedding_similarity(reference, candidates):
    if SentenceTransformer is None or util is None:
        return None
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ref_emb = model.encode(reference, convert_to_tensor=True)
    scores = {}
    for c in candidates:
        cand_emb = model.encode(c["summary"], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(ref_emb, cand_emb).item()
        scores[c["variant"]] = sim
    return scores

def main(runs_json_path, gold_json_path, output_path):
    # Load data
    generated = load_generated_summaries(runs_json_path)
    gold_summary = load_gold_summary(gold_json_path)

    # Compute metrics
    results = {}
    rouge_scores = compute_rouge(gold_summary, generated)
    if rouge_scores:
        results["rouge"] = rouge_scores
    bert_scores = compute_bertscore(gold_summary, generated)
    if bert_scores:
        results["bertscore"] = bert_scores
    emb_scores = compute_embedding_similarity(gold_summary, generated)
    if emb_scores:
        results["embedding_similarity"] = emb_scores

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    runs_json_path = r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\new\ThreadSense\outputs\experiments\run_20250827_223734_9f638fdc\runs.json"
    gold_json_path = r"C:\Users\Sriman Rakshan N\Documents\Amrita\Project_Sem_V\new\ThreadSense\outputs\thread_summary.json"
    output_path = "outputs\evaluated_metrics.json"
    main(runs_json_path, gold_json_path, output_path)
