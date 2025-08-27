import os
import re
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

import pandas as pd

# Local LLM via transformers (Falcon 7B Instruct)
from transformers import pipeline


# ==============================
# CONFIG / LOGGING
# ==============================

RUN_ID = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
OUTPUT_DIR = Path(f"outputs/experiments/{RUN_ID}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)


# ==============================
# MODEL LOADER (Falcon local)
# ==============================

def load_falcon_generator() -> Any:
    """
    Load Falcon-7B-Instruct locally using HuggingFace pipeline.
    Adjust max_new_tokens, temperature, and other knobs as needed.
    """
    logging.info("Loading Falcon-7B-Instruct locally...")
    gen = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        device_map="auto",
        torch_dtype="auto"
    )
    logging.info("Falcon loaded.")
    return gen


# ==============================
# PROMPT VARIANTS
# ==============================

PROMPT_VARIANTS = ["zero_shot", "few_shot", "chain_of_thought", "role_play"]


def _few_shot_block() -> str:
    """
    Two tiny synthetic exemplars to prime style.
    Keep small to avoid context bloat.
    """
    return (
        "### Example 1\n"
        "Root Comment: \"The devs delayed the update again.\"\n"
        "Replies:\n"
        "- Some users agree it's better to ship stable builds.\n"
        "- Others are frustrated with repeated delays.\n"
        "Reasoning:\n"
        "1) Main points: delay rationale vs user frustration.\n"
        "2) Prioritize clarity and balance.\n"
        "Good Summary: \"Users debate an update delay: some prefer stability over speed, "
        "while others are frustrated by repeated postponements. The thread balances quality concerns with expectations.\"\n\n"
        "### Example 2\n"
        "Root Comment: \"Is this feature worth the hype?\"\n"
        "Replies:\n"
        "- Mixed experiences; performance varies by setup.\n"
        "- Value depends on workflow and budget.\n"
        "Reasoning:\n"
        "1) Identify mixed evidence.\n"
        "2) Conclude with conditional recommendation.\n"
        "Good Summary: \"Opinions are split: benefits depend on individual workflows and budgets. "
        "Some see gains; others find the value marginal.\"\n\n"
    )


def build_local_prompt(variant: str, root_comment: str, replies_text: str) -> str:
    """
    Build the local (branch-level) summarization prompt for the chosen variant.
    """
    base = (
        f"Root Comment: {root_comment}\n"
        f"Replies:\n{replies_text}\n\n"
    )

    if variant == "zero_shot":
        return (
            base +
            "Task: Summarize this discussion succinctly (2–3 sentences). Focus on main points.\n"
            "Summary:"
        )

    elif variant == "few_shot":
        return (
            _few_shot_block() +
            base +
            "Task: Following the style of the above examples, give a balanced 2–3 sentence summary.\n"
            "Summary:"
        )

    elif variant == "chain_of_thought":
        return (
            base +
            "Task:\n"
            "Step 1: List key points raised.\n"
            "Step 2: Rank by importance.\n"
            "Step 3: Synthesize a concise 2–3 sentence summary.\n\n"
            "Reasoning:\n- "
        )  # We will post-process to extract the final summary if present.

    elif variant == "role_play":
        return (
            "You are a seasoned Reddit moderator who writes clear, neutral summaries to resolve confusion.\n\n" +
            base +
            "Write a 2–3 sentence moderator-style summary highlighting consensus and disagreements.\n"
            "Summary:"
        )

    else:
        raise ValueError(f"Unknown prompt variant: {variant}")


def build_global_prompt(variant: str, local_summaries: List[str]) -> str:
    """
    Build the global (thread-level) summarization prompt for the chosen variant.
    """
    locals_joined = "\n- ".join(local_summaries) if local_summaries else "None."

    if variant == "zero_shot":
        return (
            "You are given partial summaries of different branches from a Reddit thread:\n"
            f"- {locals_joined}\n\n"
            "Task: Produce a single, coherent overall summary (<= 5 sentences). Be faithful & non-redundant.\n"
            "Final Summary:"
        )

    elif variant == "few_shot":
        return (
            _few_shot_block() +
            "Branch Summaries:\n"
            f"- {locals_joined}\n\n"
            "Task: Following the style of the examples, merge overlapping ideas into a coherent final summary (<= 5 sentences).\n"
            "Final Summary:"
        )

    elif variant == "chain_of_thought":
        return (
            "Branch Summaries:\n"
            f"- {locals_joined}\n\n"
            "Task:\n"
            "1) Identify overlaps and contradictions.\n"
            "2) Order main themes logically.\n"
            "3) Write a final summary (<= 5 sentences) that preserves nuance.\n\n"
            "Reasoning:\n- "
        )

    elif variant == "role_play":
        return (
            "You are a neutral meeting facilitator. Combine the following branch summaries into a single, "
            "balanced overview that highlights consensus and key disagreements.\n\n"
            f"Branch Summaries:\n- {locals_joined}\n\n"
            "Final Summary:"
        )

    else:
        raise ValueError(f"Unknown prompt variant: {variant}")


# ==============================
# GENERATION & POST-PROCESS
# ==============================

def generate_text(gen, prompt: str,
                  max_new_tokens: int = 320,
                  temperature: float = 0.3,
                  top_p: float = 0.9,
                  repetition_penalty: float = 1.05) -> str:

    logging.info(f"Generating text (prompt len={len(prompt)} chars)...")
    try:
        outputs = gen(
            prompt,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            pad_token_id=gen.tokenizer.eos_token_id,
            eos_token_id=gen.tokenizer.eos_token_id,
            truncation=False,
            return_full_text=True,
        )
        logging.info("Generation complete.")
        return outputs[0]["generated_text"]
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return "[[GENERATION FAILED]]"


def extract_after_marker(text: str, marker: str) -> str:
    """
    If the model includes the prompt + answer, try to slice content after a marker like "Summary:" or "Final Summary:".
    """
    idx = text.lower().rfind(marker.lower())
    if idx != -1:
        return text[idx + len(marker):].strip()
    return text.strip()


def clean_summary(text: str) -> str:
    """
    Light cleanup: remove spurious headings from CoT, keep last ~5 sentences if overlong.
    """
    text = re.sub(r"(?is)reasoning:\s*", "", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 5:
        sentences = sentences[:5]
    return " ".join(s.strip() for s in sentences).strip()


# ==============================
# PIPELINE
# ==============================

@dataclass
class RunRecord:
    post_title: str
    variant: str
    root_comment: str
    local_summary: str
    global_summary: str
    thread_id: str = "thread_0"  # you can extend if you batch multiple threads


def summarize_thread_with_variant(gen, data: Dict, variant: str) -> Dict:
    """
    Executes local + global summarization for a single variant.
    """
    comments = data["comments"]
    root_comments = [c for c in comments if c.get("depth", 0) == 0]

    local_summaries: List[str] = []
    per_root: List[Dict[str, str]] = []

    for root in root_comments:
        replies = [c["text"] for c in comments if c.get("parent_id") == root["author"]]
        replies_txt = "\n".join(f"- {r}" for r in replies) if replies else "(no replies)"

        local_prompt = build_local_prompt(variant, root["text"], replies_txt)
        local_raw = generate_text(gen, local_prompt)

        if variant in {"zero_shot", "few_shot", "role_play"}:
            local_sum = extract_after_marker(local_raw, "Summary:")
        else:
            tail = extract_after_marker(local_raw, "Summary:")
            local_sum = tail if tail else local_raw

        local_sum = clean_summary(local_sum)
        local_summaries.append(local_sum)
        per_root.append({"root_comment": root["text"], "summary": local_sum})

    global_prompt = build_global_prompt(variant, local_summaries)
    global_raw = generate_text(gen, global_prompt)

    if variant in {"zero_shot", "few_shot", "role_play"}:
        global_sum = extract_after_marker(global_raw, "Final Summary:")
        if not global_sum:
            global_sum = extract_after_marker(global_raw, "Summary:")
    else:
        tail = extract_after_marker(global_raw, "Final Summary:")
        global_sum = tail if tail else global_raw

    global_sum = clean_summary(global_sum)

    return {
        "post_title": data.get("post_title", ""),
        "variant": variant,
        "local_summaries": per_root,
        "global_summary": global_sum,
    }


# ==============================
# MAIN
# ==============================

def main():
    input_path = Path("data/llm_inputs/llm_input.json")
    logging.info(f"Loading input: {input_path}")
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))

    gen = load_falcon_generator()

    all_results: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []

    for variant in PROMPT_VARIANTS:
        logging.info(f"Running variant: {variant}")
        res = summarize_thread_with_variant(gen, data, variant)
        all_results.append(res)

        # flatten per-root for CSV
        for item in res["local_summaries"]:
            flat_rows.append({
                "post_title": res["post_title"],
                "variant": variant,
                "level": "local",
                "root_comment": item["root_comment"],
                "summary": item["summary"]
            })

        flat_rows.append({
            "post_title": res["post_title"],
            "variant": variant,
            "level": "global",
            "root_comment": "",
            "summary": res["global_summary"]
        })

    # Save artifacts
    runs_json = OUTPUT_DIR / "runs.json"
    runs_csv = OUTPUT_DIR / "runs.csv"
    Path(runs_json).write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(flat_rows).to_csv(runs_csv, index=False, encoding="utf-8")
    logging.info(f"Saved: {runs_json}")
    logging.info(f"Saved: {runs_csv}")

    logging.info(f"Experiment complete. Artifacts under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()