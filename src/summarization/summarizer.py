#!/usr/bin/env python3
"""
Framework-Free Summarization Pipeline
-------------------------------------
- HuggingFace for local summaries
- HuggingFace or OpenAI GPT for global summary
- Outputs structured JSON
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from transformers import pipeline
from openai import OpenAI


# ==============================
# CONFIG
# ==============================
USE_OPENAI_FOR_GLOBAL = False  # Set True if you want GPT-4 for global summary
OPENAI_MODEL = "gpt-4o-mini"   # Or "gpt-4" if you have access


# ==============================
# SUMMARIZER CLASS
# ==============================
class ThreadSummarizer:
    def __init__(self):
        """Initialize HuggingFace and optional OpenAI client."""
        logging.info("Loading HuggingFace summarizer (bart-large-cnn)...")
        self.local_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        self.client = None
        if USE_OPENAI_FOR_GLOBAL:
            logging.info("Loading OpenAI client...")
            self.client = OpenAI()

    def summarize_local(self, root_comment: str, replies: List[str]) -> str:
        """Summarize one root + replies block."""
        text = f"Root Comment: {root_comment}\nReplies:\n" + "\n".join(replies)
        result = self.local_summarizer(text, max_length=100, min_length=25, do_sample=False)
        return result[0]['summary_text']

    def summarize_global(self, local_summaries: List[str]) -> str:
        """Summarize across all local summaries."""
        combined = "\n".join(local_summaries)

        if USE_OPENAI_FOR_GLOBAL and self.client:
            prompt = f"""
You are given partial summaries of different Reddit discussion threads:

{combined}

Step 1: Identify overlapping ideas.
Step 2: Merge into a single, coherent overall summary (max 5 sentences).

Final Summary:
"""
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()

        else:
            result = self.local_summarizer(combined, max_length=150, min_length=50, do_sample=False)
            return result[0]['summary_text']

    def summarize(self, data: Dict) -> Dict:
        """Run the full summarization pipeline."""
        comments = data["comments"]
        root_comments = [c for c in comments if c["depth"] == 0]

        results = {"post_title": data["post_title"], "local_summaries": []}
        logging.info(f"Processing {len(root_comments)} root-level discussions...")

        for root in root_comments:
            replies = [c["text"] for c in comments if c.get("parent_id") == root["author"]]
            local_summary = self.summarize_local(root["text"], replies)
            results["local_summaries"].append({
                "root_comment": root["text"],
                "summary": local_summary
            })

        local_texts = [ls["summary"] for ls in results["local_summaries"]]
        results["global_summary"] = self.summarize_global(local_texts)

        return results


# ==============================
# MAIN
# ==============================
def main():
    input_path = Path("data/llm_input.json")
    output_path = Path("outputs/summaries.json")

    logging.info(f"Loading input from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summarizer = ThreadSummarizer()
    results = summarizer.summarize(data)

    logging.info(f"Saving output to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
