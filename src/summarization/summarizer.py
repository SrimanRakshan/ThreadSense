"""
Summarization Pipeline (LangChain + Falcon Local)
-------------------------------------------------
- Runs fully locally with Falcon-7B-Instruct via HuggingFace Transformers
- Loads structured Reddit thread (JSON)
- Defines Chain-of-Thought style summarization prompts
- Runs local + global summarization
- Outputs structured JSON summaries
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline

# Transformers
from transformers import pipeline


# ==============================
# LLM LOADER (Falcon only)
# ==============================
def load_llm() -> Any:
    """Load Falcon-7B-Instruct locally using HuggingFace pipeline."""
    generator = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        device_map="auto",    # put layers on GPU if available
        torch_dtype="auto"    # automatically choose float16/32
    )
    return HuggingFacePipeline(pipeline=generator)


# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)


# ==============================
# SUMMARIZATION PIPELINE
# ==============================
class ThreadSummarizer:
    def __init__(self):
        """Initialize summarizer with Falcon LLM."""
        self.llm = load_llm()

        # Local summary chain
        self.local_prompt = PromptTemplate(
            input_variables=["root_comment", "replies"],
            template=(
                "Root Comment: {root_comment}\n"
                "Replies:\n{replies}\n\n"
                "Step 1: Identify main discussion points from this thread.\n"
                "Step 2: Summarize in 2-3 sentences.\n\n"
                "Summary:"
            )
        )
        self.local_chain = LLMChain(llm=self.llm, prompt=self.local_prompt, output_key="local_summary")

        # Global summary chain
        self.global_prompt = PromptTemplate(
            input_variables=["local_summaries"],
            template=(
                "You are given partial summaries of different discussion branches:\n"
                "{local_summaries}\n\n"
                "Step 1: Identify overlapping ideas.\n"
                "Step 2: Merge into a single, coherent overall summary (max 5 sentences).\n\n"
                "Final Summary:"
            )
        )
        self.global_chain = LLMChain(llm=self.llm, prompt=self.global_prompt, output_key="global_summary")

    def summarize(self, data: Dict) -> Dict:
        """Run the full summarization pipeline."""
        comments = data["comments"]

        # Group by root comments (depth=0)
        root_comments = [c for c in comments if c["depth"] == 0]
        results = {"post_title": data["post_title"], "local_summaries": []}

        logging.info(f"Processing {len(root_comments)} root-level discussions...")

        for root in root_comments:
            replies = [c["text"] for c in comments if c.get("parent_id") == root["author"]]

            # Local summary
            summary = self.local_chain.run(root_comment=root["text"], replies="\n".join(replies))
            results["local_summaries"].append({"root_comment": root["text"], "summary": summary})

        # Global summary
        local_text = "\n".join([ls["summary"] for ls in results["local_summaries"]])
        global_summary = self.global_chain.run(local_summaries=local_text)
        results["global_summary"] = global_summary

        return results


# ==============================
# MAIN
# ==============================
def main():
    input_path = Path("data/llm_inputs/llm_input.json")
    output_path = Path("outputs/summaries.json")

    logging.info(f"Loading input from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summarizer = ThreadSummarizer()
    results = summarizer.summarize(data)

    logging.info(f"Saving output to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
