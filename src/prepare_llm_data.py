#!/usr/bin/env python3
"""
Data Preparation for LLM Summarization
--------------------------------------
- Converts KeyBERT CSV into structured JSON for LLM input
- Extracts embeddings into a NumPy array for clustering / analysis
"""

import json
import ast
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
class Config:
    INPUT_FILE: str = "data/keybert/keybert_results.csv"
    OUTPUT_JSON: str = "data/llm_inputs/llm_input.json"
    OUTPUT_EMB: str = "data/llm_inputs/embeddings.npy"


# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)


# ==============================
# DATA PREPARATION
# ==============================
def load_and_parse_csv(file_path: str) -> pd.DataFrame:
    """Load KeyBERT CSV and parse keywords + embeddings into real lists."""
    logging.info(f"Loading CSV: {file_path}")
    df = pd.read_csv(file_path)

    # Parse keywords + embeddings safely
    df["keywords"] = df["keywords"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    df["embedding"] = df["embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    logging.info(f"Parsed {len(df)} rows from CSV")
    return df


def export_json(df: pd.DataFrame, output_file: str):
    """Export DataFrame into LLM-friendly structured JSON."""
    logging.info(f"Exporting structured JSON: {output_file}")

    comments: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        comments.append({
            "author": row["author"],
            "text": row["text"],
            "keywords": row["keywords"],
            "score": int(row["score"]),
            "depth": int(row["depth"]),
            "parent_id": None if pd.isna(row["parent_id"]) else row["parent_id"]
        })

    data = {
        "post_title": "Reddit Thread Summarization Dataset",
        "comments": comments
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logging.info(f"JSON saved to {output_file}")


def export_embeddings(df: pd.DataFrame, output_file: str):
    """Export embeddings into .npy array for clustering."""
    logging.info(f"Exporting embeddings to {output_file}")

    embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)
    np.save(output_file, embeddings)

    logging.info(f"Embeddings saved with shape {embeddings.shape}")


# ==============================
# MAIN
# ==============================
def main():
    df = load_and_parse_csv(Config.INPUT_FILE)
    export_json(df, Config.OUTPUT_JSON)
    export_embeddings(df, Config.OUTPUT_EMB)
    logging.info("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
