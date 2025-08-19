#!/usr/bin/env python3
"""
Reddit Thread Keyword Extraction Pipeline
-----------------------------------------
- Handles both nested JSON and pre-flattened comment data
- Generates sentence embeddings
- Extracts key phrases per comment using KeyBERT
- Stores results in CSV for downstream summarization
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


# ==============================
# CONFIGURATION
# ==============================
class Config:
    # File paths
    INPUT_FILE: str = "top_comment_thread.json"
    OUTPUT_FILE: str = "data/keybert_results.csv"

    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # KeyBERT parameters
    NGRAM_RANGE: tuple = (1, 2)
    TOP_N: int = 10
    STOP_WORDS: str = "english"


# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)


# ==============================
# DATA LOADER & FLATTENER
# ==============================
def load_data(file_path: Union[str, Path]) -> Any:
    """Load data from JSON or CSV."""
    file_path = Path("data/top_comment_thread.json")
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif file_path.suffix.lower() in {".csv"}:
        return pd.read_csv(file_path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def flatten_json_thread(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten nested Reddit thread JSON into a list of comments."""
    flattened_comments: List[Dict[str, Any]] = []

    def traverse(comment: Dict[str, Any], parent_id: Optional[str] = None):
        flattened_comments.append({
            "author": comment.get("author"),
            "text": comment.get("body") or comment.get("text"),
            "score": comment.get("score", 0),
            "depth": comment.get("depth", 0),
            "parent_id": parent_id
        })
        for reply in comment.get("replies", []):
            traverse(reply, parent_id=comment.get("author"))

    # Detect JSON structure
    if "selected_comment" in data:
        traverse(data["selected_comment"])
    elif isinstance(data, list):
        return data  # Already flattened
    else:
        raise ValueError("Invalid JSON structure: cannot find 'selected_comment'.")

    return flattened_comments


# ==============================
# KEYBERT PROCESSOR
# ==============================
class KeyBERTProcessor:
    def __init__(self, embedding_model: str, ngram_range: tuple, top_n: int, stop_words: str):
        self.embedding_model_name = embedding_model
        logging.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.kw_model = KeyBERT(model=self.embedding_model)
        self.ngram_range = ngram_range
        self.top_n = top_n
        self.stop_words = stop_words

    def process_comments(self, comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings & extract keywords for each comment."""
        for comment in comments:
            text = comment.get("text", "").strip()
            if not text:
                comment["keywords"] = []
                comment["embedding"] = None
                continue

            # Generate embedding
            comment["embedding"] = self.embedding_model.encode(text).tolist()

            # Extract keywords
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=self.ngram_range,
                stop_words=self.stop_words,
                top_n=self.top_n
            )
            comment["keywords"] = [kw for kw, _ in keywords]

        return comments


# ==============================
# MAIN PIPELINE
# ==============================
def main():
    logging.info("Loading data...")
    raw_data = load_data(Config.INPUT_FILE)

    logging.info("Flattening data...")
    comments = flatten_json_thread(raw_data)

    logging.info(f"Total comments: {len(comments)}")

    processor = KeyBERTProcessor(
        embedding_model=Config.EMBEDDING_MODEL,
        ngram_range=Config.NGRAM_RANGE,
        top_n=Config.TOP_N,
        stop_words=Config.STOP_WORDS
    )

    logging.info("Processing comments with KeyBERT...")
    processed_comments = processor.process_comments(comments)

    logging.info(f"Saving results to {Config.OUTPUT_FILE}...")
    df = pd.DataFrame(processed_comments)
    df.to_csv(Config.OUTPUT_FILE, index=False)

    logging.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
