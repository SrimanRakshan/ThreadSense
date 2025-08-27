import json
from pathlib import Path
from transformers import pipeline

def extract_all_comments(comment):
    texts = [comment.get("body", "")]
    for reply in comment.get("replies", []):
        texts.extend(extract_all_comments(reply))
    return texts

def extract_full_thread_text(thread):
    title = thread.get("post_title", "")
    selected_comment = thread.get("selected_comment", {})
    all_comments = extract_all_comments(selected_comment)
    full_text = title + "\n" + "\n".join(all_comments)
    return full_text

def load_falcon():
    generator = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
    return generator

def generate_summary(generator, text):
    prompt = (
        "Summarize the following Reddit thread conversation in 3-5 sentences:\n\n"
        f"{text}\n\nSummary:"
    )
    output = generator(prompt)
    # Falcon outputs list of dicts with 'generated_text'
    return output[0]['generated_text'].strip()

def main():
    input_path = Path("data/threads/top_comment_thread.json")  # Update path if needed
    output_path = Path("outputs/thread_summary.json")

    # Load thread data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract full thread text
    full_thread_text = extract_full_thread_text(data)

    print("Loaded and extracted full thread text. Generating summary...")

    # Load Falcon model
    generator = load_falcon()

    # Generate summary
    summary = generate_summary(generator, full_thread_text)

    # Prepare output dict
    results = {
        "post_title": data.get("post_title", ""),
        "summary": summary
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save summary to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to {output_path}")

if __name__ == "__main__":
    main()
