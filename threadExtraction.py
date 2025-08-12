import praw
import json
from praw.models import MoreComments

reddit = praw.Reddit(
    client_id="AEL7PpAUiOtlFdsTa5ZK-w",
    client_secret="MlOJ0isxgAXd5FKaQlqYOjz6HbLqWw",
    user_agent="ThreadSummarizer by /u/Frequent-Royal-6173",
    username="Frequent-Royal-6173",
    password="Rakshan@1234"
)

# Recursive reply extractor
def extract_replies(comment, depth=0):
    thread = {
        'author': str(comment.author),
        'body': comment.body,
        'score': comment.score,
        'depth': depth,
        'replies': []
    }
    for reply in comment.replies:
        thread['replies'].append(extract_replies(reply, depth + 1))
    return thread

# Automatically select the top comment by upvotes
def extract_top_comment_thread(submission_url=None, submission_id=None):
    if submission_url:
        submission = reddit.submission(url=submission_url)
    elif submission_id:
        submission = reddit.submission(id=submission_id)
    else:
        raise ValueError("Provide either a submission URL or submission ID.")

    submission.comments.replace_more(limit=None)
    top_comments = submission.comments[:]

    # Automatically pick the highest upvoted top-level comment
    best_comment = max(top_comments, key=lambda c: c.score)

    thread = {
        'post_title': submission.title,
        'post_url': submission.url,
        'selected_comment': {
            'author': str(best_comment.author),
            'body': best_comment.body,
            'score': best_comment.score,
            'replies': []
        }
    }

    for reply in best_comment.replies:
        thread['selected_comment']['replies'].append(extract_replies(reply, depth=1))

    return thread

# ------ Use Case ------
submission_url = "https://www.reddit.com/r/AskReddit/comments/1m5z250/whats_a_completely_legal_action_that_would/" 

# Extract and save
result = extract_top_comment_thread(submission_url=submission_url)

with open("top_comment_thread.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("✅ Top comment thread saved to top_comment_thread.json")
