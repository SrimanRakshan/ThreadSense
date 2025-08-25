import json

with open("/home/Zoro/Documents/Projects/ThreadSense/data/threads/top_comment_thread.json", "r", encoding="utf-8") as i:
    data = json.load(i)

flattenedComments = []

def traverse_comments(comment, parent_id=None):
    comment_obj = {
        "author": comment.get("author"),
        "text": comment.get("body"),
        "score": comment.get("score"),
        "depth": comment.get("depth", 0),
        "parent_id": parent_id
    }
    flattenedComments.append(comment_obj)
    for reply in comment.get("replies", []):
        traverse_comments(reply, parent_id=comment_obj["author"])

traverse_comments(data["selected_comment"])
print(f"Total comments: {len(flattenedComments)}")
print(flattenedComments)