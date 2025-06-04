import praw
import pandas as pd

# Replace these with your Reddit API credentials
reddit = praw.Reddit(
    client_id="7UuROX50vS0fTMbAx2VW1g",
    client_secret="JCLBUI9Om4qbE7iTwJ0Up5rekG6n3Q",
    user_agent="bookmark-training-script"
)

# Subreddits to pull from and their category label
subreddits = {
    "recipes": "cooking",
    "learnprogramming": "programming",
    "fashion": "fashion",
    "fitness": "fitness",
}

data = []

for subreddit_name, label in subreddits.items():
    subreddit = reddit.subreddit(subreddit_name)
    print(f"Fetching from r/{subreddit_name}...")

    for post in subreddit.top(limit=100):
        if not post.stickied and not post.over_18:
            data.append({
                "title": post.title,
                "label": label
            })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("reddit_bookmark_data.csv", index=False)
print(f"✅ Saved {len(df)} posts to reddit_bookmark_data.csv")
