"""
Advanced Automated Content Tagging for Blogs
--------------------------------------------
Includes Unique Features:
 1. TF-IDF Keyword Extraction
 2. RAKE-like Phrase Extraction
 3. Tag Confidence Score (normalized)
 4. Merged & Ranked Tag List
 5. Blog Category Prediction (Mini Classifier)
 6. Keyword Highlighting in Blog Text
 7. Tag Cloud Visualization

Outputs saved in: ./data/
"""

import os
import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ------------------------------------------
# 1) Create local data folder
# ------------------------------------------
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# ------------------------------------------
# 2) Sample blog dataset
# ------------------------------------------
blogs = [
    {"id": 1, "title": "Top 10 Python Tricks for Data Scientists",
     "content": ("Python remains a favorite among data scientists. "
                 "This post explores Python tricks for data manipulation, "
                 "pandas, NumPy, list comprehensions, and useful debugging tips."),
     "category": "Technology"},

    {"id": 2, "title": "A Weekend Guide to Exploring Kerala",
     "content": ("Kerala offers backwaters, tea gardens, and vibrant cuisine. "
                 "This travel guide covers best experiences in Alleppey, Munnar, and Kochi."),
     "category": "Travel"},

    {"id": 3, "title": "Healthy Smoothie Recipes for Busy Mornings",
     "content": ("Quick healthy smoothie recipes using bananas, berries, spinach, "
                 "and protein powders. Tips for balancing flavor and nutrition."),
     "category": "Food"},

    {"id": 4, "title": "Personal Finance 101: Building an Emergency Fund",
     "content": ("An emergency fund helps cover unexpected expenses. "
                 "This article explains budgeting, saving strategies, and investment basics."),
     "category": "Finance"},

    {"id": 5, "title": "Improving Mental Health with Daily Habits",
     "content": ("Daily habits like meditation, exercise, journaling, and sleep hygiene "
                 "can improve mental wellbeing over time."),
     "category": "Health"}
]

df = pd.DataFrame(blogs)
df.to_csv(f"{output_folder}/blogs.csv", index=False)

# ------------------------------------------
# 3) Preprocessing
# ------------------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', " ", text)
    return re.sub(r'\s+', " ", text).strip()

df["clean"] = df["content"].apply(preprocess)

# ------------------------------------------
# 4) TF-IDF Keyword Extraction
# ------------------------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["clean"])
feature_names = tfidf.get_feature_names_out()

def get_tfidf_tags(doc_idx, top_n=5):
    row = tfidf_matrix[doc_idx].toarray().flatten()
    idx = row.argsort()[::-1][:top_n]
    scores = [(feature_names[i], row[i]) for i in idx if row[i] > 0]
    return scores

df["tfidf_raw"] = [get_tfidf_tags(i) for i in range(len(df))]

# ------------------------------------------
# 5) RAKE-like Tag Extraction
# ------------------------------------------
stopwords = set(ENGLISH_STOP_WORDS)
pattern = re.compile(r"[a-z0-9]+")

def extract_phrases(text):
    words = pattern.findall(text)
    phrases = []
    curr = []
    for w in words:
        if w in stopwords:
            if curr:
                phrases.append(" ".join(curr))
                curr = []
        else:
            curr.append(w)
    if curr:
        phrases.append(" ".join(curr))
    return [p for p in phrases if 1 < len(p.split()) <= 4]

def rake_scores(text):
    phrases = extract_phrases(text)
    freq = Counter()
    degree = Counter()
    for p in phrases:
        words = p.split()
        deg = len(words) - 1
        for w in words:
            freq[w] += 1
            degree[w] += deg
    word_score = {w: (degree[w] + freq[w]) / freq[w] for w in freq}
    phrase_score = {p: sum(word_score[w] for w in p.split()) for p in phrases}
    return phrase_score

def rake_top(text, n=5):
    scores = rake_scores(text)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:n]

df["rake_raw"] = df["clean"].apply(rake_top)

# ------------------------------------------
# 6) Normalize Tag Confidence Scores
# ------------------------------------------
def normalize_scores(tag_pairs):
    if not tag_pairs:
        return []
    tags, scores = zip(*tag_pairs)
    scaler = MinMaxScaler()
    scores_norm = scaler.fit_transform([[s] for s in scores]).flatten()
    return list(zip(tags, scores_norm))

df["tfidf_scores"] = df["tfidf_raw"].apply(normalize_scores)
df["rake_scores"] = df["rake_raw"].apply(normalize_scores)

# ------------------------------------------
# 7) Merge + Rank Tags (Final Tag List)
# ------------------------------------------
def merge_tags(tfidf_tags, rake_tags):
    combined = {}

    for tag, score in tfidf_tags:
        combined[tag] = combined.get(tag, 0) + score

    for tag, score in rake_tags:
        combined[tag] = combined.get(tag, 0) + score

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked[:5]

df["final_tags"] = df.apply(lambda row: merge_tags(row["tfidf_scores"], row["rake_scores"]), axis=1)

# ------------------------------------------
# 8) Mini Category Prediction Model
# ------------------------------------------
X = df["clean"]
y = df["category"]

cv = CountVectorizer(stop_words="english")
X_vec = cv.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

df["predicted_category"] = model.predict(X_vec)

# ------------------------------------------
# 9) Keyword Highlighting
# ------------------------------------------
def highlight(text, tags):
    for tag, _ in tags:
        text = re.sub(fr"\b({tag})\b", r"**\1**", text, flags=re.IGNORECASE)
    return text

df["highlighted"] = df.apply(lambda row: highlight(row["content"], row["final_tags"]), axis=1)

# ------------------------------------------
# 10) Tag Cloud Generation
# ------------------------------------------
for idx, row in df.iterrows():
    tags_only = " ".join([t for t, s in row["final_tags"]])
    wc = WordCloud(width=600, height=400, background_color="white").generate(tags_only)
    plt.figure(figsize=(6, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(f"{output_folder}/tagcloud_blog_{row['id']}.png")
    plt.close()

# ------------------------------------------
# 11) Save Final Results
# ------------------------------------------
df.to_csv(f"{output_folder}/blogs_tags_advanced.csv", index=False)
print("Saved final dataset to data/blogs_tags_advanced.csv")
print("Tag cloud images saved in /data/")
