import os
import csv
import json
import requests

DATA_DIR = r"d:\ml\fydp\data\metadata"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Download class_labels_indices.csv
labels_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
labels_file = os.path.join(DATA_DIR, "class_labels_indices.csv")
if not os.path.exists(labels_file):
    print(f"Downloading {labels_url}...")
    r = requests.get(labels_url)
    with open(labels_file, "w", encoding="utf-8") as f:
        f.write(r.text)

# 2. Download ontology.json
ontology_url = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
ontology_file = os.path.join(DATA_DIR, "ontology.json")
if not os.path.exists(ontology_file):
    print(f"Downloading {ontology_url}...")
    r = requests.get(ontology_url)
    with open(ontology_file, "w", encoding="utf-8") as f:
        f.write(r.text)

# 3. Download balanced_train_segments.csv
bal_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
bal_file = os.path.join(DATA_DIR, "balanced_train_segments.csv")
if not os.path.exists(bal_file):
    print(f"Downloading {bal_url}...")
    r = requests.get(bal_url)
    with open(bal_file, "w", encoding="utf-8") as f:
        f.write(r.text)

# 4. Download eval_segments.csv
eval_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
eval_file = os.path.join(DATA_DIR, "eval_segments.csv")
if not os.path.exists(eval_file):
    print(f"Downloading {eval_url}...")
    r = requests.get(eval_url)
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write(r.text)

print("Downloaded core metadata successfully!")
