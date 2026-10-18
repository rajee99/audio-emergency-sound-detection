import csv
import json
import os

METADATA_DIR = r"d:\ml\fydp\data\metadata"

with open(os.path.join(METADATA_DIR, "ontology.json"), "r", encoding="utf-8") as f:
    ontology = json.load(f)
ont_by_id = {item["id"]: item for item in ontology}

with open(os.path.join(METADATA_DIR, "class_labels_indices.csv"), "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)
    mid_to_name = {row[1]: row[2] for row in reader}

def parse_segments(filename):
    segments = []
    with open(os.path.join(METADATA_DIR, filename), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            # Format: YTID, start_seconds, end_seconds, "label1,label2"
            ytid = parts[0]
            start = float(parts[1])
            end = float(parts[2])
            # The remaining elements joined are labels
            labels_raw = ",".join(parts[3:]).strip('" ')
            labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
            segments.append((ytid, start, end, labels))
    return segments

bal_segs = parse_segments("balanced_train_segments.csv")
eval_segs = parse_segments("eval_segments.csv")

print(f"Total balanced_train segments: {len(bal_segs)}")
print(f"Total eval segments: {len(eval_segs)}")

# Target candidate mappings
candidates = {
    "Screaming": ["/m/03qc9zr"],
    "Siren": ["/m/03kmc9", "/m/04qvtq", "/m/012n7d", "/m/012ndj", "/m/0dgbq"],
    "Alarm": ["/m/07pp_mv", "/m/0c3f7m", "/m/01y3hg", "/m/02mfyn"],
    "Glass Shatter": ["/m/07rn7sz", "/m/039jq"],
    "Smash / Crash": ["/m/07pjjrj"],
    "Explosion": ["/m/014zdl"],
    "Gunshot / Gunfire": ["/m/032s66", "/m/04zjc", "/m/02z32qm", "/m/0_1c", "/m/073cg4"],
    "Background / Normal": ["/m/0btp2", "/m/07pbtc8", "/m/09x0r", "/m/04rlf"]
}

def analyze_counts(segments, split_name):
    print(f"\n--- {split_name} Counts ---")
    category_counts = {cat: 0 for cat in candidates}
    mid_counts = {}
    
    for _, _, _, labels in segments:
        label_set = set(labels)
        for cat, mids in candidates.items():
            if any(m in label_set for m in mids):
                category_counts[cat] += 1
        for m in labels:
            mid_counts[m] = mid_counts.get(m, 0) + 1
            
    for cat, mids in candidates.items():
        print(f"\nCategory: {cat} (Total matching segments: {category_counts[cat]})")
        for m in mids:
            name = mid_to_name.get(m, ont_by_id.get(m, {}).get("name", "Unknown"))
            cnt = mid_counts.get(m, 0)
            print(f"   - {m:<12} {name:<32}: {cnt} clips")

analyze_counts(bal_segs, "Balanced Train")
analyze_counts(eval_segs, "Evaluation")
