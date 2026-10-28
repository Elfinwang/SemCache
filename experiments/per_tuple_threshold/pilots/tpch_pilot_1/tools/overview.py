"""One line per review: design columns, cached answer, threshold, and where the answer flipped.

    .venv/bin/python <tools>/overview.py                # all 30
    .venv/bin/python <tools>/overview.py --truth positive --clarity low
"""

import argparse

from common import load_records, load_reviews

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--truth", choices=["positive", "negative"])
parser.add_argument("--clarity", choices=["high", "medium", "low"])
parser.add_argument("--text", action="store_true", help="print each review's text under its row")
args = parser.parse_args()

reviews = load_reviews()
CATS = ("paraphrase", "intensity", "should_reuse", "should_not_reuse", "unrelated")
SHORT = ("para", "inten", "reuse", "no_reuse", "unrel")
print("flips = answers that differ from the cached one, as flipped/asked per question type")
print(f"{'tuple':14} {'truth':8} {'clarity':7} {'cached':6} {'t(d)':>5}  " + " ".join(f"{s:>8}" for s in SHORT))
for record in load_records():
    row = reviews.loc[record["tuple_id"]]
    if (args.truth and row.truth != args.truth) or (args.clarity and row.clarity != args.clarity):
        continue
    t = record["threshold"]["t_hat"]
    cells = []
    for cat in CATS:
        asked = [o for o in record["observations"] if o["category"] == cat]
        cells.append(f"{sum(not o['match'] for o in asked)}/{len(asked)}")
    print(f"{record['tuple_id']:14} {row.truth:8} {row.clarity:7} {record['cached_answer']:6} "
          f"{'-' if t is None else f'{t:.2f}':>5}  " + " ".join(f"{c:>8}" for c in cells))
    if args.text:
        print(f"    \"{row.text}\"\n")
