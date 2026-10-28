"""One review, question by question, sorted by similarity to the original question.

    .venv/bin/python <tools>/review.py tpch:pilot:23
"""

import argparse

from common import load_records, load_reviews

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("tuple_id", help="e.g. tpch:pilot:23 (00-29)")
parser.add_argument("--full", action="store_true", help="print full question text (default: first 80 chars)")
args = parser.parse_args()
width = None if args.full else 80

records = {r["tuple_id"]: r for r in load_records()}
if args.tuple_id not in records:
    raise SystemExit(f"unknown id {args.tuple_id}; choose from {sorted(records)[0]} .. {sorted(records)[-1]}")
record, row = records[args.tuple_id], load_reviews().loc[args.tuple_id]

print(f"[{row.truth} / {row.clarity} / {row.stratum}]\n{row.text}\n")
print(f"cached answer to 'clearly positive?': {record['cached_answer']}   "
      f"threshold: {record['threshold']['t_hat']} ({record['threshold']['reason']})\n")
for o in sorted(record["observations"], key=lambda o: -o["similarity"]):
    print(f"{o['similarity']:.2f}  {o['role']:4}  {o['category']:16} {o['answer']:3} "
          f"{'✓' if o['match'] else '✗'}  {o['question'][:width]}")
