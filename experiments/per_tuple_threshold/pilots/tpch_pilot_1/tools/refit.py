"""Re-fit every review's threshold from the saved answers, leaving out some question types.

Uses the same estimator as the pipeline (vCache's logistic fit) on the cached
observations, so no LLM calls are made. Prints the by-clarity table for the
original fit and the re-fit side by side.

    .venv/bin/python <tools>/refit.py                                   # drop unrelated questions
    .venv/bin/python <tools>/refit.py --drop unrelated should_not_reuse
    .venv/bin/python <tools>/refit.py --drop unrelated --tuple tpch:pilot:23
"""

import argparse
from collections import defaultdict
from statistics import mean

from common import load_records, load_reviews

from per_tuple_threshold.calibration.threshold import estimate_threshold

CATEGORIES = ("paraphrase", "intensity", "should_reuse", "should_not_reuse", "unrelated")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--drop", nargs="+", default=["unrelated"], choices=CATEGORIES,
                    help="question categories to leave out of the fit (default: unrelated)")
parser.add_argument("--tuple", help="also print the before/after for one review")
args = parser.parse_args()


def fit(observations, drop=()):
    kept = [o for o in observations if o["role"] == "fit" and o["category"] not in drop]
    return estimate_threshold([o["similarity"] for o in kept], [int(o["match"]) for o in kept])


reviews = load_reviews()
groups = defaultdict(lambda: {"before": [], "after": [], "n": 0, "fit_before": 0, "fit_after": 0})
for record in load_records():
    row = reviews.loc[record["tuple_id"]]
    before, after = fit(record["observations"]), fit(record["observations"], args.drop)
    for key in (row.clarity, f"{row.truth}/{row.clarity}"):
        g = groups[key]
        g["n"] += 1
        if before.t_hat is not None:
            g["fit_before"] += 1
            g["before"].append(before.t_hat)
        if after.t_hat is not None:
            g["fit_after"] += 1
            g["after"].append(after.t_hat)
    if args.tuple == record["tuple_id"]:
        print(f"{args.tuple}: before t={before.t_hat} ({before.reason})  ->  after t={after.t_hat} ({after.reason})\n")

fmt = lambda values: f"{mean(values):.3f}" if values else "  -  "
print(f"dropping: {', '.join(args.drop)}\n")
print(f"{'group':18} {'n':>3}  {'fitted before':>13} {'mean t before':>13}   {'fitted after':>12} {'mean t after':>12}")
for key in ["high", "medium", "low", *sorted(k for k in groups if "/" in k)]:
    if key in groups:
        g = groups[key]
        print(f"{key:18} {g['n']:>3}  {g['fit_before']:>13} {fmt(g['before']):>13}   {g['fit_after']:>12} {fmt(g['after']):>12}")
