"""TPC-H line items as review subjects: facts, strata and the v1 design columns.

No LLM is involved here. Every column is copied from TPC-H, derived from it
with SQL, or drawn from a seeded random generator (``clarity`` only):

* ``stratum``: ``returned`` (``l_returnflag = 'R'``), ``late`` (not returned,
  at least ``LATE_DAYS`` late) or ``on_time`` (not returned, not late).
  Items 1 to ``LATE_DAYS - 1`` days late are never sampled.
* ``truth``: ``positive`` for ``on_time``, ``negative`` otherwise.
* ``clarity``: ``low`` / ``medium`` / ``high``, balanced within each truth and
  independent of the facts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LATE_DAYS = 7
STRATA = ("on_time", "returned", "late")
STRATA_MIX = {"on_time": 0.5, "returned": 0.25, "late": 0.25}
CLARITY_LEVELS = ("low", "medium", "high")
TRUTH_BY_STRATUM = {"on_time": "positive", "returned": "negative", "late": "negative"}

FACTS_SQL = f"""
SELECT
    'tpch:' || l.l_orderkey || ':' || l.l_linenumber AS tuple_id,
    l.l_orderkey,
    l.l_linenumber,
    l.l_partkey AS p_partkey,
    l.l_suppkey AS s_suppkey,
    o.o_custkey AS c_custkey,
    l.l_returnflag = 'R' AS returned,
    CAST(l.l_receiptdate - l.l_commitdate AS INTEGER) AS days_late,
    l.l_discount AS discount,
    round(l.l_extendedprice / l.l_quantity, 2) AS unit_price,
    lower(string_split(p.p_type, ' ')[1]) AS grade,
    lower(string_split(p.p_type, ' ')[2]) AS finish,
    lower(string_split(p.p_type, ' ')[3]) AS material,
    p.p_container AS container,
    l.l_shipmode AS ship_mode,
    CASE
        WHEN l.l_returnflag = 'R' THEN 'returned'
        WHEN l.l_receiptdate - l.l_commitdate >= {LATE_DAYS} THEN 'late'
        WHEN l.l_receiptdate - l.l_commitdate <= 0 THEN 'on_time'
    END AS stratum
FROM lineitem AS l
JOIN part AS p ON p.p_partkey = l.l_partkey
JOIN orders AS o ON o.o_orderkey = l.l_orderkey
"""


def build_database(path: Path, scale_factor: float) -> None:
    """Generate TPC-H into a DuckDB file. Needs internet for ``INSTALL tpch``."""
    import duckdb

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as con:
        con.execute("INSTALL tpch")
        con.execute("LOAD tpch")
        con.execute(f"CALL dbgen(sf = {float(scale_factor)})")


def load_facts(path: Path) -> pd.DataFrame:
    """One row of facts per line item that falls into a stratum."""
    import duckdb

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Build it first on a machine with internet access:\n"
            "  python -m per_tuple_threshold.data.tpch_reviews.generate --dbgen-only --out <dir>"
        )
    # Plain tables after dbgen: no extension is loaded, so this works offline on GPU nodes.
    with duckdb.connect(str(path), read_only=True) as con:
        frame = con.execute(FACTS_SQL).fetchdf()
    return frame.dropna(subset=["stratum"]).reset_index(drop=True)


def stratum_counts(n: int) -> dict[str, int]:
    """Split ``n`` over the strata by ``STRATA_MIX``; rounding goes to ``on_time``."""
    counts = {name: int(n * STRATA_MIX[name]) for name in STRATA if name != "on_time"}
    counts["on_time"] = n - sum(counts.values())
    return {name: counts[name] for name in STRATA}


def sample_items(facts: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Seeded stratified sample of ``n`` line items."""
    parts = []
    for offset, (name, count) in enumerate(stratum_counts(n).items()):
        pool = facts[facts["stratum"] == name]
        if len(pool) < count:
            raise ValueError(f"Stratum {name!r} has {len(pool)} items, {count} requested; raise --sf.")
        parts.append(pool.sample(n=count, random_state=seed + offset))
    return pd.concat(parts, ignore_index=True)


def assign_design(items: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add ``truth`` (from the stratum) and ``clarity`` (balanced within each truth)."""
    rng = np.random.default_rng(seed)
    items = items.copy()
    items["truth"] = items["stratum"].map(TRUTH_BY_STRATUM)
    items["clarity"] = ""
    for _, group in items.groupby("truth"):
        levels = np.resize(np.array(CLARITY_LEVELS), len(group))
        items.loc[group.index, "clarity"] = rng.permutation(levels)
    return items
