"""
Builds a demo pair for generic_prep.py:
  data/penguins.csv     -- clean Palmer Penguins (dropna'd baseline)
  data/messy_demo.csv   -- the same data deliberately dirtied:
                             - 10-20% missing scattered into numeric AND
                               categorical feature columns
                             - a useless all-unique ID column
                             - a high-cardinality "observer_name" text column
                               (large-but-bounded name pool, so it has some
                               repeats -- exercises the "high cardinality,
                               not literally all-unique" drop rule distinctly
                               from the ID column's "all-unique" drop rule)

Run: python scripts/make_messy_demo.py
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent
RNG = np.random.default_rng(42)

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

clean = pd.read_csv(URL).dropna().reset_index(drop=True)
clean.to_csv(ROOT / "data" / "penguins.csv", index=False)
print(f"Clean baseline saved -> data/penguins.csv  ({len(clean)} rows, {len(clean.columns)} cols)")

messy = clean.copy()
n = len(messy)


def inject_missing(col: str, frac: float) -> None:
    idx = RNG.choice(n, size=int(n * frac), replace=False)
    messy.loc[idx, col] = np.nan


# --- (1) scatter missing values into numeric AND categorical columns -------
inject_missing("bill_length_mm", 0.15)
inject_missing("bill_depth_mm", 0.12)
inject_missing("body_mass_g", 0.18)
inject_missing("island", 0.10)
inject_missing("sex", 0.20)

# --- (2) useless all-unique ID column ---------------------------------------
messy.insert(0, "record_id", [f"REC-{i:04d}" for i in range(n)])

# --- (3) high-cardinality text column (bounded pool -> mostly-but-not-all
#         unique, so it lands in the "high cardinality" drop rule rather
#         than the "all-unique ID-like" drop rule) --------------------------
first_names = [
    "Amara", "Noah", "Elin", "Kaito", "Priya", "Leif", "Zara", "Matteo",
    "Ingrid", "Kofi", "Yuki", "Sana", "Owen", "Freya", "Diego", "Anke",
    "Rhys", "Nadia", "Soren", "Talia",
]
last_names = [
    "Berg", "Okafor", "Lindqvist", "Suzuki", "Kapoor", "Hoffmann", "Reyes",
    "Nakamura", "Andersen", "Mensah", "Costa", "Petrov", "Alden", "Farrow",
    "Dubois", "Lindgren", "Osei", "Marchetti", "Voss", "Halden", "Brandt",
    "Kessler", "Ravn", "Solberg", "Ademola",
]
name_pool = [f"{f} {l}" for f in first_names for l in last_names]  # 500 combos
messy["observer_name"] = RNG.choice(name_pool, size=n, replace=True)

messy.to_csv(ROOT / "data" / "messy_demo.csv", index=False)

print(f"Messy demo saved    -> data/messy_demo.csv  ({len(messy)} rows, {len(messy.columns)} cols)")
print(f"\nColumns: {list(messy.columns)}")
print("\nMissing value counts:")
print(messy.isna().sum()[messy.isna().sum() > 0])
print(f"\nrecord_id  unique: {messy['record_id'].nunique()} / {n}")
print(f"observer_name unique: {messy['observer_name'].nunique()} / {n}")
