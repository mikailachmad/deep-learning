from __future__ import annotations
import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iris.csv"

def load_raw(path: Path = DATA_PATH) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "X1": float(row["X1"]),
                    "X2": float(row["X2"]),
                    "X3": float(row["X3"]),
                    "X4": float(row["X4"]),
                    "target": 0.0 if row["label"] == "Iris-setosa" else 1.0,
                }
            )
    if len(rows) != 100:
        raise ValueError(f"Expected 100 baris data, dapat {len(rows)}")
    return rows

def core_training_index_cycle(data: list[dict]) -> list[int]:
    return list(range(0, 40)) + list(range(50, 90))

def build_training_indices(n_total_rows: int, core_cycle: list[int]) -> list[int]:
    cycle_len = len(core_cycle)
    return [core_cycle[i % cycle_len] for i in range(n_total_rows)]

def held_out_validation_indices() -> list[int]:
    return list(range(40, 50)) + list(range(90, 100))

def full_validation_indices() -> list[int]:
    return list(range(0, 100))