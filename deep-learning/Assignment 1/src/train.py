from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# KONFIGURASI
EXCEL_PATH = Path(__file__).resolve().parent.parent / "data" / "PMM-TemplateSLP_update_fixed.xlsx"
DATA_SHEET = "Data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

N_TRAIN_PER_CLASS = 40   # 40 Setosa + 40 Versicolor = 80 baris training
N_VAL_PER_CLASS = 10     # 10 Setosa + 10 Versicolor = 20 baris validasi

LEARNING_RATE = 0.1
N_EPOCHS = 5

# DATA: baca Excel + split train/validation

def load_data(path: Path = EXCEL_PATH) -> pd.DataFrame:
    """Baca sheet Data langsung dari file Excel pakai pandas."""
    df = pd.read_excel(path, sheet_name=DATA_SHEET, header=None,
                        names=["X1", "X2", "X3", "X4", "label"])
    df["target"] = (df["label"] != "Iris-setosa").astype(int)  # 0=setosa, 1=versicolor
    return df


def split_train_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    setosa = df[df["label"] == "Iris-setosa"].reset_index(drop=True)
    versicolor = df[df["label"] == "Iris-versicolor"].reset_index(drop=True)

    train_df = pd.concat(
        [setosa.iloc[:N_TRAIN_PER_CLASS], versicolor.iloc[:N_TRAIN_PER_CLASS]],
        ignore_index=True,
    )
    val_df = pd.concat(
        [
            setosa.iloc[N_TRAIN_PER_CLASS : N_TRAIN_PER_CLASS + N_VAL_PER_CLASS],
            versicolor.iloc[N_TRAIN_PER_CLASS : N_TRAIN_PER_CLASS + N_VAL_PER_CLASS],
        ],
        ignore_index=True,
    )
    return train_df, val_df

# MODEL: SLP pakai numpy
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def init_weights(value: float = 0.5) -> np.ndarray:
    """Vektor bobot awal [bias, teta1, teta2, teta3, teta4]."""
    return np.full(5, value, dtype=float)

def to_feature_vector(row) -> np.ndarray:
    """row: pandas Series dgn kolom X1..X4 -> np.array([1, X1, X2, X3, X4])."""
    return np.array([1.0, row["X1"], row["X2"], row["X3"], row["X4"]])

@dataclass
class StepResult:
    z: float
    g: float
    pred: int
    error: float
    sse: float
    correct: bool
    grad: np.ndarray  # [dbias, dteta1, dteta2, dteta3, dteta4]

def forward(weights: np.ndarray, x: np.ndarray, target: float) -> StepResult:
    """
    z       = bias + teta1*X1 + teta2*X2 + teta3*X3 + teta4*X4
    g(z)    = sigmoid(z)
    error   = g(z) - target
    dbias   = 2 * error * (1 - g(z)) * g(z)
    dteta_i = dbias * X_i
    """
    z = float(np.dot(weights, x))
    g = sigmoid(z)
    pred = 1 if g > 0.5 else 0
    error = g - target
    sse = error ** 2
    correct = pred == int(target)

    dbias_scalar = 2 * error * (1 - g) * g
    grad = dbias_scalar * x  # otomatis [dbias, dteta1..dteta4] karena x[0]=1

    return StepResult(z, g, pred, error, sse, correct, grad)

def update_weights(weights: np.ndarray, step: StepResult, lr: float) -> np.ndarray:
    return weights - lr * step.grad

# TRAINING LOOP
def run_epoch(weights: np.ndarray, df: pd.DataFrame, lr: float) -> tuple[np.ndarray, float, float]:
    """Jalankan satu epoch (training ATAU validasi) sample-demi-sample.
    Mengembalikan bobot akhir, rata-rata SSE, dan akurasi epoch tsb."""
    w = weights.copy()
    total_sse = 0.0
    total_correct = 0
    n = len(df)

    for _, row in df.iterrows():
        x = to_feature_vector(row)
        step = forward(w, x, row["target"])
        total_sse += step.sse
        total_correct += int(step.correct)
        w = update_weights(w, step, lr)

    return w, total_sse / n, total_correct / n

def train_and_validate(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        n_epochs: int = N_EPOCHS, lr: float = LEARNING_RATE) -> list[dict]:

    weights = init_weights(0.5)
    results = []

    for epoch in range(1, n_epochs + 1):
        weights, train_loss, train_acc = run_epoch(weights, train_df, lr)
        _, val_loss, val_acc = run_epoch(weights, val_df, lr) 

        results.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    return results

# OUTPUT: CSV + chart (loss & akurasi, training+validasi digabung 1 grafik)
def save_csv(results: list[dict], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Tersimpan: {path}")

def save_metric_chart(results: list[dict], train_key: str, val_key: str,
                       title: str, y_label: str, path: Path, as_percent: bool = False):
    epochs = [r["epoch"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [r[train_key] for r in results], marker="o", label="Training")
    plt.plot(epochs, [r[val_key] for r in results], marker="o", label="Validasi")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.xticks(epochs, [f"Epoch {e}" for e in epochs])
    plt.ylim(bottom=0) 
    if as_percent:
        plt.gca().yaxis.set_major_formatter(lambda y, _: f"{y*100:.0f}%")
        plt.ylim(top=1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Tersimpan: {path}")

# MAIN
def main():
    df = load_data()
    print("5 baris pertama dataset:")
    print(df.head())

    train_df, val_df = split_train_validation(df)
    print(f"\nTotal training  : {len(train_df)} baris")
    print(f"Total validation: {len(val_df)} baris\n")

    results = train_and_validate(train_df, val_df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_csv(results, OUTPUT_DIR / "results.csv")
    save_metric_chart(
        results, "train_loss", "val_loss",
        "Grafik Loss (Rata-rata SSE) per Epoch", "Rata-rata SSE",
        OUTPUT_DIR / "loss_chart.png",
    )
    save_metric_chart(
        results, "train_acc", "val_acc",
        "Grafik Akurasi per Epoch", "Akurasi",
        OUTPUT_DIR / "accuracy_chart.png", as_percent=True,
    )

if __name__ == "__main__":
    main()