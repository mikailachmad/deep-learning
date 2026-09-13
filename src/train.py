from __future__ import annotations
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from data_loader import (
    build_training_indices,
    core_training_index_cycle,
    full_validation_indices,
    held_out_validation_indices,
    load_raw,
)
from model import SLPWeights, forward, update_weights

LEARNING_RATE = 0.1
EPOCH_SIZES_TRAIN = [80, 100, 100, 100, 100]  # baris per epoch di SLP-Training
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

def run_epoch(weights: SLPWeights, data: list[dict], indices: list[int], lr: float):
    w = weights
    total_sse = 0.0
    total_correct = 0
    n = len(indices)

    for idx in indices:
        row = data[idx]
        step = forward(w, row["X1"], row["X2"], row["X3"], row["X4"], row["target"])
        total_sse += step.sse
        total_correct += int(step.correct)
        w = update_weights(w, step, lr)

    avg_sse = total_sse / n
    accuracy = total_correct / n
    return w, avg_sse, accuracy

def main():
    data = load_raw()
    core_cycle = core_training_index_cycle(data)

    total_train_rows = sum(EPOCH_SIZES_TRAIN)
    train_indices_all = build_training_indices(total_train_rows, core_cycle)

    held_out_idx = held_out_validation_indices()
    full_idx = full_validation_indices()

    train_weights = SLPWeights()  # bias = teta1..4 = 0.5
    results = []

    pos = 0
    for epoch, size in enumerate(EPOCH_SIZES_TRAIN, start=1):
        epoch_train_idx = train_indices_all[pos : pos + size]
        pos += size

        # Training (bobot lanjut, tidak reset)
        train_weights, train_loss, train_acc = run_epoch(
            train_weights, data, epoch_train_idx, LEARNING_RATE
        )

        # Validasi (bobot dicabang dari training epoch ini)
        val_idx = held_out_idx if epoch in (1, 2) else full_idx
        _, val_loss, val_acc = run_epoch(
            train_weights.copy(), data, val_idx, LEARNING_RATE
        )

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

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_csv(results)
    save_loss_chart(results)
    save_accuracy_chart(results)

def save_csv(results: list[dict]):
    path = OUTPUT_DIR / "results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Tersimpan: {path}")

def save_loss_chart(results: list[dict]):
    epochs = [r["epoch"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [r["train_loss"] for r in results], marker="o", label="Training")
    plt.plot(epochs, [r["val_loss"] for r in results], marker="o", label="Validasi")
    plt.title("Grafik Loss (Rata-rata SSE) per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Rata-rata SSE")
    plt.xticks(epochs, [f"Epoch {e}" for e in epochs])
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "loss_chart.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Tersimpan: {path}")

def save_accuracy_chart(results: list[dict]):
    epochs = [r["epoch"] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [r["train_acc"] for r in results], marker="o", label="Training")
    plt.plot(epochs, [r["val_acc"] for r in results], marker="o", label="Validasi")
    plt.title("Grafik Akurasi per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Akurasi")
    plt.xticks(epochs, [f"Epoch {e}" for e in epochs])
    plt.gca().yaxis.set_major_formatter(lambda y, _: f"{y*100:.0f}%")
    plt.legend()
    plt.tight_layout()
    path = OUTPUT_DIR / "accuracy_chart.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Tersimpan: {path}")

if __name__ == "__main__":
    main()