import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from ml.model import GestureMLP, GESTURE_CLASSES
from ml.dataset import GestureDataset, load_splits
MODEL_OUT = "models/gesture_mlp.pt"
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 10  # early stopping


def train_mlp(splits):
    train_ds = GestureDataset(splits["X_train"], splits["y_train"])
    val_ds = GestureDataset(splits["X_val"], splits["y_val"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = GestureMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()         
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

    # plot loss curve
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Training Curve")
    plt.legend()
    plt.savefig("models/training_curve.png")
    plt.close()

    # reload best checkpoint for evaluation
    model.load_state_dict(torch.load(MODEL_OUT))
    model.eval()
    return model


def evaluate_mlp(model, X_test, y_test):
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(X_tensor)
    elapsed = time.perf_counter() - start
    preds = logits.argmax(dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    latency_per_sample_ms = (elapsed / len(X_test)) * 1000

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("MLP Confusion Matrix")
    plt.xticks(range(len(GESTURE_CLASSES)), GESTURE_CLASSES, rotation=45)
    plt.yticks(range(len(GESTURE_CLASSES)), GESTURE_CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(len(GESTURE_CLASSES)):
        for j in range(len(GESTURE_CLASSES)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    print("\n=== MLP Test Report ===")
    print(classification_report(y_test, preds, target_names=GESTURE_CLASSES))
    return acc, latency_per_sample_ms


def evaluate_baselines(splits):
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    results = {}

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    start = time.perf_counter()
    rf_preds = rf.predict(X_test)
    rf_latency = ((time.perf_counter() - start) / len(X_test)) * 1000
    results["RandomForest"] = (accuracy_score(y_test, rf_preds), rf_latency)

    svm = SVC(kernel="rbf")
    svm.fit(X_train, y_train)
    start = time.perf_counter()
    svm_preds = svm.predict(X_test)
    svm_latency = ((time.perf_counter() - start) / len(X_test)) * 1000
    results["SVM"] = (accuracy_score(y_test, svm_preds), svm_latency)

    return results


def main():
    splits = load_splits()
    print(f"Train: {len(splits['X_train'])}  Val: {len(splits['X_val'])}  Test: {len(splits['X_test'])}")

    model = train_mlp(splits)
    mlp_acc, mlp_latency = evaluate_mlp(model, splits["X_test"], splits["y_test"])
    baseline_results = evaluate_baselines(splits)

    print("\n=== Final Comparison ===")
    print(f"{'Model':<15}{'Test Acc':<12}{'Latency/sample (ms)':<20}")
    print(f"{'MLP (deployed)':<15}{mlp_acc:<12.3f}{mlp_latency:<20.4f}")
    for name, (acc, latency) in baseline_results.items():
        print(f"{name:<15}{acc:<12.3f}{latency:<20.4f}")

    print(f"\nModel saved to {MODEL_OUT}")
    print("Confusion matrix -> models/confusion_matrix.png")
    print("Training curve   -> models/training_curve.png")


if __name__ == "__main__":
    main()
