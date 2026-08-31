# Course: Machine Learning & Deep Learning for Gesture Recognition

A complete, code-anchored course on the ML/DL subsystem of the **Telekinesis**
project — from raw hand landmarks to a trained, deployed gesture classifier,
plus the classical baselines used to justify the neural-network choice.

> This covers the machine-learning portion only. The vision, feature engineering,
> geometric interaction, and OS-control parts are the subject of their own course
> material.

---

## Table of Contents

1. [The Big Picture — ML in the Pipeline](#1-the-big-picture)
2. [The Machine-Learning Pipeline, Step by Step](#2-the-pipeline)
3. [Feature Engineering — Turning Landmarks Into a Vector](#3-feature-engineering)
4. [The Dataset & DataLoader](#4-dataset--dataloader)
5. [The Model — a Multi-Layer Perceptron (MLP)](#5-the-model)
6. [Loss Function — CrossEntropy](#6-loss-crossentropy)
7. [Optimizer — Adam](#7-optimizer-adam)
8. [The Training Loop](#8-the-training-loop)
9. [Evaluation & Metrics](#9-evaluation--metrics)
10. [Baselines — Random Forest & SVM](#10-baselines)
11. [Deployment — Loading & Inference](#11-deployment)
12. [Core Concepts Glossary](#12-glossary)
13. [End-to-End Walkthrough (the actual files)](#13-end-to-end-walkthrough)

---

## 1. The Big Picture

The project controls a computer from hand gestures. The **gesture-classifier
is just one link in a longer chain**:

```
landmarks (21x3)  →  [ML feature vector (63)]  →  [MLP softmax]  →  [probabilities]
      ↑                                                    │
   vision/                                              smoothed, arg-maxed
                                                         ↓
                                                   "pinch" / "point" / ...
```

Crucially, the ML model is **NOT** responsible for cursor motion, scroll
speed, zoom, or swipe detection. Those are done with *deterministic geometry*
(`interaction/geometry.py`) — hand-written math. **The ML model decides only
`"which static hand shape is this?"`** (one of 6 classes). Everything else is
classical signal processing and geometry.

So the ML task is a **small, supervised, multi-class classification problem**:

| Class | Meaning |
|-------|---------|
| `neutral` | relaxed open-ish hand |
| `point` | index finger extended (→ cursor move) |
| `pinch` | thumb + index together (→ click / drag) |
| `open_palm` | all fingers spread |
| `fist` | closed fist (→ window grab) |
| `two_fingers` | two fingers up (→ scroll) |

Because the input is just 63 engineered numbers (not raw pixels), we do **not**
need a visually sophisticated model like a Convolutional Neural Network. A
small **MLP (fully-connected network)** is the right tool: tiny, fast on CPU,
and it generalizes well on a small dataset.

---

## 2. The Pipeline

Reading the CSV → training the MLP → evaluating it, and comparing against
classical baselines, happens in `ml/train.py`. Its flow is:

```text
gesture.csv (raw 63-dim features + labels)
   │   ml/dataset.py::load_splits()
   ▼
YourChoice splits: Train / Validation / Test   (stratified, seeded)
   │
   ├──► ml/train.py::train_mlp()   → MLP (deployed model)  → models/gesture_mlp.pt
   │        with Early Stopping on validation loss
   │
   ├──► ml/train.py::evaluate_mlp() → accuracy, latency, confusion matrix, report
   │
   └──► ml/train.py::evaluate_baselines() → RandomForest, SVM  (accuracy + latency)
```

Run it with:

```bash
python -m ml.train
```

---

## 3. Feature Engineering

`features/extractor.py` turns the raw landmark array `(21, 3)` into a
63-dimensional feature vector. This step is **the most important design
decision** in the whole ML subsystem, so let's understand why.

### The problem with raw landmarks

MediaPipe gives each hand 21 landmarks `(x, y, z)`:

- `x, y` are normalized to `[0, 1]` of the frame.
- `z` is relative depth.

If we fed these raw numbers to the network, the classifier would learn to
recognize *where the hand is* and *how big it is*, rather than *what shape* it
makes. The same "point" gesture would look completely different in the top-left
corner vs. the bottom-right, or close to the camera (large) vs. far (small).

### The fix: translation + scale invariance

The extractor makes the features **invariant** to hand position and size:

```python
import numpy as np

WRIST_IDX = 0          # landmark 0 = the wrist
MIDDLE_MCP_IDX = 9     # landmark 9 = base of the middle finger

def extract_feature(landmarks: np.ndarray) -> np.ndarray:
    assert landmarks.shape == (21, 3)

    wrist = landmarks[WRIST_IDX]
    middle_mcp = landmarks[MIDDLE_MCP_IDX]

    # 1) TRANSLATION INVARIANCE: subtract the wrist from every point.
    #    Now all coordinates are relative to the wrist (wrist = origin).
    centered = landmarks - wrist                 # shape (21, 3)

    # 2) SCALE INVARIANCE: divide by the distance from wrist to middle MCP.
    #    This normalizes the whole hand to a "unit length" so that a hand
    #    that is 2x bigger produces the same features.
    scale = np.linalg.norm(middle_mcp - wrist)
    scale_ref = max(scale, 1e-6)                 # avoid division by zero

    normalized = centered / scale_ref            # shape (21, 3)

    # 3) FLATTEN to a single 63-vector (21 * 3 = 63).
    return normalized.flatten().astype(np.float32)   # shape (63,)
```

### Why pick wrist→middle-MCP as the scale?

- The **wrist** is the most stable landmark (least jitter).
- The **middle-finger MCP** (base joint, landmark 9) is roughly the "center" of
  the hand and moves fairly consistently with the hand regardless of which
  fingers are extended. So the wrist→MCP distance is a reliable proxy for
  "hand size."

### Key takeaways

- **Normalization is feature engineering.** It converts an intractable raw
  representation into one where the same gesture has the same feature vector
  regardless of position/scale.
- The MLP would struggle badly without this; a CNN could learn translation
  invariance via weight sharing — but that's unnecessary complexity once we
  normalize by hand.
- Output is `float32` because PyTorch default dtype for model weights is
  `float32`.

---

## 4. Dataset & DataLoader

### The raw CSV

`data/raw/gesture.csv` has one row per collected sample:

```text
label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
open_palm,0.0,0.0,0.0,-0.267,...,
point,...
```

That's 1 label column + 63 feature columns. (We saw ~5,100 samples with
`two_fingers` over-represented — an important real-world caveat we'll revisit.)

### `ml/dataset.py`

```python
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from ml.model import GESTURE_CLASSES

CSV_PATH = "data/raw/gesture.csv"


class GestureDataset(Dataset):
    """A thin PyTorch wrapper around numpy arrays → tensors."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)   # class indices must be long for CE loss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_splits(csv_path=CSV_PATH, test_size=0.15, val_size=0.15, random_state=42):
    df = pd.read_csv(csv_path)

    # map string labels -> integer class indices
    label_to_idx = {name: i for i, name in enumerate(GESTURE_CLASSES)}
    df = df[df["label"].isin(label_to_idx.keys())]      # filter stray labels

    # X = all columns except 'label'; y = the integer class for each row
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].map(label_to_idx).values.astype(np.int64)

    # 1) split off the test set (stratified so class ratios are preserved)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # 2) split the remaining data into train / val
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size,
        stratify=y_train_val, random_state=random_state)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
```

### Concepts in this file

**Why `torch.long` for `y`?** `CrossEntropyLoss` expects class *indices*
(integers, dtype `long`), not one-hot vectors. So `y` is stored as `long`.

**Why three splits?**
- **Train** — used to update the weights.
- **Validation** — (the *development set*) used to pick the best epoch via
  early stopping. It is **not** used to update weights.
- **Test** — completely held out, used only once at the very end, to report a
  final unbiased accuracy and compare against the Random Forest / SVM.

**`stratify=y`** — ensures the class proportions in the train/val/test sets are
the same as in the full dataset. If `two_fingers` is 24% of all data, it stays
~24% in every split. This matters when classes are imbalanced (as they are
here).

**`random_state=42`** — a fixed seed so the split is reproducible. Without a
seed, every run would produce a different split and thus different results,
which makes experiments non-reproducible.

**Why `val_size / (1 - test_size)`?** We want 15% *of the whole dataset* for
val. After test split, the remaining pool is `1 - test_size` of the data. So the
relative fraction of that pool that equals `val_size` of the whole is exactly
`val_size / (1 - test_size) ≈ 0.15 / 0.85 ≈ 0.176`.

### How the DataLoader is used (in training)

```python
from torch.utils.data import DataLoader

train_ds = GestureDataset(splits["X_train"], splits["y_train"])
val_ds   = GestureDataset(splits["X_val"],   splits["y_val"])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
```

**Batching** — instead of feeding one sample at a time, we feed mini-batches of
32. This:
- Uses the GPU/CPU more efficiently.
- Gives smoother, more stable gradient estimates than single samples.
- Is the standard modern approach (SGD on mini-batches).

**`shuffle=True` (train only)** — prevents the model from learning the order of
the data and reduces correlation between consecutive batches.

---

## 5. The Model

`ml/model.py`:

```python
import torch
import torch.nn as nn

GESTURE_CLASSES = ["neutral", "point", "pinch", "open_palm", "fist", "two_fingers"]

class GestureMLP(nn.Module):
    def __init__(self, input_dim=63, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)
```

### Anatomy

```
input (63) ──Linear──▶ 128 ─ReLU▶ ──Linear──▶ 64 ─ReLU▶ ──Linear──▶ 32 ─ReLU▶ ──Linear──▶ 6 (logits)
```

- **4 `Linear` (fully-connected) layers.**
- **3 ReLU activations** between them.
- The **last layer has NO activation** — its outputs are called **logits**
  (raw, unbounded scores). We apply softmax later (see §6).

### Why this size? (63→128→64→32→6)

| Aspect | Reasoning |
|--------|-----------|
| **Input 63** | exactly the engineered feature vector size. |
| **Expand to 128** | first layer needs capacity to mix all 63 inputs into useful intermediate features (feature interactions). |
| **Compress 128→64→32** | progressive "bottleneck" funnel — forces the network to keep only the most gesture-discriminative information and drop redundancy; this is an implicit regularizer. |
| **Output 6** | one neuron per gesture class (paired with CrossEntropyLoss). |
| **Only 3 hidden layers** | the input is small and already well-structured; a deeper net would overfit on ~5k samples and add latency. Empirically this is plenty for 6 fairly distinct hand shapes. |

### Why an MLP and not a CNN / transformer?

- **The feature is a flat 63-vector, not an image.** There is no spatial layout
  to preserve, no neighboring-pixel structure to convolve over, and no
  translation variance left (we normalized it away). A CNN's inductive biases
  (locality, translation equivariance, weight sharing) become irrelevant
  overhead.
- **Latency is king** here — it's real-time cursor control. A tiny MLP does
  inference in well under a millisecond on CPU. Deeper / convolutional /
  transformer models would cost compute for zero accuracy gain.
- **Small-data friendliness** — ~850 samples/class. A small MLP + early
  stopping generalizes; a big network would memorize the training set.

### Why ReLU?

$$ \text{ReLU}(x) = \max(0, x) $$

1. **Fixes vanishing gradients.** Sigmoid/tanh saturate (derivative → 0) and
   kill gradient flow through many layers, making deep networks slow or
   impossible to train. ReLU's derivative is 1 for all positive inputs, so
   gradients propagate cleanly.
2. **Cheap to compute** — just `max(0, x)`, ideal for real-time CPU inference.
3. **Sparsity** — ReLU zeros out negative inputs, so many activations are
   exactly 0. Sparse activations are faster and act as a mild regularizer
   (dropping unneeded units).
4. **Standard, proven default** — it is the field's default for small MLPs;
   no reason to reach for GELU/Swish at this scale.

**Caveat to know:** ReLU has a "dying ReLU" failure mode — if a unit's weights
push its input consistently negative, its gradient is 0 and it stays dead
forever. With only 3 layers + Adam + lr=1e-3 this is rarely a problem here, but
it's the tradeoff you accepted by choosing ReLU over something like LeakyReLU.

### The forward pass & logits

`forward(x)` returns the **logits** — the raw outputs of the final Linear layer.
They can be any real number. To turn them into probabilities we need softmax
(see next section).

---

## 6. Loss Function — CrossEntropy

In `train.py`:

```python
criterion = nn.CrossEntropyLoss()
```

### What it does

`nn.CrossEntropyLoss` **combines LogSoftmax + Negative Log-Likelihood** in one
numerically-stable operation. Given logits `z` and the true class `y`, it
computes:

$$ \mathcal{L} = -\log\left( \frac{e^{z_y}}{\sum_j e^{z_j}} \right) $$

which is the negative log of the **softmax probability of the true class**.

### Why softmax?

Softmax turns the 6 raw logits into a probability distribution over classes
(6 numbers in `[0,1]` that sum to 1):

```python
probs = F.softmax(logits, dim=1)   # shape (batch, 6)
```

The probability of class `i` is:

$$ p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} $$

The exponential makes it sensitive to score differences; doubling the logit gap
sharply increases the winning probability.

### Why minimize negative log-probability?

Maximizing the probability of the true class is equivalent to minimizing its
negative log. Logs turn products of probabilities (over samples/classes) into
sums, which is numerically nicer and yields the information-theoretic
cross-entropy. The model is trained to make the true class's probability → 1,
so the loss → 0.

### Notes

- No manual one-hot encoding needed — CE loss takes the true class as an
  integer index (`y` stored as `long`).
- During **training** we use `criterion(logits, y)` because it includes the
  softmax internally. During **inference** (deployment) `test_live.py` applies
  `F.softmax` explicitly to get readable probabilities.

---

## 7. Optimizer — Adam

```python
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # LR = 1e-3
```

### How gradient descent updates weights

$$ \theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta} $$

where `θ` are the weights and `η` is the **learning rate** (`LR = 1e-3 = 0.001`).

### Why Adam over plain SGD?

Plain SGD uses a single global learning rate and can oscillate or get stuck on
poorly-scaled features. **Adam (Adaptive Moment Estimation)** maintains
per-parameter adaptive learning rates using estimates of the *first moment*
(mean) and *second moment* (uncentered variance) of the gradients:

- **First moment** → like momentum (keeps direction, damps oscillation).
- **Second moment** → scales the step per-parameter (large gradients → smaller
  steps, small/rare gradients → larger steps).
- **Bias correction** during early steps.

Practical results: Adam converges faster and is far less sensitive to the
learning-rate choice than SGD — the default recommendation for this kind of
small-to-medium net.

### Why LR = 1e-3?

A common, well-behaved default for Adam. Too large → loss diverges; too small →
painfully slow convergence. 1e-3 with early stopping is a safe, effective
starting point.

---

## 8. The Training Loop

`ml/train.py::train_mlp`:

```python
model = GestureMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # 1e-3
criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
epochs_no_improve = 0
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):           # EPOCHS = 50
    # ── TRAIN PHASE ──────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()                # 1) clear old gradients
        logits = model(X_batch)              # 2) forward pass
        loss = criterion(logits, y_batch)    # 3) compute loss
        loss.backward()                      # 4) compute gradients (backprop)
        optimizer.step()                     # 5) update weights
        running_loss += loss.item() * X_batch.size(0)
    train_loss = running_loss / len(train_ds)

    # ── VALIDATION PHASE ──────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():                    # no gradients needed, faster/or 'free'
        for X_batch, y_batch in val_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # ── EARLY STOPPING ────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_OUT)   # save BEST checkpoint
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:           # PATIENCE = 10
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
            break
```

### The five canonical steps per batch

1. **`optimizer.zero_grad()`** — clear gradients from the previous batch
   (PyTorch *accumulates* gradients by default; if you forget this, gradients
   sum across batches and the updates explode).
2. **`logits = model(X_batch)`** — forward pass.
3. **`loss = criterion(logits, y_batch)`** — compute the loss.
4. **`loss.backward()`** — backpropagation: compute `∂L/∂θ` for every parameter
   using the chain rule, and store them on each tensor's `.grad`.
5. **`optimizer.step()`** — apply the Adam update using the stored gradients.

### `model.train()` vs `model.eval()`

- `train()` enables things like dropout and batchnorm *updating*.
- `eval()` disables them so validation/inference is deterministic.
- Here there's no dropout/batchnorm, so the practical difference is small —
  but it's **essential best practice**, especially since `nn.CrossEntropyLoss`
  semantics and downstream deployment rely on eval-mode determinism.

### `torch.no_grad()`

Wraps the validation forward pass so PyTorch doesn't build a computation graph
(saving memory) and doesn't track gradients (saving time). In validation we only
forward — never backprop — so gradients are waste.

### The loss curve & saving the best checkpoint

```python
plt.plot(train_losses, label="train_loss")
plt.plot(val_losses, label="val_loss")
plt.savefig("models/training_curve.png")
```

The training/validation loss curves tell you how well the model is learning and
whether it's overfitting (val loss rising while train loss falls = overfit).

We save the **model weights whenever validation loss improves** — not the last
epoch's weights. Early stopping means the saved checkpoint is the best
validation model, not an overfit one.

### Overfitting & why both losses matter

- **Underfitting**: both losses high — model too weak, or not trained long
  enough.
- **Generalization**: both low and close — ideal.
- **Overfitting**: train loss low, but val loss high or diverging — model has
  memorized training samples. Early stopping directly addresses this.

### Reproducibility

Set `random_state` in the split and rely on early stopping for stability. (For
full reproducibility you'd also seed PyTorch/numpy with `torch.manual_seed`.)

---

## 9. Evaluation & Metrics

`ml/train.py::evaluate_mlp`:

```python
def evaluate_mlp(model, X_test, y_test):
    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    # measure real inference latency on CPU
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(X_tensor)
    elapsed = time.perf_counter() - start
    preds = logits.argmax(dim=1).numpy()          # class with highest logit

    acc = accuracy_score(y_test, preds)
    latency_per_sample_ms = (elapsed / len(X_test)) * 1000

    cm = confusion_matrix(y_test, preds)          # plot -> models/confusion_matrix.png
    # ... plotting code ...
    print(classification_report(y_test, preds, target_names=GESTURE_CLASSES))
    return acc, latency_per_sample_ms
```

### The crucial detail for latency

```python
preds = logits.argmax(dim=1)      # NOT softmax!
```

Because softmax is a **monotonic** function of the logits (it doesn't change
which class has the maximum value), `argmax` on logits gives the same prediction
as `argmax` on probabilities — but without the exponentials, making it faster.

### Metrics used

1. **Accuracy** = correct / total. Simple but misleading on imbalanced data —
   if `two_fingers` dominates, a model predicting everything as `two_fingers`
   looks "accurate".
2. **Confusion matrix** — shows exactly *which* classes get confused. The
   diagonal = correct predictions; off-diagonal = specific errors. This is the
   most informative single diagnostic.
3. **Classification report** — per-class **precision, recall, f1-score,
   support**. Precision = of predicted X, how many were really X. Recall = of
   real X, how many we caught. F1 = harmonic mean. These matter when classes
   are imbalanced.
4. **Latency per sample (ms)** — the end-to-end performance-critical metric.
   The whole point of the MLP is sub-millisecond inference; this number is what
   we compare against Random Forest / SVM.

### Why is latency even measured? 

This is a **real-time system**. The classifier runs on ~every camera frame at
up to 30 fps. If the classifier took 50ms, the whole interaction pipeline would
feel laggy. Measuring and reporting per-sample latency justifies the MLP choice:
it's both accurate *and* extremely fast on CPU with no GPU.

---

## 10. Baselines — Random Forest & SVM

`ml/train.py::evaluate_baselines`:

```python
def evaluate_baselines(splits):
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
    results = {}

    # ── Random Forest ──
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    start = time.perf_counter()
    rf_preds = rf.predict(X_test)
    rf_latency = ((time.perf_counter() - start) / len(X_test)) * 1000
    results["RandomForest"] = (accuracy_score(y_test, rf_preds), rf_latency)

    # ── Support Vector Machine (RBF kernel) ──
    svm = SVC(kernel="rbf")
    svm.fit(X_train, y_train)
    start = time.perf_counter()
    svm_preds = svm.predict(X_test)
    svm_latency = ((time.perf_counter() - start) / len(X_test)) * 1000
    results["SVM"] = (accuracy_score(y_test, svm_preds), svm_latency)

    return results
```

### Why compare at all?

**To justify the deep-learning choice.** If a random forest or SVM gets the same
accuracy with less engineering, why bother with a neural net? Running all three
on the same test split and comparing **accuracy and latency** gives an
evidence-based answer. This is the disciplined, empirical approach — never
assume a model architecture is better; measure it.

### Random Forest

An **ensemble** of many decision trees, each trained on a random bootstrap
sample of the data and a random subset of features. Predictions are the majority
vote of all trees. Strengths: robust, no gradient tuning, good on
small/tabular data (of which 63-dim normalized landmarks are an example).

### Support Vector Machine (RBF kernel)

Finds a hyperplane in a high-dimensional feature space (implicitly mapped via
the RBF kernel) that maximizes the margin between classes. Good for small to
medium datasets, but prediction involves computing kernel values against many
support vectors — which gets slower with more data, explaining why its latency
can be worse than RF or the MLP.

### The takeaway, compared

```text
Model             Test Acc    Latency/sample     Verdict
----------------  ---------   ----------------   ------------------------------
MLP (deployed)    high        < 1 ms             smallest & fastest → chosen
RandomForest      comparable  low                strong baseline
SVM (RBF)         comparable  higher             slower to predict
```

Comparing all three on identical test data is how the project defends the choice
of "a neural network for a small tabular classification problem": because it's
competitive on accuracy *and* the fastest deployable option.

---

## 11. Deployment — Loading & Inference

### `ml/model.py::load_model`

```python
def load_model(path="models/gesture_mlp.pt", device="cpu"):
    model = GestureMLP()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
```

- `GestureMLP()` builds the architecture (must match exactly what was trained).
- `load_state_dict` loads the saved weights.
- `model.eval()` switches to inference mode (deterministic; disables dropout etc).
- We load **only the state dict** — the architecture (and all hyperparameters
  like layer sizes) live in the code, not the file.

### Live inference (`test_live.py`)

```python
model = load_model("models/gesture_mlp.pt")
...
feats = extract_feature(hands_lms[0])                 # (63,)
x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1, 63) — add batch dim
with torch.no_grad():
    logits = model(x)                                  # (1, 6) logits
    probs = F.softmax(logits, dim=1)                   # (1, 6) probabilities
    conf, pred_idx = torch.max(probs, dim=1)           # highest prob + its index
label = GESTURE_CLASSES[pred_idx.item()]
```

The key operations:
- **`unsqueeze(0)`** — the model expects a *batch* of samples `(B, 63)`. A single
  sample is `(63,)`, so we add a leading batch dimension → `(1, 63)`.
- **`softmax(logits, dim=1)`** — converts the 6 logits into probabilities
  (sums to 1 across the class dimension).
- **`torch.max(probs, dim=1)`** — returns the largest probability (`conf`) and
  its index (`pred_idx`).
- **`GESTURE_CLASSES[pred_idx.item()]`** — maps the index back to a human
  name like `"pinch"`.
- **`torch.no_grad()`** — inference only; no gradient graph needed.

### In `main.py`, the probabilities also go through a smoother

`main.py` doesn't just use the raw single-frame softmax. It feeds the
probability vector into an **asymmetric EMA smoother** (`_ProbSmoother`) that:

- rises fast (`alpha_rise=0.55`) when a probability increases, so new gestures
  are recognized quickly;
- falls slowly (`alpha_fall=0.15`) when it decreases, so brief classifier
  hiccups (1–3 frames) don't interrupt an ongoing drag / scroll / fist.

This is the layer that turns *frame-by-frame noisy* classifier outputs into
*temporally stable* behavior. It's a cheap, deterministic smoothing that ML
alone doesn't provide.

---

## 12. Glossary

| Term | Meaning |
|------|---------|
| **MLP** | Multi-Layer Perceptron — a feedforward fully-connected neural network. |
| **Neuron / unit** | computes `w·x + b`, then applies an activation. |
| **Layer / hidden layer** | a set of neurons; "hidden" = not input/output. |
| **Weight `w`, bias `b`** | learnable parameters of a `Linear` layer. |
| **Activation function** | nonlinearity (here ReLU) after each hidden layer. |
| **Logits** | raw pre-softmax outputs of the final layer (any real number). |
| **Softmax** | turns logits into a probability distribution over classes. |
| **CrossEntropyLoss** | LogSoftmax + NLL combined; the classification objective. |
| **Backpropagation** | chain-rule computation of gradients through the net. |
| **Optimizer / Adam** | rule for updating weights from gradients (adaptive per-param LR). |
| **Learning rate (LR)** | step size for parameter updates (here 1e-3). |
| **Epoch** | one full pass over the training set. |
| **Batch / mini-batch** | a subset of samples processed together (here 32). |
| **Gradient descent** | iteratively move weights to reduce loss. |
| **Overfitting** | model memorizes training data, fails to generalize. |
| **Early stopping** | stop training when validation performance stops improving. |
| **Patience** | number of epochs to wait for improvement before stopping (here 10). |
| **Train / val / test split** | learn on train, tune on val, evaluate once on test. |
| **Stratified split** | preserves class proportions across splits. |
| **Accuracy** | fraction of correct predictions. |
| **Confusion matrix** | grid showing correct vs. confused predictions. |
| **Precision / Recall / F1** | per-class correctness metrics for imbalanced data. |
| **Random Forest** | ensemble of decision trees. |
| **SVM (RBF)** | max-margin classifier using a radial-basis kernel. |
| **Feature vector** | input to the model (here 63 normalized numbers). |
| **Normalization** | making features invariant to position/scale/units. |
| **state_dict** | a dict of a model's learnable parameters (weights & biases). |
| **Inference** | using a trained model to make predictions (as opposed to training). |
| **Tensor** | PyTorch's n-dimensional array; the basic data type. |
| **`unsqueeze`** | add a dimension (here a batch dimension). |
| **`torch.no_grad()`** | disable gradient tracking for fast, memory-light inference. |

---

## 13. End-to-End Walkthrough

For completeness, here is the full path the data and gradients take, mapped to
the real files, plus the exact commands.

### Data journey (offline)

```bash
# 1. collect gesture samples → data/raw/gesture.csv
python collect_data.py

# 2. load + split, build model, train, evaluate, compare baselines
python -m ml.train
```

`ml/train.py`'s `main()`:

```python
def main():
    splits = load_splits()
    print(f"Train: {len(splits['X_train'])}  Val: {len(splits['X_val'])}  Test: {len(splits['X_test'])}")

    model = train_mlp(splits)                       # trains + early-stops + saves best
    mlp_acc, mlp_latency = evaluate_mlp(model, splits["X_test"], splits["y_test"])
    baseline_results = evaluate_baselines(splits)   # RandomForest + SVM

    print("=== Final Comparison ===")
    # ...table: Model | Test Acc | Latency/sample (ms)...
```

### Model journey (online / runtime)

```bash
# classifier-only validation, no OS control
python test_live.py

# full pipeline with real OS control
python main.py
```

At runtime the ML steps per frame are:

```text
hands_landmarks[0]  (21,3)
   │  features/extractor.py::extract_feature
   ▼
feature (63,)
   │  → tensor, unsqueeze → (1,63)
   ▼
GestureMLP.forward  → logits (1,6)
   │  F.softmax(dim=1)
   ▼
probabilities (1,6)
   │  _ProbSmoother (asymmetric EMA) → argmax
   ▼
"pinch" / "point" / ...  (+ confidence)  →  InteractionEngine
```

---

## Final Summary of the ML/DL Design

1. **Task**: 6-way classification of hand *shape* from 63 engineered features.
2. **Features are normalized** (translation + scale invariant) so the model
   learns shape, not position/size.
3. **A small MLP** (63→128→64→32→6, ReLU) is the right architecture: tiny,
   CPU-fast, generalizes on a small dataset, accurate on 6 distinct shapes.
4. **CrossEntropy loss + Adam (lr 1e-3) + mini-batches (32)** is the standard,
   robust training setup.
5. **Train/val/test stratified splits** + **early stopping** guard against
   overfitting and give an honest final evaluation.
6. **RandomForest & SVM baselines** are benchmarked on the same test split to
   *prove* the MLP is the best accuracy-vs-latency choice.
7. **Deployment** is simple: load the state dict, `eval()`, softmax + argmax —
   fast enough for real-time per-frame inference.
