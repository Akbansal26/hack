"""
A small, dependency-free LSTM (forward + backprop-through-time + Adam),
implemented directly in NumPy. This environment can't install a full deep
learning framework (torch/tensorflow), so this is a compact but real
implementation -- useful for a hackathon demo of "engineered features vs.
sequence model" and small enough to train in seconds on CPU.

Input:  raw (unengineered) SEQ_LEN-step windows of the 4 sensors, normalized.
Output: probability that a failure occurs within HORIZON steps after the
        window's last timestep (same label definition as the baseline).
"""
import numpy as np
import pandas as pd

from features import SENSORS, HORIZON, add_label

SEQ_LEN = 24
HIDDEN = 16
EPOCHS = 25
LR = 0.03
BATCH = 64
RNG_SEED = 11


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyLSTM:
    """Single-layer LSTM -> linear -> sigmoid, batched, hand-written BPTT."""

    def __init__(self, n_features, hidden=HIDDEN, seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        z = hidden + n_features
        scale = 1.0 / np.sqrt(z)
        self.hidden = hidden
        # combined gate weights: [forget, input, candidate, output]
        self.W = {g: rng.normal(0, scale, (z, hidden)) for g in "fioc"}
        self.b = {g: np.zeros(hidden) for g in "fioc"}
        self.b["f"] += 1.0  # forget-gate bias trick: remember by default
        self.Wy = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1))
        self.by = np.zeros(1)
        # Adam moment buffers
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.t = 0

    def _params(self):
        p = {f"W{g}": self.W[g] for g in "fioc"}
        p.update({f"b{g}": self.b[g] for g in "fioc"})
        p["Wy"], p["by"] = self.Wy, self.by
        return p

    def forward(self, X):
        """X: (batch, seq_len, n_features) -> proba (batch,), cache for backward."""
        B, T, F = X.shape
        H = self.hidden
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = {"X": X, "h": [h], "c": [c], "gates": []}
        for t in range(T):
            xt = X[:, t, :]
            z = np.concatenate([h, xt], axis=1)
            f = sigmoid(z @ self.W["f"] + self.b["f"])
            i = sigmoid(z @ self.W["i"] + self.b["i"])
            o = sigmoid(z @ self.W["o"] + self.b["o"])
            g = np.tanh(z @ self.W["c"] + self.b["c"])
            c = f * c + i * g
            h = o * np.tanh(c)
            cache["gates"].append((f, i, o, g, z))
            cache["h"].append(h)
            cache["c"].append(c)
        logits = h @ self.Wy + self.by
        proba = sigmoid(logits).ravel()
        cache["proba"] = proba
        return proba, cache

    def backward(self, cache, y):
        B, T, F = cache["X"].shape
        H = self.hidden
        grads = {k: np.zeros_like(v) for k, v in self._params().items()}

        dlogits = (cache["proba"] - y).reshape(-1, 1) / B  # dL/dlogits, mean BCE
        h_last = cache["h"][-1]
        grads["Wy"] = h_last.T @ dlogits
        grads["by"] = dlogits.sum(axis=0)
        dh_next = dlogits @ self.Wy.T
        dc_next = np.zeros((B, H))

        for t in reversed(range(T)):
            f, i, o, g, z = cache["gates"][t]
            c_prev = cache["c"][t]
            c_t = cache["c"][t + 1]
            dh = dh_next
            do = dh * np.tanh(c_t)
            dc = dc_next + dh * o * (1 - np.tanh(c_t) ** 2)
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            df_raw = df * f * (1 - f)
            di_raw = di * i * (1 - i)
            do_raw = do * o * (1 - o)
            dg_raw = dg * (1 - g ** 2)

            grads["Wf"] += z.T @ df_raw
            grads["Wi"] += z.T @ di_raw
            grads["Wo"] += z.T @ do_raw
            grads["Wc"] += z.T @ dg_raw
            grads["bf"] += df_raw.sum(axis=0)
            grads["bi"] += di_raw.sum(axis=0)
            grads["bo"] += do_raw.sum(axis=0)
            grads["bc"] += dg_raw.sum(axis=0)

            dz = (df_raw @ self.W["f"].T + di_raw @ self.W["i"].T
                  + do_raw @ self.W["o"].T + dg_raw @ self.W["c"].T)
            dh_next = dz[:, :H]
            dc_next = dc_prev
        return grads

    def adam_step(self, grads, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        params = self._params()
        for k in params:
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * grads[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (grads[k] ** 2)
            mhat = self.m[k] / (1 - beta1 ** self.t)
            vhat = self.v[k] / (1 - beta2 ** self.t)
            update = lr * mhat / (np.sqrt(vhat) + eps)
            if k.startswith("W") and k != "Wy":
                self.W[k[1]] -= update
            elif k.startswith("b") and k != "by":
                self.b[k[1]] -= update
            elif k == "Wy":
                self.Wy -= update
            elif k == "by":
                self.by -= update

    def predict_proba(self, X, batch=256):
        out = []
        for i in range(0, len(X), batch):
            proba, _ = self.forward(X[i:i + batch])
            out.append(proba)
        return np.concatenate(out)


def build_sequences(raw_csv="sensor_data.csv", seq_len=SEQ_LEN, horizon=HORIZON):
    df = pd.read_csv(raw_csv, parse_dates=["timestamp"])
    df = add_label(df, horizon=horizon)

    # normalize per-sensor globally using healthy-period stats (first 72h across all machines)
    healthy = df[df["step"] < 72]
    means = healthy[SENSORS].mean()
    stds = healthy[SENSORS].std() + 1e-6

    X_list, y_list, meta = [], [], []
    for mid, sub in df.groupby("machine_id"):
        sub = sub.sort_values("step").reset_index(drop=True)
        vals = ((sub[SENSORS] - means) / stds).to_numpy()
        labels = sub["label"].to_numpy()
        steps = sub["step"].to_numpy()
        for end in range(seq_len - 1, len(sub)):
            X_list.append(vals[end - seq_len + 1: end + 1])
            y_list.append(labels[end])
            meta.append((mid, steps[end]))
    X = np.stack(X_list).astype(np.float64)
    y = np.array(y_list, dtype=np.float64)
    meta = pd.DataFrame(meta, columns=["machine_id", "step"])
    return X, y, meta, (means, stds)


def train_test_split_by_machine(meta, test_fraction=0.3, seed=7):
    machines = np.array(sorted(meta["machine_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(machines)
    n_test = max(1, int(len(machines) * test_fraction))
    test_machines = set(machines[:n_test])
    is_test = meta["machine_id"].isin(test_machines).to_numpy()
    return ~is_test, is_test, test_machines


def main():
    print("Building raw sequence windows...")
    X, y, meta, _ = build_sequences()
    train_mask, test_mask, test_machines = train_test_split_by_machine(meta)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"train windows: {len(X_train)} (pos rate {y_train.mean():.3f}), "
          f"test windows: {len(X_test)} (pos rate {y_test.mean():.3f})")

    model = NumpyLSTM(n_features=X.shape[2], hidden=HIDDEN)
    n = len(X_train)
    rng = np.random.default_rng(3)

    for epoch in range(EPOCHS):
        idx = rng.permutation(n)
        losses = []
        for start in range(0, n, BATCH):
            b = idx[start:start + BATCH]
            proba, cache = model.forward(X_train[b])
            eps = 1e-9
            loss = -np.mean(y_train[b] * np.log(proba + eps) + (1 - y_train[b]) * np.log(1 - proba + eps))
            grads = model.backward(cache, y_train[b])
            model.adam_step(grads)
            losses.append(loss)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            test_proba = model.predict_proba(X_test)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, test_proba) if len(set(y_test)) > 1 else float("nan")
            print(f"epoch {epoch:02d}  train_loss={np.mean(losses):.4f}  test_auc={auc:.4f}")

    test_proba = model.predict_proba(X_test)
    meta_test = meta[test_mask].copy()
    meta_test["lstm_proba"] = test_proba
    meta_test["label"] = y_test
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    pred = (test_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    metrics = {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "false_alarm_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0.0),
        "roc_auc": float(roc_auc_score(y_test, test_proba)) if len(set(y_test)) > 1 else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

    # Threshold curves for LSTM
    lstm_threshold_curves = []
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        th_p = (test_proba >= thresh).astype(int)
        th_tn, th_fp, th_fn, th_tp = confusion_matrix(y_test, th_p, labels=[0, 1]).ravel()
        lstm_threshold_curves.append({
            "threshold": thresh,
            "precision": float(precision_score(y_test, th_p, zero_division=0)),
            "recall": float(recall_score(y_test, th_p, zero_division=0)),
            "f1": float(f1_score(y_test, th_p, zero_division=0)),
            "false_alarm_rate": float(th_fp / (th_fp + th_tn) if (th_fp + th_tn) > 0 else 0.0),
        })
    metrics["threshold_curves"] = lstm_threshold_curves

    import os
    import json
    os.makedirs("results", exist_ok=True)
    
    meta_test.to_csv("lstm_test_scored.csv", index=False, encoding="utf-8")
    meta_test.to_csv(os.path.join("results", "lstm_test_scored.csv"), index=False, encoding="utf-8")
    
    with open("lstm_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join("results", "lstm_results.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("LSTM evaluated successfully:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
