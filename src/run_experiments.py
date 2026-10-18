#!/usr/bin/env python3
"""Reproduce all experiments and figures for Machine Learning Assignment 2.

Student: Jiya Yadav (M26CSE004)
Seed: 26004, obtained by retaining the numeric roll-number characters in order.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


SEED = 26004
N_CLASSES = 40
B_VALUES = [1, 5, 10, 25, 50, 100, 200]
BOOST_DEPTHS = [1, 2, 3, 4]
BOOST_ROUNDS = [10, 25, 50, 100]


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def aligned_proba(tree, X, n_classes=N_CLASSES):
    out = np.zeros((len(X), n_classes), dtype=float)
    raw = tree.predict_proba(X)
    out[:, tree.classes_.astype(int)] = raw
    return out


def vote_matrix(predictions, n_classes=N_CLASSES, weights=None):
    predictions = np.asarray(predictions)
    n_models, n_samples = predictions.shape
    votes = np.zeros((n_samples, n_classes), dtype=float)
    if weights is None:
        weights = np.ones(n_models, dtype=float)
    for row, weight in zip(predictions, weights):
        votes[np.arange(n_samples), row.astype(int)] += weight
    return votes


def winning_margin(votes):
    ordered = np.sort(votes, axis=1)
    denom = votes.sum(axis=1)
    denom[denom == 0] = 1.0
    return (ordered[:, -1] - ordered[:, -2]) / denom


def confidence(votes):
    denom = votes.sum(axis=1)
    denom[denom == 0] = 1.0
    return votes.max(axis=1) / denom


def disagreement(predictions):
    p = np.asarray(predictions)
    if len(p) < 2:
        return 0.0
    vals = []
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            vals.append(np.mean(p[i] != p[j]))
    return float(np.mean(vals))


def leaf_statistics(tree, X_train):
    leaf_ids = tree.apply(X_train)
    counts = np.array(list(Counter(leaf_ids).values()))
    return {
        "n_leaves": int(tree.get_n_leaves()),
        "mean_samples_per_occupied_leaf": float(counts.mean()),
        "singleton_leaf_fraction": float(np.mean(counts == 1)),
    }


def split_by_identity(target, seed=SEED):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for identity in range(N_CLASSES):
        idx = np.flatnonzero(target == identity)
        perm = rng.permutation(idx)
        train.extend(perm[:6])
        val.extend(perm[6:8])
        test.extend(perm[8:10])
    return np.array(train), np.array(val), np.array(test)


def _fit_bag_tree(X, y, sample, params, random_state):
    t0 = time.perf_counter()
    tree = DecisionTreeClassifier(max_depth=params["max_depth"],
                                  min_samples_leaf=params["min_samples_leaf"],
                                  random_state=random_state)
    tree.fit(X[sample], y[sample])
    return tree, time.perf_counter() - t0


@dataclass
class BaggingModel:
    trees: list
    bootstrap_indices: list
    fit_times: list

    @classmethod
    def fit(cls, X, y, n_estimators, params, seed, n_jobs=1):
        rng = np.random.default_rng(seed)
        indices = [rng.integers(0, len(y), size=len(y)) for _ in range(n_estimators)]
        if n_jobs == 1:
            fitted = [_fit_bag_tree(X, y, sample, params, seed + b + 1)
                      for b, sample in enumerate(indices)]
        else:
            fitted = Parallel(n_jobs=n_jobs, max_nbytes="2M")(
                delayed(_fit_bag_tree)(X, y, sample, params, seed + b + 1)
                for b, sample in enumerate(indices)
            )
        trees = [x[0] for x in fitted]
        fit_times = [x[1] for x in fitted]
        return cls(trees, indices, fit_times)

    def predictions(self, X, n=None):
        if n is None:
            n = len(self.trees)
        return np.asarray([tree.predict(X).astype(int) for tree in self.trees[:n]])

    def scores(self, X, n=None):
        p = self.predictions(X, n)
        return vote_matrix(p)

    def predict(self, X, n=None):
        return self.scores(X, n).argmax(axis=1)


@dataclass
class SAMMEModel:
    trees: list
    alphas: np.ndarray
    history: list
    stopped_reason: str

    @classmethod
    def fit(cls, X, y, max_rounds, depth, seed, n_classes=N_CLASSES):
        n = len(y)
        weights = np.full(n, 1.0 / n)
        trees, alphas, history = [], [], []
        rng = np.random.default_rng(seed)
        stopped_reason = "maximum rounds reached"
        for t in range(max_rounds):
            tree = DecisionTreeClassifier(max_depth=depth, random_state=int(rng.integers(1, 2**31 - 1)))
            t0 = time.perf_counter()
            tree.fit(X, y, sample_weight=weights)
            fit_time = time.perf_counter() - t0
            pred = tree.predict(X).astype(int)
            miss = pred != y
            err = float(np.dot(weights, miss))
            threshold = 1.0 - 1.0 / n_classes
            if err >= threshold - 1e-12:
                stopped_reason = f"round {t + 1}: weak learner error {err:.6f} was not better than {threshold:.6f}"
                break
            safe_err = float(np.clip(err, 1e-12, 1.0 - 1e-12))
            alpha = math.log((1.0 - safe_err) / safe_err) + math.log(n_classes - 1)
            weights *= np.exp(alpha * miss)
            weights /= weights.sum()
            trees.append(tree)
            alphas.append(alpha)
            history.append(
                {
                    "round": t + 1,
                    "weighted_error": err,
                    "alpha": alpha,
                    "training_error": float(np.mean(miss)),
                    "max_sample_weight": float(weights.max()),
                    "top10_weight": float(np.sort(weights)[-10:].sum()),
                    "fit_time": fit_time,
                    "weights": weights.copy(),
                }
            )
            if err <= 1e-12:
                stopped_reason = f"round {t + 1}: perfect weak learner"
                break
        return cls(trees, np.asarray(alphas), history, stopped_reason)

    def scores(self, X, n=None):
        if n is None:
            n = len(self.trees)
        if n == 0:
            return np.zeros((len(X), N_CLASSES))
        pred = np.asarray([tree.predict(X).astype(int) for tree in self.trees[:n]])
        return vote_matrix(pred, weights=self.alphas[:n])

    def predict(self, X, n=None):
        return self.scores(X, n).argmax(axis=1)


def run_resample_rep(rep, Xtr, ytr, Xte, yte, tree_params, final_B, final_depth,
                     final_rounds, depth_candidates):
    """Run one independent Q3 resample. Kept top-level for joblib workers."""
    rng = np.random.default_rng(SEED + 3000 + rep)
    identity_positions = [np.flatnonzero(ytr == c) for c in range(N_CLASSES)]
    sampled_pos = np.concatenate([rng.choice(pos, size=6, replace=True) for pos in identity_positions])
    Xr, yr = Xtr[sampled_pos], ytr[sampled_pos]
    rep_seed = SEED + 10000 + rep * 1000
    st = DecisionTreeClassifier(max_depth=tree_params["max_depth"],
                                min_samples_leaf=tree_params["min_samples_leaf"],
                                random_state=rep_seed).fit(Xr, yr)
    sb = BaggingModel.fit(Xr, yr, final_B, tree_params, rep_seed + 100)
    sm = SAMMEModel.fit(Xr, yr, final_rounds, final_depth, rep_seed + 400)
    pv_tree = aligned_proba(st, Xte)
    bag_pred = sb.predictions(Xte, final_B)
    pv_bag = vote_matrix(bag_pred) / final_B
    pv_boost_raw = sm.scores(Xte, min(final_rounds, len(sm.trees)))
    pv_boost = pv_boost_raw / np.maximum(pv_boost_raw.sum(axis=1, keepdims=True), 1e-12)
    result = {
        "rep": rep,
        "selected_vectors": {"single tree": pv_tree, "bagging": pv_bag, "boosting": pv_boost},
        "selected_acc": {"single tree": accuracy_score(yte, pv_tree.argmax(axis=1)),
                         "bagging": accuracy_score(yte, pv_bag.argmax(axis=1)),
                         "boosting": accuracy_score(yte, pv_boost.argmax(axis=1))},
        "selected_disagreement": disagreement(bag_pred),
        "selected_leaf": leaf_statistics(st, Xr),
        "depth": {},
    }
    for offset, depth in enumerate(depth_candidates):
        label = "None" if depth is None else str(depth)
        if depth == tree_params["max_depth"] and final_B == 100:
            dt, db, bp, sv, bv = st, sb, bag_pred, pv_tree, pv_bag
        else:
            dp = {"max_depth": depth, "min_samples_leaf": tree_params["min_samples_leaf"]}
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=tree_params["min_samples_leaf"],
                                        random_state=rep_seed + 500 + offset).fit(Xr, yr)
            db = BaggingModel.fit(Xr, yr, 100, dp, rep_seed + 600 + offset * 100)
            sv = aligned_proba(dt, Xte)
            bp = db.predictions(Xte, 100)
            bv = vote_matrix(bp) / 100.0
        ls = leaf_statistics(dt, Xr)
        result["depth"][label] = {
            "single": sv, "bag100": bv,
            "single_acc": accuracy_score(yte, sv.argmax(axis=1)),
            "bag_acc": accuracy_score(yte, bv.argmax(axis=1)),
            "leaf_count": ls["n_leaves"],
            "singleton_fraction": ls["singleton_leaf_fraction"],
            "disagreement": disagreement(bp),
        }
    return result


def savefig(path):
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def make_toy_example(outdir):
    X = np.arange(9, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0])
    w0 = np.full(len(y), 1.0 / len(y))
    tree = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X, y, sample_weight=w0)
    pred = tree.predict(X).astype(int)
    miss = pred != y
    err = float(np.dot(w0, miss))
    alpha = math.log((1 - err) / err) + math.log(2)
    w1 = w0 * np.exp(alpha * miss)
    w1 /= w1.sum()
    table = pd.DataFrame(
        {"sample": np.arange(1, 10), "x": X.ravel(), "class": y, "w_initial": w0, "prediction": pred,
         "incorrect": miss.astype(int), "w_updated": w1}
    )
    table.to_csv(outdir / "toy_samme_round.csv", index=False)
    return {"weighted_error": err, "alpha": alpha, "table": table.to_dict(orient="records")}


def bias_variance(pred_vectors, y_true):
    arr = np.asarray(pred_vectors)
    mean_p = arr.mean(axis=0)
    target = np.eye(N_CLASSES)[y_true]
    bias2 = float(np.mean(np.sum((mean_p - target) ** 2, axis=1)))
    variance = float(np.mean(np.sum((arr - mean_p[None, :, :]) ** 2, axis=2)))
    return bias2, variance


def select_examples(y, tree_pred, bag_votes, preferred=2):
    bag_pred = bag_votes.argmax(axis=1)
    corrected = np.flatnonzero((tree_pred != y) & (bag_pred == y))
    both_wrong = np.flatnonzero((tree_pred != y) & (bag_pred != y))
    return corrected[:preferred], both_wrong[:preferred]


def plot_face_cases(path, faces, y, tree_pred, bag_votes, corrected, both_wrong, title):
    chosen = list(corrected) + list(both_wrong)
    labels = ["bagging corrects tree"] * len(corrected) + ["both models wrong"] * len(both_wrong)
    if not chosen:
        return False
    fig, axes = plt.subplots(1, len(chosen), figsize=(3.0 * len(chosen), 3.2), squeeze=False)
    for ax, idx, label in zip(axes[0], chosen, labels):
        top = np.argsort(bag_votes[idx])[-3:][::-1]
        vote_text = ", ".join(f"{c}:{int(bag_votes[idx, c])}" for c in top)
        ax.imshow(faces[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{label}\ntrue {y[idx]}, tree {tree_pred[idx]}\nbag votes {vote_text}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    savefig(path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    root = args.root.resolve()
    for folder in ["data", "results", "figures", "logs", "config"]:
        (root / folder).mkdir(parents=True, exist_ok=True)

    overall_t0 = time.perf_counter()
    faces = fetch_olivetti_faces(data_home=root / "data" / "sklearn_cache", shuffle=False, download_if_missing=True)
    X = faces.data.astype(np.float32)
    images = faces.images.astype(np.float32)
    y = faces.target.astype(int)
    train_idx, val_idx, test_idx = split_by_identity(y)
    np.savez(root / "data" / "split_seed26004.npz", train=train_idx, validation=val_idx, test=test_idx)
    split_df = pd.DataFrame({"index": np.r_[train_idx, val_idx, test_idx],
                             "split": ["train"] * len(train_idx) + ["validation"] * len(val_idx) + ["test"] * len(test_idx)})
    split_df["identity"] = y[split_df["index"].to_numpy()]
    split_df.to_csv(root / "data" / "split_seed26004.csv", index=False)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]
    masked_images = images[test_idx].copy()
    masked_images[:, 24:40, 24:40] = 0.0
    Xmask = masked_images.reshape(len(masked_images), -1)

    # Q1(a): full validation-only grid.
    baseline_rows = []
    fitted_grid = {}
    for depth in [2, 4, 8, 16, None]:
        for leaf in [1, 2, 4]:
            t0 = time.perf_counter()
            model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
            model.fit(Xtr, ytr)
            fit_time = time.perf_counter() - t0
            stats = leaf_statistics(model, Xtr)
            row = {"max_depth": "None" if depth is None else depth, "min_samples_leaf": leaf,
                   "train_accuracy": model.score(Xtr, ytr), "validation_accuracy": model.score(Xva, yva),
                   "test_accuracy": model.score(Xte, yte), "fit_time_seconds": fit_time, **stats}
            baseline_rows.append(row)
            fitted_grid[(depth, leaf)] = model
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(root / "results" / "tree_grid.csv", index=False)
    # Validation accuracy is the only performance criterion. Simpler models break exact ties.
    depth_order = {2: 0, 4: 1, 8: 2, 16: 3, None: 4}
    selected_key = max(fitted_grid, key=lambda k: (fitted_grid[k].score(Xva, yva), -depth_order[k[0]], k[1]))
    tree = fitted_grid[selected_key]
    tree_params = {"max_depth": selected_key[0], "min_samples_leaf": selected_key[1]}
    tree_stats = leaf_statistics(tree, Xtr)

    # Q1(b,c): one 200-tree sequence, evaluated by prefixes.
    bag = BaggingModel.fit(Xtr, ytr, max(B_VALUES), tree_params, SEED + 1000, n_jobs=args.jobs)
    pred_tr_all = bag.predictions(Xtr)
    pred_va_all = bag.predictions(Xva)
    pred_te_all = bag.predictions(Xte)
    bag_rows = []
    for B in B_VALUES:
        tr_votes = vote_matrix(pred_tr_all[:B])
        va_votes = vote_matrix(pred_va_all[:B])
        te_votes = vote_matrix(pred_te_all[:B])
        oob_votes = np.zeros((len(ytr), N_CLASSES), dtype=float)
        oob_count = np.zeros(len(ytr), dtype=int)
        distinct = []
        for pred, sample in zip(pred_tr_all[:B], bag.bootstrap_indices[:B]):
            seen = np.zeros(len(ytr), dtype=bool)
            seen[np.unique(sample)] = True
            oob = ~seen
            oob_votes[np.flatnonzero(oob), pred[oob]] += 1
            oob_count[oob] += 1
            distinct.append(len(np.unique(sample)))
        covered = oob_count > 0
        oob_acc = accuracy_score(ytr[covered], oob_votes[covered].argmax(axis=1)) if covered.any() else np.nan
        bag_rows.append({
            "B": B,
            "train_accuracy": accuracy_score(ytr, tr_votes.argmax(axis=1)),
            "validation_accuracy": accuracy_score(yva, va_votes.argmax(axis=1)),
            "test_accuracy": accuracy_score(yte, te_votes.argmax(axis=1)),
            "mean_distinct_bootstrap_examples": np.mean(distinct),
            "theoretical_distinct_examples": len(ytr) * (1 - (1 - 1 / len(ytr)) ** len(ytr)),
            "oob_accuracy": oob_acc,
            "oob_coverage": np.mean(covered),
            "pairwise_disagreement_validation": disagreement(pred_va_all[:B]),
            "mean_vote_margin_validation": np.mean(winning_margin(va_votes)),
            "cumulative_fit_time_seconds": np.sum(bag.fit_times[:B]),
        })
    bag_df = pd.DataFrame(bag_rows)
    bag_df.to_csv(root / "results" / "bagging_by_B.csv", index=False)
    best_bag_row = bag_df.sort_values(["validation_accuracy", "B"], ascending=[False, True]).iloc[0]
    final_B = int(best_bag_row.B)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    axes[0].plot(bag_df.B, bag_df.oob_accuracy, "o-", color="#3366AA")
    axes[0].set_ylabel("OOB accuracy")
    axes[1].plot(bag_df.B, bag_df.pairwise_disagreement_validation, "o-", color="#AA3377")
    axes[1].set_ylabel("Pairwise disagreement")
    axes[2].plot(bag_df.B, bag_df.mean_vote_margin_validation, "o-", color="#228833")
    axes[2].set_ylabel("Mean vote margin")
    for ax in axes:
        ax.set_xlabel("Number of trees B")
        ax.set_xscale("log")
        ax.grid(alpha=.25)
    fig.tight_layout()
    savefig(root / "figures" / "bagging_metrics.pdf")

    q1d_rows = []
    for split_name, Xeval in [("clean", Xte), ("masked", Xmask)]:
        tp = tree.predict(Xeval).astype(int)
        tv = aligned_proba(tree, Xeval)
        bv = bag.scores(Xeval, final_B)
        q1d_rows.extend([
            {"model": "single tree", "test_set": split_name, "accuracy": accuracy_score(yte, tp),
             "mean_confidence": float(np.mean(confidence(tv)))},
            {"model": "bagging", "test_set": split_name, "accuracy": accuracy_score(yte, bv.argmax(axis=1)),
             "mean_confidence": float(np.mean(confidence(bv)))},
        ])
    pd.DataFrame(q1d_rows).to_csv(root / "results" / "tree_bag_clean_masked.csv", index=False)
    masked_tree_pred = tree.predict(Xmask).astype(int)
    masked_bag_votes = bag.scores(Xmask, final_B)
    corrected, both_wrong = select_examples(yte, masked_tree_pred, masked_bag_votes)
    source_faces, face_title = masked_images, "Masked test examples"
    if len(corrected) < 2 or len(both_wrong) < 2:
        clean_tree_pred = tree.predict(Xte).astype(int)
        clean_bag_votes = bag.scores(Xte, final_B)
        corrected, both_wrong = select_examples(yte, clean_tree_pred, clean_bag_votes)
        source_faces, face_title = images[test_idx], "Clean test examples"
        masked_tree_pred, masked_bag_votes = clean_tree_pred, clean_bag_votes
    plot_face_cases(root / "figures" / "bagging_examples.pdf", source_faces, yte, masked_tree_pred,
                    masked_bag_votes, corrected, both_wrong, face_title)
    case_rows = []
    for kind, arr in [("bagging_corrects_tree", corrected), ("both_wrong", both_wrong)]:
        for idx in arr:
            case_rows.append({"case": kind, "test_position": int(idx), "dataset_index": int(test_idx[idx]),
                              "true": int(yte[idx]), "tree_prediction": int(masked_tree_pred[idx]),
                              "bag_prediction": int(masked_bag_votes[idx].argmax()),
                              "bag_vote_counts": {str(k): int(v) for k, v in enumerate(masked_bag_votes[idx]) if v}})

    # Q2(a): one fully logged toy round.
    toy = make_toy_example(root / "results")

    # Q2(b): tune depth and rounds on clean validation only.
    boost_models = {}
    boost_rows = []
    for depth in BOOST_DEPTHS:
        model = SAMMEModel.fit(Xtr, ytr, max(BOOST_ROUNDS), depth, SEED + 2000 + depth)
        boost_models[depth] = model
        for rounds in BOOST_ROUNDS:
            used = min(rounds, len(model.trees))
            hist = model.history[used - 1] if used else None
            boost_rows.append({
                "depth": depth, "requested_rounds": rounds, "rounds_used": used,
                "train_accuracy": accuracy_score(ytr, model.predict(Xtr, used)),
                "validation_accuracy": accuracy_score(yva, model.predict(Xva, used)),
                "test_accuracy": accuracy_score(yte, model.predict(Xte, used)),
                "weighted_training_error_last": hist["weighted_error"] if hist else np.nan,
                "learner_weight_last": hist["alpha"] if hist else np.nan,
                "maximum_sample_weight_last": hist["max_sample_weight"] if hist else np.nan,
                "top10_weight_last": hist["top10_weight"] if hist else np.nan,
                "cumulative_fit_time_seconds": sum(h["fit_time"] for h in model.history[:used]),
                "stopped_reason": model.stopped_reason,
            })
    boost_df = pd.DataFrame(boost_rows)
    boost_df.to_csv(root / "results" / "boosting_grid.csv", index=False)
    best_boost_row = boost_df.sort_values(["validation_accuracy", "requested_rounds", "depth"],
                                          ascending=[False, True, True]).iloc[0]
    final_depth = int(best_boost_row.depth)
    final_rounds = int(best_boost_row.rounds_used)
    boost = boost_models[final_depth]

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.8))
    h = boost.history[:final_rounds]
    rr = [x["round"] for x in h]
    axes[0, 0].plot(rr, [x["weighted_error"] for x in h], color="#3366AA")
    axes[0, 0].set_ylabel("Weighted error")
    axes[0, 1].plot(rr, [x["alpha"] for x in h], color="#AA3377")
    axes[0, 1].set_ylabel("Learner weight")
    axes[1, 0].plot(rr, [x["max_sample_weight"] for x in h], color="#228833")
    axes[1, 0].set_ylabel("Maximum sample weight")
    axes[1, 1].plot(rr, [x["top10_weight"] for x in h], color="#CC6677")
    axes[1, 1].set_ylabel("Weight on 10 heaviest")
    for ax in axes.ravel():
        ax.set_xlabel("Round")
        ax.grid(alpha=.25)
    fig.tight_layout()
    savefig(root / "figures" / "boosting_dynamics.pdf")

    compare_rows = []
    for learners in [25, 50, 100]:
        for model_name in ["bagging", "boosting"]:
            if model_name == "bagging":
                used = learners
                predict = lambda z, u=used: bag.predict(z, u)
                fit_t = float(np.sum(bag.fit_times[:used]))
            else:
                used = min(learners, len(boost.trees))
                predict = lambda z, u=used: boost.predict(z, u)
                fit_t = float(sum(hh["fit_time"] for hh in boost.history[:used]))
            compare_rows.append({"model": model_name, "learners": learners, "learners_used": used,
                                 "train_accuracy": accuracy_score(ytr, predict(Xtr)),
                                 "validation_accuracy": accuracy_score(yva, predict(Xva)),
                                 "clean_test_accuracy": accuracy_score(yte, predict(Xte)),
                                 "training_time_seconds": fit_t})
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(root / "results" / "bagging_vs_boosting.csv", index=False)

    # A training image with the largest weight at any selected round.
    weight_history = np.asarray([hh["weights"] for hh in boost.history[:final_rounds]])
    heavy_train_position = int(np.unravel_index(np.argmax(weight_history), weight_history.shape)[1])
    heavy_identity = int(ytr[heavy_train_position])
    heavy_weights = weight_history[:, heavy_train_position]
    heavy_predictions = np.asarray([t.predict(Xtr[heavy_train_position:heavy_train_position + 1])[0]
                                    for t in boost.trees[:final_rounds]])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].imshow(images[train_idx[heavy_train_position]], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Training face: true class {heavy_identity}")
    axes[0].axis("off")
    axes[1].plot(np.arange(1, final_rounds + 1), heavy_weights, color="#AA3377", label="sample weight")
    wrong_rounds = np.flatnonzero(heavy_predictions != heavy_identity) + 1
    if len(wrong_rounds):
        axes[1].scatter(wrong_rounds, heavy_weights[wrong_rounds - 1], s=16, color="#CC3311", label="misclassified")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Normalised sample weight")
    axes[1].grid(alpha=.25)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    savefig(root / "figures" / "boosting_heavy_face.pdf")

    # Q2(d): three frozen models on clean and masked tests.
    q2d_rows = []
    for split_name, Xeval in [("clean", Xte), ("masked", Xmask)]:
        scores_by_model = {
            "single tree": aligned_proba(tree, Xeval),
            "bagging": bag.scores(Xeval, final_B),
            "boosting": boost.scores(Xeval, final_rounds),
        }
        for model_name, scores in scores_by_model.items():
            q2d_rows.append({"model": model_name, "test_set": split_name,
                             "accuracy": accuracy_score(yte, scores.argmax(axis=1)),
                             "mean_winning_vote_or_probability": float(np.mean(confidence(scores)))})
    q2d_df = pd.DataFrame(q2d_rows)
    q2d_df.to_csv(root / "results" / "frozen_models_clean_masked.csv", index=False)

    bag_clean_pred = bag.predict(Xte, final_B)
    bag_mask_pred = bag.predict(Xmask, final_B)
    boost_clean_pred = boost.predict(Xte, final_rounds)
    boost_mask_pred = boost.predict(Xmask, final_rounds)
    response_candidates = np.flatnonzero((bag_mask_pred != boost_mask_pred) &
                                         ((bag_clean_pred == yte) | (boost_clean_pred == yte)))
    response_idx = int(response_candidates[0]) if len(response_candidates) else int(np.flatnonzero(bag_mask_pred != boost_mask_pred)[0])
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8))
    axes[0].imshow(images[test_idx[response_idx]], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Clean")
    axes[1].imshow(masked_images[response_idx], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Masked")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"True {yte[response_idx]} | bag {bag_clean_pred[response_idx]}→{bag_mask_pred[response_idx]} | "
                 f"boost {boost_clean_pred[response_idx]}→{boost_mask_pred[response_idx]}", fontsize=9)
    fig.tight_layout()
    savefig(root / "figures" / "bag_boost_mask_response.pdf")

    # Q3: 30 within-identity resamples. Training data are fixed; only row multiplicities change.
    selected_vectors = {"single tree": [], "bagging": [], "boosting": []}
    selected_acc = {k: [] for k in selected_vectors}
    selected_disagreement = []
    selected_leaf = []
    depth_candidates = [2, selected_key[0], None]
    # Preserve three distinct settings if the selected tree is shallow or unbounded.
    if len({str(x) for x in depth_candidates}) < 3:
        depth_candidates = [2, 8, None]
    depth_labels = ["None" if d is None else str(d) for d in depth_candidates]
    depth_stats = {label: {"single": [], "bag100": [], "single_acc": [], "bag_acc": [],
                           "leaf_count": [], "singleton_fraction": [], "disagreement": []}
                   for label in depth_labels}
    print(f"running {args.repetitions} resamples with {args.jobs} parallel workers", flush=True)
    rep_results = Parallel(n_jobs=args.jobs, verbose=10, max_nbytes="2M")(
        delayed(run_resample_rep)(rep, Xtr, ytr, Xte, yte, tree_params, final_B,
                                  final_depth, final_rounds, depth_candidates)
        for rep in range(args.repetitions)
    )
    for result in sorted(rep_results, key=lambda x: x["rep"]):
        for name in selected_vectors:
            selected_vectors[name].append(result["selected_vectors"][name])
            selected_acc[name].append(result["selected_acc"][name])
        selected_disagreement.append(result["selected_disagreement"])
        selected_leaf.append(result["selected_leaf"])
        for label in depth_labels:
            for key in depth_stats[label]:
                depth_stats[label][key].append(result["depth"][label][key])

    bv_rows = []
    for name in selected_vectors:
        b2, var = bias_variance(selected_vectors[name], yte)
        bv_rows.append({"model": name, "bias_squared": b2, "variance": var,
                        "mean_test_accuracy": np.mean(selected_acc[name]),
                        "std_test_accuracy": np.std(selected_acc[name], ddof=1)})
    bv_df = pd.DataFrame(bv_rows)
    bv_df.to_csv(root / "results" / "selected_model_bias_variance.csv", index=False)
    depth_rows = []
    for label in depth_labels:
        s_b2, s_var = bias_variance(depth_stats[label]["single"], yte)
        b_b2, b_var = bias_variance(depth_stats[label]["bag100"], yte)
        for name, b2, var, acc in [("single tree", s_b2, s_var, depth_stats[label]["single_acc"]),
                                   ("bagging-100", b_b2, b_var, depth_stats[label]["bag_acc"])]:
            depth_rows.append({"depth": label, "model": name, "bias_squared": b2, "variance": var,
                               "mean_test_accuracy": np.mean(acc), "std_test_accuracy": np.std(acc, ddof=1),
                               "mean_leaf_count": np.mean(depth_stats[label]["leaf_count"]),
                               "mean_singleton_leaf_fraction": np.mean(depth_stats[label]["singleton_fraction"]),
                               "mean_pairwise_disagreement": np.mean(depth_stats[label]["disagreement"])})
    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(root / "results" / "depth_bias_variance.csv", index=False)

    x = np.arange(len(depth_labels))
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    for model_name, marker, color in [("single tree", "o", "#3366AA"), ("bagging-100", "s", "#AA3377")]:
        d = depth_df[depth_df.model == model_name]
        axes[0].plot(x, d.bias_squared, marker=marker, color=color, label=model_name)
        axes[1].plot(x, d.variance, marker=marker, color=color, label=model_name)
        axes[2].plot(x, d.mean_test_accuracy, marker=marker, color=color, label=model_name)
    for ax, title in zip(axes, ["Bias squared", "Variance", "Mean test accuracy"]):
        ax.set_xticks(x, depth_labels)
        ax.set_xlabel("Maximum depth")
        ax.set_title(title)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    savefig(root / "figures" / "depth_bias_variance.pdf")

    # Final requested table combines fixed-test clean/masked performance and resampling measurements.
    final_rows = []
    for model_name in ["single tree", "bagging", "boosting"]:
        clean = q2d_df[(q2d_df.model == model_name) & (q2d_df.test_set == "clean")].iloc[0]
        masked = q2d_df[(q2d_df.model == model_name) & (q2d_df.test_set == "masked")].iloc[0]
        bv = bv_df[bv_df.model == model_name].iloc[0]
        final_rows.append({"model": model_name, "clean_test_accuracy": clean.accuracy,
                           "masked_test_accuracy": masked.accuracy, "bias_squared": bv.bias_squared,
                           "variance": bv.variance})
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(root / "results" / "final_summary.csv", index=False)

    summary = {
        "student": {"name": "Jiya Yadav", "roll_number": "M26CSE004", "seed": SEED},
        "dataset": {"samples": len(X), "identities": N_CLASSES, "image_shape": [64, 64],
                    "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
                    "mask": "rows 24:40 and columns 24:40 set to zero"},
        "selected_tree": {**tree_params, **tree_stats,
                          "train_accuracy": tree.score(Xtr, ytr), "validation_accuracy": tree.score(Xva, yva),
                          "clean_test_accuracy": tree.score(Xte, yte)},
        "selected_bagging": {"B": final_B, **best_bag_row.to_dict()},
        "selected_boosting": {"depth": final_depth, "rounds": final_rounds, **best_boost_row.to_dict()},
        "toy_samme": toy,
        "bagging_examples": case_rows,
        "heavy_boosting_face": {"training_position": heavy_train_position,
                                "dataset_index": int(train_idx[heavy_train_position]), "true_class": heavy_identity,
                                "maximum_weight": float(heavy_weights.max()),
                                "misclassified_rounds": wrong_rounds.tolist()},
        "mask_response_face": {"test_position": response_idx, "dataset_index": int(test_idx[response_idx]),
                               "true": int(yte[response_idx]), "bag_clean": int(bag_clean_pred[response_idx]),
                               "bag_masked": int(bag_mask_pred[response_idx]),
                               "boost_clean": int(boost_clean_pred[response_idx]),
                               "boost_masked": int(boost_mask_pred[response_idx])},
        "selected_ensemble_disagreement_mean_over_resamples": np.mean(selected_disagreement),
        "selected_tree_leaf_stats_mean_over_resamples": {
            "leaves": np.mean([x["n_leaves"] for x in selected_leaf]),
            "singleton_fraction": np.mean([x["singleton_leaf_fraction"] for x in selected_leaf]),
        },
        "runtime_seconds": time.perf_counter() - overall_t0,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    with open(root / "results" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=jsonable)
    with open(root / "logs" / "final_run.log", "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2, default=jsonable))
        f.write("\n")


if __name__ == "__main__":
    main()
