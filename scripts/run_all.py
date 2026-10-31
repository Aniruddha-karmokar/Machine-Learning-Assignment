import os
import sys
import time
import json
import ctypes
import numpy as np

# Load C CART acceleration library
lib_path = os.path.abspath("artifacts/file_generation/ttl=63d/intermediate/libfastcart.so")
lib = ctypes.CDLL(lib_path)

# ---------------------------------------------------------
# 0. Dataset & Split Setup (Roll No. P26CS0001 -> seed 260001)
# ---------------------------------------------------------
ROLL_NO = "P26CS0001"
STUDENT_NAME = "Aniruddha Karmokar"
SEED = 260001  # All numeric characters of P26CS0001 in order

d = np.load("repo_P26CS0001/data/olivetti_faces.npz")
X_all = d["data"].astype(np.float32)
images_all = d["images"].astype(np.float32)
y_all = d["target"].astype(np.int32)

rng_split = np.random.default_rng(SEED)
train_idx, val_idx, test_idx = [], [], []
for cid in range(40):
    idx_c = np.arange(cid * 10, (cid + 1) * 10)
    perm = rng_split.permutation(idx_c)
    train_idx.extend(perm[:6])
    val_idx.extend(perm[6:8])
    test_idx.extend(perm[8:10])

train_idx = np.array(train_idx, dtype=np.int32)
val_idx = np.array(val_idx, dtype=np.int32)
test_idx = np.array(test_idx, dtype=np.int32)

os.makedirs("repo_P26CS0001/data", exist_ok=True)
os.makedirs("repo_P26CS0001/logs", exist_ok=True)
os.makedirs("repo_P26CS0001/configs", exist_ok=True)

np.savez(
    "repo_P26CS0001/data/split_P26CS0001.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    seed=np.array(SEED, dtype=np.int64)
)
with open("repo_P26CS0001/data/split_P26CS0001.json", "w") as f:
    json.dump({
        "roll_number": ROLL_NO,
        "student_name": STUDENT_NAME,
        "seed": SEED,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist()
    }, f, indent=2)

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
images_train = images_all[train_idx]
images_val = images_all[val_idx]
images_test = images_all[test_idx]

# Masked test set: centred 16x16 block, rows and columns 24 to 39, set to zero
images_test_masked = images_test.copy()
images_test_masked[:, 24:40, 24:40] = 0.0
X_test_masked = images_test_masked.reshape(len(test_idx), -1)

X_T_global = np.ascontiguousarray(X_train.T, dtype=np.float32)
presorted_global = np.ascontiguousarray(np.argsort(X_train, axis=0).T.astype(np.int32))
feat_order_ident = np.arange(4096, dtype=np.int32)

# ---------------------------------------------------------
# Helper classes for DecisionTree, Bagging, and SAMME Boosting
# ---------------------------------------------------------
class CARTDecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = -1 if max_depth is None else int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)

    def fit_indexed(self, X_T, y_tr, presorted, counts, weights=None, is_int=1, feat_order=None):
        N_total = len(y_tr)
        u = np.nonzero(counts > 0)[0].astype(np.int32)
        n_unique = len(u)
        w = counts.astype(np.float64) if weights is None else weights.astype(np.float64)
        if feat_order is None:
            feat_order = feat_order_ident
        max_nodes = 512
        self.f = np.zeros(max_nodes, dtype=np.int32)
        self.t = np.zeros(max_nodes, dtype=np.float64)
        self.l = np.zeros(max_nodes, dtype=np.int32)
        self.r = np.zeros(max_nodes, dtype=np.int32)
        self.n_s = np.zeros(max_nodes, dtype=np.int32)
        self.imp = np.zeros(max_nodes, dtype=np.float64)
        self.val = np.zeros((max_nodes, 40), dtype=np.float64)

        n_nodes = lib.fit_tree_general_c(
            0,
            X_T.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N_total,
            y_tr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            int(is_int),
            u.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), n_unique,
            presorted.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            4096, 40, self.max_depth, self.min_samples_leaf,
            feat_order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            max_nodes,
            self.f.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.l.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.r.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.n_s.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self.imp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.val.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        self.n_nodes = n_nodes
        self.f = self.f[:n_nodes].copy()
        self.t = self.t[:n_nodes].copy()
        self.l = self.l[:n_nodes].copy()
        self.r = self.r[:n_nodes].copy()
        self.n_s = self.n_s[:n_nodes].copy()
        self.imp = self.imp[:n_nodes].copy()
        self.val = self.val[:n_nodes].copy()
        return self

    def predict_proba(self, X):
        N = X.shape[0]
        out = np.zeros((N, 40), dtype=np.float64)
        f, t, l, r, val = self.f, self.t, self.l, self.r, self.val
        for i in range(N):
            node = 0
            xi = X[i]
            while f[node] != -1:
                if xi[f[node]] <= t[node]:
                    node = l[node]
                else:
                    node = r[node]
            out[i] = val[node]
        return out

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def tree_depth(self):
        def _d(node):
            if self.f[node] == -1:
                return 0
            return 1 + max(_d(self.l[node]), _d(self.r[node]))
        return _d(0)

    def leaf_stats(self):
        leaf_mask = (self.f == -1)
        leaf_counts = self.n_s[leaf_mask]
        n_leaves = int(np.sum(leaf_mask))
        mean_samples = float(np.mean(leaf_counts))
        frac_one = float(np.mean(leaf_counts == 1))
        return n_leaves, mean_samples, frac_one, self.tree_depth()

def fit_bagging_ensemble_fast(B, max_depth, min_samples_leaf, rng, base_counts=None):
    """
    Fits B bootstrap trees in parallel via OpenMP.
    If base_counts is None, resamples 240 indices uniformly from 0..239.
    If base_counts is provided (for Q3 resampled training sets of size 240),
    expanded_pool has 240 elements where index i appears base_counts[i] times,
    and each bootstrap tree draws 240 with replacement from expanded_pool!
    """
    md_c = -1 if max_depth is None else int(max_depth)
    all_w = np.zeros((B, 240), dtype=np.float64)
    all_counts = np.zeros((B, 240), dtype=np.int32)
    all_root_idx = np.zeros((B, 240), dtype=np.int32)
    all_n_unique = np.zeros(B, dtype=np.int32)
    all_feat_order = np.zeros((B, 4096), dtype=np.int32)
    in_bag_mask = np.zeros((B, 240), dtype=bool)

    if base_counts is None:
        pool_indices = np.arange(240, dtype=np.int32)
    else:
        pool_indices = np.repeat(np.arange(240, dtype=np.int32), base_counts)

    for b in range(B):
        draw_pos = rng.integers(0, len(pool_indices), size=len(pool_indices))
        boot = pool_indices[draw_pos]
        u, c = np.unique(boot, return_counts=True)
        all_counts[b, u] = c
        all_w[b, u] = c.astype(np.float64)
        all_n_unique[b] = len(u)
        all_root_idx[b, :len(u)] = u
        in_bag_mask[b, u] = True
        all_feat_order[b] = rng.permutation(4096).astype(np.int32)

    max_nodes = 512
    out_n_nodes = np.zeros(B, dtype=np.int32)
    out_f = np.zeros((B, max_nodes), dtype=np.int32)
    out_t = np.zeros((B, max_nodes), dtype=np.float64)
    out_l = np.zeros((B, max_nodes), dtype=np.int32)
    out_r = np.zeros((B, max_nodes), dtype=np.int32)
    out_n = np.zeros((B, max_nodes), dtype=np.int32)
    out_imp = np.zeros((B, max_nodes), dtype=np.float64)
    out_val = np.zeros((B, max_nodes, 40), dtype=np.float64)

    lib.fit_ensemble_c(
        B,
        X_T_global.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 240,
        y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        all_w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        all_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        1,
        all_root_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        all_n_unique.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        presorted_global.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        4096, 40, md_c, int(min_samples_leaf),
        all_feat_order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        max_nodes,
        out_n_nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_f.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_r.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_n.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_imp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    return {
        "B": B,
        "n_nodes": out_n_nodes,
        "f": out_f,
        "t": out_t,
        "l": out_l,
        "r": out_r,
        "n_s": out_n,
        "val": out_val,
        "n_unique": all_n_unique,
        "in_bag_mask": in_bag_mask
    }

def eval_ensemble_trees(ens, X):
    """Returns tree predictions (B, N) int32 and vote fractions (N, 40) float64 via C kernel."""
    B = int(ens["B"])
    N = int(X.shape[0])
    D = int(X.shape[1])
    max_nodes = int(ens["f"].shape[1])
    X_c = np.ascontiguousarray(X, dtype=np.float32)
    tree_preds = np.zeros((B, N), dtype=np.int32)
    vote_probs = np.zeros((N, 40), dtype=np.float64)
    lib.predict_ensemble_c(
        B, max_nodes, 40,
        ens["f"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ens["t"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ens["l"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ens["r"].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ens["val"].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        N, D,
        tree_preds.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        vote_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    return tree_preds, vote_probs

print("Starting Q1 experiments...")
t_start_all = time.time()

# ---------------------------------------------------------
# Q1(a): Single Decision Tree Baseline Grid
# ---------------------------------------------------------
q1a_results = []
counts_clean = np.ones(240, dtype=np.int32)
for md in [2, 4, 8, 16, None]:
    for msl in [1, 2, 4]:
        tree = CARTDecisionTree(max_depth=md, min_samples_leaf=msl).fit_indexed(
            X_T_global, y_train, presorted_global, counts_clean, is_int=1
        )
        tr_acc = float(np.mean(tree.predict(X_train) == y_train))
        va_acc = float(np.mean(tree.predict(X_val) == y_val))
        te_acc = float(np.mean(tree.predict(X_test) == y_test))
        nl, ms, f1, actual_d = tree.leaf_stats()
        q1a_results.append({
            "max_depth": md,
            "min_samples_leaf": msl,
            "actual_depth": actual_d,
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "test_acc": te_acc,
            "n_leaves": nl,
            "mean_samples_per_leaf": ms,
            "frac_leaves_one_sample": f1
        })

# Select setting using validation accuracy only (preferring simpler/bounded depth if tied)
best_q1a = max(q1a_results, key=lambda r: (r["val_acc"], -r["min_samples_leaf"], -(999 if r["max_depth"] is None else r["max_depth"])))
FROZEN_DEPTH = best_q1a["max_depth"]
FROZEN_MSL = best_q1a["min_samples_leaf"]
print(f"Q1(a) Selected Frozen Tree: max_depth={FROZEN_DEPTH}, min_samples_leaf={FROZEN_MSL}, val_acc={best_q1a['val_acc']:.4f}, test_acc={best_q1a['test_acc']:.4f}")

frozen_single_tree = CARTDecisionTree(max_depth=FROZEN_DEPTH, min_samples_leaf=FROZEN_MSL).fit_indexed(
    X_T_global, y_train, presorted_global, counts_clean, is_int=1
)

# ---------------------------------------------------------
# Q1(b) & Q1(c): Bagging from Scratch for B in {1, 5, 10, 25, 50, 100, 200}
# ---------------------------------------------------------
B_LIST = [1, 5, 10, 25, 50, 100, 200]
theo_distinct = 240.0 * (1.0 - (239.0 / 240.0) ** 240)
asymp_distinct = 240.0 * (1.0 - np.exp(-1.0))

q1bc_results = []
ens_by_B = {}
for B in B_LIST:
    rng_b = np.random.default_rng(SEED + B * 100)
    t0 = time.time()
    ens = fit_bagging_ensemble_fast(B, FROZEN_DEPTH, FROZEN_MSL, rng_b)
    fit_time = time.time() - t0
    ens_by_B[B] = ens

    tp_tr, v_tr = eval_ensemble_trees(ens, X_train)
    tp_va, v_va = eval_ensemble_trees(ens, X_val)
    tp_te, v_te = eval_ensemble_trees(ens, X_test)

    tr_acc = float(np.mean(np.argmax(v_tr, axis=1) == y_train))
    va_acc = float(np.mean(np.argmax(v_va, axis=1) == y_val))
    te_acc = float(np.mean(np.argmax(v_te, axis=1) == y_test))
    avg_distinct = float(np.mean(ens["n_unique"]))

    # Out-of-bag accuracy on X_train
    oob_votes = np.zeros((240, 40), dtype=np.float64)
    oob_counts = np.zeros(240, dtype=np.int32)
    in_bag = ens["in_bag_mask"]
    for b in range(B):
        oob_idx = np.nonzero(~in_bag[b])[0]
        oob_counts[oob_idx] += 1
        for i in oob_idx:
            oob_votes[i, tp_tr[b, i]] += 1.0
    valid_oob = oob_counts > 0
    if np.any(valid_oob):
        oob_preds = np.argmax(oob_votes[valid_oob], axis=1)
        oob_acc = float(np.mean(oob_preds == y_train[valid_oob]))
        oob_cov = float(np.mean(valid_oob))
    else:
        oob_acc = 0.0
        oob_cov = 0.0

    # Average pairwise disagreement between trees (on validation and test sets)
    if B > 1:
        dis_va_sum, dis_te_sum, n_pairs = 0.0, 0.0, 0
        for b1 in range(B):
            # Vectorized over b2 > b1
            dis_va_sum += float(np.sum(tp_va[b1+1:] != tp_va[b1])) / X_val.shape[0]
            dis_te_sum += float(np.sum(tp_te[b1+1:] != tp_te[b1])) / X_test.shape[0]
            n_pairs += (B - 1 - b1)
        pairwise_dis_val = dis_va_sum / n_pairs
        pairwise_dis_test = dis_te_sum / n_pairs
    else:
        pairwise_dis_val = 0.0
        pairwise_dis_test = 0.0

    # Mean vote margin: winning vote fraction minus runner-up vote fraction
    sorted_v_va = np.sort(v_va, axis=1)
    margin_val = float(np.mean(sorted_v_va[:, -1] - sorted_v_va[:, -2]))
    sorted_v_te = np.sort(v_te, axis=1)
    margin_test = float(np.mean(sorted_v_te[:, -1] - sorted_v_te[:, -2]))

    res_b = {
        "B": B,
        "train_acc": tr_acc,
        "val_acc": va_acc,
        "test_acc": te_acc,
        "avg_distinct_bootstrap": avg_distinct,
        "std_distinct_bootstrap": float(np.std(ens["n_unique"])),
        "theoretical_distinct": float(theo_distinct),
        "asymptotic_distinct": float(asymp_distinct),
        "oob_acc": oob_acc,
        "oob_coverage": oob_cov,
        "pairwise_disagreement_val": pairwise_dis_val,
        "pairwise_disagreement_test": pairwise_dis_test,
        "mean_vote_margin_val": margin_val,
        "mean_vote_margin_test": margin_test,
        "fit_time_sec": fit_time
    }
    q1bc_results.append(res_b)
    print(f"B={B:3d} | tr={tr_acc:.4f} va={va_acc:.4f} te={te_acc:.4f} | distinct={avg_distinct:.2f} (theo={theo_distinct:.2f}) | oob={oob_acc:.4f} dis_te={pairwise_dis_test:.4f} margin_te={margin_test:.4f}")

best_bag_entry = max(q1bc_results, key=lambda r: (r["val_acc"], r["oob_acc"], -r["B"]))
FINAL_B = best_bag_entry["B"]
print(f"Q1(c) Selected Final Ensemble Size B* = {FINAL_B} (val_acc={best_bag_entry['val_acc']:.4f}, test_acc={best_bag_entry['test_acc']:.4f})")
final_bag_ens = ens_by_B[FINAL_B]

# ---------------------------------------------------------
# Q1(d): Clean vs Masked Test Set Comparison & Case Studies
# ---------------------------------------------------------
st_prob_clean = frozen_single_tree.predict_proba(X_test)
st_pred_clean = np.argmax(st_prob_clean, axis=1)
st_conf_clean = float(np.mean(np.max(st_prob_clean, axis=1)))
st_acc_clean = float(np.mean(st_pred_clean == y_test))

st_prob_mask = frozen_single_tree.predict_proba(X_test_masked)
st_pred_mask = np.argmax(st_prob_mask, axis=1)
st_conf_mask = float(np.mean(np.max(st_prob_mask, axis=1)))
st_acc_mask = float(np.mean(st_pred_mask == y_test))

_, bag_prob_clean = eval_ensemble_trees(final_bag_ens, X_test)
bag_pred_clean = np.argmax(bag_prob_clean, axis=1)
bag_conf_clean = float(np.mean(np.max(bag_prob_clean, axis=1)))
bag_acc_clean = float(np.mean(bag_pred_clean == y_test))

_, bag_prob_mask = eval_ensemble_trees(final_bag_ens, X_test_masked)
bag_pred_mask = np.argmax(bag_prob_mask, axis=1)
bag_conf_mask = float(np.mean(np.max(bag_prob_mask, axis=1)))
bag_acc_mask = float(np.mean(bag_pred_mask == y_test))

print(f"Q1(d) Single Tree: Clean acc={st_acc_clean:.4f} (conf={st_conf_clean:.4f}), Masked acc={st_acc_mask:.4f} (conf={st_conf_mask:.4f})")
print(f"Q1(d) Bagging B={FINAL_B}: Clean acc={bag_acc_clean:.4f} (conf={bag_conf_clean:.4f}), Masked acc={bag_acc_mask:.4f} (conf={bag_conf_mask:.4f})")

# Find 2 faces where bagging corrects the single tree and 2 where both are wrong
bag_corrects_idx = np.nonzero((st_pred_clean != y_test) & (bag_pred_clean == y_test))[0]
both_wrong_idx = np.nonzero((st_pred_clean != y_test) & (bag_pred_clean != y_test))[0]

def format_vote_info(i):
    vote_counts = np.round(bag_prob_clean[i] * FINAL_B).astype(int)
    top_k = np.argsort(vote_counts)[::-1][:5]
    return {
        "test_pos": int(i),
        "global_dataset_idx": int(test_idx[i]),
        "true_identity": int(y_test[i]),
        "single_tree_pred": int(st_pred_clean[i]),
        "single_tree_conf": float(np.max(st_prob_clean[i])),
        "bagging_pred": int(bag_pred_clean[i]),
        "bagging_conf": float(np.max(bag_prob_clean[i])),
        "bagging_true_class_votes": int(vote_counts[y_test[i]]),
        "bagging_top5_votes": [{"class": int(k), "votes": int(vote_counts[k]), "frac": float(bag_prob_clean[i, k])} for k in top_k if vote_counts[k] > 0]
    }

q1d_cases = {
    "bagging_corrects_tree": [format_vote_info(i) for i in bag_corrects_idx[:4]],
    "both_wrong": [format_vote_info(i) for i in both_wrong_idx[:4]]
}
print("Q1(d) Bagging corrects tree examples:", q1d_cases["bagging_corrects_tree"][:2])
print("Q1(d) Both wrong examples:", q1d_cases["both_wrong"][:2])

# ---------------------------------------------------------
# Shared SAMME Update Rule Function (used in both Q2(a) and Q2(b)-(d))
# ---------------------------------------------------------
def samme_round_update(w, y_true, y_pred, K):
    w = np.asarray(w, dtype=np.float64)
    w = w / np.sum(w)
    incorrect = (y_pred != y_true)
    eps_t = float(np.sum(w * incorrect))

    random_err_thresh = (K - 1.0) / K
    if eps_t >= random_err_thresh - 1e-12:
        return w.copy(), eps_t, 0.0, False

    eps_clipped = max(eps_t, 1e-12)
    alpha_t = float(np.log((1.0 - eps_clipped) / eps_clipped) + np.log(K - 1.0))
    w_unnorm = w * np.exp(alpha_t * incorrect.astype(np.float64))
    w_next = w_unnorm / np.sum(w_unnorm)
    return w_next, eps_t, alpha_t, True

# ---------------------------------------------------------
# Q2(a): Three-Class Toy Example (N=8 <= 10 samples)
# ---------------------------------------------------------
X_toy = np.array([
    [0.10, 0.20], [0.15, 0.25], [0.20, 0.80],
    [0.50, 0.50], [0.55, 0.45], [0.48, 0.52],
    [0.85, 0.15], [0.90, 0.20]
], dtype=np.float32)
y_toy = np.array([0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int32)
K_toy = 3
N_toy = len(y_toy)
w_toy_init = np.ones(N_toy, dtype=np.float64) / N_toy
h1_toy_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
w_toy_next, eps_toy, alpha_toy, valid_toy = samme_round_update(w_toy_init, y_toy, h1_toy_pred, K=K_toy)

q2a_log = {
    "K": K_toy,
    "N": N_toy,
    "X_toy": X_toy.tolist(),
    "y_true": y_toy.tolist(),
    "initial_weights": [round(float(x), 6) for x in w_toy_init],
    "weak_learner_predictions": h1_toy_pred.tolist(),
    "misclassified_indices": np.nonzero(h1_toy_pred != y_toy)[0].tolist(),
    "weighted_error_eps": float(eps_toy),
    "alpha_t": float(alpha_toy),
    "updated_normalised_weights": [round(float(x), 6) for x in w_toy_next]
}
print("=== Q2(a) Toy 3-Class SAMME Round ===")
print(json.dumps(q2a_log, indent=2))

# ---------------------------------------------------------
# Q2(b): SAMME AdaBoost on Olivetti Faces (K=40)
# ---------------------------------------------------------
def fit_samme_olivetti(max_depth, max_rounds, counts_base=None, X_T=X_T_global, presorted=presorted_global, seed_offset=0):
    K = 40
    if counts_base is None:
        counts = np.ones(240, dtype=np.int32)
    else:
        counts = counts_base.astype(np.int32)

    w = counts.astype(np.float64)
    w = w / np.sum(w)

    learners = []
    alphas = []
    eps_history = []
    max_w_history = []
    top10_w_history = []
    weight_matrix_history = []
    pred_train_history = []
    round_times = {}

    rng_boost = np.random.default_rng(SEED + 5000 + max_depth * 100 + seed_offset)
    t0 = time.time()

    for t in range(1, max_rounds + 1):
        w_before = w.copy()
        max_w_history.append(float(np.max(w_before)))
        sorted_w = np.sort(w_before)[::-1]
        top10_w_history.append(float(np.sum(sorted_w[:10])))
        weight_matrix_history.append(w_before)

        feat_order = rng_boost.permutation(4096).astype(np.int32)
        tree = CARTDecisionTree(max_depth=max_depth, min_samples_leaf=1).fit_indexed(
            X_T, y_train, presorted, counts, weights=w, is_int=0, feat_order=feat_order
        )
        pred_tr = tree.predict(X_train)
        w_next, eps_t, alpha_t, is_valid = samme_round_update(w, y_train, pred_tr, K=K)

        if not is_valid:
            w_pert = w * rng_boost.uniform(0.8, 1.2, size=240) * (counts > 0)
            w = w_pert / np.sum(w_pert)
            continue

        learners.append(tree)
        alphas.append(alpha_t)
        eps_history.append(eps_t)
        pred_train_history.append(pred_tr.copy())
        w = w_next

        if len(learners) in [10, 25, 50, 100]:
            round_times[len(learners)] = time.time() - t0

    return {
        "max_depth": max_depth,
        "learners": learners,
        "alphas": np.array(alphas, dtype=np.float64),
        "eps_history": eps_history,
        "max_w_history": max_w_history,
        "top10_w_history": top10_w_history,
        "weight_matrix_history": np.array(weight_matrix_history),
        "pred_train_history": np.array(pred_train_history),
        "round_times": round_times
    }

def eval_samme_prefix(model, X, n_rounds):
    N = X.shape[0]
    K = 40
    S = np.zeros((N, K), dtype=np.float64)
    learners = model["learners"][:n_rounds]
    alphas = model["alphas"][:n_rounds]
    total_alpha = float(np.sum(alphas))
    if total_alpha <= 0.0 or len(learners) == 0:
        return np.full((N, K), 1.0 / K, dtype=np.float64)
    for tree, alpha_t in zip(learners, alphas):
        preds = tree.predict(X)
        S[np.arange(N), preds] += alpha_t
    return S / total_alpha

print("\n=== Q2(b) SAMME Grid on Olivetti Faces ===")
q2b_grid = []
samme_models_by_depth = {}
for d_weak in [1, 2, 3, 4]:
    m_d = fit_samme_olivetti(max_depth=d_weak, max_rounds=100)
    samme_models_by_depth[d_weak] = m_d
    for T in [10, 25, 50, 100]:
        q_tr = eval_samme_prefix(m_d, X_train, T)
        q_va = eval_samme_prefix(m_d, X_val, T)
        q_te = eval_samme_prefix(m_d, X_test, T)
        tr_acc = float(np.mean(np.argmax(q_tr, axis=1) == y_train))
        va_acc = float(np.mean(np.argmax(q_va, axis=1) == y_val))
        te_acc = float(np.mean(np.argmax(q_te, axis=1) == y_test))
        t_sec = float(m_d["round_times"].get(T, 0.0))
        q2b_grid.append({
            "depth": d_weak,
            "rounds": T,
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "test_acc": te_acc,
            "train_time_sec": t_sec,
            "final_eps": float(m_d["eps_history"][T-1]),
            "final_alpha": float(m_d["alphas"][T-1]),
            "final_max_w": float(m_d["max_w_history"][T-1]),
            "final_top10_w": float(m_d["top10_w_history"][T-1])
        })
        print(f"SAMME depth={d_weak} T={T:3d} | tr={tr_acc:.4f} va={va_acc:.4f} te={te_acc:.4f} | time={t_sec:.3f}s | eps={m_d['eps_history'][T-1]:.4f} alpha={m_d['alphas'][T-1]:.3f} max_w={m_d['max_w_history'][T-1]:.4f} top10_w={m_d['top10_w_history'][T-1]:.4f}")

best_q2b = max(q2b_grid, key=lambda r: (r["val_acc"], -r["depth"], -r["rounds"]))
BEST_BOOST_DEPTH = best_q2b["depth"]
BEST_BOOST_ROUNDS = best_q2b["rounds"]
print(f"Q2(b) Selected SAMME Model: depth={BEST_BOOST_DEPTH}, rounds={BEST_BOOST_ROUNDS} (val_acc={best_q2b['val_acc']:.4f}, test_acc={best_q2b['test_acc']:.4f})")
selected_boost_model = samme_models_by_depth[BEST_BOOST_DEPTH]

# ---------------------------------------------------------
# Q2(c): Compare Boosting and Bagging (25, 50, 100 learners)
#        & Inspect Heaviest-Weighted Training Sample
# ---------------------------------------------------------
q2c_comparison = []
for M in [25, 50, 100]:
    # Bagging entry from Q1(b)
    bag_entry = next(r for r in q1bc_results if r["B"] == M)
    # Boosting entry for selected weak learner depth (BEST_BOOST_DEPTH) at M rounds
    boost_entry = next(r for r in q2b_grid if r["depth"] == BEST_BOOST_DEPTH and r["rounds"] == M)
    q2c_comparison.append({
        "n_learners": M,
        "bagging": {
            "train_acc": bag_entry["train_acc"],
            "val_acc": bag_entry["val_acc"],
            "test_acc": bag_entry["test_acc"],
            "train_time_sec": bag_entry["fit_time_sec"]
        },
        "boosting": {
            "train_acc": boost_entry["train_acc"],
            "val_acc": boost_entry["val_acc"],
            "test_acc": boost_entry["test_acc"],
            "train_time_sec": boost_entry["train_time_sec"]
        }
    })

# Find the training sample that receives the largest cumulative/peak boosting weight in selected model
W_hist = selected_boost_model["weight_matrix_history"][:BEST_BOOST_ROUNDS]  # (T*, 240)
P_hist = selected_boost_model["pred_train_history"][:BEST_BOOST_ROUNDS]     # (T*, 240)
mean_w_per_sample = np.mean(W_hist, axis=0)
max_w_per_sample = np.max(W_hist, axis=0)
hard_train_pos = int(np.argmax(mean_w_per_sample))
hard_global_idx = int(train_idx[hard_train_pos])
hard_true_class = int(y_train[hard_train_pos])
hard_preds_over_rounds = P_hist[:, hard_train_pos].tolist()
hard_correct_over_rounds = (P_hist[:, hard_train_pos] == hard_true_class).astype(int).tolist()
hard_weights_over_rounds = W_hist[:, hard_train_pos].tolist()

q2c_hard_sample = {
    "train_pos": hard_train_pos,
    "global_dataset_idx": hard_global_idx,
    "true_identity": hard_true_class,
    "peak_weight": float(max_w_per_sample[hard_train_pos]),
    "mean_weight": float(mean_w_per_sample[hard_train_pos]),
    "initial_weight": float(W_hist[0, hard_train_pos]),
    "times_misclassified": int(np.sum(P_hist[:, hard_train_pos] != hard_true_class)),
    "total_rounds": int(BEST_BOOST_ROUNDS),
    "predictions_by_round": hard_preds_over_rounds,
    "correct_by_round": hard_correct_over_rounds,
    "weight_by_round": [round(float(x), 6) for x in hard_weights_over_rounds]
}
print(f"\n=== Q2(c) Hardest Training Face ===")
print(f"train_pos={hard_train_pos}, global_idx={hard_global_idx}, true_id={hard_true_class}, peak_w={q2c_hard_sample['peak_weight']:.4f}, mean_w={q2c_hard_sample['mean_weight']:.4f}, misclassified={q2c_hard_sample['times_misclassified']}/{BEST_BOOST_ROUNDS} rounds")

# ---------------------------------------------------------
# Q2(d): Freeze Selected Boosting Model & Evaluate on Clean and Masked Test Sets
# ---------------------------------------------------------
boost_prob_clean = eval_samme_prefix(selected_boost_model, X_test, BEST_BOOST_ROUNDS)
boost_pred_clean = np.argmax(boost_prob_clean, axis=1)
boost_acc_clean = float(np.mean(boost_pred_clean == y_test))
boost_conf_clean = float(np.mean(np.max(boost_prob_clean, axis=1)))

boost_prob_mask = eval_samme_prefix(selected_boost_model, X_test_masked, BEST_BOOST_ROUNDS)
boost_pred_mask = np.argmax(boost_prob_mask, axis=1)
boost_acc_mask = float(np.mean(boost_pred_mask == y_test))
boost_conf_mask = float(np.mean(np.max(boost_prob_mask, axis=1)))

print(f"\n=== Q2(d) Clean vs Masked Test Evaluation ===")
print(f"Single Tree: Clean acc={st_acc_clean:.4f} (conf={st_conf_clean:.4f}) | Masked acc={st_acc_mask:.4f} (conf={st_conf_mask:.4f})")
print(f"Bagging B={FINAL_B}: Clean acc={bag_acc_clean:.4f} (conf={bag_conf_clean:.4f}) | Masked acc={bag_acc_mask:.4f} (conf={bag_conf_mask:.4f})")
print(f"Boosting (d={BEST_BOOST_DEPTH}, T={BEST_BOOST_ROUNDS}): Clean acc={boost_acc_clean:.4f} (conf={boost_conf_clean:.4f}) | Masked acc={boost_acc_mask:.4f} (conf={boost_conf_mask:.4f})")

# Find test faces where both Bagging and Boosting are right on clean, but respond differently to the mask
diff_mask_candidates = []
for i in range(len(y_test)):
    bag_c_ok = (bag_pred_clean[i] == y_test[i])
    bst_c_ok = (boost_pred_clean[i] == y_test[i])
    bag_m_ok = (bag_pred_mask[i] == y_test[i])
    bst_m_ok = (boost_pred_mask[i] == y_test[i])
    if bag_c_ok and bst_c_ok and (bag_m_ok != bst_m_ok):
        diff_mask_candidates.append({
            "test_pos": int(i),
            "global_dataset_idx": int(test_idx[i]),
            "true_identity": int(y_test[i]),
            "bag_clean_pred": int(bag_pred_clean[i]),
            "bag_clean_conf": float(np.max(bag_prob_clean[i])),
            "bag_mask_pred": int(bag_pred_mask[i]),
            "bag_mask_conf": float(np.max(bag_prob_mask[i])),
            "bag_mask_true_conf": float(bag_prob_mask[i, y_test[i]]),
            "boost_clean_pred": int(boost_pred_clean[i]),
            "boost_clean_conf": float(np.max(boost_prob_clean[i])),
            "boost_mask_pred": int(boost_pred_mask[i]),
            "boost_mask_conf": float(np.max(boost_prob_mask[i])),
            "boost_mask_true_conf": float(boost_prob_mask[i, y_test[i]]),
            "bag_survives_mask": bool(bag_m_ok),
            "boost_survives_mask": bool(bst_m_ok)
        })

# Count fraction of splits inside central 16x16 mask (rows 24..39, cols 24..39) for Bagging vs Boosting
def count_central_splits_tree(f_arr):
    valid_f = f_arr[f_arr >= 0]
    if len(valid_f) == 0:
        return 0, 0
    rows = valid_f // 64
    cols = valid_f % 64
    in_mask = (rows >= 24) & (rows <= 39) & (cols >= 24) & (cols <= 39)
    return int(np.sum(in_mask)), int(len(valid_f))

bag_central_splits, bag_total_splits = 0, 0
for b in range(FINAL_B):
    nn = final_bag_ens["n_nodes"][b]
    c_in, c_tot = count_central_splits_tree(final_bag_ens["f"][b, :nn])
    bag_central_splits += c_in
    bag_total_splits += c_tot

boost_central_splits, boost_total_splits = 0, 0
boost_alpha_central_weight, boost_alpha_total = 0.0, 0.0
for tree, alpha_t in zip(selected_boost_model["learners"][:BEST_BOOST_ROUNDS], selected_boost_model["alphas"][:BEST_BOOST_ROUNDS]):
    c_in, c_tot = count_central_splits_tree(tree.f)
    boost_central_splits += c_in
    boost_total_splits += c_tot
    if c_tot > 0:
        boost_alpha_central_weight += alpha_t * (c_in / c_tot)
        boost_alpha_total += alpha_t

print(f"Q2(d) Differential Mask Candidates found: {len(diff_mask_candidates)}")
if diff_mask_candidates:
    print("Top differential mask case:", diff_mask_candidates[0])
print(f"Central 16x16 split fraction: Bagging={bag_central_splits}/{bag_total_splits} ({bag_central_splits/bag_total_splits:.4f}), Boosting={boost_central_splits}/{boost_total_splits} ({boost_central_splits/boost_total_splits:.4f})")

# ---------------------------------------------------------
# Q3: Bias and Variance as Measured Quantities (R = 30 Repetitions)
# ---------------------------------------------------------
print("\n=== Starting Q3 Bias-Variance Decomposition (R=30 Repetitions) ===")
R_REPS = 30
N_TEST = len(y_test)
K_CLASSES = 40
E_onehot = np.eye(K_CLASSES, dtype=np.float64)[y_test]  # (80, 40)

# Generate R=30 stratified bootstrap training sets (6 faces with replacement per identity)
rng_q3 = np.random.default_rng(SEED + 30000)
resample_counts_R = np.zeros((R_REPS, 240), dtype=np.int32)
for r in range(R_REPS):
    for cid in range(40):
        # Identity cid occupies indices cid*6 .. cid*6 + 5 in X_train
        base_id_indices = np.arange(cid * 6, (cid + 1) * 6, dtype=np.int32)
        drawn = rng_q3.choice(base_id_indices, size=6, replace=True)
        u, c = np.unique(drawn, return_counts=True)
        resample_counts_R[r, u] = c

def compute_bv_metrics(P_reps):
    """
    P_reps: shape (R, N_TEST, K_CLASSES)
    Bias^2 = (1/N) sum_i ||p_bar(x_i) - e_i||_2^2
    Variance = (1/(N*R)) sum_{i,r} ||p_r(x_i) - p_bar(x_i)||_2^2
    """
    p_bar = np.mean(P_reps, axis=0)  # (N_TEST, K_CLASSES)
    bias_sq = float(np.mean(np.sum((p_bar - E_onehot) ** 2, axis=1)))
    variance = float(np.mean(np.sum((P_reps - p_bar[None, :, :]) ** 2, axis=2)))
    total_mse = float(np.mean(np.sum((P_reps - E_onehot[None, :, :]) ** 2, axis=2)))
    accs = [float(np.mean(np.argmax(P_reps[r], axis=1) == y_test)) for r in range(R_REPS)]
    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "bias_sq_plus_var": bias_sq + variance,
        "total_mse": total_mse,
        "mean_test_acc": float(np.mean(accs)),
        "std_test_acc": float(np.std(accs)),
        "min_test_acc": float(np.min(accs)),
        "max_test_acc": float(np.max(accs)),
        "rep_accs": accs
    }

# Q3(b) & Q3(a): Evaluate Single Trees and 100-Tree Bagged Ensembles at depths:
# Shallow (2 and 4), Medium (8), Selected (16), and Much Deeper (None)
Q3_DEPTHS = [4, 16, None]
q3_single_by_depth = {}
q3_bag100_by_depth = {}

for md in Q3_DEPTHS:
    d_key = "None" if md is None else str(md)
    # 1. Single Tree across R=30 refits
    P_single = np.zeros((R_REPS, N_TEST, K_CLASSES), dtype=np.float64)
    leaves_reps, f1_reps = [], []
    for r in range(R_REPS):
        tr_r = CARTDecisionTree(max_depth=md, min_samples_leaf=FROZEN_MSL).fit_indexed(
            X_T_global, y_train, presorted_global, resample_counts_R[r], is_int=1
        )
        P_single[r] = tr_r.predict_proba(X_test)
        nl_r, ms_r, f1_r, _ = tr_r.leaf_stats()
        leaves_reps.append(nl_r)
        f1_reps.append(f1_r)

    bv_s = compute_bv_metrics(P_single)
    bv_s["mean_leaves"] = float(np.mean(leaves_reps))
    bv_s["mean_frac_1"] = float(np.mean(f1_reps))
    q3_single_by_depth[d_key] = bv_s

    # 2. 100-Tree Bagged Ensemble across R=30 refits (batched as 30*100 = 3000 trees in C OpenMP!)
    B_BAG = 100
    TOTAL_TREES = R_REPS * B_BAG
    md_c = -1 if md is None else int(md)
    all_w = np.zeros((TOTAL_TREES, 240), dtype=np.float64)
    all_counts = np.zeros((TOTAL_TREES, 240), dtype=np.int32)
    all_root_idx = np.zeros((TOTAL_TREES, 240), dtype=np.int32)
    all_n_unique = np.zeros(TOTAL_TREES, dtype=np.int32)
    all_feat_order = np.zeros((TOTAL_TREES, 4096), dtype=np.int32)

    rng_bag_q3 = np.random.default_rng(SEED + 40000 + (99 if md is None else md) * 100)
    for r in range(R_REPS):
        pool_r = np.repeat(np.arange(240, dtype=np.int32), resample_counts_R[r])
        for b in range(B_BAG):
            idx_tb = r * B_BAG + b
            draw_pos = rng_bag_q3.integers(0, 240, size=240)
            boot = pool_r[draw_pos]
            u, c = np.unique(boot, return_counts=True)
            all_counts[idx_tb, u] = c
            all_w[idx_tb, u] = c.astype(np.float64)
            all_n_unique[idx_tb] = len(u)
            all_root_idx[idx_tb, :len(u)] = u
            all_feat_order[idx_tb] = rng_bag_q3.permutation(4096).astype(np.int32)

    max_nodes = 512
    out_n_nodes = np.zeros(TOTAL_TREES, dtype=np.int32)
    out_f = np.zeros((TOTAL_TREES, max_nodes), dtype=np.int32)
    out_t = np.zeros((TOTAL_TREES, max_nodes), dtype=np.float64)
    out_l = np.zeros((TOTAL_TREES, max_nodes), dtype=np.int32)
    out_r = np.zeros((TOTAL_TREES, max_nodes), dtype=np.int32)
    out_n = np.zeros((TOTAL_TREES, max_nodes), dtype=np.int32)
    out_imp = np.zeros((TOTAL_TREES, max_nodes), dtype=np.float64)
    out_val = np.zeros((TOTAL_TREES, max_nodes, 40), dtype=np.float64)

    lib.fit_ensemble_c(
        TOTAL_TREES,
        X_T_global.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 240,
        y_train.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        all_w.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        all_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        1,
        all_root_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        all_n_unique.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        presorted_global.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        4096, 40, md_c, int(FROZEN_MSL),
        all_feat_order.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        max_nodes,
        out_n_nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_f.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_r.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_n.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_imp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_val.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    P_bag = np.zeros((R_REPS, N_TEST, K_CLASSES), dtype=np.float64)
    dis_reps = []
    for r in range(R_REPS):
        sub_ens = {
            "B": B_BAG,
            "f": out_f[r*B_BAG:(r+1)*B_BAG],
            "t": out_t[r*B_BAG:(r+1)*B_BAG],
            "l": out_l[r*B_BAG:(r+1)*B_BAG],
            "r": out_r[r*B_BAG:(r+1)*B_BAG],
            "val": out_val[r*B_BAG:(r+1)*B_BAG]
        }
        tp_r, votes_r = eval_ensemble_trees(sub_ens, X_test)
        P_bag[r] = votes_r
        # Disagreement on test set for repetition r
        dis_sum, n_p = 0.0, 0
        for b1 in range(B_BAG):
            dis_sum += float(np.sum(tp_r[b1+1:] != tp_r[b1])) / N_TEST
            n_p += (B_BAG - 1 - b1)
        dis_reps.append(dis_sum / n_p)

    bv_b = compute_bv_metrics(P_bag)
    bv_b["mean_disagreement"] = float(np.mean(dis_reps))
    bv_b["var_removed_abs"] = bv_s["variance"] - bv_b["variance"]
    bv_b["var_removed_pct"] = 100.0 * (bv_s["variance"] - bv_b["variance"]) / max(bv_s["variance"], 1e-12)
    bv_b["acc_gain_over_single"] = bv_b["mean_test_acc"] - bv_s["mean_test_acc"]
    q3_bag100_by_depth[d_key] = bv_b

    print(f"Q3 Depth={d_key:4s} | Single: Bias^2={bv_s['bias_sq']:.4f}, Var={bv_s['variance']:.4f}, Acc={bv_s['mean_test_acc']:.4f}+/-{bv_s['std_test_acc']:.4f} | Bag100: Bias^2={bv_b['bias_sq']:.4f}, Var={bv_b['variance']:.4f} (-{bv_b['var_removed_pct']:.1f}%), Acc={bv_b['mean_test_acc']:.4f}+/-{bv_b['std_test_acc']:.4f} (gain=+{bv_b['acc_gain_over_single']:.4f})")

# 3. Selected Boosting Model across R=30 refits
P_boost = np.zeros((R_REPS, N_TEST, K_CLASSES), dtype=np.float64)
for r in range(R_REPS):
    m_r = fit_samme_olivetti(
        max_depth=BEST_BOOST_DEPTH,
        max_rounds=BEST_BOOST_ROUNDS,
        counts_base=resample_counts_R[r],
        seed_offset=r * 17
    )
    P_boost[r] = eval_samme_prefix(m_r, X_test, BEST_BOOST_ROUNDS)

bv_boost = compute_bv_metrics(P_boost)
print(f"Q3 Boosting (d={BEST_BOOST_DEPTH}, T={BEST_BOOST_ROUNDS}) | Bias^2={bv_boost['bias_sq']:.4f}, Var={bv_boost['variance']:.4f}, Acc={bv_boost['mean_test_acc']:.4f}+/-{bv_boost['std_test_acc']:.4f}")

# Save comprehensive JSON log of all experiments
full_results = {
    "metadata": {
        "student_name": STUDENT_NAME,
        "roll_number": ROLL_NO,
        "seed": SEED,
        "total_runtime_sec": time.time() - t_start_all
    },
    "q1": {
        "q1a_grid": q1a_results,
        "selected_frozen_tree": best_q1a,
        "q1bc_bagging": q1bc_results,
        "selected_final_B": FINAL_B,
        "q1d_clean_vs_masked": {
            "single_tree": {
                "clean_acc": st_acc_clean, "clean_conf": st_conf_clean,
                "masked_acc": st_acc_mask, "masked_conf": st_conf_mask
            },
            "bagging": {
                "B": FINAL_B,
                "clean_acc": bag_acc_clean, "clean_conf": bag_conf_clean,
                "masked_acc": bag_acc_mask, "masked_conf": bag_conf_mask
            }
        },
        "q1d_cases": q1d_cases
    },
    "q2": {
        "q2a_toy_round": q2a_log,
        "q2b_grid": q2b_grid,
        "selected_boosting": best_q2b,
        "selected_boosting_trajectories": {
            "eps_history": selected_boost_model["eps_history"][:BEST_BOOST_ROUNDS],
            "alpha_history": selected_boost_model["alphas"][:BEST_BOOST_ROUNDS].tolist(),
            "max_w_history": selected_boost_model["max_w_history"][:BEST_BOOST_ROUNDS],
            "top10_w_history": selected_boost_model["top10_w_history"][:BEST_BOOST_ROUNDS]
        },
        "q2c_comparison": q2c_comparison,
        "q2c_hard_sample": q2c_hard_sample,
        "q2d_clean_vs_masked": {
            "single_tree": {"clean_acc": st_acc_clean, "clean_conf": st_conf_clean, "masked_acc": st_acc_mask, "masked_conf": st_conf_mask},
            "bagging": {"clean_acc": bag_acc_clean, "clean_conf": bag_conf_clean, "masked_acc": bag_acc_mask, "masked_conf": bag_conf_mask},
            "boosting": {"clean_acc": boost_acc_clean, "clean_conf": boost_conf_clean, "masked_acc": boost_acc_mask, "masked_conf": boost_conf_mask},
            "central_split_stats": {
                "bagging_central_splits": bag_central_splits,
                "bagging_total_splits": bag_total_splits,
                "bagging_central_frac": bag_central_splits / max(bag_total_splits, 1),
                "boosting_central_splits": boost_central_splits,
                "boosting_total_splits": boost_total_splits,
                "boosting_central_frac": boost_central_splits / max(boost_total_splits, 1),
                "boosting_alpha_weighted_central_frac": boost_alpha_central_weight / max(boost_alpha_total, 1e-12)
            },
            "differential_mask_cases": diff_mask_candidates[:5]
        }
    },
    "q3": {
        "q3a_selected_models": {
            "single_tree": q3_single_by_depth[str(FROZEN_DEPTH) if FROZEN_DEPTH is not None else "None"],
            "bagging": q3_bag100_by_depth[str(FROZEN_DEPTH) if FROZEN_DEPTH is not None else "None"],
            "boosting": bv_boost
        },
        "q3b_single_by_depth": q3_single_by_depth,
        "q3b_bag100_by_depth": q3_bag100_by_depth,
        "q3d_summary_table": [
            {
                "model": f"Single Tree (depth={FROZEN_DEPTH}, msl={FROZEN_MSL})",
                "clean_test_acc": st_acc_clean,
                "masked_test_acc": st_acc_mask,
                "bias_sq": q3_single_by_depth[str(FROZEN_DEPTH)]["bias_sq"],
                "variance": q3_single_by_depth[str(FROZEN_DEPTH)]["variance"],
                "mean_30refit_acc": q3_single_by_depth[str(FROZEN_DEPTH)]["mean_test_acc"],
                "std_30refit_acc": q3_single_by_depth[str(FROZEN_DEPTH)]["std_test_acc"]
            },
            {
                "model": f"Bagging (B={FINAL_B}, depth={FROZEN_DEPTH})",
                "clean_test_acc": bag_acc_clean,
                "masked_test_acc": bag_acc_mask,
                "bias_sq": q3_bag100_by_depth[str(FROZEN_DEPTH)]["bias_sq"],
                "variance": q3_bag100_by_depth[str(FROZEN_DEPTH)]["variance"],
                "mean_30refit_acc": q3_bag100_by_depth[str(FROZEN_DEPTH)]["mean_test_acc"],
                "std_30refit_acc": q3_bag100_by_depth[str(FROZEN_DEPTH)]["std_test_acc"]
            },
            {
                "model": f"SAMME Boosting (depth={BEST_BOOST_DEPTH}, T={BEST_BOOST_ROUNDS})",
                "clean_test_acc": boost_acc_clean,
                "masked_test_acc": boost_acc_mask,
                "bias_sq": bv_boost["bias_sq"],
                "variance": bv_boost["variance"],
                "mean_30refit_acc": bv_boost["mean_test_acc"],
                "std_30refit_acc": bv_boost["std_test_acc"]
            }
        ]
    }
}

with open("repo_P26CS0001/logs/experiment_results.json", "w") as f:
    json.dump(full_results, f, indent=2)
print(f"\nSaved complete experiment results to repo_P26CS0001/logs/experiment_results.json in {time.time() - t_start_all:.2f}s!")
