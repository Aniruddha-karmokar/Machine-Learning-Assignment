import numpy as np

try:
    from sklearn.tree import DecisionTreeClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class BaggingFromScratch:
    """
    Bootstrap Aggregating (Bagging) classifier implemented from scratch using
    bootstrap samples of original training set size N and a frozen DecisionTreeClassifier setting.
    """

    def __init__(self, n_estimators: int = 100, max_depth=16, min_samples_leaf: int = 1, seed: int = 260001):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self.trees = []
        self.in_bag_masks = []
        self.distinct_counts = []
        self.n_classes = 40

    def fit(self, X: np.ndarray, y: np.ndarray, tree_factory=None):
        N = X.shape[0]
        rng = np.random.default_rng(self.seed)
        self.trees = []
        self.in_bag_masks = []
        self.distinct_counts = []

        for b in range(self.n_estimators):
            boot_idx = rng.integers(0, N, size=N)
            unique_idx = np.unique(boot_idx)
            self.distinct_counts.append(len(unique_idx))

            mask = np.zeros(N, dtype=bool)
            mask[unique_idx] = True
            self.in_bag_masks.append(mask)

            tree_seed = int(rng.integers(0, 2**31 - 1))
            if tree_factory is not None:
                clf = tree_factory(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=tree_seed)
            elif HAS_SKLEARN:
                clf = DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=tree_seed,
                )
            else:
                raise RuntimeError("Provide tree_factory when scikit-learn is not installed.")

            clf.fit(X[boot_idx], y[boot_idx])
            self.trees.append(clf)
        return self

    def predict_tree_matrix(self, X: np.ndarray) -> np.ndarray:
        """Returns (B, N) matrix of class predictions from each individual tree."""
        return np.array([t.predict(X) for t in self.trees], dtype=np.int32)

    def predict_vote_fractions(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, K) matrix of vote fractions across the B trees."""
        N = X.shape[0]
        tp = self.predict_tree_matrix(X)
        votes = np.zeros((N, self.n_classes), dtype=np.float64)
        for b in range(self.n_estimators):
            votes[np.arange(N), tp[b]] += 1.0
        return votes / self.n_estimators

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_vote_fractions(X), axis=1)

    def compute_oob_accuracy(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        N = X_train.shape[0]
        tp = self.predict_tree_matrix(X_train)
        oob_votes = np.zeros((N, self.n_classes), dtype=np.float64)
        oob_counts = np.zeros(N, dtype=np.int32)
        for b in range(self.n_estimators):
            oob_idx = np.nonzero(~self.in_bag_masks[b])[0]
            oob_counts[oob_idx] += 1
            oob_votes[oob_idx, tp[b, oob_idx]] += 1.0
        valid = oob_counts > 0
        if not np.any(valid):
            return 0.0
        return float(np.mean(np.argmax(oob_votes[valid], axis=1) == y_train[valid]))

    def compute_pairwise_disagreement(self, X: np.ndarray) -> float:
        B = self.n_estimators
        if B <= 1:
            return 0.0
        tp = self.predict_tree_matrix(X)
        N = X.shape[0]
        dis_sum, n_pairs = 0.0, 0
        for b1 in range(B):
            dis_sum += float(np.sum(tp[b1 + 1:] != tp[b1])) / N
            n_pairs += (B - 1 - b1)
        return dis_sum / n_pairs

    def compute_mean_vote_margin(self, X: np.ndarray) -> float:
        vf = self.predict_vote_fractions(X)
        sorted_vf = np.sort(vf, axis=1)
        return float(np.mean(sorted_vf[:, -1] - sorted_vf[:, -2]))
