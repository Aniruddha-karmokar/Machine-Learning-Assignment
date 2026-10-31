import numpy as np

try:
    from sklearn.tree import DecisionTreeClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def samme_round_update(w: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, K: int):
    """
    Single SAMME AdaBoost weight update rule shared between the 3-class toy example
    and the 40-class Olivetti Faces dataset:
        eps_t = sum_i w_i * 1[h_t(x_i) != y_i] / sum_i w_i
        alpha_t = log((1 - eps_t) / eps_t) + log(K - 1)
        w_i^{(t+1)} = w_i^{(t)} * exp(alpha_t * 1[h_t(x_i) != y_i]), normalized.
    Explicitly handles weak learners not better than random multiclass prediction
    (eps_t >= (K - 1) / K).
    """
    w = np.asarray(w, dtype=np.float64)
    w = w / np.sum(w)
    incorrect = (y_pred != y_true)
    eps_t = float(np.sum(w * incorrect))

    if eps_t >= (K - 1.0) / K - 1e-12:
        return w.copy(), eps_t, 0.0, False

    eps_clipped = max(eps_t, 1e-12)
    alpha_t = float(np.log((1.0 - eps_clipped) / eps_clipped) + np.log(K - 1.0))
    w_next = w * np.exp(alpha_t * incorrect.astype(np.float64))
    w_next /= np.sum(w_next)
    return w_next, eps_t, alpha_t, True


class SAMMEAdaBoostFromScratch:
    """Multiclass SAMME AdaBoost implemented from scratch."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 4, n_classes: int = 40, seed: int = 260001):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.n_classes = int(n_classes)
        self.seed = int(seed)
        self.learners = []
        self.alphas = []
        self.eps_history = []
        self.max_w_history = []
        self.top10_w_history = []
        self.weight_matrix_history = []

    def fit(self, X: np.ndarray, y: np.ndarray, tree_factory=None):
        N = X.shape[0]
        w = np.ones(N, dtype=np.float64) / N
        rng = np.random.default_rng(self.seed)

        self.learners = []
        self.alphas = []
        self.eps_history = []
        self.max_w_history = []
        self.top10_w_history = []
        self.weight_matrix_history = []

        for t in range(self.n_estimators):
            self.max_w_history.append(float(np.max(w)))
            self.top10_w_history.append(float(np.sum(np.sort(w)[::-1][:10])))
            self.weight_matrix_history.append(w.copy())

            tree_seed = int(rng.integers(0, 2**31 - 1))
            if tree_factory is not None:
                clf = tree_factory(max_depth=self.max_depth, min_samples_leaf=1, random_state=tree_seed)
            elif HAS_SKLEARN:
                clf = DecisionTreeClassifier(criterion="gini", max_depth=self.max_depth, min_samples_leaf=1, random_state=tree_seed)
            else:
                raise RuntimeError("Provide tree_factory when scikit-learn is not installed.")

            clf.fit(X, y, sample_weight=w)
            y_pred = clf.predict(X)
            w_next, eps_t, alpha_t, is_valid = samme_round_update(w, y, y_pred, K=self.n_classes)
            if not is_valid:
                w = w * rng.uniform(0.85, 1.15, size=N)
                w /= np.sum(w)
                continue

            self.learners.append(clf)
            self.alphas.append(alpha_t)
            self.eps_history.append(eps_t)
            w = w_next

        self.alphas = np.array(self.alphas, dtype=np.float64)
        self.weight_matrix_history = np.array(self.weight_matrix_history, dtype=np.float64)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns normalised weighted vote q_k(x) = S_k(x) / sum_t alpha_t."""
        N = X.shape[0]
        S = np.zeros((N, self.n_classes), dtype=np.float64)
        total_alpha = float(np.sum(self.alphas))
        if total_alpha <= 0.0 or len(self.learners) == 0:
            return np.full((N, self.n_classes), 1.0 / self.n_classes, dtype=np.float64)
        for clf, alpha_t in zip(self.learners, self.alphas):
            preds = clf.predict(X)
            S[np.arange(N), preds] += alpha_t
        return S / total_alpha

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
