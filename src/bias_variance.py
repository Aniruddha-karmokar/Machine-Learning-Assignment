import numpy as np


def compute_measured_bias_variance(P_reps: np.ndarray, y_test: np.ndarray, n_classes: int = 40):
    """
    Computes empirical Bias^2 and Variance over R repetitions as defined in Q3:
        Bias^2   = (1 / N) * sum_{i=1}^N ||p_bar(x_i) - e_i||_2^2
        Variance = (1 / (N * R)) * sum_{i=1}^N sum_{r=1}^R ||p_r(x_i) - p_bar(x_i)||_2^2
    where p_r(x_i) is the predicted class vector on repetition r, p_bar(x_i) is its mean
    over the R repetitions, and e_i is the one-hot target vector.
    """
    R, N, K = P_reps.shape
    assert K == n_classes
    e_onehot = np.eye(n_classes, dtype=np.float64)[y_test]
    p_bar = np.mean(P_reps, axis=0)

    bias_sq = float(np.mean(np.sum((p_bar - e_onehot) ** 2, axis=1)))
    variance = float(np.mean(np.sum((P_reps - p_bar[None, :, :]) ** 2, axis=2)))
    accs = [float(np.mean(np.argmax(P_reps[r], axis=1) == y_test)) for r in range(R)]
    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "total_mse": bias_sq + variance,
        "mean_test_acc": float(np.mean(accs)),
        "std_test_acc": float(np.std(accs)),
        "rep_accuracies": accs,
    }
