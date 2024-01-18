# ADPTED FROM: https://github.com/yandexdataschool/roc_comparison/
import numpy as np
import scipy.stats


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    Computes midranks.

    Args:
        x: 1D numpy array

    Returns:
        Array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple:
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.

    Args:
        predictions_sorted_transposed: 2D numpy array[n_classifiers, n_examples]
            sorted such as the examples with label "1" are first
        label_1_count: Number of examples with label "1"

    Returns:
        Tuple containing AUC value and DeLong covariance

    Reference:
    @article{sun2014fast,
        title={Fast Implementation of DeLong's Algorithm for
               Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
        author={Xu Sun and Weichao Xu},
        journal={IEEE Signal Processing Letters},
        volume={21},
        number={11},
        pages={1389--1393},
        year={2014},
        publisher={IEEE}
    }
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_p_value(aucs: np.ndarray, sigma: np.ndarray) -> float:
    """
    Computes p-value for the null hypothesis that two ROC AUCs are identical.

    Args:
        aucs: 1D array of AUCs
        sigma: AUC DeLong covariances

    Returns:
        Test p-value
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 2 * (1 - scipy.stats.norm.cdf(z, loc=0, scale=1))


def compute_ground_truth_statistics(ground_truth: np.ndarray) -> tuple:
    """
    Computes ground truth statistics for DeLong test.

    Args:
        ground_truth: Numpy array of 0s and 1s

    Returns:
        Tuple containing order and label_1_count
    """
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth: np.ndarray, predictions: np.ndarray) -> tuple:
    """
    Computes ROC AUC variance for a single set of predictions.

    Args:
        ground_truth: Numpy array of 0s and 1s
        predictions: Numpy array of floats of the probability of being class 1

    Returns:
        Tuple containing AUC value and DeLong covariance
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fast_delong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1
    return aucs[0], delongcov


def delong_roc_test(
    ground_truth: np.ndarray, predictions_one: np.ndarray, predictions_two: np.ndarray
) -> float:
    """
    Computes p-value for the null hypothesis that two ROC AUCs are the same.

    Args:
        ground_truth: Numpy array of 0s and 1s
        predictions_one: Predictions of the first model, numpy array of floats of the probability of being class 1
        predictions_two: Predictions of the second model, numpy array of floats of the probability of being class 1

    Returns:
        Logarithm of p-value
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[
        :, order
    ]
    aucs, delongcov = fast_delong(predictions_sorted_transposed, label_1_count)
    return calc_p_value(aucs, delongcov)
