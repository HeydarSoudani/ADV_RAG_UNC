
from dataclasses import dataclass
import numpy as np

@dataclass
class RiskCoverageResult:
    thresholds: np.ndarray
    coverage: np.ndarray
    accuracy: np.ndarray
    risk: np.ndarray
    aurc: float

def _safe_div(num, den):
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den!=0)

def compute_risk_coverage(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray | None = None) -> RiskCoverageResult:
    """
    Compute risk–coverage by sweeping a threshold over scores.
    scores: confidence in [0,1], shape (N,)
    labels: correctness {0,1}, shape (N,)
    thresholds: optional thresholds to sweep. If None, uses sorted unique scores (plus 1 above max to include 0 coverage).
    Returns coverage, accuracy, risk arrays aligned with thresholds, and AURC (lower is better).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    assert scores.shape == labels.shape, "scores and labels must have same shape"
    N = scores.shape[0]

    if thresholds is None:
        # Dense grid yields smooth curves; you can also use unique score values.
        thresholds = np.linspace(0.0, 1.0, 1001)

    coverage = np.empty_like(thresholds, dtype=float)
    accuracy = np.empty_like(thresholds, dtype=float)

    for j, t in enumerate(thresholds):
        mask = scores >= t
        k = np.sum(mask)
        coverage[j] = k / N
        if k == 0:
            accuracy[j] = 0.0
        else:
            accuracy[j] = np.sum(labels[mask]) / k

    risk = 1.0 - accuracy

    # AURC via trapezoidal integration over coverage domain
    # Sort by coverage just in case
    order = np.argsort(coverage)
    cov_sorted = coverage[order]
    risk_sorted = risk[order]
    aurc = np.trapz(risk_sorted, cov_sorted)

    return RiskCoverageResult(thresholds=thresholds, coverage=coverage, accuracy=accuracy, risk=risk, aurc=aurc)

def selective_accuracy_at_coverages(scores: np.ndarray, labels: np.ndarray, coverages: np.ndarray):
    """
    For each desired coverage c in (0,1], compute accuracy when answering only the top-c fraction by score.
    Returns accuracy array of same shape as coverages.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    N = scores.shape[0]
    order = np.argsort(-scores)  # descending by confidence
    sorted_labels = labels[order]
    cum_correct = np.cumsum(sorted_labels)
    idxs = np.clip((coverages * N).astype(int), 1, N)  # at least 1 sample when c>0
    acc = cum_correct[idxs - 1] / idxs
    return acc

@dataclass
class ReliabilityResult:
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    bin_counts: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    ece: float

def reliability_diagram(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> ReliabilityResult:
    """
    Compute reliability diagram stats and ECE (Expected Calibration Error).
    Returns per-bin counts, mean confidence, accuracy, and ECE.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    eps = 1e-12
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(scores, bin_edges, right=True)  # bins: 0..n_bins-1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_counts = np.zeros(n_bins, dtype=int)
    bin_confidence = np.zeros(n_bins, dtype=float)
    bin_accuracy = np.zeros(n_bins, dtype=float)

    for b in range(n_bins):
        mask = bin_ids == b
        cnt = np.sum(mask)
        bin_counts[b] = cnt
        if cnt > 0:
            bin_confidence[b] = np.mean(scores[mask])
            bin_accuracy[b] = np.mean(labels[mask])
        else:
            bin_confidence[b] = 0.0
            bin_accuracy[b] = 0.0

    # ECE: weighted average of |acc - conf| by bin mass
    weights = bin_counts / (np.sum(bin_counts) + eps)
    ece = np.sum(weights * np.abs(bin_accuracy - bin_confidence))

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return ReliabilityResult(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        bin_counts=bin_counts,
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        ece=ece,
    )
