import numpy as np
from scipy import stats
from typing import Sequence, Tuple


def t_test_binary(x: Sequence[int], y: Sequence[int], alpha=0.05, equal_var: bool = False) -> Tuple[float, float]:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)

    if x.size == 0 or y.size == 0:
        raise ValueError("Both samples must be non-empty.")
    if not (np.isin(x, [0, 1]).all() and np.isin(y, [0, 1]).all()):
        raise ValueError("Inputs must contain only 0 and 1.")

    t_stat, p_value = stats.ttest_ind(x, y, equal_var=equal_var)
    
    print(f"t_test_binary test statistic: {t_stat}, p-value: {p_value}")
    if p_value < alpha:
        print("[t_test_binary] The difference in performance is statistically significant.")
    else:
        print("[t_test_binary] The difference in performance is not statistically significant.")
    
    return float(t_stat), float(p_value)


def wilcoxon_rank_sum_binary(x: Sequence[int], y: Sequence[int], alpha=0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)

    if x.size == 0 or y.size == 0:
        raise ValueError("Both samples must be non-empty.")
    if not (np.isin(x, [0, 1]).all() and np.isin(y, [0, 1]).all()):
        raise ValueError("Inputs must contain only 0 and 1.")

    u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    
    print(f"wilcoxon_rank_sum_binary test statistic: {u_stat}, p-value: {p_value}")
    if p_value < alpha:
        print("[wilcoxon_rank_sum_binary] The difference in performance is statistically significant.")
    else:
        print("[wilcoxon_rank_sum_binary] The difference in performance is not statistically significant.")
    
    return float(u_stat), float(p_value)


def sign_test(x: Sequence[float], y: Sequence[float]) -> Tuple[int, int, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    diff = y - x
    k_pos = int(np.sum(diff > 0))
    k_neg = int(np.sum(diff < 0))
    n = k_pos + k_neg
    if n == 0:
        raise ValueError("All pairs are ties; sign test is undefined.")

    # exact two-sided binomial test with p=0.5 under H0
    res = stats.binomtest(k_pos, n=n, p=0.5, alternative="two-sided")
    
    return k_pos, n, float(res.pvalue)

