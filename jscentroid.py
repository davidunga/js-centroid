import numpy as np

"""
Jensen-Shannon centroid computation, following method of:
Nielsen, F. (2020). On a Generalization of the Jensen–Shannon Divergence and the Jensen–Shannon Centroid.
Entropy, 22(2), 221. https://doi.org/10.3390/e22020221
"""


class CCCPError(Exception):
    pass


def jensen_shannon_centroid(probas: np.ndarray,
                            max_itrs: int = 100,
                            converge_tol: float = 1e-10) -> np.ndarray:
    """
    Computes the Jensen-Shannon centroid for a set of probability densities.
    Args:
        probas: (N,D) numpy array, where each row is a probability density over D categories.
        max_itrs: Maximum number of iteration steps.
        converge_tol: Tolerance for convergence, measured by the maximum absolute difference.
    Returns:
        (D,) numpy array representing the centroid w.r.t. Jensen-Shannon divergence.
    """

    if probas.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    if not np.allclose(probas.sum(axis=1), 1):
        raise ValueError("Probabilities must sum to 1.")

    probas = probas[:, :-1]
    theta = probas.mean(axis=0)

    success = False
    for itr in range(max_itrs):
        theta_before = theta
        theta = .5 * (probas + theta)
        norm = np.maximum(1 - theta.sum(axis=1, keepdims=True), 1e-12)
        thetas_prod = np.exp(np.log(theta / norm).mean(axis=0))
        theta = thetas_prod / (1 + thetas_prod.sum())
        if np.abs(theta_before - theta).max() < converge_tol:
            success = True
            break

    if not success:
        raise CCCPError(f"Did not converge. max_itrs={max_itrs}, converge_tol={converge_tol}")

    centroid = np.append(theta, 1 - theta.sum())
    return centroid
