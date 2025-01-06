import numpy as np
from jscentroid import jensen_shannon_centroid

def run_example(n_distribs: int = 5, n_categories: int = 8, seed: int = 1):

    def _jsdiv(p, q):
        m = (p + q) / 2
        kl_ = p * np.log(p / m) + q * np.log(q / m)
        return kl_.sum() / 2

    rng = np.random.default_rng(seed)
    probas = rng.uniform(low=1e-5, high=1, size=(n_distribs, n_categories))
    probas /= probas.sum(axis=1, keepdims=True)

    centroid = jensen_shannon_centroid(probas)
    naive_mean = probas.mean(axis=0)

    sse_centroid = sum(_jsdiv(p, centroid) for p in probas)
    sse_naive = sum(_jsdiv(p, naive_mean) for p in probas)

    print(f"Sum of JS Divergences w.r.t centroid= {sse_centroid}")
    print(f"Sum of JS Divergences w.r.t naive mean= {sse_naive}")
    print(f"Relative difference= {(sse_naive-sse_centroid)/sse_centroid:.3%}")

if __name__ == "__main__":
    run_example(n_distribs=5, n_categories=8, seed=1)
