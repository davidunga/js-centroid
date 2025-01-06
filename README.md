## Jensen-Shannon Centroid

The **Jensen-Shannon (JS) centroid** of a set of probability distributions $\(P_1, P_2, \ldots, P_N\)$ is the distribution $P^*$ that minimizes the sum of Jensen-Shannon divergences between itself and the distributions:

$$
P^* = \arg\min_{P} \sum_{i=1}^{N} \text{JS}(P_i \| P)
$$

This repository is an implementation of the method described in:

**Nielsen, F. (2020).**  
*On a Generalization of the Jensen–Shannon Divergence and the Jensen–Shannon Centroid.*  
*Entropy, 22(2), 221.* [https://doi.org/10.3390/e22020221](https://doi.org/10.3390/e22020221)

## Example Usage

```python
from jscentroid import jensen_shannon_centroid
import numpy as np

probas = np.array([[0.2, 0.8], [0.5, 0.5], [0.7, 0.3]])
centroid = jensen_shannon_centroid(probas)
print("Jensen-Shannon Centroid:", centroid)
```
