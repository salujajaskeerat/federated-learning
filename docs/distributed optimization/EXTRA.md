---
layout: default
title : Exact First Order Algorithm (EXTRA)
parent: Distributed Optimization Algorithms
nav_order: 2
---

<script type="text/javascript" 
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"> MathJax.Hub.Config({ "HTML-CSS": { availableFonts: ["TeX"], }, tex2jax: { inlineMath: [['$', '$'], ["\\(", "\\)"]] }, displayMath: [['$$', '$$'], ['\[', '\]']], TeX: { extensions: ["AMSmath.js", "AMSsymbols.js", "color.js"], equationNumbers: { autoNumber: "AMS" } }, showProcessingMessages: false, messageStyle: "none", imageFont: null, "AssistiveMML": { disabled: true } }); </script>

- [Paper](http://arxiv.org/abs/1404.6264)

- We try the EXTRA(Exact First Order) algorithm. This is a discrete-time algorithm with fixed step sizes. The notation largely being the same as above, $\alpha$ is now a constant and doesn't depend on the epoch.

- We use a random guess as $x_i(0)$, then we calculate:
$x_i(1) = \left(\sum_{j=1}^{4}w_{ij} \cdot x_j(k) \right)  − \alpha \cdot \nabla f_i(x_i(0))$

- Now, we calculate $x_i(k+2)$ iteratively using $x_j(k+1)$ and $x_j(k) \;\;\; \forall j \in [n]$ 


- $x_i(k+2) := x_i(k+1) + \left( \sum_{j=1}^{n}w_{ij} \cdot x_j(k + 1) \right) \;− \left(\sum_{j=1} \tilde{w}_{ij} \cdot x_j (k)\right) − α \cdot \nabla f_i(x_i(k+1) − \nabla f_i(x_i(k)))$

where $\tilde{W}$ is weight mixing matrix which is also doubly stochastic. We assume $\tilde{W} = \frac{W+I}{2}$ as mentioned in EXTRA paper [here](http://arxiv.org/abs/1404.6264).

- In vectorised form,  $X(k)$ denotes a matrix with i-th row row being $x_i(k)^T$ and similarly, we define $\nabla f(k)$ with i-th row being the sub-gradient at the i-th node i.e. $\nabla f_i(x_i(k))^T$. Note that both $X(k), \nabla f(k) \in \mathbb{R}^{n \times n}$, n being the number of nodes/dimension of $x_i$

- Here, we denote first step by
$X(1) := W \cdot X(0) - \alpha \cdot \nabla f(0)$

- And after that:

$X(k+2) := (I+W) \cdot X(k+1) - \tilde{W} \cdot X(k) -\alpha \cdot \left( \nabla f(k+1) - \nabla f(k) \right) $

## Python Implementation
```python
def EXTRA(W:np.array,X:np.array,iter:np.int32,subgradient,alpha):
    if(X.ndim!=2):
        return "dimension of input vector wrong"
    values =[]
    values.append(X)
    n ,m= X.shape
    X = np.dot(W,X) - alpha * subgradient_assemble(subgradient,X)
    values.append(X)
    W_tilde = (np.eye(n) + W)/2
    D1 = subgradient_assemble(subgradient,values[-2])
    for i in range(1,iter):
        D0 = D1
        D1 = subgradient_assemble(subgradient,values[-1])
        X_new = 2*np.dot(W_tilde,values[-1]) - np.dot(W_tilde,values[-2]) - alpha*(D1 - D0)
        values.append(X_new)
        X=X_new

    return np.array(values)
```
## Examples
* ###   Qudratic function 
  * $\min f(x) = \sum_{i=1}^{i=4}{(x_i-i)^2} , x^j=(x^j_1,x^j_2,x^j_3,x^j_4) \in \mathcal{R}^4$
  * Assuming $f_i(x) = (x_i-i)^2$
  * optimal solution : $(1,2,3,4)$
  * taking $\alpha = 0.32$ found by Hyper-parameter tuning

* <img src="../Experiments/../../Experiments/distributed%20learning/results/EXTRA-1.png">