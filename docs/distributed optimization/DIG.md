---
layout: default
title : Distributed In-exact Gradient Algorithm (DIG)
parent: Distributed Optimization Algorithms
nav_order: 2
---

<script type="text/javascript" 
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"> MathJax.Hub.Config({ "HTML-CSS": { availableFonts: ["TeX"], }, tex2jax: { inlineMath: [['$', '$'], ["\\(", "\\)"]] }, displayMath: [['$$', '$$'], ['\[', '\]']], TeX: { extensions: ["AMSmath.js", "AMSsymbols.js", "color.js"], equationNumbers: { autoNumber: "AMS" } }, showProcessingMessages: false, messageStyle: "none", imageFont: null, "AssistiveMML": { disabled: true } }); </script>

- [Paper](http://arxiv.org/abs/2001.00870)

- For DIG algorithm, we calculate another vector $y_i(k)$ at k-th step for each node $i$. This is the estimate of the average gradient at time step k.

- We assume $x_i(0)$ randomly and assign $y_i(k) := \nabla f_i(x_i(0))$

- We get iterative steps:

$x_i(k + 1) := \left(\sum_{j=1}^{4}w_{ij} \cdot x_j(k) \right)  − \alpha(k) \cdot y_i(k)$

$y_i(k + 1) := \left(\sum_{j=1}^{4}w_{ij} \cdot y_j(k) \right)  + \nabla f_i(x_i(k+1)) - \nabla f_i(x_i(k))$

- In vectorised form,  $X(k)$ denotes a matrix with i-th row row being $x_i(k)^T$ and similarly, we define $\nabla f(k)$ with i-th row being the sub-gradient at the i-th node i.e. $\nabla f_i(x_i(k))^T$. Note that both $X(k), \nabla f(k) \in \mathbb{R}^{n \times n}$, n being the number of nodes/dimension of $x_i$

- Taking $Y(k)$ to be the matrix with i-th row as $y_i(k)^T$
We get, initially

$Y(0) = \nabla f(X(0))$

and then,

$X(k+1) = W \cdot X(k) - \alpha \cdot Y$

$Y(k+1) = W \cdot Y(k) + \nabla f(k+1) - \nabla f(k)$
## Python Implementation
```python
def DIG(W:np.array,X:np.array,iter:np.int32,subgradient,alpha):
    if(X.ndim!=2):
        return "dimension of input vector wrong"
    values =[]
    D1 = Y = subgradient_assemble(subgradient, X)
    values.append(X)
    for i in range(iter):
        X_new = np.dot(W,X) - alpha * Y
        D0 = D1
        D1 = subgradient_assemble(subgradient,X_new)
        Y_new = np.dot(W,Y) + D1 - D0
        values.append(X_new)
        X=X_new
        Y=Y_new

    return np.array(values)
```
## Examples
* ###   Qudratic function 
  * $\min f(x) = \sum_{i=1}^{i=4}{(x_i-i)^2} , x^j=(x^j_1,x^j_2,x^j_3,x^j_4) \in \mathcal{R}^4$
  * Assuming $f_i(x) = (x_i-i)^2$
  * optimal solution : $(1,2,3,4)$
  * taking $\alpha = 0.256$ found by Hyper-parameter tuning

* <img src="../Experiments/../../Experiments/distributed%20learning/results/DIG-1.png">