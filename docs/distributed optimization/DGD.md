---
layout: default
title : Distributed Gradient descent Algorithm (DGD)
parent: Distributed Optimization Algorithms
nav_order: 2
---

<script type="text/javascript" 
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"> MathJax.Hub.Config({ "HTML-CSS": { availableFonts: ["TeX"], }, tex2jax: { inlineMath: [['$', '$'], ["\\(", "\\)"]] }, displayMath: [['$$', '$$'], ['\[', '\]']], TeX: { extensions: ["AMSmath.js", "AMSsymbols.js", "color.js"], equationNumbers: { autoNumber: "AMS" } }, showProcessingMessages: false, messageStyle: "none", imageFont: null, "AssistiveMML": { disabled: true } }); </script>

- [Paper](http://ieeexplore.ieee.org/document/4749425/)

- Nedic and Ozdaglar(2009)

- Each agent performs the consensus step and then descent along the local subgradient direction of it's own **convex objective function**

- $x^i{(k+1)} = \sum_{}^{}{w_{ij}x^j{(k)}} - \alpha(k) \Delta f_i(x^i)$

- $x^i \in \mathcal{R}^n$ : agents i's estimate at time k
- Condition for convergence
  - $\sum_{k=0}^{\infty}{\alpha(k)} = \infty$
  - $\sum_{k=0}^{\infty}{(\alpha(k))^2} < \infty$
  - $\alpha(k)$ is non increasing sequence
- Assumptions
  - Sub-gradient are unbounded

- In vectorised form,  $X(k)$ denotes a matrix with i-th row row being $x_i(k)^T$ and similarly, we define $\nabla f(k)$ with i-th row being the sub-gradient at the i-th node i.e. $\nabla f_i(x_i(k))^T$. Note that both $X(k), \nabla f(k) \in \mathbb{R}^{n \times n}$, n being the number of nodes/dimension of $x_i$

So, after vectorising, we get:

$X(k+1) = W \cdot X(k) - \alpha(k) \cdot \nabla f(k)$

## Python Implementation
```python

def DGD(W:np.array,X:np.array,iter:np.int32,subgradient,alpha):
    if(X.ndim!=2):
        return "dimension of input vector wrong"
    values =[]
    values.append(X)
    n ,m= X.shape
    for i in range(iter):
        X_new = np.zeros_like(X)
        # Do DGD step for each agent
        for ith_agent in range(0,n):
            consensus = np.zeros_like(X[ith_agent])
            for j in range(0, m):
                consensus += W[ith_agent, j]*X[j]
            D = subgradient(ith_agent,X[ith_agent])
            X_new[ith_agent]= consensus - alpha(i)*D
        values.append(X_new)
        X=X_new
    return np.array(values)
```



## Examples 

* ###   Qudratic function 
  * $\min f(x) = \sum_{i=1}^{i=4}{(x_i-i)^2} , x^j=(x^j_1,x^j_2,x^j_3,x^j_4) \in \mathcal{R}^4$
  * Assuming $f_i(x) = (x_i-i)^2$
  * optimal solution : $(1,2,3,4)$
  * taking $\alpha(k) = \frac{a}{b+k}$ where a,b are constants

  * <img src="../Experiments/../../Experiments/distributed%20learning/results/DGD-1a.png">
  * <img src="../Experiments/../../Experiments/distributed%20learning/results/DGD-1c.png">
  
  * <img src="../Experiments/../../Experiments/distributed%20learning/results/DGD-1f.png">
    