---
layout: default
title : Distributed Gradient descent Algorithm (DGD)
parent: Distributed Optimization Algorithms
nav_order: 2
---

<script type="text/javascript" 
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS_HTML"> MathJax.Hub.Config({ "HTML-CSS": { availableFonts: ["TeX"], }, tex2jax: { inlineMath: [['$', '$'], ["\\(", "\\)"]] }, displayMath: [['$$', '$$'], ['\[', '\]']], TeX: { extensions: ["AMSmath.js", "AMSsymbols.js", "color.js"], equationNumbers: { autoNumber: "AMS" } }, showProcessingMessages: false, messageStyle: "none", imageFont: null, "AssistiveMML": { disabled: true } }); </script>

- Nedic and Ozdaglar(2009)

- Each agent performs the consensus step and then descent along the local subgradient direction of it's own **convex objective function**

- $x_i{(k+1)} = \sum_{}^{}{w_{ij}x_j{(k)}} - \alpha(k) \Delta f_i(x_i)$

- $x_i \in \mathcal{R}^n$ : agents i's estimate at time k
- Condition for convergence
  - $\sum_{k=0}^{\infty}{\alpha(k)} = \infty$
  - $\sum_{k=0}^{\infty}{(\alpha(k))^2} < \infty$
  - $\alpha(k)$ is non increasing sequence
- Assumptions
  - Sub-gradient are unbounded


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

