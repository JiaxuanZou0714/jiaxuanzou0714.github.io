---
layout: post
title: "Tensor Programs (Part 2): From Tensor Programs to μP"
date: 2026-03-02 19:33:00
description: "This article systematically reviews the core theoretical derivations of the maximal update parameterization (μP) derived from Tensor Programs. The most fundamental and core insight of Tensor Programs theory in deriving neural network scaling laws is that one must strictly distinguish and apply the law of large numbers (LLN) and the central limit theorem (CLT) based on the different generation mechanisms of weight tensors."
tags: [deep-learning, tensor-programs, muP, feature-learning]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/tensor-programs-mup-intuition/
ref: tensor-programs-mup-intuition
related_posts: false
---


This article systematically reviews the core theoretical derivation of maximal update parameterization ($\mu$P) derived from Tensor Programs. In deriving neural network scaling laws, the most fundamental and central insight of Tensor Programs theory is that one must strictly distinguish and apply the law of large numbers (LLN) and the central limit theorem (CLT) according to the different generation mechanisms of weight tensors. If you haven't seen the starting point of this series, you can first read [Tensor Programs (Part 1): From the Spectral Conditions of Feature Learning to μP](/en/blog/2026/spectral-condition-feature-learning/); that article explains why $\mu$P is needed from the spectral condition, while this article further clarifies the scaling rules for initialization and learning rate.

## 1. Probability Theory Foundations and Core Statistical Theorems

In the theoretical analysis of infinitely wide neural networks (width $n \to \infty$), the output of a network layer is essentially a sum of a large number of random variables. The key to determining the asymptotic scale of these summands lies in whether they are correlated and whether their expectation is zero.

>Law of large numbers (LLN): If $x_1, \dots, x_n, \dots$ "look like" random independent samples of a random variable $X$, then the empirical mean converges to the expectation:
>$$
>\frac{1}{n} \sum_{i=1}^n x_i \to \mathbb{E}[X], \quad \text{as } n \to \infty
>$$

>Central limit theorem (CLT): Under the same conditions as above, the normalized fluctuations converge to a Gaussian distribution:
>$$
>\frac{1}{\sqrt{n}} \sum_{i=1}^n (x_i - \mathbb{E}[X]) \to \mathcal{N}(0, \sigma(X)), \quad \text{as } n \to \infty
>$$
>where $\sigma(X)$ is the standard deviation of the random variable $X$.

Core intuition: Based on the above two theorems, we can derive a basic intuition about the sum of a large number of random variables $\sum_{i=1}^n x_i$. When $n$ is large, the "typical size" of this sum (which can be understood as the order of magnitude it occupies most of the time) is:

$$
\sum_{i=1}^n x_i \text{ has typical size }
\begin{cases}
\Theta(n) & \text{if } \mathbb{E}[X] \neq 0 \\
\Theta(\sqrt{n}) & \text{if } \mathbb{E}[X] = 0
\end{cases}
$$

This constitutes the basic criterion for deriving $\mu$P from Tensor Programs:
* Use CLT for initialization scaling: at initialization, weights are random variables sampled independently and identically distributed from a specific distribution, with expectation strictly zero. The sum of zero-mean independent variables is dominated by the central limit theorem, producing a scale of $\Theta(\sqrt{n})$.
* Use LLN for gradient and learning rate scaling: during training, the weight update is an outer product computed from forward activations and backward gradients. The variables involved have strong intrinsic correlation, and the expectation of the product term is nonzero. The sum of nonzero-mean variables is dominated by the law of large numbers, producing a scale of $\Theta(n)$.

> Again, when to use LLN and when to use CLT is one of the core insights of Tensor Programs.

## 2. Random Variable Representation and Coordinate Typical Size

To rigorously describe the distributional characteristics of vectors, we introduce the following notation system.

Definition: We say that a vector $v \in \mathbb{R}^n$ has coordinates of size $\Theta(n^a)$ (or simply $\Theta(n^a)$ coordinates) if $\|v\|^2/n = \Theta(n^{2a})$ as $n \to \infty$. In the case where the coordinates are approximately independent and identically distributed, this intuitively means that each component of $v$ has typical size $\Theta(n^a)$.

Empirical distribution random variable $Z$: For each vector $v$ with $\Theta(1)$ coordinate size, we can associate a random variable $Z^v$. This random variable is independent of $n$ and represents the empirical distribution of the coordinates of $v$ in the infinite width limit. Its key property is that if vectors $u$ and $v$ are correlated, then the corresponding random variables $Z^u$ and $Z^v$ will also be correlated, and their inner product over the entire dimension converges to the expectation of the product of these two random variables:
$$
\lim_{n \to \infty} \frac{v^\top u}{n} = \mathbb{E}[Z^u Z^v]
$$
This notation allows us to rigorously convert inner products of high-dimensional vectors into expectation computations of scalar random variables.

## 3. Learning Rate Scaling (Applicability of LLN)

Assume that all coordinates of the input vector $x \in \mathbb{R}^n$ have size $\Theta(1)$. We derive the precise scaling required for various matrices $A$ acting on $x$ so that $Ax$ still maintains $\Theta(1)$ coordinates.

### 3.1 Linear Tensor Product Matrix (Deriving SGD Update Scaling)

In gradient descent, a single step of weight update takes the form of an outer product. Given vectors $u, v, x \in \mathbb{R}^n$ with approximately independent and identically distributed coordinates (of size $\Theta(1)$). Construct the outer product:

$$
A \triangleq u \otimes v / n = u v^\top / n
$$

Compute the result of $A$ acting on $x$:

$$
Ax = u \frac{v^\top x}{n} \approx c u, \quad \text{where } c = \mathbb{E}[Z^v Z^x]
$$

Since in gradient descent dynamics, $v$ (such as the previous layer activation) and $x$ are correlated, $\mathbb{E}[Z^v Z^x] \neq 0$. According to the law of large numbers (LLN), the coordinates of $Ax$ are also approximately independent and identically distributed, with a distribution similar to:

$$
Z^{Ax} \triangleq Z^u \mathbb{E}[Z^v Z^x]
$$

Similarly, if $A$ is a sum of $k$ outer products $A = \sum_{i=1}^k u^i \otimes v^i / n$, then:

$$
Ax = \sum_{i=1}^k u^i \frac{(v^i)^\top x}{n}, \quad \text{with coordinates distributed as } Z^{Ax} = \sum_{i=1}^k Z^{u^i} \mathbb{E}[Z^{v^i} Z^x]
$$

Deep insight: Since the coordinates of $u$ and $v$ are both $\Theta(1)$, the elements of the original unscaled outer product matrix $u v^\top$ are naturally of size $\Theta(1)$. If used directly to update the network, the law of large numbers in the summation would cause the output $Ax$ to explode to $\Theta(n)$ level. To keep the result $Ax$ at $\Theta(1)$, a $1/n$ scaling factor must be introduced in the formula, making the coordinate size of $A$ become $\Theta(1/n)$. In the actual SGD update formula $\Delta W = - \eta \nabla W$, the elements of the gradient matrix $\nabla W$ are already at the $\Theta(1)$ level, so this $1/n$ factor necessary for system stability naturally and only can be placed in the learning rate $\eta$. This rigorously proves why SGD requires a learning rate of $\Theta(1/n)$.

### 3.2 Nonlinear Tensor Product Matrix (Deriving Adam Update Scaling)

When using adaptive optimizers such as Adam, the gradient is normalized coordinate-wise before being applied. This normalized update matrix $A$ takes the form of a nonlinear tensor product:

$$
A_{\alpha\beta} = \psi(u_\alpha^1, \dots, u_\alpha^k, v_\beta^1, \dots, v_\beta^k)
$$

Taking Adam as an example, each gradient update is $\mu/\sigma$, where $\mu$ and $\sigma^2$ are moving averages of the gradient. If the unnormalized gradient is the outer product $u^i \otimes v^i$, then the coordinates of the update are:

$$
(\mu/\sigma)_{\alpha\beta} = \psi(\dots) \triangleq \frac{\sum_i \gamma_i u_\alpha^i v_\beta^i}{\sqrt{\sum_i \omega_i (u_\alpha^i v_\beta^i)^2}}
$$

Now suppose $\psi = n^{-1} \bar{\psi}$, where $\bar{\psi}$ is independent of $n$. Then for $x \in \mathbb{R}^n$ with $\Theta(1)$ coordinates, by the law of large numbers:

$$
(Ax)_\alpha = \frac{1}{n} \sum_{\beta=1}^n \bar{\psi}(u_\alpha^1, \dots, u_\alpha^k, v_\beta^1, \dots, v_\beta^k) x_\beta \approx \mathbb{E}[\bar{\psi}(u_\alpha^1, \dots, u_\alpha^k, Z^{v^1}, \dots, Z^{v^k}) Z^x] \triangleq \Psi(u_\alpha^1, \dots, u_\alpha^k)
$$

As long as there is correlation between the input terms, this expectation $\Psi$ is a nonzero $\Theta(1)$ constant, making $(Ax)_\alpha$ approximately an independent and identically distributed random variable of size $\Theta(1)$:

$$
Z^{Ax} \triangleq \Psi(Z^{u^1}, \dots, Z^{u^k})
$$

Key insight: The core mechanism of the Adam algorithm is to force each element of its update matrix to be normalized to $\Theta(1)$. As derived above, any matrix with elements of size $\Theta(1)$ and correlated with the input will, due to the cumulative effect of the law of large numbers, amplify the activations of the next layer by a factor of $n$. To counteract this explosion, we set $\psi = n^{-1} \bar{\psi}$ in the mathematical construction, i.e., we force the coordinate size of matrix $A$ to be $\Theta(1/n)$. In practice, since the update step size produced by Adam is itself fixed at $\Theta(1)$, we must intervene externally—namely, strictly scale Adam's base learning rate to $\Theta(1/n)$—to close this theoretical requirement.

### 3.3 Initialization Scaling (Applicability of CLT)

Consider $A \in \mathbb{R}^{n \times n}$ as a random Gaussian matrix, $A_{\alpha\beta} \sim \mathcal{N}(0, 1/n)$, with coordinates approximately independent and identically distributed and of size $\Theta(1/\sqrt{n})$.

If $x$ is independent of $A$ (with zero expectation), by the central limit theorem (CLT), the variance of the sum is $\Theta(1)$, so $Ax$ has coordinates of size $\Theta(1)$.

If $x$ is correlated with $A$, consider a limiting case $x = A^\top \mathbf{1}$ (where $\mathbf{1} \in \mathbb{R}^n$ is the all-ones vector, with coordinates of size $\Theta(1)$). For each index $\alpha$:

$$
(AA^\top \mathbf{1})_\alpha = \sum_{\beta, \gamma} A_{\alpha\beta} A_{\gamma\beta} = \sum_{\beta} A_{\alpha\beta}^2 + \sum_{\beta} \sum_{\gamma \neq \alpha} A_{\alpha\beta} A_{\gamma\beta}
$$

Since $\mathbb{E}[A_{\alpha\beta}^2] = 1/n$, by the law of large numbers, the first term $\sum_{\beta} A_{\alpha\beta}^2 \approx 1$.
For the second term, there are $n$ summands of the form $\sum_{\gamma \neq \alpha} A_{\alpha\beta} A_{\gamma\beta}$, which are independent and identically distributed with variance $\frac{n-1}{n^2} = \Theta(1/n)$. By the central limit theorem, this sum is expected to be $\mathcal{N}(0, 1)$.
Therefore, $(AA^\top \mathbf{1})_\alpha$ looks like $1 + \mathcal{N}(0, 1) = \mathcal{N}(1, 1)$, maintaining its scale at $\Theta(1)$.
This again proves that to handle sums of independent/weakly correlated random variables driven by the CLT, regardless of subsequent weak correlations due to backpropagation, the variance of the Gaussian initialization matrix must be set to $\Theta(1/n)$, i.e., coordinate size $\Theta(1/\sqrt{n})$.

## 4. Summary and Parameterization Comparison Table

Based on the underlying mathematical mechanics above, we can clearly see that different types of network layers must follow different statistical physics laws during initialization and training. The specific scaling laws and their justifications are summarized as follows:

| Layer Type | Parameter Shape | Scaling Target | Dominant Mechanism | Required Coordinate Scale | Hyperparameter Setting Requirement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Hidden layer initialization | $W \in \mathbb{R}^{n \times n}$ | Ensure forward activations do not explode | Central limit theorem (CLT)<br>sum of zero-mean, independent random variables | $\Theta(1/\sqrt{n})$ | Weight variance must be set to $\Theta(1/n)$ |
| Hidden layer update | $\Delta W \in \mathbb{R}^{n \times n}$ | Ensure maximal feature learning without divergence | Law of large numbers (LLN)<br>sum of nonzero-mean, highly correlated variables | $\Theta(1/n)$ | Learning rate must be set to $\Theta(1/n)$ |
| Output layer initialization | $W \in \mathbb{R}^{1 \times n}$ | Ensure scalar output stable at $\Theta(1)$ | Law of large numbers (LLN)<br>to maintain mathematical consistency with update magnitude | $\Theta(1/n)$ | Weight variance must be set to $\Theta(1/n^2)$ |
| Output layer update | $\Delta W \in \mathbb{R}^{1 \times n}$ | Ensure numerical fidelity of output | Law of large numbers (LLN)<br>sum of nonzero-mean, highly correlated variables | $\Theta(1/n)$ | Learning rate must be set to $\Theta(1/n)$ |
| Input layer initialization | $W \in \mathbb{R}^{n \times d}$ | Finite sum over constant dimension $d$ | None (constant-order operation) | $\Theta(1)$ | Weight variance set to $\Theta(1)$ |
| Input layer update | $\Delta W \in \mathbb{R}^{n \times d}$ | Finite sum over constant dimension $d$ | None (constant-order operation) | $\Theta(1)$ | Learning rate set to $\Theta(1)$ |
{: .table .table-striped .table-sm .w-auto .mx-auto style="font-size: 0.8em;"}
If this article is viewed as a probabilistic version under the Tensor Programs framework, the corresponding geometric version can be found in ["On the Sphere: From Spherical Dynamics to μP"](/en/blog/2026/spherical-dynamics-mup/). That article bypasses the formal derivations of LLN/CLT and directly starts from spherical dynamics under RMSNorm, obtaining the same learning rate scaling conclusions; on this basis, further incorporating optimizers and norm constraints, one can continue to ["On the Sphere: μP Scaling for Optimizers with Hyperball Mechanisms"](/en/blog/2026/spherical-hyperball/).

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026mup_intuition,
  title={Tensor Programs (二)：从Tensor Programs到 μP},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/tensor-programs-mup-intuition/}
}
```

## References


[1] Yang, G., Hu, E. J., Babuschkin, I., et al. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. *arXiv preprint arXiv:2203.03466*.
