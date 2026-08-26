---
source_sha: c9b530fbeaa06fa0
layout: post
title: "Frobenius Norm Estimation of the Update Matrices for Adam and Muon Optimizers"
date: 2026-03-08 12:00:00
description: "This article rigorously derives and estimates the Frobenius norm of the update matrices of the Adam and Muon optimizers in a single iteration step, and explores the influence of matrix shape on the order of magnitude of the norm."
tags: [optimizer, adam, muon, frobenius-norm]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/optimizer-update-matrix-norm/
ref: optimizer-update-matrix-norm
related_posts: false
---

In deep learning optimization, understanding the scale of an optimizer's single-step update matrix is crucial for setting the learning rate and analyzing training stability. This article aims to estimate the Frobenius norm of the update matrices of the Adam and Muon optimizers through rigorous mathematical derivation. The derivation starts from the element-wise update sequence of the parameter matrix and examines in turn the theoretical absolute upper bound under extreme gradient sequences, the statistical expectation in the regime of pure random noise, and the direct effect of matrix shape and dimension on the final order of magnitude of the norm.

A direct motivation for this article is to supplement the previous [On the Hypersphere: μP Scaling of Optimizers with the Hyperball Mechanism](/en/blog/2026/spherical-hyperball/). In that article, both the AdamH and MuonH derivations used the order-of-magnitude assumption $\lVert u_t \rVert_F = \Theta(n)$; the goal of this article is to isolate this assumption and provide a more explicit derivation and estimation of the Frobenius norm of the update matrices for Adam and Muon, thereby clarifying the prerequisites for subsequent learning rate scaling analysis.

## 1. Basic Setup

Let the parameter matrix to be optimized be $W_t \in \mathbb{R}^{m \times n}$, with total number of elements $d = mn$. At iteration $t$, the gradient of the loss function with respect to this parameter matrix is $G_t \in \mathbb{R}^{m \times n}$. The base learning rate is denoted by $\alpha$.

## 2. Norm Estimation of the Adam Optimizer Update Matrix

According to the definition of the Adam algorithm, the update formulas for the first moment estimate $M_t$ and the second moment estimate $V_t$ are as follows:

$$
M_t = \beta_1 M_{t-1} + (1 - \beta_1) G_t
$$

$$
V_t = \beta_2 V_{t-1} + (1 - \beta_2) G_t^{\odot 2}
$$

where $G_t^{\odot 2}$ denotes the element-wise square of the matrix, and $\beta_1, \beta_2 \in [0, 1)$ are the exponential decay rates. The bias-corrected moment estimates are:

$$
\hat{M}_t = \frac{M_t}{1 - \beta_1^t}, \qquad \hat{V}_t = \frac{V_t}{1 - \beta_2^t}
$$

The overall update $\Delta W_t$ of the parameter matrix is defined as:

$$
\Delta W_t = - \alpha \frac{\hat{M}_t}{\sqrt{\hat{V}_t} + \varepsilon}
$$

The Frobenius norm of the update matrix is precisely defined as:

$$
\lVert \Delta W_t \rVert_F = \alpha \left( \sum_{i,j} \frac{\hat{M}_{t,ij}^2}{\left(\sqrt{\hat{V}_{t,ij}} + \varepsilon\right)^2} \right)^{1/2}
$$

### 2.1 Element-wise Theoretical Absolute Upper Bound

Define the effective ratio for each element as $$r_{ij} = \frac{\lvert \hat{M}_{t,ij} \rvert}{\sqrt{\hat{V}_{t,ij}} + \varepsilon}$$. Ignoring the tiny smoothing constant $$\varepsilon$$, expand $$\hat{M}_{t,ij}$$ and $$\hat{V}_{t,ij}$$ as weighted sums over historical gradient elements $$G_{k,ij}$$:

$$
\hat{M}_{t,ij} = \frac{1 - \beta_1}{1 - \beta_1^t} \sum_{k=1}^t \beta_1^{t-k} G_{k,ij}
$$

$$
\hat{V}_{t,ij} = \frac{1 - \beta_2}{1 - \beta_2^t} \sum_{k=1}^t \beta_2^{t-k} G_{k,ij}^2
$$

Apply the Cauchy-Schwarz inequality to the summation expression for $\hat{M}_{t,ij}$, rewriting it as a product of two factors and bounding it as follows:

$$
\left( \sum_{k=1}^t \beta_1^{t-k} G_{k,ij} \right)^2 \le \left( \sum_{k=1}^t \frac{\beta_1^{2(t-k)}}{\beta_2^{t-k}} \right) \left( \sum_{k=1}^t \beta_2^{t-k} G_{k,ij}^2 \right)
$$

The second summation term on the right constitutes the unnormalized form of $$\hat{V}_{t,ij}$$. The first summation term on the right is the sum of a geometric series. Requiring the hyperparameters to satisfy $$\frac{\beta_1^2}{\beta_2} < 1$$, substitute back into the $$\hat{M}_{t,ij}^2$$ expression and rearrange to obtain an upper bound on the squared absolute value:

$$
\frac{\hat{M}_{t,ij}^2}{\hat{V}_{t,ij}} \le \frac{(1 - \beta_1)^2 (1 - \beta_2^t)}{(1 - \beta_1^t)^2 (1 - \beta_2) \left( 1 - \frac{\beta_1^2}{\beta_2} \right)} \left( 1 - \left( \frac{\beta_1^2}{\beta_2} \right)^t \right)
$$

As $t \to \infty$, the individual bias-correction terms converge to $1$, and the limit is:

$$
\lim_{t \to \infty} \frac{\hat{M}_{t,ij}^2}{\hat{V}_{t,ij}} \le \frac{(1 - \beta_1)^2}{1 - \beta_2} \frac{\beta_2}{\beta_2 - \beta_1^2}
$$

Substituting the default parameters $\beta_1 = 0.9, \beta_2 = 0.999$, this constant evaluates to approximately $52.857$. That is, the single-element update ratio is strictly bounded by $\lvert r_{ij} \rvert \le 7.27$. This proves the universality of $\lvert r_{ij} \rvert = \mathcal{O}(1)$.

### 2.2 Expected Estimate under Stationary Stochastic Gradients

If the gradients exhibit zero-mean independent random fluctuations, assume $G_{k,ij}$ are independent and identically distributed random variables with $E[G_{k,ij}] = 0$ and $\operatorname{Var}(G_{k,ij}) = \sigma^2$. Under this assumption, the variance of $$\hat{M}_{t,ij}$$ asymptotically converges to:

$$
\lim_{t \to \infty} \operatorname{Var}(\hat{M}_{t,ij}) = \frac{1 - \beta_1}{1 + \beta_1} \sigma^2
$$

Since $\beta_2$ is close to $1$, the second moment is highly concentrated around $\sigma^2$ after averaging over many samples. The expectation of the mean square value of the scale factor is approximately:

$$
E \left[ \frac{\hat{M}_{t,ij}^2}{\hat{V}_{t,ij}} \right] \approx \frac{\operatorname{Var}(\hat{M}_{t,ij})}{\sigma^2} = \frac{1 - \beta_1}{1 + \beta_1}
$$

### 2.3 Order of Magnitude of the Overall Frobenius Norm of the Matrix

According to the previous derivations, whether in the theoretical extreme case or in the random steady state, the root mean square of the element-wise update ratio is $\mathcal{O}(1)$. If all $d = mn$ elements in the matrix are in a dense active state, the overall norm is:

$$
\lVert \Delta W_t \rVert_F \approx \alpha \sqrt{\sum_{i,j} E[r_{ij}^2]} = \mathcal{O}(\alpha \sqrt{mn})
$$

If there is a sparse update, with only $k_{\mathrm{eff}}$ coordinates significantly nonzero, then:

$$
\lVert \Delta W_t \rVert_F = \mathcal{O}(\alpha \sqrt{k_{\mathrm{eff}}})
$$

## 3. Norm Estimation of the Muon Optimizer Update Matrix

The core of the Muon optimizer lies in extracting the orthogonal component of the momentum matrix. Let the momentum matrix be $M_t = U S V^\top$; the approximate orthogonalization operation (such as the Newton-Schulz iteration) is equivalent to outputting the closest semi-orthogonal matrix $U V^\top$.

### 3.1 Original Unscaled Orthogonalized Matrix

After orthogonalization, all nonzero singular values of the matrix are $1$. Let $r = \operatorname{rank}(M_t)$; the norm of the base orthogonalized matrix is:

$$
\lVert U V^\top \rVert_F = \sqrt{r}
$$

If the momentum matrix is of full rank, i.e., $r = \min(m, n)$, then the order of magnitude of the original single-step update is:

$$
\lVert \Delta W_t \rVert_F = \Theta(\alpha \sqrt{\min(m, n)})
$$

At this point, its element-wise root mean square magnitude is $\Theta\left(\frac{\alpha}{\sqrt{\max(m, n)}}\right)$.

### 3.2 Engineering Scaled Version of the Update Matrix

To allow Muon to be a seamless replacement for standard optimizers, its actual engineering implementation introduces an explicit scalar scaling factor, raising the root mean square of the update matrix to $\Theta(1)$ to align with the base learning rate scale.

After root mean square alignment scaling, the element-wise root mean square of the update matrix becomes $\Theta(1)$. Working backwards, the Frobenius norm of the overall update matrix is then explicitly amplified to:

$$
\lVert \Delta W_t \rVert_F = \Theta(\alpha \sqrt{mn})
$$

## 4. Analysis of Feature Width and Matrix Dimensions

When analyzing a specific network architecture (e.g., setting the hidden layer feature width to $n$), the weight matrix often appears as an $n \times n$ square matrix. In this case, $m = n$, and the total number of elements is $d = n^2$.

For the Adam optimizer and the engineering scaled version of the Muon optimizer, substituting $d = n^2$ into the previous conclusions, the Frobenius norm of the update matrix is:

$$
\lVert \Delta W_t \rVert_F = \Theta(\alpha \sqrt{n^2}) = \Theta(\alpha n)
$$

This explains why, under the square matrix assumption, the order of magnitude of the update matrix norm scales linearly with width $n$ as $\Theta(n)$, rather than $\Theta(\sqrt{n})$. Only the purely mathematically defined unscaled orthogonalized matrix has magnitude $\Theta(\sqrt{n})$.

## 5. Conclusion

The matrix norms of different optimizers during a single-step update are significantly influenced by the adaptive mechanism and orthogonalization scaling. The relevant asymptotic magnitudes are summarized in the table below:

| Optimizer Algorithm | Update State | Element-wise RMS Magnitude | Frobenius Norm Magnitude (Rectangular $m \times n$) | Frobenius Norm Magnitude (Square $n \times n$) |
| :--- | :---: | :---: | :---: | ---: |
| **Adam** | Dense update | $\mathcal{O}(1)$ | $\mathcal{O}(\alpha \sqrt{mn})$ | $\mathcal{O}(\alpha n)$ |
| **Adam** | Sparse update ($k_{\mathrm{eff}}$) | $\mathcal{O}(1)$ | $\mathcal{O}(\alpha \sqrt{k_{\mathrm{eff}}})$ | - |
| **Muon** (Theoretical original) | Full-rank polar decomposition | $\Theta(1/\sqrt{\max(m,n)})$ | $\Theta(\alpha \sqrt{\min(m, n)})$ | $\Theta(\alpha \sqrt{n})$ |
| **Muon** (Engineering scaled) | Root mean square alignment | $\Theta(1)$ | $\Theta(\alpha \sqrt{mn})$ | $\Theta(\alpha n)$ |
{: .table .table-striped .table-sm .w-auto .mx-auto style="font-size: 0.8em;"}

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026optimizer-update-matrix-norm,
  title={Adam 与 Muon 优化器更新矩阵的 Frobenius 范数估计},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/optimizer-update-matrix-norm/}
}
```