---
source_sha: 43ec354b1ec23362
layout: post
title: "Bias and Fluctuations of the Spectral Norm of Random Gaussian Matrices at Finite Width"
date: 2026-03-11 12:00:00
description: "Starting from Wishart random matrix theory, this article derives the expansion of the spectral norm of a Gaussian matrix with element variance 1/n at finite width, showing that it not only converges to the macroscopic limit 2, but also carries a bias of order $n^{-2/3}$ and Tracy-Widom type random fluctuations."
tags: [random-matrix, spectral-norm, wishart, tracy-widom, finite-width]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/finite-width-spectral-norm/
ref: finite-width-spectral-norm
related_posts: false
---

> **Preface**: In the previous blog posts on muP, we often had to deal with quantities such as random matrices, spectral norms, or Frobenius norms. For these quantities, a core insight of Tensor Programs is that in the large-width limit, the asymptotic behavior of these quantities is often very stable. The author, Greg Yang, said that when characterizing scaling laws, we actually want to characterize the behavior of the network in the limit state (width/depth/training time limits), so we naturally use the law of large numbers and the central limit theorem to analyze the limiting behavior of these quantities. However, I believe that finite-width networks inevitably introduce some systematic biases and random fluctuations, which are ignored in the large-width limit but may be very important in actual networks (for example, they may affect our muP scaling or optimizer design). This article takes the spectral norm as an example to analyze its behavior at finite width.

In many width-scaling theories, we are accustomed to treating the spectral norm of a randomly initialized matrix as a stable $\Theta(1)$ quantity; for example, when the matrix elements follow a Gaussian distribution with zero mean and variance $1/n$, intuitively its spectral norm should be "close to a constant." But if we truly care about finite-width networks, we cannot stop at the large-width limit, because for random matrices, the spectral norm itself is not a deterministic quantity without fluctuations. Not only is it random, but that randomness does not follow the most common central limit theorem scaling; instead it has a finer edge-fluctuation structure.

This article starts from the matrix form and discusses an $n\times n$ Gaussian random matrix

$$
W=\frac{1}{\sqrt n}X,\qquad X_{ij}\overset{\text{i.i.d.}}{\sim}\mathcal N(0,1),
$$

that is, the case $W_{ij}\sim\mathcal N(0,1/n)$. Our goal is to estimate the behavior of its spectral norm $\|W\|_2$ at finite width and decompose it into three parts: the macroscopic leading limit, the finite-width bias, and the finite-width random fluctuation.


## 1. From the spectral norm to the [Wishart matrix](https://zh.wikipedia.org/wiki/%E5%A8%81%E6%B2%99%E7%89%B9%E5%88%86%E4%BD%88)

The most natural entry point for the spectral norm is the covariance matrix:

$$
S=W^\top W=\frac1n X^\top X.
$$

Thus

$$
\|W\|_2=\sqrt{\lambda_{\max}(S)}.
$$

This step is crucial. Once we switch to $S$, the problem becomes the classic largest-eigenvalue problem for a [Wishart matrix](https://zh.wikipedia.org/wiki/%E5%A8%81%E6%B2%99%E7%89%B9%E5%88%86%E4%BD%88), which is precisely one of the most thoroughly developed objects in random matrix theory.

In the large-width limit $n\to\infty$, the empirical spectral distribution of $S$ follows the [Marchenko–Pastur law](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution#). For the square case here, the aspect ratio is $1$, and the spectral support is $[0,4]$. Furthermore, we have

$$
\lambda_{\max}(S)\xrightarrow{\text{a.s.}}4,
$$

Therefore

$$
\|W\|_2\xrightarrow{\text{a.s.}}2.
$$

This gives the most common zeroth-order conclusion: when the width is large enough, for a Gaussian matrix with element variance $1/n$, the leading order of its spectral norm is stable around $2$. Many scaling analyses are built on exactly this $\Theta(1)$ macroscopic scale.

## 2. Beyond the macroscopic limit: edge fluctuations of the largest eigenvalue

But "tends to 2" does not mean "equals 2." If we care about finite-width corrections, we must study how $\lambda_{\max}(S)$ fluctuates near the edge $4$.

For a standard real [Wishart matrix](https://zh.wikipedia.org/wiki/%E5%A8%81%E6%B2%99%E7%89%B9%E5%88%86%E4%BD%88) $X^\top X$, [Johnstone (2001)](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-2/On-the-distribution-of-the-largest-eigenvalue-in-principal-components/10.1214/aos/1009210544.full?utm_source=chatgpt.com) proved: if

$$
\mu_{np}=(\sqrt{n-1}+\sqrt p)^2,\qquad
\sigma_{np}=(\sqrt{n-1}+\sqrt p)\Bigl(\frac1{\sqrt{n-1}}+\frac1{\sqrt p}\Bigr)^{1/3},
$$

then

$$
\frac{\lambda_{\max}(X^\top X)-\mu_{np}}{\sigma_{np}}
\xrightarrow{d} TW_1.
$$

In the square case \(p=n\) we care about here,

$$
\mu_{nn}=4n+o(n^{1/3}),\qquad \sigma_{nn}=2^{4/3}n^{1/3}(1+o(1)),
$$

Therefore, equivalently, we can write

$$
\frac{\lambda_{\max}(X^\top X)-4n}{2^{4/3}n^{1/3}}
\xrightarrow{d}TW_1.
$$

where $TW_1$ denotes the order-1 [Tracy–Widom distribution](https://en.wikipedia.org/wiki/Tracy%E2%80%93Widom_distribution).

Since $S=\frac1n X^\top X$, rewriting the above formula for $S$ gives

$$
\frac{n^{2/3}(\lambda_{\max}(S)-4)}{2^{4/3}} =: \xi_n \xrightarrow{d}TW_1,
$$

Thus

> $$
> \lambda_{\max}(S)=4+2^{4/3}n^{-2/3}\xi_n+o_p(n^{-2/3}),\qquad \xi_n \Rightarrow TW_1.
> $$

This formula already reveals the most important point in finite-width analysis: the correction scale of the largest eigenvalue is not the common $n^{-1/2}$ from the CLT, but the finer $n^{-2/3}$.


## 3. Finite-width expansion of the spectral norm

What we really care about is the spectral norm itself, not $\lambda_{\max}(S)$. Therefore we need to propagate the above result back to

$$
\|W\|_2=\sqrt{\lambda_{\max}(S)}.
$$

Perform a Taylor expansion of the function $f(x)=\sqrt x$ around $x=4$:

$$
\sqrt{4+h}=2+\frac14 h-\frac1{64}h^2+\mathcal O(h^3).
$$

Let

$$
h=2^{4/3}n^{-2/3}\xi_n + o_p(n^{-2/3}),
$$

Substituting gives

$$
\|W\|_2
=2+\frac14\left(2^{4/3}n^{-2/3}\xi_n\right)+o_p(n^{-2/3}),
$$

That is,

$$
\|W\|_2
=2+2^{-2/3}n^{-2/3}\,\xi_n+o_p(n^{-2/3}),
\qquad \xi_n \Rightarrow TW_1.
$$

This is the finite-width expansion we wanted. It is richer than the simple statement “the spectral norm is approximately 2” because it tells us:

1. The limit is $2$;
2. The first-order correction is of order $n^{-2/3}$;
3. This correction term is itself random and follows Tracy–Widom-type statistics.

Furthermore, the mean and variance of the [Tracy–Widom distribution for $\beta=1$](https://en.wikipedia.org/wiki/Tracy%E2%80%93Widom_distribution) are

$$
\mu_{TW}\approx -1.206,\qquad \sigma_{TW}^2\approx 1.608,
$$

Then we immediately obtain

> $$
> \|W\|_2
> =
> \underbrace{2}_{\text{macroscopic limit}}
> +
> \underbrace{2^{-2/3}\mu_{TW}n^{-2/3}}_{\text{finite-width bias}}
> +
> \underbrace{2^{-2/3}n^{-2/3}(\xi_n-\mu_{TW})}_{\text{finite-width fluctuation}}
> +
> o_p(n^{-2/3}).
> $$

Taking expectations gives

$$
\mathbb E[\|W\|_2]\approx 2+2^{-2/3}\mu_{TW}n^{-2/3},
$$

and

$$
\mathrm{Std}(\|W\|_2)\approx 2^{-2/3}\sigma_{TW}n^{-2/3}.
$$

These two conclusions correspond respectively to the systematic bias and random fluctuation at finite width, and are exactly what the numerical verification below will check.

## 4. Numerical verification

The theoretical conclusions above can be checked directly with numerical experiments. For each width $n$, independently generate Gaussian random matrices multiple times

$$
W^{(1)},W^{(2)},\dots,W^{(M)},
$$

and record the spectral norm of the $k$-th experiment as

$$
s_k=\|W^{(k)}\|_2.
$$

The three statistics in the figure are all computed from this sample set $\{s_k\}_{k=1}^M$:

$$
\mathrm{Mean}(n)=\frac1M\sum_{k=1}^M s_k,
$$

$$
\mathrm{Bias}(n)=2-\mathrm{Mean}(n),
$$

$$
\mathrm{Std}(n)=\sqrt{\frac1M\sum_{k=1}^M\bigl(s_k-\mathrm{Mean}(n)\bigr)^2}.
$$

Here bias is defined as the “positive deviation relative to the infinite-width limit $2$”, i.e., how much the mean is below $2$; therefore the bias in the figure is positive, and theoretically satisfies

$$
\mathrm{Bias}(n)\approx -2^{-2/3}\mu_{TW}n^{-2/3},
$$

while the standard deviation satisfies

$$
\mathrm{Std}(n)\approx 2^{-2/3}\sigma_{TW}n^{-2/3}.
$$

The left, middle, and right panels of the figure below correspond to $\mathrm{Mean}(n)$, $\mathrm{Bias}(n)$, and $\mathrm{Std}(n)$, respectively. Thus this set of figures does not merely show loosely that “the spectral norm converges”, but separately tests three things: the mean approaches $2$ from below, the systematic bias decays as $n^{-2/3}$, and the standard deviation of the random fluctuations decays at the same scale.

{% include figure.liquid
  path="assets/img/post-03-11/image.png"
  class="img-fluid rounded z-depth-1 mx-auto d-block"
  width="100%"
  max-width="1050px"
  sizes="(min-width: 1200px) 1050px, 95vw"
  zoomable=true
  alt="Numerical Verification of the Spectral Norm of Random Gaussian Matrices at Finite Width"
  caption="The left panel shows the sample mean $\mathrm{Mean}(n)$, the middle panel shows the bias $\mathrm{Bias}(n)=2-\mathrm{Mean}(n)$ relative to the limit value $2$, and the right panel shows the sample standard deviation $\mathrm{Std}(n)$."
%}

These plots also make it easier to see that the finite-width effect is not simply a matter of "a bit more random noise". The mean curve itself shows a stable downward bias.

## 5. A summary formulation better suited to subsequent modeling

If this result is to be plugged into the network forward pass, normalization, or optimization dynamics, a convenient expression is:

$$
\|W\|_2 = 2 + \delta_n^{\text{bias}} + \delta_n^{\text{fluc}} + o_p(n^{-2/3}),
$$

where

$$
\delta_n^{\text{bias}} = 2^{-2/3}\mu_{TW}n^{-2/3},
\qquad
\delta_n^{\text{fluc}} = 2^{-2/3}n^{-2/3}(\xi_n-\mu_{TW}).
$$

The advantage of writing it this way is that later, if some quantity depends on the random matrix $\|W\|_2$, it can be directly decomposed into:
1. the macroscopic limit term;
2. the deterministic shift due to finite width;
3. the random perturbation due to finite width.

## References

[1] [V. A. Marchenko and L. A. Pastur, *Distribution of eigenvalues for some sets of random matrices*, Mathematics of the USSR-Sbornik, 1967.](http://www.ledoit.net/V_A_Mar%C4%8Denko_1967_Math._USSR_Sb._1_457.pdf)

[2] [Z. D. Bai and Y. Q. Yin, *Limit of the smallest eigenvalue of a large dimensional sample covariance matrix*, Annals of Probability, 1993.](https://projecteuclid.org/journals/annals-of-probability/volume-21/issue-3/Limit-of-the-Smallest-Eigenvalue-of-a-Large-Dimensional-Sample/10.1214/aop/1176989118.full)

[3] [I. M. Johnstone, *On the distribution of the largest eigenvalue in principal components analysis*, Annals of Statistics, 2001.](https://www.jstor.org/stable/2674106)

[4] [C. A. Tracy and H. Widom, *Level-spacing distributions and the Airy kernel*, Communications in Mathematical Physics, 1994.](https://arxiv.org/abs/hep-th/9211141)

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026finite-width-spectral-norm,
  title={有限宽度下随机高斯矩阵谱范数的偏置与涨落},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/finite-width-spectral-norm/}
}
```