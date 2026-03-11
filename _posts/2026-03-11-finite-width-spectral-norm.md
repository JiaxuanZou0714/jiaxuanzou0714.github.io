---
layout: post
title: "有限宽度下随机高斯矩阵谱范数的偏置与涨落"
date: 2026-03-11 12:00:00
description: "本文从 Wishart 随机矩阵理论出发，推导元素方差为 1/n 的高斯矩阵谱范数在有限宽度下的展开式，说明其不仅收敛到宏观极限 2，还带有 $n^{-2/3}$ 级别的偏置和 Tracy-Widom 型随机涨落。"
tags: [random-matrix, spectral-norm, wishart, tracy-widom, finite-width]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

> **写在前面**：在前面的几篇 blog 对 mup 的探讨中，我们经常要处理随机矩阵、矩阵的谱范数或者 Frobenius 范数等量。对于这些量，Tensor Program 一个核心的 insight 就是，在大宽度极限下，这些量的渐进性态往往非常稳定。作者 greg yang 说，我们在刻画 scaling law 时候，实际上是想要试图刻画网络在极限状态下的性态（宽度/深度/训练时间极限），于是我们很自然地使用大数定律和中心极限定理来分析这些量的极限行为。但是我认为这种有限宽度的网络必然会带来一些系统性的偏置和随机的涨落，这些效应在大宽度极限里是被忽略掉的，但在实际网络中却可能是非常重要的（比如它可能影响我们的 mup 缩放，或者优化器的设计）。本文就以谱范数为例，来分析一下它在有限宽度下的行为。

在很多宽度缩放理论里，我们习惯把随机初始化矩阵的谱范数看作一个稳定的 $\Theta(1)$ 量；例如，当矩阵元素服从零均值、方差为 $1/n$ 的高斯分布时，直觉上它的谱范数应该“接近常数”。但如果我们真的关心有限宽度网络，就不能只停留在大宽度极限上，因为对于随机矩阵，谱范数本身并不是一个没有波动的确定性量。它不仅有随机性，而且这种随机性并不服从最常见的中心极限定理尺度，而是带有更精细的边缘涨落结构。

本文从矩阵形式出发，讨论一个 $n\times n$ 高斯随机矩阵

$$
W=\frac{1}{\sqrt n}X,\qquad X_{ij}\overset{\text{i.i.d.}}{\sim}\mathcal N(0,1),
$$

也就是 $W_{ij}\sim\mathcal N(0,1/n)$ 的情形。我们的目标是估计其谱范数 $\|W\|_2$ 在有限宽度下的行为，并把它拆成三部分：宏观主极限、有限宽度偏置，以及有限宽度随机涨落。


## 1. 从谱范数到 [Wishart 矩阵](https://zh.wikipedia.org/wiki/%E5%A8%81%E6%B2%99%E7%89%B9%E5%88%86%E4%BD%88)

谱范数最自然的切入点是协方差矩阵：

$$
S=W^\top W=\frac1n X^\top X.
$$

于是

$$
\|W\|_2=\sqrt{\lambda_{\max}(S)}.
$$

这一步很关键。因为一旦转到 $S$，问题就变成了典型的 [Wishart 矩阵](https://zh.wikipedia.org/wiki/%E5%A8%81%E6%B2%99%E7%89%B9%E5%88%86%E4%BD%88)最大特征值问题，而这正是随机矩阵理论最成熟的对象之一。

在大宽度极限 $n\to\infty$ 下，$S$ 的经验谱分布服从 [Marchenko–Pastur 定律](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution#)。对于这里的正方形情形，纵横比为 $1$，谱支撑集为 $[0,4]$。进一步，有，

$$
\lambda_{\max}(S)\xrightarrow{\text{a.s.}}4,
$$

因此

$$
\|W\|_2\xrightarrow{\text{a.s.}}2.
$$

这给出了最常见的零阶结论：当宽度足够大时，元素方差为 $1/n$ 的高斯矩阵，其谱范数主量级稳定在 $2$ 附近。很多缩放分析正是建立在这个 $\Theta(1)$ 的宏观标度之上的。

## 2. 宏观极限之外：最大特征值的边缘涨落

但“趋于 2”并不等于“等于 2”。如果我们关心有限宽度修正，就必须研究 $\lambda_{\max}(S)$ 在边缘 $4$ 附近如何波动。

对标准实 [Wishart 矩阵](https://zh.wikipedia.org/wiki/%E5%A8%81%E6%B2%99%E7%89%B9%E5%88%86%E4%BD%88) $X^\top X$，[Johnstone (2001)](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-2/On-the-distribution-of-the-largest-eigenvalue-in-principal-components/10.1214/aos/1009210544.full?utm_source=chatgpt.com) 证明：若

$$
\mu_{np}=(\sqrt{n-1}+\sqrt p)^2,\qquad
\sigma_{np}=(\sqrt{n-1}+\sqrt p)\Bigl(\frac1{\sqrt{n-1}}+\frac1{\sqrt p}\Bigr)^{1/3},
$$

则

$$
\frac{\lambda_{\max}(X^\top X)-\mu_{np}}{\sigma_{np}}
\xrightarrow{d} TW_1.
$$

在本文关心的正方形情形 \(p=n\) 下，

$$
\mu_{nn}=4n+o(n^{1/3}),\qquad \sigma_{nn}=2^{4/3}n^{1/3}(1+o(1)),
$$

因此等价地可写成

$$
\frac{\lambda_{\max}(X^\top X)-4n}{2^{4/3}n^{1/3}}
\xrightarrow{d}TW_1.
$$

其中 $TW_1$ 表示一阶 [Tracy–Widom 分布](https://en.wikipedia.org/wiki/Tracy%E2%80%93Widom_distribution)。

由于 $S=\frac1n X^\top X$，把上式改写到 $S$ 上可得

$$
\frac{n^{2/3}(\lambda_{\max}(S)-4)}{2^{4/3}}\xrightarrow{d}TW_1,
$$

于是

> $$
> \lambda_{\max}(S)=4+2^{4/3}n^{-2/3}TW_1+o_p(n^{-2/3}).
> $$

这个式子已经揭示了有限宽度分析里最重要的一点：最大特征值的修正尺度不是常见的由 CLT 带来的 $n^{-1/2}$，而是更精细的$n^{-2/3}$。


## 3. 谱范数的有限宽度展开

我们真正关心的是谱范数本身，而不是 $\lambda_{\max}(S)$。因此还需要把上面的结果传回到

$$
\|W\|_2=\sqrt{\lambda_{\max}(S)}.
$$

对函数 $f(x)=\sqrt x$ 在 $x=4$ 附近做泰勒展开：

$$
\sqrt{4+h}=2+\frac14 h-\frac1{64}h^2+\mathcal O(h^3).
$$

令

$$
h=2^{4/3}n^{-2/3}TW_1,
$$

代入后得到

$$
\|W\|_2
=2+\frac14\left(2^{4/3}n^{-2/3}TW_1\right)+\mathcal O_p(n^{-4/3}),
$$

也就是

$$
\|W\|_2
=2+2^{-2/3}n^{-2/3}TW_1+\mathcal O_p(n^{-4/3}).
$$

这就是我们想要的有限宽度展开式。它比单纯的“谱范数约等于 2”更丰富，因为它告诉我们：

1. 极限是 $2$；
2. 第一阶修正是 $n^{-2/3}$ 级别；
3. 这个修正项本身是随机的，并且服从 Tracy–Widom 型统计。

进一步，[$\beta=1$ 的 Tracy–Widom 分布](https://en.wikipedia.org/wiki/Tracy%E2%80%93Widom_distribution)的均值与方差为

$$
\mu_{TW}\approx -1.206,\qquad \sigma_{TW}^2\approx 1.608,
$$

那么立刻得到

> $$
> \|W\|_2
> =
> \underbrace{2}_{\text{宏观极限}}
> +
> \underbrace{2^{-2/3}\mu_{TW}n^{-2/3}}_{\text{有限宽度偏置}}
> +
> \underbrace{2^{-2/3}n^{-2/3}(TW_1-\mu_{TW})}_{\text{有限宽度随机涨落}}
> +
> \mathcal O_p(n^{-4/3}).
> $$

取期望可得

$$
\mathbb E[\|W\|_2]\approx 2+2^{-2/3}\mu_{TW}n^{-2/3},
$$

以及

$$
\mathrm{Std}(\|W\|_2)\approx 2^{-2/3}\sigma_{TW}n^{-2/3}.
$$

这两条结论分别对应有限宽度下的系统性偏置和随机涨落，也正是后面数值验证要检查的对象。

## 4. 数值验证

上面的理论结论可以直接用数值实验来检查。对每个宽度 $n$，独立生成多次高斯随机矩阵

$$
W^{(1)},W^{(2)},\dots,W^{(M)},
$$

并记第 $k$ 次实验的谱范数为

$$
s_k=\|W^{(k)}\|_2.
$$

图中的三个统计量都是对这组样本 $\{s_k\}_{k=1}^M$ 计算出来的：

$$
\mathrm{Mean}(n)=\frac1M\sum_{k=1}^M s_k,
$$

$$
\mathrm{Bias}(n)=2-\mathrm{Mean}(n),
$$

$$
\mathrm{Std}(n)=\sqrt{\frac1M\sum_{k=1}^M\bigl(s_k-\mathrm{Mean}(n)\bigr)^2}.
$$

这里把 bias 定义为“相对于无穷宽极限值 $2$ 的正偏差量”，也就是均值比 $2$ 低多少；因此图中 bias 为正，且理论上满足

$$
\mathrm{Bias}(n)\approx -2^{-2/3}\mu_{TW}n^{-2/3},
$$

而标准差满足

$$
\mathrm{Std}(n)\approx 2^{-2/3}\sigma_{TW}n^{-2/3}.
$$

下图左、中、右三栏分别对应 $\mathrm{Mean}(n)$、$\mathrm{Bias}(n)$ 和 $\mathrm{Std}(n)$。因此这组图并不只是笼统地展示“谱范数会收敛”，而是在分别检验三件事：均值从下方逼近 $2$，系统性偏置按 $n^{-2/3}$ 衰减，随机涨落的标准差也按同一标度衰减。

{% include figure.liquid
  path="assets/img/post-03-11/image.png"
  class="img-fluid rounded z-depth-1 mx-auto d-block"
  width="100%"
  max-width="1050px"
  sizes="(min-width: 1200px) 1050px, 95vw"
  zoomable=true
  alt="有限宽度下随机高斯矩阵谱范数的数值验证"
  caption="左图为样本均值 $\mathrm{Mean}(n)$，中图为相对极限值 $2$ 的偏置 $\mathrm{Bias}(n)=2-\mathrm{Mean}(n)$，右图为样本标准差 $\mathrm{Std}(n)$。"
%}

从这组图上也能更直观地看出，有限宽度效应并不只是“多一点随机噪声”这么简单。均值曲线本身就存在稳定的下偏。

## 5. 一个更适合后续建模的总结写法

如果后续要把这个结果代入网络前向、归一化或者优化动力学中，一个方便的表达式是：

$$
\|W\|_2 = 2 + \delta_n^{\text{bias}} + \delta_n^{\text{fluc}} + \mathcal O_p(n^{-4/3}),
$$

其中

$$
\delta_n^{\text{bias}} = 2^{-2/3}\mu_{TW}n^{-2/3},
\qquad
\delta_n^{\text{fluc}} = 2^{-2/3}n^{-2/3}(TW_1-\mu_{TW}).
$$

这样写的好处是，后面一旦某个量依赖于随机矩阵 $\|W\|_2$，就可以直接把它拆成：
1. 宏观极限项；
2. 有限宽度导致的确定性漂移；
3. 有限宽度导致的随机扰动。

## 参考文献

[1] [V. A. Marchenko and L. A. Pastur, *Distribution of eigenvalues for some sets of random matrices*, Mathematics of the USSR-Sbornik, 1967.](http://www.ledoit.net/V_A_Mar%C4%8Denko_1967_Math._USSR_Sb._1_457.pdf)

[2] [Z. D. Bai and Y. Q. Yin, *Limit of the smallest eigenvalue of a large dimensional sample covariance matrix*, Annals of Probability, 1993.](https://projecteuclid.org/journals/annals-of-probability/volume-21/issue-3/Limit-of-the-Smallest-Eigenvalue-of-a-Large-Dimensional-Sample/10.1214/aop/1176989118.full)

[3] [I. M. Johnstone, *On the distribution of the largest eigenvalue in principal components analysis*, Annals of Statistics, 2001.](https://www.jstor.org/stable/2674106)

[4] [C. A. Tracy and H. Widom, *Level-spacing distributions and the Airy kernel*, Communications in Mathematical Physics, 1994.](https://arxiv.org/abs/hep-th/9211141)

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026finite-width-spectral-norm,
  title={有限宽度下随机高斯矩阵谱范数的偏置与涨落},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/finite-width-spectral-norm/}
}
```