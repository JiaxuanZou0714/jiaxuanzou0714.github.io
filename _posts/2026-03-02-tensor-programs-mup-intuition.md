---
layout: post
title: "Tensor Programs (二)：从Tensor Programs到 μP"
date: 2026-03-02 19:33:00
description: "本文对 Tensor Programs 导出的极大更新参数化（μP）的核心理论推导进行系统性梳理。Tensor Programs 理论在推导神经网络缩放法则时，其最基础且最核心的洞察在于：必须根据权重张量生成机制的不同，严格区分并应用大数定律（LLN）与中心极限定理（CLT）。"
tags: [deep-learning, tensor-programs, muP, feature-learning]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---


本文对 Tensor Programs 导出的极大更新参数化（$\mu$P）的核心理论推导进行系统性梳理。Tensor Programs 理论在推导神经网络缩放法则时，其最基础且最核心的洞察在于：必须根据权重张量生成机制的不同，严格区分并应用大数定律（LLN）与中心极限定理（CLT）。如果你还没看过这个系列的起点，可以先读 [《Tensor Programs (一)：从Feature Learning 的谱条件到 μP》]({% post_url 2026-02-14-spectral-condition-feature-learning %})；那一篇从谱条件出发解释了为什么需要 $\mu$P，而本文则进一步把初始化与学习率的缩放规律拆开讲清楚。

## 1. 概率论基础与核心统计定理

在无限宽神经网络（宽度 $n \to \infty$）的理论分析中，网络层的输出本质上是大量随机变量的求和。决定这些求和项渐进尺度的关键，在于它们是否具有相关性以及期望是否为零。

>大数定律（LLN）：如果 $x_1, \dots, x_n, \dots$ “看起来像”随机变量 $X$ 的随机独立样本，则经验均值收敛于期望：
>$$
>\frac{1}{n} \sum_{i=1}^n x_i \to \mathbb{E}[X], \quad \text{as } n \to \infty
>$$

>中心极限定理（CLT）：在上述相同情况下，归一化的波动收敛于高斯分布：
>$$
>\frac{1}{\sqrt{n}} \sum_{i=1}^n (x_i - \mathbb{E}[X]) \to \mathcal{N}(0, \sigma(X)), \quad \text{as } n \to \infty
>$$
>其中 $\sigma(X)$ 是随机变量 $X$ 的标准差。

核心直觉：基于上述两个定理，我们可以得出一个关于大量随机变量之和 $\sum_{i=1}^n x_i$ 的基础直觉。当 $n$ 很大时，该求和的“典型大小”（可以理解为绝大多数时间内所处的量级）为：

$$
\sum_{i=1}^n x_i \text{ has typical size }
\begin{cases}
\Theta(n) & \text{if } \mathbb{E}[X] \neq 0 \\
\Theta(\sqrt{n}) & \text{if } \mathbb{E}[X] = 0
\end{cases}
$$

这构成了从 Tensor Programs 导出 $\mu$P 的基本准则：
* 推导初始化缩放使用 CLT：初始化时，权重是从特定分布中独立同分布采样的随机变量，期望严格为零。零均值独立变量的求和由中心极限定理主导，产生 $\Theta(\sqrt{n})$ 的尺度。
* 推导梯度与学习率缩放使用 LLN：训练时，权重的更新量是由前向激活值与反向梯度计算得到的外积。参与运算的变量存在强烈的内在相关性，乘积项的期望非零。非零均值变量的求和由大数定律主导，产生 $\Theta(n)$ 的尺度。

> 再次强调，何时使用LLN，何时使用CLT，是Tensor Programs的核心 insight 之一。

## 2. 随机变量表示与坐标典型大小

为了严谨地描述向量的分布特征，我们引入以下符号体系。

定义：我们称向量 $v \in \mathbb{R}^n$ 具有 $\Theta(n^a)$ 大小的坐标（或简称 $\Theta(n^a)$ 坐标），如果 $\|v\|^2/n = \Theta(n^{2a})$ 与 $n \to \infty$ 相当。在坐标近似独立同分布的情况下，这直观地意味着 $v$ 的每个分量具有典型大小 $\Theta(n^a)$。

经验分布随机变量 $Z$：对于每个具有 $\Theta(1)$ 坐标大小的向量 $v$，我们可以关联一个随机变量 $Z^v$。该随机变量与 $n$ 独立，并表示 $v$ 的坐标在无限宽度极限下的经验分布。其核心性质在于：如果向量 $u$ 与 $v$ 相关，则对应的随机变量 $Z^u$ 也将与 $Z^v$ 相关，并且它们在整个维度上的内积收敛于这两个随机变量乘积的期望：
$$
\lim_{n \to \infty} \frac{v^\top u}{n} = \mathbb{E}[Z^u Z^v]
$$
这一符号的引入，使我们能够将高维向量的内积严谨地转化为标量随机变量的期望计算。

## 3. 学习率缩放（LLN适用范围）

设定输入向量 $x \in \mathbb{R}^n$ 的所有坐标大小均为 $\Theta(1)$。我们要推导各类矩阵 $A$ 作用于 $x$ 时，为使 $Ax$ 仍保持 $\Theta(1)$ 坐标所需的精确尺度。

### 3.1 线性张量积矩阵（推导 SGD 更新缩放）

在梯度下降中，权重的单步更新呈现外积形式。给定具有近似独立同分布坐标（大小为 $\Theta(1)$）的向量 $u, v, x \in \mathbb{R}^n$。构造外积：

$$
A \triangleq u \otimes v / n = u v^\top / n
$$

计算 $A$ 作用于 $x$ 的结果：

$$
Ax = u \frac{v^\top x}{n} \approx c u, \quad \text{where } c = \mathbb{E}[Z^v Z^x]
$$

由于在梯度下降的动力学中，$v$（如前一层激活值）与 $x$ 是相关的，$\mathbb{E}[Z^v Z^x] \neq 0$。根据大数定律（LLN），$Ax$ 的坐标也具有近似独立同分布的坐标，其分布类似于：

$$
Z^{Ax} \triangleq Z^u \mathbb{E}[Z^v Z^x]
$$

同样地，如果 $A$ 是 $k$ 个外积之和 $A = \sum_{i=1}^k u^i \otimes v^i / n$，那么：

$$
Ax = \sum_{i=1}^k u^i \frac{(v^i)^\top x}{n}, \quad \text{with coordinates distributed as } Z^{Ax} = \sum_{i=1}^k Z^{u^i} \mathbb{E}[Z^{v^i} Z^x]
$$

深刻洞察：由于 $u$ 和 $v$ 的坐标大小均为 $\Theta(1)$，原始的未经缩放的外积矩阵 $u v^\top$ 的每个元素大小自然是 $\Theta(1)$。如果用它直接更新网络，求和过程中的大数定律会导致输出 $Ax$ 爆炸到 $\Theta(n)$ 级别。为了使结果 $Ax$ 保持 $\Theta(1)$，公式中必须引入一个 $1/n$ 的缩放因子，使得 $A$ 的坐标大小变为 $\Theta(1/n)$。在 SGD 的实际更新公式 $\Delta W = - \eta \nabla W$ 中，梯度矩阵 $\nabla W$ 的元素已经是 $\Theta(1)$ 级别，因此这个维持系统稳定所必需的 $1/n$ 因子，自然且只能被放置在学习率 $\eta$ 中。这就严格证明了 SGD 为什么需要 $\Theta(1/n)$ 的学习率。

### 3.2 非线性张量积矩阵（推导 Adam 更新缩放）

当使用 Adam 等自适应优化器时，梯度在应用之前会进行逐坐标规范化。这种规范化后的更新矩阵 $A$ 采取如下非线性张量积的形式：

$$
A_{\alpha\beta} = \psi(u_\alpha^1, \dots, u_\alpha^k, v_\beta^1, \dots, v_\beta^k)
$$

以 Adam 为例，每次梯度更新为 $\mu/\sigma$，其中 $\mu$ 和 $\sigma^2$ 是梯度的移动平均。如果未归一化的梯度是外积 $u^i \otimes v^i$，那么更新的坐标为：

$$
(\mu/\sigma)_{\alpha\beta} = \psi(\dots) \triangleq \frac{\sum_i \gamma_i u_\alpha^i v_\beta^i}{\sqrt{\sum_i \omega_i (u_\alpha^i v_\beta^i)^2}}
$$

现在假设 $\psi = n^{-1} \bar{\psi}$，其中 $\bar{\psi}$ 与 $n$ 无关。那么对于具有 $\Theta(1)$ 坐标的 $x \in \mathbb{R}^n$，根据大数定律：

$$
(Ax)_\alpha = \frac{1}{n} \sum_{\beta=1}^n \bar{\psi}(u_\alpha^1, \dots, u_\alpha^k, v_\beta^1, \dots, v_\beta^k) x_\beta \approx \mathbb{E}[\bar{\psi}(u_\alpha^1, \dots, u_\alpha^k, Z^{v^1}, \dots, Z^{v^k}) Z^x] \triangleq \Psi(u_\alpha^1, \dots, u_\alpha^k)
$$

只要输入项之间存在相关性，该期望 $\Psi$ 即为一个非零的 $\Theta(1)$ 常数，从而使得 $(Ax)_\alpha$ 也近似为大小为 $\Theta(1)$ 的独立同分布随机变量：

$$
Z^{Ax} \triangleq \Psi(Z^{u^1}, \dots, Z^{u^k})
$$

深刻洞察：Adam 算法的核心机制是将其更新矩阵的每个元素强制归一化为 $\Theta(1)$ 级别。如上文推导所示，任何元素大小为 $\Theta(1)$ 且与输入存在相关性的矩阵，由于大数定律的累加效应，都会使下一层的激活值扩大 $n$ 倍。为了抵消这种爆炸，我们在数学构造中设定了 $\psi = n^{-1} \bar{\psi}$，即强制要求矩阵 $A$ 的坐标大小必须为 $\Theta(1/n)$。在实践中，由于 Adam 吐出的更新步长本身固定为 $\Theta(1)$，我们必须在外部人为干预——即把 Adam 的基础学习率严格缩放为 $\Theta(1/n)$，才能闭合这一理论要求。

### 3.3 初始化缩放（CLT适用范围）

考虑 $A \in \mathbb{R}^{n \times n}$ 为随机高斯矩阵，$A_{\alpha\beta} \sim \mathcal{N}(0, 1/n)$，$A$ 的坐标近似独立同分布且大小为 $\Theta(1/\sqrt{n})$。

如果 $x$ 与 $A$ 独立（期望为零），根据中心极限定理（CLT），求和的方差为 $\Theta(1)$，故 $Ax$ 具有 $\Theta(1)$ 的坐标。

如果 $x$ 与 $A$ 相关，我们考虑一个极限情况 $x = A^\top \mathbf{1}$（$\mathbf{1} \in \mathbb{R}^n$ 为全 1 向量，坐标大小 $\Theta(1)$）。对于每个索引 $\alpha$：

$$
(AA^\top \mathbf{1})_\alpha = \sum_{\beta, \gamma} A_{\alpha\beta} A_{\gamma\beta} = \sum_{\beta} A_{\alpha\beta}^2 + \sum_{\beta} \sum_{\gamma \neq \alpha} A_{\alpha\beta} A_{\gamma\beta}
$$

由于 $\mathbb{E}[A_{\alpha\beta}^2] = 1/n$，根据大数定律，第一项 $\sum_{\beta} A_{\alpha\beta}^2 \approx 1$。
对于第二项，形如 $\sum_{\gamma \neq \alpha} A_{\alpha\beta} A_{\gamma\beta}$ 的求和项共有 $n$ 个，它们独立同分布且方差为 $\frac{n-1}{n^2} = \Theta(1/n)$。根据中心极限定理，这部分加总预期为 $\mathcal{N}(0, 1)$。
因此，$(AA^\top \mathbf{1})_\alpha$ 看起来像 $1 + \mathcal{N}(0, 1) = \mathcal{N}(1, 1)$，其尺度维持在 $\Theta(1)$。
这再次证明，为了处理由 CLT 驱动的独立/弱相关随机变量加和，无论后续是否存在由于反向传播带来的弱相关，高斯初始化矩阵的方差必须设定为 $\Theta(1/n)$，即坐标大小 $\Theta(1/\sqrt{n})$。

## 4. 总结与参数化对照表

基于上述底层的数学力学，我们可以清晰地看到不同类型的网络层在初始化和训练阶段必须遵循不同的统计物理法则。具体缩放法则及其依据总结如下：

| 层类型 | 参数形状 | 缩放目标 | 主导力学机制 | 必须的坐标尺度 | 超参数设定要求 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 隐藏层初始化 | $W \in \mathbb{R}^{n \times n}$ | 保证前向激活不爆炸 | 中心极限定理 (CLT)<br>零均值、独立随机变量求和 | $\Theta(1/\sqrt{n})$ | 权重方差必须设为 $\Theta(1/n)$ |
| 隐藏层更新 | $\Delta W \in \mathbb{R}^{n \times n}$ | 保证极大特征学习且不发散 | 大数定律 (LLN)<br>非零均值、高度相关变量求和 | $\Theta(1/n)$ | 学习率必须设为 $\Theta(1/n)$ |
| 输出层初始化 | $W \in \mathbb{R}^{1 \times n}$ | 保证标量输出稳定在 $\Theta(1)$ | 大数定律 (LLN)<br>为与更新量级保持数学一致性 | $\Theta(1/n)$ | 权重方差必须设为 $\Theta(1/n^2)$ |
| 输出层更新 | $\Delta W \in \mathbb{R}^{1 \times n}$ | 保证输出的数值保真度 | 大数定律 (LLN)<br>非零均值、高度相关变量求和 | $\Theta(1/n)$ | 学习率必须设为 $\Theta(1/n)$ |
| 输入层初始化 | $W \in \mathbb{R}^{n \times d}$ | 常数维度 $d$ 内的有限求和 | 无 (常数级操作) | $\Theta(1)$ | 权重方差设为 $\Theta(1)$ |
| 输入层更新 | $\Delta W \in \mathbb{R}^{n \times d}$ | 常数维度 $d$ 内的有限求和 | 无 (常数级操作) | $\Theta(1)$ | 学习率设为 $\Theta(1)$ |
{: .table .table-striped .table-sm .w-auto .mx-auto style="font-size: 0.8em;"}
如果把本文看作 Tensor Programs 路线下的概率论版本，那么与之对应的几何版本可以参考 [《球面之上：从球面动力学到 μP》]({% post_url 2026-03-04-spherical-dynamics-mup %})。那篇文章绕开 LLN / CLT 的形式化推导，直接从 RMSNorm 下的球面动力学出发，得到同样的学习率缩放结论；而在此基础上进一步加入优化器与范数约束，则可继续看 [《球面之上：带有 Hyperball 机制的优化器的 μP 缩放》]({% post_url 2026-03-06-spherical-hyperball %})。

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026mup_intuition,
  title={Tensor Programs (二)：从Tensor Programs到 μP},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/tensor-programs-mup-intuition/}
}
```

## 参考文献


[1] Yang, G., Hu, E. J., Babuschkin, I., et al. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. *arXiv preprint arXiv:2203.03466*.
