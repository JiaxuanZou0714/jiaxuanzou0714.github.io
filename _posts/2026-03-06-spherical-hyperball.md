---
layout: post
title: "球面之上：带有 Hyperball 机制的优化器的 μP 缩放"
date: 2026-03-07 10:24:00
description: "从连续时间球面动力学视角的第一性原理出发，探讨权重范数的内生依赖对超参数对齐的破坏，并严格推导各类 Hyperball 变体优化器实现特征空间对齐的底层数学机制。"
tags: [deep-learning, spherical-dynamics, muP, optimizer]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

在现代神经网络训练中，跨模型规模迁移超参数始终是核心问题。对含归一化（如rmsnorm）的架构而言，关键不再是参数本身，而是特征在超球面上的演化 [[2]](https://arxiv.org/abs/2006.08419)。

由于归一化特征满足 $\lVert z \rVert_2 = \sqrt{n}$，跨宽度对齐时无需再关心特征模长本身，$z$ 的分量已经满足 $\lvert z_i \rvert = \Theta(1)$，只需保证：
> 保证归一化特征 $z$ 的演化速率 $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$ 始终维持在稳定量级。

下文先说明标准优化器的问题，再推导 Wen et al. [[1]](https://tinyurl.com/muonh) 提出的 Hyperball 系列优化器的缩放规律。如果你还没看过这条几何路线的前一篇，可以先读 [《球面之上：从球面动力学到 μP》]({% post_url 2026-03-04-spherical-dynamics-mup %})；那篇文章先在没有 Hyperball 约束的情况下，建立了 RMSNorm 架构中球面动力学与 $\mu$P 学习率缩放之间的基本对应关系。

## 1. 基础设定与连续时间球面映射

设网络隐层宽度为 $n$，关键变量的坐标与范数量级如下。

输入满足 $\lVert x \rVert_2^2 = \Theta(n)$，其逐坐标分量量级为 $x_j = \Theta(1)$。

定义未归一化特征 $y_t = W_t x$。在输入方向于 $W_t$ 子空间中各向同性分散的假设下，$y_t$ 的 $L_2$ 范数平方为：

$$
\lVert y_t \rVert_2^2 = x^T W_t^T W_t x = \Theta\left(\frac{\lVert x \rVert_2^2}{n} \text{Tr}(W_t^T W_t)\right)
$$

由 $\lVert x \rVert_2^2 = \Theta(n)$ 以及 $\text{Tr}(W_t^T W_t) = \lVert W_t \rVert_F^2$，得到：

$$
\lVert y_t \rVert_2^2 = \Theta\left(\frac{n}{n} \lVert W_t \rVert_F^2\right) = \Theta(\lVert W_t \rVert_F^2)
$$

于是：

$$
\lVert y_t \rVert_2 = \Theta(\lVert W_t \rVert_F)
$$

应用 rmsnorm 后，向后传递的特征为：

$$
z = \sqrt{n}\frac{y_t}{\lVert y_t \rVert_2}
$$

对应的雅可比矩阵为：

$$
J = \frac{\partial z}{\partial y_t} = \frac{\sqrt{n}}{\lVert y_t \rVert_2}P_y, \quad P_y = I - \frac{y_t y_t^T}{\lVert y_t \rVert_2^2}
$$

其中 $P_y$ 是到以 $y_t$ 为法向量之超平面的正交投影。引入更新矩阵 $U_t$ 后，特征的球面动力学可写为：

$$
\frac{dz}{dt} = \frac{\sqrt{n}}{\lVert y_t \rVert_2} P_y \frac{dy_t}{dt} = \frac{\sqrt{n}}{\Theta(\lVert W_t \rVert_F)} P_y \left(-\eta U_t x\right)
$$

要求坐标分量量级对齐到 $\Theta(1)$，便得到无范数约束时的通用对齐方程：

$$
\eta = \frac{\Theta(\lVert W_t \rVert_F)}{\sqrt{n} \lvert (P_y (U_t x))_i \rvert}
$$

## 2. 标准优化器的内生半径依赖与动力学平衡困境 [[2]](https://arxiv.org/abs/2006.08419)

由上式可见，真正决定特征球面角速度的不是基础学习率 $\eta$，而是有效球面步长 $\eta_{\mathrm{eff}}^{(i)}(t)$：

$$
\eta_{\mathrm{eff}}^{(i)}(t) := \eta \frac{\sqrt{n} \lvert (P_y (U_t x))_i \rvert}{\lVert W_t \rVert_F}
$$

这里，权重矩阵的弗罗贝尼乌斯范数 $\lVert W_t \rVert_F$ 扮演控制球面角速度的内生半径。对带解耦权重衰减的标准优化器而言，这个半径会随差分方程演化。考虑权重衰减系数 $\lambda$ 下的随机梯度下降更新：

$$
W_{t+1} = W_t - \eta \left( \frac{\partial \mathcal{L}}{\partial W_t} + \lambda W_t \right)
$$

对上式两端取弗罗贝尼乌斯范数平方并展开：

$$
\lVert W_{t+1} \rVert_F^2 = (1 - \eta \lambda)^2 \lVert W_t \rVert_F^2 - 2 \eta (1 - \eta \lambda) \left\langle W_t, \frac{\partial \mathcal{L}}{\partial W_t} \right\rangle + \eta^2 \left\lVert \frac{\partial \mathcal{L}}{\partial W_t} \right\rVert_F^2
$$

归一化机制赋予权重矩阵尺度不变性 [[2]](https://arxiv.org/abs/2006.08419)，网络输出不随权重量级变化，因此梯度张量与当前权重张量正交，即 $\langle W_t, \frac{\partial \mathcal{L}}{\partial W_t} \rangle = 0$。于是交叉项为零：

$$
\lVert W_{t+1} \rVert_F^2 = (1 - \eta \lambda)^2 \lVert W_t \rVert_F^2 + \eta^2 \left\lVert \frac{\partial \mathcal{L}}{\partial W_t} \right\rVert_F^2
$$

为消去梯度范数对当前权重范数的依赖，引入单元梯度 $\tilde{G}_t = \lVert W_t \rVert_F \frac{\partial \mathcal{L}}{\partial W_t}$。代入后对两端取平方根，并在 $\eta \lambda \ll 1$ 下做泰勒展开，可得单步内生半径的演化主方程：

$$
\lVert W_{t+1} \rVert_F \approx \lVert W_t \rVert_F - \lambda \eta \lVert W_t \rVert_F + \frac{\eta^2}{2 \lVert W_t \rVert_F^3} \lVert \tilde{G}_t \rVert_F^2
$$

在驻态下，内生半径的期望保持不变，即 $\mathbb{E}[\lVert W_{t+1} \rVert_F] = \mathbb{E}[\lVert W_t \rVert_F]$。记 $L = \mathbb{E}[\lVert \tilde{G}_t \rVert_F^2 \mid W_t]$ 为期望单元梯度范数平方，令增量为零即可得到渐近极限 $w^*$：

$$
w^* = \sqrt[4]{\frac{L\eta}{2\lambda}}
$$

要分析 $w^*$ 对宽度 $n$ 的依赖，需要先求单元梯度 $\tilde{G}_t$ 的范数量级。设损失函数对归一化特征 $z$ 的梯度为 $g_z$。由链式法则，未归一化特征 $y_t$ 接收到的梯度为 $\frac{\partial \mathcal{L}}{\partial y_t} = \frac{\sqrt{n}}{\lVert y_t \rVert_2} P_y g_z$。再求对权重张量 $W_t$ 的梯度，并代入 $\tilde{G}_t$ 的定义，结合各向同性假设下的恒等式 $\lVert y_t \rVert_2 = \frac{\lVert x \rVert_2}{\sqrt{n}} \lVert W_t \rVert_F$，有：

$$
\tilde{G}_t = \lVert W_t \rVert_F \frac{\sqrt{n}}{\frac{\lVert x \rVert_2}{\sqrt{n}} \lVert W_t \rVert_F} (P_y g_z) x^T = \frac{n}{\lVert x \rVert_2} (P_y g_z) x^T
$$

其弗罗贝尼乌斯范数平方为：

$$
\lVert \tilde{G}_t \rVert_F^2 = \frac{n^2}{\lVert x \rVert_2^2} \lVert P_y g_z \rVert_2^2 \lVert x \rVert_2^2 = n^2 \lVert P_y g_z \rVert_2^2
$$

设上游传回的归一化梯度满足 $(g_z)_i = \Theta(1)$，则 $\lVert g_z \rVert_2^2 = \Theta(n)$。又因为 $P_y$ 是超平面正交投影，其对范数只带来 $\Theta(1)$ 级衰减，所以 $\lVert P_y g_z \rVert_2^2 = \Theta(n)$。于是期望单元梯度范数平方 $L$ 的标度为：

$$
L = \mathbb{E}[\lVert \tilde{G}_t \rVert_F^2 \mid W_t] = n^2 \Theta(n) = \Theta(n^3)
$$

把 $L$ 的量级代入 $w^*$，并设权重衰减系数 $\lambda = \Theta(1)$，得到驻点处的内生半径标度：

$$
\lVert W_t \rVert_F = w^* = \Theta\left(\sqrt[4]{n^3 \eta}\right) = \Theta(n^{3/4} \eta^{1/4})
$$

为了满足对齐基准 $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$，再看随机梯度下降的更新方向 $U_t = \frac{\partial \mathcal{L}}{\partial W_t} = \frac{\sqrt{n}}{\lVert y_t \rVert_2} (P_y g_z) x^T$。它对输入特征 $x$ 的作用为：

$$
U_t x = \frac{\sqrt{n}}{\lVert y_t \rVert_2} (P_y g_z) x^T x = \frac{\sqrt{n} \lVert x \rVert_2^2}{\lVert y_t \rVert_2} (P_y g_z)
$$

代入 $\lVert x \rVert_2^2 = \Theta(n)$、$\lVert y_t \rVert_2 = \Theta(\lVert W_t \rVert_F)$，并利用 $P_y^2 = P_y$：

$$
P_y (U_t x) = \frac{n \sqrt{n}}{\Theta(\lVert W_t \rVert_F)} (P_y g_z)
$$

于是坐标分量的绝对值量级为 $\lvert (P_y (U_t x))_i \rvert = \Theta\left(\frac{n \sqrt{n}}{\lVert W_t \rVert_F}\right)$。代入特征演化对齐方程：

$$
\frac{\eta \sqrt{n}}{\lVert W_t \rVert_F} \Theta\left(\frac{n \sqrt{n}}{\lVert W_t \rVert_F}\right) = \Theta\left(\frac{\eta n^2}{\lVert W_t \rVert_F^2}\right) = \Theta(1)
$$

将驻点处的内生半径标度 $\lVert W_t \rVert_F = \Theta(n^{3/4} \eta^{1/4})$ 代入上述约束：

$$
\frac{\eta n^2}{(n^{3/4} \eta^{1/4})^2} = \frac{\eta n^2}{n^{3/2} \eta^{1/2}} = \eta^{1/2} n^{1/2} = \Theta(1)
$$

求解得跨尺度对齐所需的学习率缩放规律：

$$
\eta = \Theta\left(\frac{1}{n}\right)
$$

再代回内生半径表达式，可得系统在动力学平衡状态下的范数量级：

$$
w^* = \Theta\left(n^{3/4} (n^{-1})^{1/4}\right) = \Theta(n^{1/2}) = \Theta(\sqrt{n})
$$

这说明：如果系统能瞬时达到动力学平衡，那么当 $\eta = \Theta(1/n)$ 时，标准优化的内生权重范数会自动收敛到 $\Theta(\sqrt{n})$，与标准初始化方差对应的量级一致。其核心 insight 与 mup 是一致的。

但在实际工程里，依赖这一机制逼近自然驻点来维持超参数对齐，会遇到三个问题：

1. 收敛有延迟：$w^*$ 是渐近极限，权重范数不会瞬间到达平衡点。训练初期 $\lVert W_t \rVert_F$ 还没收敛到 $\Theta(\sqrt{n})$，球面上的有效步长就是错的。
2. 学习率一变就失衡：现代训练普遍使用多阶段学习率调度。$\eta$ 一旦衰减，对应的平衡点 $w^*$ 立刻改变，但权重范数需要时间追上新的平衡点。在这段过渡期内，对齐条件不成立。
3. 正交假设不总成立：以上推导依赖 $\langle W_t, \frac{\partial \mathcal{L}}{\partial W_t} \rangle = 0$，即梯度严格正交于权重。但在带残差连接的网络中，这个条件并不严格满足，交叉项不为零，各层的权重范数会各自漂移，无法统一对齐。

## 3. Hyperball 约束与统一的主方程

为从根本上消除内生半径 $\lVert W_t \rVert_F$ 对动力学的干扰，Wen et al. [[1]](https://tinyurl.com/muonh) 提出的 Hyperball 机制显式引入切空间投影算子 $\Pi_W$，把权重更新方向约束在以初始范数为半径的超球面上。

在离散更新规则的连续时间极限下，权重动力学方程为：

$$
\frac{dW}{dt} = -\eta \lVert W_0 \rVert_F \Pi_W\left(\frac{u_t}{\lVert u_t \rVert_F}\right)
$$

在标准初始化下，矩阵元素采样方差为 $\frac{1}{n}$，因此初始常量 $R = \lVert W_0 \rVert_F = \Theta(\sqrt{n})$。切空间投影算子 $\Pi_W$ 保证权重矩阵的弗罗贝尼乌斯范数在任意时间 $t$ 恒定不变：

$$
\lVert W_t \rVert_F = \lVert W_0 \rVert_F = \Theta(\sqrt{n})
$$

因此，动态分母变成了时间与宽度上的不变量：

$$
\lVert y_t \rVert_2 = \Theta(\lVert W_t \rVert_F) = \Theta(\sqrt{n})
$$

利用投影算子的线性性质 $P_y(\Pi_W(A)x) = P_y(Ax)$，代回第一节的雅可比矩阵后，前置系数化简为常数 $\frac{\sqrt{n}}{\Theta(\sqrt{n})} = \Theta(1)$。于是得到控制所有 Hyperball 变体动力学的统一主方程：

$$
\frac{dz}{dt} = -\eta \Theta(\sqrt{n}) \frac{1}{\lVert u_t \rVert_F} P_y (u_t x)
$$

为了让 $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$ 成立，学习率 $\eta$ 必须满足：

$$
\eta = \frac{\lVert u_t \rVert_F}{\Theta(\sqrt{n}) \lvert (P_y (u_t x))_i \rvert}
$$

| | 无 hyperball  | 有 Hyperball  |
| :--- | :--- | :--- |
| 学习率约束 | $\eta = \Theta\left(\frac{\lVert W_t \rVert_F}{\sqrt{n} \lvert (P_y (U_t x))_i \rvert}\right)$ | $\eta = \Theta\left(\frac{\lVert u_t \rVert_F}{\sqrt{n} \lvert (P_y (u_t x))_i \rvert}\right)$ |
{: .table .table-striped .table-sm style="font-size: 0.5em;"}


> 这种转换切断了超参数对齐与系统收敛状态之间的耦合。没有 Hyperball 时，要得到正确的球面角速度，网络必须依靠权重衰减与梯度正交性达到平衡态 $\lVert W_t \rVert_F = \Theta(\sqrt{n})$；有了 Hyperball 后，$\lVert u_t \rVert_F$ 被显式提到学习率分子中，$\eta$ 的设定只取决于当前更新步骤的梯度结构，而不再受权重范数历史轨迹的制约。

## 4. 具体 Hyperball 变体的对齐推导

下面在特定假设下分析不同优化器的更新特性，并给出各自的对齐学习率。

### 4.1 SGDH 的对齐推导

设上游梯度 $g = \nabla_z L$ 的坐标量级为 $\Theta(1)$。基础梯度更新矩阵为 $u_t = \Theta(1) (P_y g) x^T$。由于 $\lVert P_y g \rVert_2 = \Theta(\sqrt{n})$ 且 $\lVert x \rVert_2 = \Theta(\sqrt{n})$，有：

$$
\lVert u_t \rVert_F = \Theta(1) \lVert P_y g \rVert_2 \lVert x \rVert_2 = \Theta(n)
$$

再看更新矩阵对输入特征向量的作用：

$$
u_t x = \Theta(1) (P_y g) (x^T x) = \Theta(1) (P_y g) \Theta(n) = \Theta(n) P_y g
$$

经正交投影算子作用：

$$
P_y (u_t x) = P_y (\Theta(n) P_y g) = \Theta(n) P_y g
$$

由前提假设 $\lvert (P_y g)_i \rvert = \Theta(1)$，故 $\lvert (P_y (u_t x))_i \rvert = \Theta(n)$。代入主方程：

$$
\eta = \frac{\Theta(n)}{\Theta(\sqrt{n}) \Theta(n)} = \Theta\left(\frac{1}{\sqrt{n}}\right)
$$

### 4.2 AdamH 的对齐推导

这里对 AdamH 与后文 MuonH 共同使用的 $\lVert u_t \rVert_F = \Theta(n)$ 假设，只保留结果本身。更细的 Frobenius 范数估计可参见 [《Adam 与 Muon 优化器更新矩阵的 Frobenius 范数估计》]({% post_url 2026-03-08-optimizer-update-matrix-norm %})，那篇文章专门把这一步单独展开说明。

提取梯度的符号矩阵后，更新矩阵 $u_t$ 含有 $n^2$ 个绝对值为 $1$ 的元素，因此其范数为：

$$
\lVert u_t \rVert_F = \sqrt{n^2} = n = \Theta(n)
$$

在引入动量或大批量时，更新矩阵的符号与输入向量 $x$ 的坐标分布近似独立。由中心极限定理，$n$ 个独立项线性组合后给出 $\lvert (u_t x)_i \rvert = \Theta(\sqrt{n})$，因此 $\lvert (P_y (u_t x))_i \rvert = \Theta(\sqrt{n})$。代入主方程：

$$
\eta = \frac{\Theta(n)}{\Theta(\sqrt{n}) \Theta(\sqrt{n})} = \Theta(1)
$$

### 4.3 MuonH 的对齐推导与各向同性优势

在 Muon 的实际工程实现中，正交化后的更新量还会经过学习率调整，使不同矩阵形状下的更新量均方根量级与标准优化器一致。

在这一设定下，更新矩阵的弗罗贝尼乌斯范数主导阶为：

$$
\lVert u_t \rVert_F = \Theta(n)
$$

再看该更新矩阵对当前输入特征 $x$ 的作用。由于正交化与均方根对齐，经过切空间投影后，各坐标分量的典型量级为：

$$
\lvert (P_y (u_t x))_i \rvert = \Theta(\sqrt{n})
$$

代入主方程，得到最优学习率：

$$
\eta = \frac{\Theta(n)}{\Theta(\sqrt{n}) \Theta(\sqrt{n})} = \Theta(1)
$$

这说明未约束 Muon 发生漂移的根源在于：内生半径无法在不同宽度下自动对齐；而 MuonH 强制 $\lVert W_t \rVert_F = \Theta(\sqrt{n})$ 后，主导项便对齐到常数学习率。

进一步比较 MuonH 与 AdamH，MuonH 往往能给出更精确的跨尺度对齐。原因在于：Adam 即便加入 Hyperball 约束，更新矩阵 $u_t$ 仍依赖逐元素自适应归一化，因此投影分量 $\lvert (P_y (u_t x))_i \rvert$ 仍有残留各向异性；而 Muon 的正交化操作会展平更新矩阵的奇异值结构，使其对当前特征的角向作用更接近各向同性。因此在范数被显式固定后，MuonH 在不同宽度间的残差对齐误差更小。

## 5. 全局缩放定律汇总

基于上述特征空间主方程，Hyperball 系列优化器在不同几何与统计假设下实现特征角速度对齐所需的学习率 $\eta$ 汇总如下：

| Hyperball 优化器变体 | 更新矩阵范数 $\lVert u_t \rVert_F$ | 所需 $\eta$ |
| :--- | :--- | :--- |
| SGDH | $\Theta(n)$ | $\Theta(1/\sqrt{n})$ |
| AdamH | $\Theta(n)$ | $\Theta(1)$ |
| MuonH | $\Theta(n)$ | $\Theta(1)$ |
{: .table .table-striped .table-sm style="font-size: 0.5em;"}

## 6. 结语

传统优化依赖内生权重范数去寻找自然平衡点，但这一机制深度耦合了网络宽度、调度策略和模型架构，因此在模型规模变化时无法保证球面角速度的跨尺度一致性。Hyperball 通过超球面上的几何投影约束消除了这种内生依赖，使雅可比前置系数化简为标量常数。推导表明，只有把 $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$ 作为统一对齐基准，并切断内生权重范数与超参数之间的耦合，优化器超参数的缩放规律才能被清楚刻画。若想继续补齐本文里 AdamH / MuonH 使用的更新矩阵范数假设，可以继续读 [《Adam 与 Muon 优化器更新矩阵的 Frobenius 范数估计》]({% post_url 2026-03-08-optimizer-update-matrix-norm %})。

## 参考文献

[1] Wen, K., Dang, X., Lyu, K., Ma, T., & Liang, P. (2025). Fantastic Pretraining Optimizers and Where to Find Them 2.1: Hyperball Optimization. https://tinyurl.com/muonh

[2] Wan, R., Zhu, Z., Zhang, X., & Sun, J. (2020). Spherical Motion Dynamics: Learning Dynamics of Neural Network with Normalization, Weight Decay, and SGD. arXiv preprint arXiv:2006.08419. https://arxiv.org/abs/2006.08419

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026sphericalhyperball,
  title={球面之上：带有 Hyperball 机制的优化器的 μP 缩放},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={[https://jiaxuanzou0714.github.io/blog/2026/spherical-hyperball/](https://jiaxuanzou0714.github.io/blog/2026/spherical-hyperball/)}
}
```