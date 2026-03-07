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

在现代神经网络的训练中，跨越不同模型规模的超参数迁移是一个核心挑战。在包含归一化的架构中，我们的注意力应该从网络参数的优化过程转向特征在超球面上的演化 [[2]](https://arxiv.org/abs/2006.08419)。

在包含归一化的现代神经网络架构中，归一化特征的量级 $\lVert z \rVert_2 = \sqrt{n}$ 已经被约束。因此，为了实现跨越不同网络宽度的超参数迁移，我们不再需要对齐整个训练过程中特征 $z$ 的量级，而仅需：
> 保证归一化特征 $z$ 的演化速率 $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$ 始终维持在稳定量级。

本文将揭示标准优化器的内生缺陷，并严格推导 Wen et al. [[1]](https://tinyurl.com/muonh) 提出的 Hyperball 系列优化器的缩放规律。

## 1. 基础设定与连续时间球面映射

设网络隐层宽度为 $n$。前向传播中的关键变量及其坐标和范数量级设定如下：

输入满足 $\lVert x \rVert_2^2 = \Theta(n)$，其逐坐标分量量级为 $x_j = \Theta(1)$。

未归一化特征定义为 $y_t = W_t x$。假设输入特征 $x$ 的方向在 $W_t$ 的子空间中处于各向同性分散状态，计算 $y_t$ 的 $L_2$ 范数平方：

$$
\lVert y_t \rVert_2^2 = x^T W_t^T W_t x = \Theta\left(\frac{\lVert x \rVert_2^2}{n} \text{Tr}(W_t^T W_t)\right)
$$

由于 $\lVert x \rVert_2^2 = \Theta(n)$ 且 $\text{Tr}(W_t^T W_t) = \lVert W_t \rVert_F^2$，可得：

$$
\lVert y_t \rVert_2^2 = \Theta\left(\frac{n}{n} \lVert W_t \rVert_F^2\right) = \Theta(\lVert W_t \rVert_F^2)
$$

开方后得到核心依赖关系：

$$
\lVert y_t \rVert_2 = \Theta(\lVert W_t \rVert_F)
$$

应用 rmsnorm 后，向后传递的特征严格定义为：

$$
z = \sqrt{n}\frac{y_t}{\lVert y_t \rVert_2}
$$

从归一化特征 $z$ 到未归一化特征 $y_t$ 的雅可比矩阵计算如下：

$$
J = \frac{\partial z}{\partial y_t} = \frac{\sqrt{n}}{\lVert y_t \rVert_2}P_y, \quad P_y = I - \frac{y_t y_t^T}{\lVert y_t \rVert_2^2}
$$

其中 $P_y$ 是将向量正交投影到以 $y_t$ 为法向量的超平面上的投影算子。引入更新矩阵 $U_t$，特征的球面动力学表示为：

$$
\frac{dz}{dt} = \frac{\sqrt{n}}{\lVert y_t \rVert_2} P_y \frac{dy_t}{dt} = \frac{\sqrt{n}}{\Theta(\lVert W_t \rVert_F)} P_y \left(-\eta U_t x\right)
$$

提取坐标分量的量级，要求其对齐至 $\Theta(1)$，我们得到没有任何范数约束前提下的通用对齐方程：

$$
\eta = \frac{\Theta(\lVert W_t \rVert_F)}{\sqrt{n} \lvert (P_y (U_t x))_i \rvert}
$$

## 2. 标准优化器的内生半径依赖与动力学平衡困境

从上述通用对齐方程可以看出，真正决定特征球面角速度的并非基础学习率 $\eta$，而是有效球面步长 $\eta_{\mathrm{eff}}^{(i)}(t)$：

$$
\eta_{\mathrm{eff}}^{(i)}(t) := \eta \frac{\sqrt{n} \lvert (P_y (U_t x))_i \rvert}{\lVert W_t \rVert_F}
$$

在此公式中，权重矩阵的弗罗贝尼乌斯范数 $\lVert W_t \rVert_F$ 充当了控制球面角速度的内生半径。对于带有解耦权重衰减的标准优化器，该内生半径并非固定常数，而是由具体的差分方程驱动演化。考察包含权重衰减系数 $\lambda$ 的随机梯度下降更新规则：

$$
W_{t+1} = W_t - \eta \left( \frac{\partial \mathcal{L}}{\partial W_t} + \lambda W_t \right)
$$

为解析权重范数的演化规律，对上式两端取弗罗贝尼乌斯范数的平方展开：

$$
\lVert W_{t+1} \rVert_F^2 = (1 - \eta \lambda)^2 \lVert W_t \rVert_F^2 - 2 \eta (1 - \eta \lambda) \left\langle W_t, \frac{\partial \mathcal{L}}{\partial W_t} \right\rangle + \eta^2 \left\lVert \frac{\partial \mathcal{L}}{\partial W_t} \right\rVert_F^2
$$

由于归一化机制赋予了权重矩阵严格的尺度不变性 [[2]](https://arxiv.org/abs/2006.08419)，网络输出不随权重量级的改变而改变，因此梯度张量始终严格正交于当前权重张量，即满足 $\langle W_t, \frac{\partial \mathcal{L}}{\partial W_t} \rangle = 0$。该正交性质使得展开式中的交叉项严格为零：

$$
\lVert W_{t+1} \rVert_F^2 = (1 - \eta \lambda)^2 \lVert W_t \rVert_F^2 + \eta^2 \left\lVert \frac{\partial \mathcal{L}}{\partial W_t} \right\rVert_F^2
$$

为了消解梯度范数对当前权重范数的依赖，引入单元梯度 $\tilde{G}_t = \lVert W_t \rVert_F \frac{\partial \mathcal{L}}{\partial W_t}$。将其代入并对等式两端取平方根，在 $\eta \lambda \ll 1$ 的条件下进行泰勒级数展开，可得单步内生半径的演化主方程：

$$
\lVert W_{t+1} \rVert_F \approx \lVert W_t \rVert_F - \lambda \eta \lVert W_t \rVert_F + \frac{\eta^2}{2 \lVert W_t \rVert_F^3} \lVert \tilde{G}_t \rVert_F^2
$$

当系统演化至驻态时，内生半径的期望保持平稳，即 $\mathbb{E}[\lVert W_{t+1} \rVert_F] = \mathbb{E}[\lVert W_t \rVert_F]$。设 $L = \mathbb{E}[\lVert \tilde{G}_t \rVert_F^2 \mid W_t]$ 为期望单元梯度范数的平方。令上述差分项的增量为零，即可严格解得内生半径的渐近理论极限 $w^*$：

$$
w^* = \sqrt[4]{\frac{L\eta}{2\lambda}}
$$

为解析渐近极限 $w^*$ 关于网络宽度 $n$ 的依赖，需要严格推导单元梯度 $\tilde{G}_t$ 的范数量级。设损失函数对归一化特征 $z$ 的梯度为 $g_z$。应用链式法则，未归一化特征 $y_t$ 接收到的梯度为 $\frac{\partial \mathcal{L}}{\partial y_t} = \frac{\sqrt{n}}{\lVert y_t \rVert_2} P_y g_z$。进而求解对权重张量 $W_t$ 的梯度，并代入单元梯度 $\tilde{G}_t$ 的定义式，利用输入特征在各向同性假设下的恒等式 $\lVert y_t \rVert_2 = \frac{\lVert x \rVert_2}{\sqrt{n}} \lVert W_t \rVert_F$，可得：

$$
\tilde{G}_t = \lVert W_t \rVert_F \frac{\sqrt{n}}{\frac{\lVert x \rVert_2}{\sqrt{n}} \lVert W_t \rVert_F} (P_y g_z) x^T = \frac{n}{\lVert x \rVert_2} (P_y g_z) x^T
$$

计算其弗罗贝尼乌斯范数的平方：

$$
\lVert \tilde{G}_t \rVert_F^2 = \frac{n^2}{\lVert x \rVert_2^2} \lVert P_y g_z \rVert_2^2 \lVert x \rVert_2^2 = n^2 \lVert P_y g_z \rVert_2^2
$$

设定上游传回的归一化梯度坐标量级为 $(g_z)_i = \Theta(1)$，故其平方范数 $\lVert g_z \rVert_2^2 = \Theta(n)$。由于 $P_y$ 为超平面正交投影算子，其对范数的衰减仅为 $\Theta(1)$，因此 $\lVert P_y g_z \rVert_2^2 = \Theta(n)$。由此得到期望单元梯度范数平方 $L$ 的严格标度：

$$
L = \mathbb{E}[\lVert \tilde{G}_t \rVert_F^2 \mid W_t] = n^2 \Theta(n) = \Theta(n^3)
$$

将 $L$ 的量级代入 $w^*$ 的表达式中，假设权重衰减系数 $\lambda = \Theta(1)$，得到动力学驻点处的内生半径标度：

$$
\lVert W_t \rVert_F = w^* = \Theta\left(\sqrt[4]{n^3 \eta}\right) = \Theta(n^{3/4} \eta^{1/4})
$$

要求特征演化速率满足对齐基准 $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$。对于随机梯度下降，更新方向 $U_t = \frac{\partial \mathcal{L}}{\partial W_t} = \frac{\sqrt{n}}{\lVert y_t \rVert_2} (P_y g_z) x^T$。其对输入特征 $x$ 的作用为：

$$
U_t x = \frac{\sqrt{n}}{\lVert y_t \rVert_2} (P_y g_z) x^T x = \frac{\sqrt{n} \lVert x \rVert_2^2}{\lVert y_t \rVert_2} (P_y g_z)
$$

代入 $\lVert x \rVert_2^2 = \Theta(n)$ 及 $\lVert y_t \rVert_2 = \Theta(\lVert W_t \rVert_F)$，并应用投影算子的幂等性 $P_y^2 = P_y$：

$$
P_y (U_t x) = \frac{n \sqrt{n}}{\Theta(\lVert W_t \rVert_F)} (P_y g_z)
$$

提取坐标分量的绝对值量级 $\lvert (P_y (U_t x))_i \rvert = \Theta\left(\frac{n \sqrt{n}}{\lVert W_t \rVert_F}\right)$。代入特征演化对齐方程：

$$
\frac{\eta \sqrt{n}}{\lVert W_t \rVert_F} \Theta\left(\frac{n \sqrt{n}}{\lVert W_t \rVert_F}\right) = \Theta\left(\frac{\eta n^2}{\lVert W_t \rVert_F^2}\right) = \Theta(1)
$$

将驻点处的内生半径标度 $\lVert W_t \rVert_F = \Theta(n^{3/4} \eta^{1/4})$ 代入上述约束：

$$
\frac{\eta n^2}{(n^{3/4} \eta^{1/4})^2} = \frac{\eta n^2}{n^{3/2} \eta^{1/2}} = \eta^{1/2} n^{1/2} = \Theta(1)
$$

求解可得实现跨尺度对齐所需的基础学习率精确缩放规律：

$$
\eta = \Theta\left(\frac{1}{n}\right)
$$

将此学习率缩放规律回代至内生半径的表达式，可得系统在动力学平衡状态下的最终范数量级：

$$
w^* = \Theta\left(n^{3/4} (n^{-1})^{1/4}\right) = \Theta(n^{1/2}) = \Theta(\sqrt{n})
$$

上述推导揭示了一个深刻的数学事实：如果系统能够瞬时达到动力学平衡，在设定 $\eta = \Theta(1/n)$ 时，标准优化的内生权重范数会自动收敛至 $\Theta(\sqrt{n})$，这与标准初始化方差所对应的量级完全一致。上述整段的推导的核心 insight 其实还是和 mup 类似的。

然而，依赖此演化机制逼近自然驻点来维持超参数对齐，在实际工程中面临三个无法逾越的动力学困境：

1. 收敛有延迟：$w^*$ 是渐近极限，权重范数不会瞬间到达平衡点。训练初期 $\lVert W_t \rVert_F$ 还没收敛到 $\Theta(\sqrt{n})$，球面上的有效步长就是错的。
2. 学习率一变就失衡：现代训练普遍使用多阶段学习率调度。$\eta$ 一旦衰减，对应的平衡点 $w^*$ 立刻改变，但权重范数需要时间追上新的平衡点。在这段过渡期内，对齐条件不成立。
3. 正交假设不总成立：以上推导依赖 $\langle W_t, \frac{\partial \mathcal{L}}{\partial W_t} \rangle = 0$，即梯度严格正交于权重。但在带残差连接的网络中，这个条件并不严格满足，交叉项不为零，各层的权重范数会各自漂移，无法统一对齐。

## 3. Hyperball 约束与统一的主方程

为了从根本上消除内生半径 $\lVert W_t \rVert_F$ 演化对动力学的破坏，Wen et al. [[1]](https://tinyurl.com/muonh) 提出的 Hyperball 机制显式引入切空间投影算子 $\Pi_W$，将权重更新方向严格约束在以初始范数为半径的超球面上。

离散更新规则的连续时间极限下，其权重动力学方程为：

$$
\frac{dW}{dt} = -\eta \lVert W_0 \rVert_F \Pi_W\left(\frac{u_t}{\lVert u_t \rVert_F}\right)
$$

在标准初始化下，矩阵元素采样方差为 $\frac{1}{n}$，由此可得初始常量 $R = \lVert W_0 \rVert_F = \Theta(\sqrt{n})$。切空间投影算子 $\Pi_W$ 严格保证了权重矩阵的弗罗贝尼乌斯范数在任意时间 $t$ 恒定不变：

$$
\lVert W_t \rVert_F = \lVert W_0 \rVert_F = \Theta(\sqrt{n})
$$

由于权重范数被强制固定，动态分母被转化为时间与宽度的不变量：

$$
\lVert y_t \rVert_2 = \Theta(\lVert W_t \rVert_F) = \Theta(\sqrt{n})
$$

利用投影算子的线性性质 $P_y(\Pi_W(A)x) = P_y(Ax)$，并将其代入第一节确立的雅可比矩阵，前置系数化简为常数 $\frac{\sqrt{n}}{\Theta(\sqrt{n})} = \Theta(1)$。由此，我们得到了控制所有 Hyperball 变体动力学行为的统一主方程：

$$
\frac{dz}{dt} = -\eta \Theta(\sqrt{n}) \frac{1}{\lVert u_t \rVert_F} P_y (u_t x)
$$

为了使得 $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$ 严格成立，学习率 $\eta$ 必须满足如下等式约束：

$$
\eta = \frac{\lVert u_t \rVert_F}{\Theta(\sqrt{n}) \lvert (P_y (u_t x))_i \rvert}
$$

| | 无 hyperball  | 有 Hyperball  |
| :--- | :--- | :--- |
| 学习率约束 | $\eta = \Theta\left(\frac{\lVert W_t \rVert_F}{\sqrt{n} \lvert (P_y (U_t x))_i \rvert}\right)$ | $\eta = \Theta\left(\frac{\lVert u_t \rVert_F}{\sqrt{n} \lvert (P_y (u_t x))_i \rvert}\right)$ |
{: .table .table-striped .table-sm style="font-size: 0.5em;"}


> 这种转换彻底切断了超参数对齐与系统收敛状态之间的耦合。在无 Hyperball 的系统中，实现正确的球面角速度要求网络必须通过权重衰减与梯度的正交性达到平衡态 $\lVert W_t \rVert_F = \Theta(\sqrt{n})$。而在 Hyperball 系统中，$\lVert u_t \rVert_F$ 这一项被作为显式标量提取到了学习率的分子上，使得 $\eta$ 的设置仅依赖于当前更新步骤的梯度结构属性，不再受制于权重范数的历史累积状态。

## 4. 具体 Hyperball 变体的对齐推导

基于上文推导，本节将通过解析不同优化器在特定假设下的更新特性，严格推导其对应的对齐学习率。

### 4.1 SGDH 的对齐推导

设上游传回的梯度 $g = \nabla_z L$ 的坐标量级为 $\Theta(1)$。基础梯度更新矩阵为 $u_t = \Theta(1) (P_y g) x^T$。首先计算该矩阵的弗罗贝尼乌斯范数，由于 $\lVert P_y g \rVert_2 = \Theta(\sqrt{n})$ 且 $\lVert x \rVert_2 = \Theta(\sqrt{n})$：

$$
\lVert u_t \rVert_F = \Theta(1) \lVert P_y g \rVert_2 \lVert x \rVert_2 = \Theta(n)
$$

考察更新矩阵对输入特征向量的线性作用：

$$
u_t x = \Theta(1) (P_y g) (x^T x) = \Theta(1) (P_y g) \Theta(n) = \Theta(n) P_y g
$$

经切空间超平面的正交投影算子作用：

$$
P_y (u_t x) = P_y (\Theta(n) P_y g) = \Theta(n) P_y g
$$

由于前提假设 $\lvert (P_y g)_i \rvert = \Theta(1)$，故分量绝对值为 $\lvert (P_y (u_t x))_i \rvert = \Theta(n)$。代入主方程求解：

$$
\eta = \frac{\Theta(n)}{\Theta(\sqrt{n}) \Theta(n)} = \Theta\left(\frac{1}{\sqrt{n}}\right)
$$

### 4.2 AdamH 的对齐推导

通过提取梯度的符号矩阵，更新矩阵 $u_t$ 包含 $n^2$ 个绝对值为 $1$ 的元素，其范数严格恒定：

$$
\lVert u_t \rVert_F = \sqrt{n^2} = n = \Theta(n)
$$

在引入动量或大批量设定下，更新矩阵的符号与当前输入向量 $x$ 的坐标系分布相互独立。根据中心极限定理，这 $n$ 个独立项的线性组合使得分量量级衰减为 $\lvert (u_t x)_i \rvert = \Theta(\sqrt{n})$。由此 $\lvert (P_y (u_t x))_i \rvert = \Theta(\sqrt{n})$。代入主方程求解：

$$
\eta = \frac{\Theta(n)}{\Theta(\sqrt{n}) \Theta(\sqrt{n})} = \Theta(1)
$$

### 4.3 MuonH 的对齐推导与各向同性优势

在 Muon 优化器的实际工程实现中，会对正交化后的更新量进行学习率调整，使得不同矩阵形状下的更新量均方根量级与标准优化器保持一致。

在此实践设定下，更新矩阵的弗罗贝尼乌斯范数的主导阶近似为：

$$
\lVert u_t \rVert_F = \Theta(n)
$$

同时，考察该更新矩阵对当前输入特征 $x$ 的作用。由于正交化操作以及均方根对齐，其作用于特征向量并经切空间投影后，各坐标分量的典型量级为：

$$
\lvert (P_y (u_t x))_i \rvert = \Theta(\sqrt{n})
$$

将上述量级代入主方程求解最优学习率：

$$
\eta = \frac{\Theta(n)}{\Theta(\sqrt{n}) \Theta(\sqrt{n})} = \Theta(1)
$$

该推导阐明了未约束 Muon 发生漂移的本质：内生半径在不同宽度下未能自动对齐；而 MuonH 强制 $\lVert W_t \rVert_F = \Theta(\sqrt{n})$ 后，主导项随即对齐至常数学习率。

进一步将 MuonH 与 AdamH 对比，MuonH 往往展现出更为精确的跨尺度对齐特性。其内在机制在于：Adam 即使采用 Hyperball 约束，其更新矩阵 $u_t$ 仍依赖于逐元素的自适应归一化，导致投影分量 $\lvert (P_y (u_t x))_i \rvert$ 存在残留的各向异性误差；相反，Muon 的正交化操作将更新矩阵的奇异值结构展平，使得其对当前特征的角向作用更接近各向同性。因此，当范数被显式固定后，MuonH 在不同宽度间的残差对齐误差得以显著降低。

## 5. 全局缩放定律汇总

基于自顶向下的特征空间主方程，Hyperball 系列优化器在不同几何与统计假设下，为实现特征角速度对齐所需的最优学习率 $\eta$ 汇总如下：

| Hyperball 优化器变体 | 更新矩阵范数 $\lVert u_t \rVert_F$ | 所需 $\eta$ |
| :--- | :--- | :--- |
| SGDH | $\Theta(n)$ | $\Theta(1/\sqrt{n})$ |
| AdamH | $\Theta(n)$ | $\Theta(1)$ |
| MuonH | $\Theta(n)$ | $\Theta(1)$ |
{: .table .table-striped .table-sm style="font-size: 0.5em;"}

## 6. 结语

传统优化理论中依赖内生权重范数寻找自然平衡点的做法，由于该平衡机制深度耦合了网络宽度、调度策略与模型架构，在网络规模变化时无法保证跨尺度下球面角速度的一致性。Hyperball 通过超球面上严格的几何投影约束，将内生依赖从球面动力学中消除，使雅可比前置系数彻底化简为标量常数。推导表明，只有确立了 $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$ 这一恒定的对齐基准，并消除内生权重范数与具体超参数之间的多重耦合，优化器超参数缩悉规律的数学机制才能被严密解析。

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