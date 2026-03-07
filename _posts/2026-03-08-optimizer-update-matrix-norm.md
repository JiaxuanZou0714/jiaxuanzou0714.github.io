---
layout: post
title: "Adam 与 Muon 优化器更新矩阵的 Frobenius 范数估计"
date: 2026-03-08 12:00:00
description: "本文严密推导并估计了 Adam 与 Muon 优化器在单步迭代中更新矩阵的 Frobenius 范数，并探讨了矩阵形状对范数量级的影响。"
tags: [optimizer, adam, muon, frobenius-norm]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

在深度学习的优化过程中，理解优化器单步更新矩阵的尺度对于学习率设定与训练稳定性分析至关重要。本文旨在通过严谨的数学推导，估计 Adam 与 Muon 优化器更新矩阵的 Frobenius 范数。推导过程将从参数矩阵的逐元素更新序列出发，分别探讨极端梯度序列下的理论绝对上界、纯随机噪声状态下的统计期望，以及矩阵形状维度对最终范数量级的直接影响。

这篇文章的一个直接动机，是对上一篇 [《球面之上：带有 Hyperball 机制的优化器的 μP 缩放》]({% post_url 2026-03-06-spherical-hyperball %}) 的补充。在那篇文章里，AdamH 与 MuonH 两部分推导都使用了 $\lVert u_t \rVert_F = \Theta(n)$ 这一量级假设；而本文的目标，就是把这个假设单独拆出来，对 Adam 与 Muon 的更新矩阵 Frobenius 范数做更明确的推导与估计，从而把后续学习率缩放分析的前提写清楚。

## 1. 基础设定

设需要优化的参数矩阵为 $W_t \in \mathbb{R}^{m \times n}$，其元素总数为 $d = mn$。在第 $t$ 步迭代中，损失函数对该参数矩阵的梯度为 $G_t \in \mathbb{R}^{m \times n}$。基础学习率记为 $\alpha$。

## 2. Adam 优化器更新矩阵范数估计

根据 Adam 算法的定义，一阶矩估计 $M_t$ 和二阶矩估计 $V_t$ 的更新公式如下：

$$
M_t = \beta_1 M_{t-1} + (1 - \beta_1) G_t
$$

$$
V_t = \beta_2 V_{t-1} + (1 - \beta_2) G_t^{\odot 2}
$$

其中 $G_t^{\odot 2}$ 表示矩阵元素的逐元素平方，$\beta_1, \beta_2 \in [0, 1)$ 为指数衰减率。包含偏差校正的修正矩估计为：

$$
\hat{M}_t = \frac{M_t}{1 - \beta_1^t}, \qquad \hat{V}_t = \frac{V_t}{1 - \beta_2^t}
$$

参数矩阵的整体更新量 $\Delta W_t$ 定义为：

$$
\Delta W_t = - \alpha \frac{\hat{M}_t}{\sqrt{\hat{V}_t} + \varepsilon}
$$

更新矩阵的 Frobenius 范数精确定义为：

$$
\lVert \Delta W_t \rVert_F = \alpha \left( \sum_{i,j} \frac{\hat{M}_{t,ij}^2}{\left(\sqrt{\hat{V}_{t,ij}} + \varepsilon\right)^2} \right)^{1/2}
$$

### 2.1 逐元素理论绝对上界

定义每个元素的有效比值为 $$r_{ij} = \frac{\lvert \hat{M}_{t,ij} \rvert}{\sqrt{\hat{V}_{t,ij}} + \varepsilon}$$。忽略极小的平滑常数 $$\varepsilon$$，将 $$\hat{M}_{t,ij}$$ 和 $$\hat{V}_{t,ij}$$ 展开为关于历史梯度元素 $$G_{k,ij}$$ 的加权求和形式：

$$
\hat{M}_{t,ij} = \frac{1 - \beta_1}{1 - \beta_1^t} \sum_{k=1}^t \beta_1^{t-k} G_{k,ij}
$$

$$
\hat{V}_{t,ij} = \frac{1 - \beta_2}{1 - \beta_2^t} \sum_{k=1}^t \beta_2^{t-k} G_{k,ij}^2
$$

对 $\hat{M}_{t,ij}$ 的求和表达式应用 Cauchy-Schwarz 不等式，将其重写为两项乘积并放缩：

$$
\left( \sum_{k=1}^t \beta_1^{t-k} G_{k,ij} \right)^2 \le \left( \sum_{k=1}^t \frac{\beta_1^{2(t-k)}}{\beta_2^{t-k}} \right) \left( \sum_{k=1}^t \beta_2^{t-k} G_{k,ij}^2 \right)
$$

右侧第二个求和项即构成 $$\hat{V}_{t,ij}$$ 的非归一化形式。右侧第一个求和项为等比数列求和。要求设定的超参数满足 $$\frac{\beta_1^2}{\beta_2} < 1$$，代回 $$\hat{M}_{t,ij}^2$$ 表达式并移项，得到绝对值平方的上界：

$$
\frac{\hat{M}_{t,ij}^2}{\hat{V}_{t,ij}} \le \frac{(1 - \beta_1)^2 (1 - \beta_2^t)}{(1 - \beta_1^t)^2 (1 - \beta_2) \left( 1 - \frac{\beta_1^2}{\beta_2} \right)} \left( 1 - \left( \frac{\beta_1^2}{\beta_2} \right)^t \right)
$$

当 $t \to \infty$ 时，各项偏差校正项收敛于 $1$，极限为：

$$
\lim_{t \to \infty} \frac{\hat{M}_{t,ij}^2}{\hat{V}_{t,ij}} \le \frac{(1 - \beta_1)^2}{1 - \beta_2} \frac{\beta_2}{\beta_2 - \beta_1^2}
$$

代入默认参数 $\beta_1 = 0.9, \beta_2 = 0.999$，计算得该常数约为 $52.857$。即单元素更新比值严格受限于 $\lvert r_{ij} \rvert \le 7.27$。这证明了 $\lvert r_{ij} \rvert = \mathcal{O}(1)$ 的普适性。

### 2.2 平稳随机梯度下的期望估计

若梯度呈现零均值的独立随机波动，假设 $G_{k,ij}$ 为独立同分布随机变量，$E[G_{k,ij}] = 0$，$\operatorname{Var}(G_{k,ij}) = \sigma^2$。在此假设下，$$\hat{M}_{t,ij}$$ 的方差渐近收敛为：

$$
\lim_{t \to \infty} \operatorname{Var}(\hat{M}_{t,ij}) = \frac{1 - \beta_1}{1 + \beta_1} \sigma^2
$$

由于 $\beta_2$ 接近 $1$，二阶矩经过大样本平均高度集中于 $\sigma^2$。尺度因子均方值的期望近似为：

$$
E \left[ \frac{\hat{M}_{t,ij}^2}{\hat{V}_{t,ij}} \right] \approx \frac{\operatorname{Var}(\hat{M}_{t,ij})}{\sigma^2} = \frac{1 - \beta_1}{1 + \beta_1}
$$

### 2.3 矩阵整体 Frobenius 范数量级

根据前述推导，无论是在理论极端情况还是随机稳态下，单元素的更新比值均方根均为 $\mathcal{O}(1)$。若矩阵内 $d = mn$ 个元素均处于稠密活跃状态，整体范数为：

$$
\lVert \Delta W_t \rVert_F \approx \alpha \sqrt{\sum_{i,j} E[r_{ij}^2]} = \mathcal{O}(\alpha \sqrt{mn})
$$

若存在稀疏更新，仅有 $k_{\mathrm{eff}}$ 个坐标显著非零，则：

$$
\lVert \Delta W_t \rVert_F = \mathcal{O}(\alpha \sqrt{k_{\mathrm{eff}}})
$$

## 3. Muon 优化器更新矩阵范数估计

Muon 优化器的核心在于提取动量矩阵的正交成分。设动量矩阵 $M_t = U S V^\top$，近似正交化操作（如 Newton-Schulz 迭代）等价于输出最接近的半正交矩阵 $U V^\top$。

### 3.1 原始无缩放正交化矩阵

正交化后矩阵的非零奇异值全为 $1$。记 $r = \operatorname{rank}(M_t)$，基础正交化矩阵范数为：

$$
\lVert U V^\top \rVert_F = \sqrt{r}
$$

若动量矩阵满秩，即 $r = \min(m, n)$，则原始单步更新量级为：

$$
\lVert \Delta W_t \rVert_F = \Theta(\alpha \sqrt{\min(m, n)})
$$

此时，其逐元素均方根量级为 $\Theta\left(\frac{\alpha}{\sqrt{\max(m, n)}}\right)$。

### 3.2 工程缩放版更新矩阵

为使 Muon 能与标准优化器无缝替换，其实际工程实现会引入显式标量放大因子，将更新矩阵的均方根提升至 $\Theta(1)$ 以对齐基础学习率尺度。

经过均方根对齐缩放后，更新矩阵的逐元素均方根变为 $\Theta(1)$。根据反向计算，此时整体更新矩阵的 Frobenius 范数显式放大为：

$$
\lVert \Delta W_t \rVert_F = \Theta(\alpha \sqrt{mn})
$$

## 4. 特征宽度与矩阵维度分析

当分析特定网络架构（如设隐层特征宽度为 $n$）时，权重矩阵常表现为 $n \times n$ 的方阵。此时 $m = n$，总元素数量 $d = n^2$。

对于 Adam 优化器与工程缩放版的 Muon 优化器，将 $d = n^2$ 代入前述结论，更新矩阵的 Frobenius 范数为：

$$
\lVert \Delta W_t \rVert_F = \Theta(\alpha \sqrt{n^2}) = \Theta(\alpha n)
$$

这解释了在方阵假设下，为何更新矩阵范数的量级呈现关于宽度 $n$ 的线性量级 $\Theta(n)$，而非 $\Theta(\sqrt{n})$。而仅有纯数学定义的未缩放正交矩阵，其量级为 $\Theta(\sqrt{n})$。

## 5. 结论

不同优化器在单步更新时的矩阵范数受自适应机制与正交化缩放的显著影响。相关渐进量级总结如下表所示：

| 优化器算法 | 更新状态 | 逐元素 RMS 量级 | Frobenius 范数量级 (矩形 $m \times n$) | Frobenius 范数量级 (方阵 $n \times n$) |
| :--- | :---: | :---: | :---: | ---: |
| **Adam** | 稠密更新 | $\mathcal{O}(1)$ | $\mathcal{O}(\alpha \sqrt{mn})$ | $\mathcal{O}(\alpha n)$ |
| **Adam** | 稀疏更新 ($k_{\mathrm{eff}}$) | $\mathcal{O}(1)$ | $\mathcal{O}(\alpha \sqrt{k_{\mathrm{eff}}})$ | - |
| **Muon** (理论原始) | 满秩极分解 | $\Theta(1/\sqrt{\max(m,n)})$ | $\Theta(\alpha \sqrt{\min(m, n)})$ | $\Theta(\alpha \sqrt{n})$ |
| **Muon** (工程缩放) | 均方根对齐 | $\Theta(1)$ | $\Theta(\alpha \sqrt{mn})$ | $\Theta(\alpha n)$ |
{: .table .table-striped .table-sm style="font-size: 0.3em;"}

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026optimizer-update-matrix-norm,
  title={Adam 与 Muon 优化器更新矩阵的 Frobenius 范数估计},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/optimizer-update-matrix-norm/}
}
```