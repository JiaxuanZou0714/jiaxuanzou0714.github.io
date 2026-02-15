---
layout: post
title: "Tensor Programs (一)：Feature Learning 的谱条件"
date: 2026-02-14 17:00:00
description: "本文介绍 Greg Yang 的 Tensor Programs 系列的入门论文——A Spectral Condition for Feature Learning，从谱范数的视角推导出 feature learning 所需的 scaling 条件，并由此重新推导 maximal update parametrization（μP）。"
tags: [deep-learning, tensor-programs, muP, feature-learning]
categories: [deep-learning]
featured: true
giscus_comments: true
toc:
  sidebar: left
---

> 这是 Tensor Programs 系列导读的第一篇。整个系列旨在向读者介绍 Greg Yang 发起的 [Tensor Programs](https://thegregyang.com/) 研究计划——一个试图为深度学习中的宽度极限、feature learning 和超参数迁移提供统一数学基础的宏大框架。本文选择了 Greg Yang 本人推荐的入门论文 *A Spectral Condition for Feature Learning* [1] 作为起点。

## 0. Why Tensor Programs?

深度学习的核心魔力在于 Feature Learning：模型能够从原始数据中自动学习到分层的、语义丰富的表示。这种能力使得神经网络超越了传统的核方法，让 LLM 的成功成为可能。

然而，现在大多数的理论研究仍依赖于 NTK 这一套数学工具。NTK 描述了神经网络在 infinite-width 极限下的行为，它将神经网络视为一个固定的核函数，其预测值本质上是训练样本的核函数加权求和（即核回归，或者说初始化处的随机特征的线性回归）：

$$
f(\boldsymbol{x}) = \sum_{i=1}^N \alpha_i K_{\text{NTK}}(\boldsymbol{x}, \boldsymbol{x}_i), \quad \text{其中 } K_{\text{NTK}}(\boldsymbol{x}, \boldsymbol{x}_i) = \langle \nabla f(\boldsymbol{x}; \boldsymbol{\theta}_0), \nabla f(\boldsymbol{x}_i; \boldsymbol{\theta}_0) \rangle.
$$

NTK 框架的缺点在于，在它的setting下，训练后的权重仍然在初始化附近（也叫做lazy learning，因为这样才能够做泰勒展开的一阶近似）。此时，神经网络实际上是在初始化得到的随机特征上做线性回归，且特征在整个训练阶段都不改变，完全脱离真实情况下的 feature learning。

而现实情况是，我们需要 feature learning，同时我们也在不断扩大模型宽度，但如果采用 NTK 的 setting，我们将失去 feature learning 能力（落入 lazy learning regime）。这引出了一个至关重要的问题：**当我们想要训练宽度更大的模型（scale up）时，如何才能保持模型的 feature learning 能力？**

这正是 Tensor Programs 系列研究试图回答的核心问题。Tensor Programs 不仅仅是一个简单的参数化技巧，它是一个宏大的数学框架，旨在精确描述神经网络计算在无限宽极限下的行为。在这个框架下，Greg Yang 等人推导出了一套在无限宽极限下能够保持特征学习能力的参数化方案——即著名的 Maximal Update Parametrization (μP)。

换句话说，μP 只是 Tensor Programs 这一伟大数学框架的一个副产品（尽管是极其有用的副产品，它允许我们在小模型上调参并零样本迁移到大模型，即hyperparameter transfer）。要真正理解 μP 的本质，我们需要回到更基础的数学规律中去。而本文要介绍的论文 [1]，正提供了一个切入点，通过谱范数来直观理解 feature learning 保持条件，以及如何通过这个保持条件导出 μP。

---

## 1. What is Feature Learning?

考虑一个 $$L$$ 层 MLP。记第 $$\ell$$ 层的隐藏表示为 $$\boldsymbol{h}_\ell(\boldsymbol{x}) \in \mathbb{R}^{n_\ell}$$，经过一步梯度下降后的变化为 $$\Delta \boldsymbol{h}_\ell(\boldsymbol{x})$$。我们希望神经网络发生 feature learning，说白了就是希望特征发生显著的改变，所以我们把 feature learning 定义如下。

> Feature Learning：对所有隐藏层 $$\ell$$，当各层宽度 $$n_\ell \to \infty$$ 时：
> $$
> \|\boldsymbol{h}_\ell(\boldsymbol{x})\|_2 = \Theta(\sqrt{n_\ell}), \qquad \|\Delta \boldsymbol{h}_\ell(\boldsymbol{x})\|_2 = \Theta(\sqrt{n_\ell}).
> $$

换句话说，特征向量每个分量的"典型大小"是 $$\Theta(1)$$（即常数量级，不随网络宽度 $$n$$ 变化），这是合理的，因为常见的激活函数被设计成接受 $$O(1)$$ 的输入并产生 $$O(1)$$ 的输出。同时，训练中的更新量的分量也是 $$\Theta(1)$$ 量级。这也是合理的——更大的更新会在宽度增大时爆炸（导致训练不稳定，发散），更小的更新会在宽度增大时消失（导致特征一直不变，即 lazy learning）。

---

## 2. Spectral Norm vs. Frobenius Norm

在讨论 scaling 之前，需要区分两种矩阵范数。对一个 $$m \times n$$ 矩阵 $$\boldsymbol{A}$$：

- 谱范数：$$\|\boldsymbol{A}\|_* = \max_{\|\boldsymbol{v}\|_2=1} \|\boldsymbol{A}\boldsymbol{v}\|_2 = \sigma_{\max}(\boldsymbol{A})$$。
- Frobenius 范数：$$\|\boldsymbol{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$$。

它们之间的关系是 $$\|\boldsymbol{A}\|_* \leq \|\boldsymbol{A}\|_F \leq \sqrt{\mathrm{rank}(\boldsymbol{A})} \cdot \|\boldsymbol{A}\|_*$$。

关键差异在于：对于 iid Gaussian 随机矩阵 $$\boldsymbol{A} \in \mathbb{R}^{m \times n}$$（元素方差为 $$\sigma^2$$），

$$
\|\boldsymbol{A}\|_F \approx \sigma \sqrt{mn}, \qquad \|\boldsymbol{A}\|_* \approx \sigma(\sqrt{m} + \sqrt{n}).
$$

推导过程可见苏剑林的博客（[随机矩阵的谱范数的快速估计](https://spaces.ac.cn/archives/11335)）。

Frobenius 范数本质上在度量矩阵所有元素的"总能量"，它与矩阵的维度线性增长；而谱范数度量的是矩阵作为线性算子的最大放大倍数，更直接地反映了 $$\boldsymbol{A}\boldsymbol{v}$$ 的行为。从这个角度来看，由于神经网络里有大量形如 $$\boldsymbol{A}\boldsymbol{v}$$ 的线性运算，谱范数似乎是更合理的度量。

这个差异也解释了为什么基于 Frobenius 范数或元素大小的 scaling 方案与基于谱范数的方案可能给出不同的结论。

---

## 3. 从 Feature Learning 推导谱条件

有了 feature learning 的目标定义和谱范数这个工具，我们可以开始推导：到底需要满足什么条件，才能保证 feature learning 发生？

### 3.1 前向传播：$$\scriptsize\|\boldsymbol{h}_\ell(\boldsymbol{x})\|_2 = \Theta(\sqrt{n_\ell})$$

先考虑一个简化的 setting，考虑一个 $$L$$ 层线性 MLP：$$\boldsymbol{h}_\ell(\boldsymbol{x}) = \boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}(\boldsymbol{x})$$，输入满足 $$\|\boldsymbol{x}\|_2 = \Theta(\sqrt{n_0})$$。

我们的目标是让 $$\|\boldsymbol{h}_\ell\|_2 = \Theta(\sqrt{n_\ell})$$。每一层做的事情就是 $$\boldsymbol{h}_\ell = \boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}$$，而谱范数的次乘性告诉我们：

$$
\|\boldsymbol{h}_\ell\|_2 = \|\boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}\|_2 \leq \|\boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_{\ell-1}\|_2.
$$

假设 $$\|\boldsymbol{h}_{\ell-1}\|_2 = \Theta(\sqrt{n_{\ell-1}})$$ 已经成立（归纳假设），那么要让 $$\|\boldsymbol{h}_\ell\|_2 = \Theta(\sqrt{n_\ell})$$，我们需要线性算子的"放大倍数"恰好是：

$$
\|\boldsymbol{W}_\ell\|_* = \Theta\!\left(\sqrt{\frac{n_\ell}{n_{\ell-1}}}\right).
$$

这就是从 feature learning 的数值稳定性要求，自然推导出的对权重矩阵谱范数的约束。

但仅有上界还不够——上面的不等式只说明了谱范数不能太大，否则特征会爆炸。次乘性的上界够紧吗？会不会实际上 $$\|\boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}\|_2 \ll \|\boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_{\ell-1}\|_2$$，导致特征逐层消失？其实不会，我们有以下的观察。

> Claim 1：对于随机初始化的权重矩阵（高斯或半正交），当 fan-out $$\geq$$ fan-in，即 $$n_\ell \geq n_{\ell-1}$$ 时，有
> $$
> \|\boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}\|_2 = \Theta(\|\boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_{\ell-1}\|_2).
> $$

这一点对高斯初始化用大数定律即可验证：$$\|\boldsymbol{W}_\ell \boldsymbol{h}\|^2 = \sum_{i=1}^{n_\ell} (\sum_j W_{ij} h_j)^2$$，每个 $$\sum_j W_{ij} h_j$$ 的方差为 $$\sigma_\ell^2 \|\boldsymbol{h}\|^2$$，对 $$n_\ell$$ 个独立项求和后浓度集中于期望 $$n_\ell \sigma_\ell^2 \|\boldsymbol{h}\|^2$$。结合 $$\|\boldsymbol{W}_\ell\|_* \approx \sigma_\ell(\sqrt{n_\ell} + \sqrt{n_{\ell-1}}) = \Theta(\sigma_\ell \sqrt{n_\ell})$$（当 $$n_\ell \geq n_{\ell-1}$$），这就保证了下界和上界同阶。

于是，次乘性的上界在随机初始化下是紧的，$$\|\boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$，是充要条件。

### 3.2 梯度更新：$$\scriptsize\|\Delta \boldsymbol{h}_\ell\|_2 = \Theta(\sqrt{n_\ell})$$

Feature learning 不仅要求初始特征的量级正确，还要求训练过程中的特征更新 $$\Delta \boldsymbol{h}_\ell$$ 也是 $$\Theta(\sqrt{n_\ell})$$。类似的推导会给出对 $$\Delta \boldsymbol{W}_\ell$$ 的约束。

对于 batch size 为 1 的梯度下降，第 $$\ell$$ 层的权重更新为：

$$
\Delta \boldsymbol{W}_\ell = -\eta_\ell \nabla_{\boldsymbol{W}_\ell} \mathcal{L}.
$$

其中 $$\eta_\ell$$ 是第 $$\ell$$ 层的学习率。现在的问题是：$$\nabla_{\boldsymbol{W}_\ell} \mathcal{L}$$ 长什么样？

记反传信号 $$\boldsymbol{\delta}_\ell = \partial \mathcal{L} / \partial \boldsymbol{h}_\ell \in \mathbb{R}^{n_\ell}$$。由于前向传播是 $$\boldsymbol{h}_\ell = \boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}$$，对权重矩阵的第 $$(i,j)$$ 个元素用链式法则：

$$
\frac{\partial \mathcal{L}}{\partial W_{\ell,ij}} = \frac{\partial \mathcal{L}}{\partial h_{\ell,i}} \cdot \frac{\partial h_{\ell,i}}{\partial W_{\ell,ij}} = \delta_{\ell,i} \cdot h_{\ell-1,j}.
$$

写成矩阵形式，就是 $$\nabla_{\boldsymbol{W}_\ell} \mathcal{L} = \boldsymbol{\delta}_\ell \boldsymbol{h}_{\ell-1}^\top$$。代回更新公式：

$$
\Delta \boldsymbol{W}_\ell = -\eta_\ell \boldsymbol{\delta}_\ell \boldsymbol{h}_{\ell-1}^\top.
$$

这是一个秩一矩阵（两个向量的外积）。由于这是秩一矩阵，谱范数等于 Frobenius 范数：

$$
\|\Delta \boldsymbol{W}_\ell\|_* = \|\Delta \boldsymbol{W}_\ell\|_F = \eta_\ell \|\boldsymbol{\delta}_\ell\|_2 \cdot \|\boldsymbol{h}_{\ell-1}\|_2.
$$

这里有一个非常漂亮的性质。
> Claim 2：由于 $$\Delta \boldsymbol{W}_\ell$$ 的右奇异向量恰好是 $$\boldsymbol{h}_{\ell-1}$$，所以
> $$
> \|\Delta \boldsymbol{W}_\ell \cdot \boldsymbol{h}_{\ell-1}\|_2 = \|\Delta \boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_{\ell-1}\|_2.
> $$

次乘性在这里取等号——梯度更新和输入特征是完美对齐的。这说明训练的梯度更新恰好沿着最能影响当前特征的方向。

因此，要让 $$\|\Delta \boldsymbol{h}_\ell\|_2 = \|\Delta \boldsymbol{W}_\ell \cdot \boldsymbol{h}_{\ell-1}\|_2 = \Theta(\sqrt{n_\ell})$$，结合 $$\|\boldsymbol{h}_{\ell-1}\|_2 = \Theta(\sqrt{n_{\ell-1}})$$，我们需要：

$$
\|\Delta \boldsymbol{W}_\ell\|_* = \Theta\!\left(\sqrt{\frac{n_\ell}{n_{\ell-1}}}\right).
$$

### 3.3 总结：Spectral Scaling Condition

综合前面两个方向的推导，我们得到了论文的核心结果：

> Spectral Scaling Condition：对于每一层 $$\ell = 1, \ldots, L$$，要求
>
> $$
> \|\boldsymbol{W}_\ell\|_* = \Theta\!\left(\sqrt{\frac{n_\ell}{n_{\ell-1}}}\right), \qquad \|\Delta \boldsymbol{W}_\ell\|_* = \Theta\!\left(\sqrt{\frac{n_\ell}{n_{\ell-1}}}\right).
> $$

这个条件的含义现在非常清晰了：权重矩阵 $$\boldsymbol{W}_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$$ 作为将 $$n_{\ell-1}$$ 维向量映射到 $$n_\ell$$ 维向量的线性算子，其"放大倍数"（谱范数）需要恰好匹配输入输出的维度比。太大则特征爆炸，太小则特征消失或学习停滞。

### 3.4 推广：从 Toy Model 到真实网络

虽然上述推导基于简化假设，但原论文 [1] 证明了 Spectral Scaling Condition 的结论在更广泛的情形下依然成立。

- 非线性激活函数：加入非线性激活 $$\boldsymbol{h}'_\ell = \phi(\boldsymbol{h}_\ell)$$ 后，只要激活函数不改变特征向量的量级（即满足 $$\|\boldsymbol{h}'_\ell\|_2 = \Theta(\|\boldsymbol{h}_\ell\|_2)$$），那么 $$\Delta \boldsymbol{W}_\ell$$ 依然是秩一矩阵，且满足完美对齐性质 $$\|\Delta \boldsymbol{W}_\ell \boldsymbol{h}'_{\ell-1}\|_2 = \|\Delta \boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}'_{\ell-1}\|_2$$。因此线性情况下的结论完全适用。

- 批量大小 > 1：当 $$B > 1$$ 时，更新量 $$\Delta \boldsymbol{W}_\ell = \frac{1}{B} \sum \Delta \boldsymbol{W}_\ell^{(i)}$$ 不再是秩一矩阵，无法与所有输入向量完美对齐。但只要 $$B$$ 与宽度 $$n$$ 无关，且更新项之间不发生恶意的完全抵消，我们依然有 Scaling 意义上的对齐：

    $$
    \|\Delta \boldsymbol{W}_\ell \boldsymbol{h}_\ell(\boldsymbol{x}_i)\|_2 = \Theta(\|\Delta \boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_\ell(\boldsymbol{x}_i)\|_2)
    $$

    这就足以保证谱缩放条件依然有效。有趣的是，论文观察到即使在大 batch size 下，更新矩阵依然保持数值上的低秩结构。
    
    {% include figure.liquid 
        path="assets/img/post-02-14/low_rank.png" 
        class="img-fluid rounded z-depth-1 mx-auto d-block" 
        width="auto" 
        zoomable=true   
        alt="更新矩阵的数值低秩结构" 
    %}

- 多步训练：梯度的演化依赖于"谱范数大小正确"和"传递特征大小正确"这两个性质。论文指出，只要更新量不与初始权重发生极端的完美抵消（$$\scriptsize\|\boldsymbol{W} + \Delta \boldsymbol{W}\|_* = \Theta(\|\boldsymbol{W}\|_* + \|\Delta \boldsymbol{W}\|_*)$$），那么一步更新后的权重将保持上述性质。归纳可知，feature learning 在后续训练步骤中依然成立。

- 自适应优化器 (Adam)：对于 Adam 等逐元素处理梯度的优化器，论文在附录中证明，当宽度较大时，逐元素非线性处理能保持矩阵的 Frobenius 范数（最多相差常数倍），且梯度仍表现出类似独立向量外积的性质，因此结论同样适用。

---

## 4. 从谱条件到 μP

要满足 Spectral Scaling Condition，最直接的方法是对权重和梯度进行谱归一化。例如，我们可以强制设定：

$$
\boldsymbol{W}_\ell = \sigma \sqrt{\frac{n_\ell}{n_{\ell-1}}} \frac{\boldsymbol{W}'_\ell}{\|\boldsymbol{W}'_\ell\|_*}, \qquad \Delta \boldsymbol{W}_\ell = -\eta \sqrt{\frac{n_\ell}{n_{\ell-1}}} \frac{\nabla_{\boldsymbol{W}_\ell} \mathcal{L}}{\|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_*}.
$$

虽然这种方法能快速验证理论，但计算大规模矩阵的谱范数（最大奇异值）代价极其高昂，在实际训练中不可行。

幸运的是，我们不需要显式计算谱范数。论文展示了可以通过分析随机矩阵的 scaling 规律，选择合适的逐层初始化方差 $$\sigma_\ell$$ 和学习率 $$\eta_\ell$$，从而自动满足 Spectral Scaling Condition。这就是 μP 的本质。

### 4.1 初始化标度

假设 $$\boldsymbol{W}_\ell = \sigma_\ell \boldsymbol{W}'_\ell$$，其中 $$\boldsymbol{W}'_\ell$$ 的元素为 iid 标准正态。由随机矩阵理论：

$$
\|\boldsymbol{W}_\ell\|_* \approx \sigma_\ell (\sqrt{n_\ell} + \sqrt{n_{\ell-1}}).
$$

要让 $$\|\boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$，需要：

$$
\sigma_\ell = \Theta\!\left(\frac{\sqrt{n_\ell / n_{\ell-1}}}{\sqrt{n_\ell} + \sqrt{n_{\ell-1}}}\right) = \Theta\!\left(\frac{1}{n_{\ell-1}}\right) \quad \text{（当隐藏层等宽 $n_\ell = n$ 时）}.
$$

### 4.2 学习率标度



如何确定学习率 $$\eta_\ell$$ 以满足 $$\|\Delta \boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$？这里的关键挑战在于确定梯度 $$\|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_*$$ 的 scaling。

我们可以通过对损失函数 $$\mathcal{L}$$ 进行一阶泰勒展开来推导。
每一次梯度更新 $$\Delta \boldsymbol{W}_\ell$$ 旨在引起输出 $$\Delta \boldsymbol{h}_L(\boldsymbol{x})$$ 的变化，进而导致损失函数发生 $$\Theta(1)$$ 量级的变化（$$\Delta \mathcal{L} = \Theta(1)$$）。

利用迹内积的性质，损失变化量可以近似为：

$$
\Delta \mathcal{L} \approx \langle \Delta \boldsymbol{W}_\ell, \nabla_{\boldsymbol{W}_\ell} \mathcal{L} \rangle = \Theta(\|\Delta \boldsymbol{W}_\ell\|_F \cdot \|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_F) = \Theta(\|\Delta \boldsymbol{W}_\ell\|_* \cdot \|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_*).
$$

这里利用了我们在低秩更新下的观察：由于矩阵近似秩一（或低秩），其 Frobenius 范数与谱范数同阶。

代入我们期望的 $$\Delta \mathcal{L} = \Theta(1)$$ 和谱缩放条件 $$\|\Delta \boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$，我们可以直接解出梯度的 scaling：

$$
\|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_* = \Theta(\sqrt{n_{\ell-1} / n_\ell}).
$$

既然 $$\Delta \boldsymbol{W}_\ell = -\eta_\ell \nabla_{\boldsymbol{W}_\ell} \mathcal{L}$$，要满足 $$\|\Delta \boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$，学习率必须设定为：

$$
\eta_\ell = \frac{\|\Delta \boldsymbol{W}_\ell\|_*}{\|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_*} = \Theta\left(\frac{n_\ell}{n_{\ell-1}}\right).
$$

这给出了 μP 学习率 scaling 的直观解释：对于标准的 $$n_\ell = n$$ 隐藏层，学习率应为 $$\Theta(1)$$；而对于输出层（假设 $$n_L=1$$），学习率应为 $$\Theta(1/n)$$。

### 4.3 Spectral Parametrization

将初始化和学习率的推导结果结合起来，论文总结出了 Spectral Parametrization，也是该论文的主要贡献之一。

> 如果每一层 $\ell$ 的初始化缩放和学习率按照以下方式选择，则谱缩放条件成立且实现特征学习：
> $$
> \sigma_\ell = \Theta \left( \frac{1}{\sqrt{n_{\ell-1}}} \min \left\{ 1, \sqrt{\frac{n_\ell}{n_{\ell-1}}} \right\} \right), \qquad \eta_\ell = \Theta \left( \frac{n_\ell}{n_{\ell-1}} \right).
> $$

这个统一的公式涵盖了所有层的情况：
- 对于隐藏层（通常 $n_\ell \approx n_{\ell-1}$），$\sigma_\ell = \Theta(1/n_{\ell-1})$，$\eta_\ell = \Theta(1)$。
- 对于输出层（$n_\ell \ll n_{\ell-1}$，例如 $n_L=1$），$\sigma_\ell = \Theta(1/n_{\ell-1})$，$\eta_\ell = \Theta(1/n_{\ell-1})$。

这与 μP 表（Yang et al., 2021 的 Table 3）完全一致。换句话说，谱 scaling 条件提供了 μP 的一种等价但更直观的推导方式。

---

## 5. 与其他参数化方案的对比

### 5.1 Standard Parametrization（SP）

主流的 Kaiming/Xavier/LeCun 初始化使用 $$\sigma_\ell = \Theta(1/\sqrt{n_{\ell-1}})$$，配合与宽度无关的学习率。

这意味着：

$$
\|\boldsymbol{W}_\ell\|_* \approx \frac{1}{\sqrt{n_{\ell-1}}} (\sqrt{n_\ell} + \sqrt{n_{\ell-1}}) = \Theta\!\left(1 + \sqrt{\frac{n_\ell}{n_{\ell-1}}}\right).
$$

当 $$n_\ell \gg n_{\ell-1}$$ 时这远大于 $$\sqrt{n_\ell / n_{\ell-1}}$$，但更关键的是，SP 在输出层（fan-out $$\ll$$ fan-in）的谱范数偏大，导致宽度变大时输出可能发散。

而 SP 使用固定学习率（与宽度无关），对于宽的隐藏层来说，学习率实际上偏小——更新的谱范数随宽度衰减，feature learning 不足。

### 5.2 Neural Tangent Parametrization（NTP）

NTP 将权重参数化为 $$\boldsymbol{W}_\ell / \sqrt{n_{\ell-1}}$$，使用与宽度无关的学习率。可以验证它等价于 $$\sigma_\ell = \Theta(1/\sqrt{n_{\ell-1}})$$，$$\eta_\ell = \Theta(1/n_{\ell-1})$$。

输出层的 $$\sigma_L$$ 比 μP 大了 $$\sqrt{n_{L-1}}$$ 倍，这通过反向传播放大了所有中间层的梯度；然而，$$\eta_\ell = 1/n_{\ell-1}$$ 的过小学习率又将这个放大效应压回去了。最终结果是：

$$
\|\Delta \boldsymbol{W}_\ell\|_* \propto \frac{\sqrt{n_{L-1}}}{n_{\ell-1}} \to 0 \quad (n \to \infty).
$$

权重更新的谱范数随宽度衰减至零——这正是 lazy learning / kernel regime 的特征：特征冻结，网络行为退化为 NTK。

### 5.3 小结

| 方案 | 初始化 $$\sigma_\ell$$（隐藏层） | 学习率 $$\eta_\ell$$（隐藏层） | Feature Learning? |
| --- | --- | --- | --- |
| SP | $$1/\sqrt{n}$$ | $$\Theta(1)$$ | ✗ |
| NTP | $$1/\sqrt{n}$$ | $$1/n$$ | ✗ |
| μP / Spectral | $$1/n$$ | $$\Theta(1)$$ | ✓ |
{: .table .table-striped}

{% include figure.liquid 
    path="assets/img/post-02-14/experiments.png" 
    class="img-fluid rounded z-depth-1 mx-auto d-block" 
    width="auto" 
    zoomable=true   
    alt="不同参数化方案下的训练表现" 
%}
---

## 6. 实验验证

为了验证上述理论推导，论文在不同宽度的 MLP 上进行了实验。下图展示了在 NTP 和 μP 缩放下，网络内部特征和权重变化的 scaling 行为。

{% include figure.liquid 
    path="assets/img/post-02-14/experiments.png" 
    class="img-fluid rounded z-depth-1 mx-auto d-block" 
    width="auto" 
    zoomable=true   
    alt="不同参数化方案下的训练表现" 
%}

可以看到：
1. 特征变化量（左上 A 图）：在 μP 缩放下，特征变化量 $$\frac{\|\boldsymbol{h}_2(\boldsymbol{x}) - \boldsymbol{h}_2^0(\boldsymbol{x})\|_2}{\|\boldsymbol{h}_2^0(\boldsymbol{x})\|_2}$$ 保持常数级别 $$\Theta(1)$$，与宽度无关；而在 NTP 缩放下，特征变化量随宽度增加呈 $$n^{-1/2}$$ 衰减。这意味着在 NTP 缩放下，随着模型变宽，特征学习会逐渐消失，最终退化为 Lazy Regime。
2. 权重变化量（右上 B图）：在 μP 缩放下，权重的谱范数变化 $$\frac{\|\boldsymbol{W}_2 - \boldsymbol{W}_2^0\|_*}{\|\boldsymbol{W}_2^0\|_*}$$ 也不随宽度衰减（保持 $$\Theta(1)$$），而 NTP 缩放下则显著衰减。

这证实了只有 μP 能够在大宽度的极限下保持非平凡的特征学习。

---

## 7. 如何理解唯一的最大缩放

Spectral Scaling Condition 给出的是唯一的最大缩放（maximal scaling）。

具体而言，如果任何 $$\|\boldsymbol{W}_\ell\|_*$$ 或 $$\|\Delta \boldsymbol{W}_\ell\|_*$$ 超过了 $$\Theta(\sqrt{n_\ell / n_{\ell-1}})$$，那么随着宽度增大，训练会发散。反过来，过小的 scaling 会导致 feature learning 不充分，甚至落入 lazy learning regime。μP（谱条件）正是让每一层的 feature learning 都尽可能充分的唯一解。

这也是"maximal update parametrization"中"maximal"一词的来源。

---

## 8. 与 Tensor Programs 系列的关系

这篇论文展示了一种绕过 Tensor Programs 形式化体系的、用基本线性代数推导 μP 的方法。但需要注意的是，Tensor Programs 的真正威力在于它的普适性：它可以处理任意架构（不只是 MLP）、任意优化器（不只是 SGD）、以及训练的全过程（不只是一步）。

| 问题 | 本文方法 | Tensor Programs |
| --- | --- | --- |
| 适用架构 | MLP（可推广） | 任意 TP 可表达的架构 |
| 适用优化器 | SGD（可推广到 Adam） | 任意自适应优化器 |
| 训练步数 | 一步 → 多步 | 无穷步（极限定理） |
| 推导难度 | 基本线性代数 | 需要 Master Theorem |
{: .table .table-striped}

在后续的系列文章中，我们将逐步深入 Tensor Programs 的形式化框架，理解 Master Theorem 如何给出任意神经网络计算的无穷宽极限。

---

## 参考文献

[1] Bernstein, J., Newhouse, L., Lee, J., Yang, G. (2024). A Spectral Condition for Feature Learning. *arXiv preprint arXiv:2310.17813*.

[2] Yang, G. & Hu, E. J. (2021). Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks. *ICML 2021*. arXiv:2011.14522.

[3] Yang, G., Hu, E. J., Babuschkin, I., et al. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. *arXiv preprint arXiv:2203.03466*.

[4] Yang, G., Schnabel, T., Li, Z., & Du, S. S. (2023). Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks. *arXiv preprint arXiv:2310.02244*.

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026spectral,
  title={Tensor Programs (一)：Feature Learning 的谱条件},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/spectral-condition-feature-learning/}
}
```
