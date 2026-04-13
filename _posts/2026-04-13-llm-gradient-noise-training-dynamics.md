---
layout: post
title: "在 LLM 语境下，梯度里的噪声会如何影响 training dynamics？"
date: 2026-04-14 00:00:00
description: "从理论上分析现代大规模预训练后期的噪声主导现象，推导基于广义范数的行归一化优化器在此区域内如何本质地改变梯度下降的稳态误差与动力学。"
tags: [optimization, deep-learning, llm, scaling-law]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

近期社区中涌现出多项基于行归一化的优化器工作，在某些设定下展现出优于此前广泛使用的正交化优化器的性能。从传统的优化视角来看，行归一化或参数的符号更新似乎是平滑优化中一种略显粗糙的近似方法。然而，在现代大规模模型预训练的后期阶段，真实的梯度信号几乎被随机采样引入的巨大噪声所淹没，训练系统进入了严重的噪声主导区域。本文旨在通过严密的数学推导，证明在存在行间异方差的极度高噪环境中，这类基于共轭范数的归一化方法并非通过提升梯度方向的精度产生收益，而是通过改变稳态误差的阶数以及对噪声进行隐式的逆方差加权，从而改变了优化过程的动力学性质。

## 1. 最小模型与行归一化的几何本质

我们首先定义一个用于分析的最小随机优化模型。考察一个矩阵参数块 $$W \in \mathbb{R}^{m \times d}$$。设第 $$r$$ 行的随机梯度为：

$$
g_r = s_r + \xi_r
$$

其中真实梯度定义为 $$s_r := \nabla_r F(W)$$，且假设噪声条件期望为 $$\mathbb{E}[\xi_r \mid W] = 0$$。

目前常见的按块或按行归一化方法，以及涉及梯度的分数幂更新方法，其一般形式可以写为：

$$
u_t = \frac{\text{sign}(g_t) \odot \lvert g_t \rvert^p}{\lVert g_t \rVert_{p+1}^p}
$$

我们需要首先证明，上述公式严格对应于共轭范数空间下的最速下降法。设定优化目标为在给定步长约束下，使得函数沿梯度方向的局部线性近似下降量最大。即求解以下优化问题：

$$
v^* = \arg\min_{\lVert v \rVert_q \le 1} \langle g_t, v \rangle
$$

根据 Hölder 不等式，对于任意满足 $$\frac{1}{p+1} + \frac{1}{q} = 1$$ 的共轭指数，存在：

$$
\lvert\langle g_t, v \rangle\rvert \le \lVert g_t \rVert_{p+1} \lVert v \rVert_q
$$

由于目标是使得内积达到负的最大值，不等式必须取等号。Hölder 不等式取等号的条件是存在常数 $$c > 0$$ 使得：

$$
\lvert v_i \rvert^q = c \lvert g_{t,i} \rvert^{p+1}
$$

由共轭条件 $$\frac{1}{p+1} + \frac{1}{q} = 1$$可得$$q = \frac{p+1}{p}$$。代入上式解出分量的绝对值：

$$
\lvert v_i \rvert = c^{\frac{1}{q}} \lvert g_{t,i} \rvert^{\frac{p+1}{q}} = c^{\frac{p}{p+1}} \lvert g_{t,i} \rvert^p
$$

将此关系代入范数约束 $$\lVert v \rVert_q = 1$$ 求解常数项：

$$
\sum_i \lvert v_i \rvert^q = c \sum_i \lvert g_{t,i} \rvert^{pq} = c \sum_i \lvert g_{t,i} \rvert^{p+1} = c \lVert g_t \rVert_{p+1}^{p+1} = 1
$$

解得 $$c = \lVert g_t \rVert_{p+1}^{-(p+1)}$$。将其代回 $$\lvert v_i \rvert$$ 的表达式：

$$
\lvert v_i \rvert = \left( \lVert g_t \rVert_{p+1}^{-(p+1)} \right)^{\frac{p}{p+1}} \lvert g_{t,i} \rvert^p = \frac{\lvert g_{t,i} \rvert^p}{\lVert g_t \rVert_{p+1}^p}
$$

结合内积取负值所需的符号条件 $$\text{sign}(v_i) = -\text{sign}(g_{t,i})$$，剔除负号后即严格导出了前述的更新公式。当 $$p=1$$时，该公式退化为严格的按行$$L_2$$范数归一化；当$$p=0$$ 时，该公式退化为按元素的符号更新。

## 2. 纯行范数归一化并不改善行内信噪比

一个直观但错误的假设是，行归一化修正了参数在高维空间中的更新方向。对于严格的行 $$L_2$$归一化，更新量为$$u_r = \frac{g_r}{\lVert g_r \rVert_2}$$。其方向与原始随机梯度 $$g_r$$完全一致。因此，计算其与真实梯度$$s_r$$ 的方向余弦：

$$
\cos(u_r, s_r) = \frac{\langle u_r, s_r \rangle}{\lVert u_r \rVert_2 \lVert s_r \rVert_2} = \frac{\langle g_r, s_r \rangle}{\lVert g_r \rVert_2 \lVert s_r \rVert_2} = \cos(g_r, s_r)
$$

这意味着，归一化操作本身在瞬时步上并未提升真实梯度的方向余弦。其在现代优化语境下的所有数学收益，必定来源于对不同行间相对步长的重新分配，以及在极高噪声区域丢弃行范数这一特定通道。

## 3. 低噪声与高噪声区域的动力学分叉

为了理解为何该类算法仅在训练后期展现出优势，需要分别对低噪声与高噪声两种状态进行扰动展开。

### 3.1 信号主导区域

假设系统处于 $$\lVert s_r \rVert \gg \lVert \xi_r \rVert$$的状态。令$$\hat{s} = \frac{s}{\lVert s \rVert}$$，并将噪声正交分解为 $$\xi = \alpha \hat{s} + \xi_{\perp}$$，其中 $$\hat{s}^T \xi_{\perp} = 0$$。更新量可以写为：

$$
u = \frac{s + \xi}{\lVert s + \xi \rVert} = \frac{(\lVert s \rVert + \alpha)\hat{s} + \xi_{\perp}}{\sqrt{(\lVert s \rVert + \alpha)^2 + \lVert \xi_{\perp} \rVert^2}}
$$

提取分母的主导项：

$$
u = \left( \hat{s} + \frac{\xi_{\perp}}{\lVert s \rVert + \alpha} \right) \left( 1 + \frac{\lVert \xi_{\perp} \rVert^2}{(\lVert s \rVert + \alpha)^2} \right)^{-1/2}
$$

利用二项式级数展开 $$\frac{1}{\lVert s \rVert + \alpha} = \frac{1}{\lVert s \rVert} - \frac{\alpha}{\lVert s \rVert^2} + \mathcal{O}\left(\frac{\alpha^2}{\lVert s \rVert^3}\right)$$以及$$(1+z)^{-1/2} = 1 - \frac{1}{2}z + \mathcal{O}(z^2)$$，可以得到一阶近似：

$$
u = \hat{s} + \frac{\xi_{\perp}}{\lVert s \rVert} - \frac{\alpha \xi_{\perp}}{\lVert s \rVert^2} - \frac{\lVert \xi_{\perp} \rVert^2}{2\lVert s \rVert^2}\hat{s} + \mathcal{O}\left(\frac{\lVert \xi \rVert^3}{\lVert s \rVert^3}\right)
$$

对噪声取条件期望，由于 $$\mathbb{E}[\alpha \xi_{\perp}] = 0$$，一阶项中与真实梯度平行的幅值噪声 $$\alpha \hat{s}$$ 被完全消除：

$$
\mathbb{E}[u \mid W] = \hat{s} - \frac{\mathbb{E}\lVert \xi_{\perp} \rVert^2}{2\lVert s \rVert^2}\hat{s} + \mathcal{O}\left(\frac{\mathbb{E}\lVert \xi \rVert^3}{\lVert s \rVert^3}\right)
$$

在信号主导区域，归一化主要保留了方向而舍弃了真实幅值信息，这在需要精细收敛的传统优化设定下往往会导致收敛速度受损。

### 3.2 噪声主导区域

假设系统处于 $$\lVert s_r \rVert \ll \sigma_r \sqrt{d}$$的状态。定义映射$$\phi(x) = \frac{x}{\lVert x \rVert_2}$$，其对应的雅可比矩阵为 $$J_{\phi}(x) = \frac{1}{\lVert x \rVert_2} \left( I - \frac{x x^T}{\lVert x \rVert_2^2} \right)$$。
对 $$u = \phi(s + \xi)$$在$$\xi$$ 处进行泰勒展开：

$$
u = \phi(\xi) + J_{\phi}(\xi)s + \mathcal{O}\left(\frac{\lVert s \rVert^2}{\lVert \xi \rVert^2}\right)
$$

如果噪声分布关于零点对称，则 $$\mathbb{E}[\phi(\xi) \mid W] = 0$$。假设噪声为各向同性高斯分布 $$\xi \sim \mathcal{N}(0, \sigma^2 I_d)$$，由旋转对称性可求得雅可比矩阵的期望：

$$
\mathbb{E}[J_{\phi}(\xi)] = \frac{\kappa_d}{\sigma} I_d
$$

其中常量 $$\kappa_d$$ 的显式解为：

$$
\kappa_d = \frac{d-1}{d} \mathbb{E}\left[\frac{1}{\lVert z \rVert_2}\right] = \frac{d-1}{d\sqrt{2}} \frac{\Gamma(\frac{d-1}{2})}{\Gamma(\frac{d}{2})} \sim d^{-1/2}
$$

因此，更新方向的期望漂移变为：

$$
\mathbb{E}[u_r \mid W] \approx \frac{\kappa_d}{\sigma_r} s_r
$$

这一推导表明，在高噪声区域，行归一化的有效漂移不再与真实梯度的大小成正比，而是自动依据 $$1/\sigma_r$$ 对高方差的行进行了幅度下调。

## 4. 稳态误差半径的降阶性质

为了进一步阐明动力学性质的变化，考察局部二次模型 $$f(x) = \frac{h}{2} (x - x^*)^2$$及其在最优值附近的稳态方差。设误差$$e_t = x_t - x^*$$，观测梯度 $$g_t = h e_t + \xi_t$$，其中 $$\xi_t \sim \mathcal{N}(0, \sigma^2)$$。

对于非归一化的传统更新，误差演化为：

$$
e_{t+1} = (1 - \eta h)e_t - \eta \xi_t
$$

其平稳分布的方差满足：

$$
\text{Var}(e_{\infty}) = \frac{\eta^2 \sigma^2}{1 - (1 - \eta h)^2} \approx \frac{\eta \sigma^2}{2h}
$$

即稳态方差对噪声标准差的依赖为 $$\mathcal{O}(\eta \sigma^2 / h)$$。

对于无量纲的归一化更新（在一维情况退化为符号更新），误差演化为：

$$
e_{t+1} = e_t - \eta \text{sign}(h e_t + \xi_t)
$$

其期望漂移项为：

$$
\mathbb{E}[\text{sign}(h e_t + \xi_t) \mid e_t] = 2\Phi\left(\frac{h e_t}{\sigma}\right) - 1 \approx \sqrt{\frac{2}{\pi}} \frac{h e_t}{\sigma}
$$

演化方程可以近似改写为：

$$
e_{t+1} \approx \left( 1 - \eta h \sqrt{\frac{2}{\pi}} \frac{1}{\sigma} \right) e_t + \eta \zeta_t
$$

其中 $$\text{Var}(\zeta_t) \approx 1$$。求解该过程的平稳方差：

$$
\text{Var}(e_{\infty}) \approx \frac{\eta^2}{2 \eta h \sqrt{\frac{2}{\pi}} \frac{1}{\sigma}} = \frac{\eta \sigma}{2h} \sqrt{\frac{\pi}{2}}
$$

稳态方差对噪声标准差的依赖下降为 $$\mathcal{O}(\eta \sigma / h)$$。这种从二次到一次的依赖关系改变，是系统能够在后期免于发散且维持较大恒定学习率的核心理论支撑。

## 5. 行间异方差条件下的期望下降量上界

在矩阵具有多行且存在严重激活异常值的现代网络架构中，不同行的梯度噪声往往存在显著的异方差特性。设定损失函数满足平滑系数为 $$L$$ 的利普希茨条件。

对于非归一化更新，利用期望下降量不等式：

$$
\mathbb{E}[\Delta F] \le -\eta \lVert S \rVert_F^2 + \frac{L \eta^2}{2} \left( \lVert S \rVert_F^2 + d \sum_{r=1}^m \sigma_r^2 \right)
$$

其最优步长受限于全局最大的噪声方差之和。若各行真实信号强度均值相近为 $$a$$，其极值处的有效进展近似为：

$$
\Delta_{sgd}^* \approx -\frac{m^2 a^2}{2 L d \sum_{r=1}^m \sigma_r^2}
$$

对于行归一化更新，整个矩阵的更新范数被强行有界化 $$\lVert U \rVert_F^2 = m$$。结合第三节的推导，期望下降量为：

$$
\mathbb{E}[\Delta F] \le -\eta \kappa_d \sum_{r=1}^m \frac{\lVert s_r \rVert_2^2}{\sigma_r} + \frac{L \eta^2}{2} m
$$

其极值处的有效进展近似为：

$$
\Delta_{row}^* \approx -\frac{\kappa_d^2 a^2}{2 L m} \left( \sum_{r=1}^m \frac{1}{\sigma_r} \right)^2
$$

计算两者有效进展的比值：

$$
\frac{\lvert \Delta_{row}^* \rvert}{\lvert \Delta_{sgd}^* \rvert} \approx d \kappa_d^2 \frac{ \left( \sum_r \sigma_r^{-1} \right)^2 \left( \sum_r \sigma_r^2 \right) }{m^3}
$$

根据幂平均不等式，在 $$\sigma_r$$表现出剧烈分布差异时，该比值将远大于$$1$$。这从数学上证明了，行归一化通过阻断不同行之间的二次罚项交叉干扰，能够利用异方差特性产生严格的理论优势。

## 参考文献

[1] Shazeer, N., & Stern, M. (2018). Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. Proceedings of the 35th International Conference on Machine Learning, PMLR 80:4596-4604.

[2] Bernstein, J., et al. (2025). Muon: A Momentum-Based Orthogonalized Optimizer. arXiv preprint arXiv:2510.05491.

[3] Zhao, Y., et al. (2026). RMNP: Row-wise Matrix Normalization Preconditioner for Large Language Models. arXiv preprint arXiv:2603.20527.

[4] Li, H., et al. (2026). MANO: Manifold Optimization for Neural Network Training. arXiv preprint.

[5] Wang, S., et al. (2026). GradPower: Batch Size and Noise-Dependent Power Normalization. arXiv preprint arXiv:2505.24275.

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026noise-training-dynamics,
  title={在 LLM 语境下，梯度里的噪声会如何影响 training dynamics？},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={[https://jiaxuanzou0714.github.io/blog/2026/noise-training-dynamics/](https://jiaxuanzou0714.github.io/blog/2026/noise-training-dynamics/)}
}
```