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

近期社区中涌现出多项基于行归一化或参数分块归一化的优化器工作，在某些设定下展现出优于此前广泛使用的正交化优化器的性能。从传统的优化理论视角来看，行归一化或参数的符号更新似乎是平滑优化中一种略显粗糙的近似方法。在信号主导的传统优化分析中，算法设计的核心往往在于更精确地估计梯度的真实方向与曲率。然而，在现代大规模语言模型预训练的后期阶段，真实的梯度信号几乎被随机采样引入的巨大噪声所淹没，训练系统进入了严重的噪声主导区域。

因此，针对此类场景的优化器设计与训练动力学研究，必须脱离传统的信号导向框架。本文旨在通过严密的数学推导证明：在存在行间异方差的极度高噪环境中，各类基于共轭范数的归一化方法并非通过提升梯度方向的精度产生收益，而是通过零次齐次映射的雅可比矩阵缩放性质，本质地改变了优化过程的稳态误差阶数，并对不同参数块的噪声进行了隐式的逆方差加权，从而彻底改变了高噪声区域的动力学性质。

## 1. 最小模型与共轭范数下的最速下降

我们首先定义一个用于分析的最小随机优化模型。考察被划分为 $m$ 个参数块的变量，记第 $r$ 个参数块为 $x_r \in \mathbb{R}^{d_r}$。设第 $r$ 块的随机梯度为：

$$
g_r = s_r + \xi_r = s_r + \sigma_r z_r
$$

其中真实梯度定义为 $s_r := \nabla_r F(x)$，噪声项满足条件期望 $\mathbb{E}[z_r \mid x] = 0$ 且方差尺度由 $\sigma_r$ 决定。定义每个参数块的局部信噪比为 $\rho_r := \frac{\lVert s_r \rVert_2}{\sigma_r \sqrt{d_r}}$。本文关注的训练后期场景严格满足 $\rho_r \ll 1$。

目前常见的按块或按行归一化方法，以及涉及梯度的分数幂更新方法，其一般形式可以写为：

$$
u_r = \frac{\operatorname{sign}(g_r) \odot \lvert g_r \rvert^p}{\lVert g_r \rVert_{p+1}^p}
$$

该公式严格对应于共轭范数空间下的最速下降法。设定优化目标为在给定范数步长约束下，使得目标函数沿更新方向的局部线性近似下降量最大。即求解以下优化问题：

$$
v^* = \arg\min_{\lVert v \rVert_q \le 1} \langle g_r, v \rangle
$$

根据 Hölder 不等式，对于任意满足 $\frac{1}{p+1} + \frac{1}{q} = 1$ 的共轭指数，存在：

$$
\lvert\langle g_r, v \rangle\rvert \le \lVert g_r \rVert_{p+1} \lVert v \rVert_q
$$

由于目标是使得内积达到负的最大值，不等式必须取等号。Hölder 不等式取等号的条件是存在常数 $c > 0$ 使得：

$$
\lvert v_{r,i} \rvert^q = c \lvert g_{r,i} \rvert^{p+1}
$$

由共轭条件可得 $q = \frac{p+1}{p}$。代入上式解出分量的绝对值：

$$
\lvert v_{r,i} \rvert = c^{\frac{1}{q}} \lvert g_{r,i} \rvert^{\frac{p+1}{q}} = c^{\frac{p}{p+1}} \lvert g_{r,i} \rvert^p
$$

将此关系代入范数约束 $\lVert v \rVert_q = 1$ 求解常数项 $c$：

$$
\sum_i \lvert v_{r,i} \rvert^q = c \sum_i \lvert g_{r,i} \rvert^{pq} = c \sum_i \lvert g_{r,i} \rvert^{p+1} = c \lVert g_r \rVert_{p+1}^{p+1} = 1
$$

解得 $c = \lVert g_r \rVert_{p+1}^{-(p+1)}$。将其代回 $\lvert v_{r,i} \rvert$ 的表达式：

$$
\lvert v_{r,i} \rvert = \left( \lVert g_r \rVert_{p+1}^{-(p+1)} \right)^{\frac{p}{p+1}} \lvert g_{r,i} \rvert^p = \frac{\lvert g_{r,i} \rvert^p}{\lVert g_r \rVert_{p+1}^p}
$$

结合内积取负值所需的符号条件 $\operatorname{sign}(v_{r,i}) = -\operatorname{sign}(g_{r,i})$，剔除负号后即严格导出了前述的更新公式。当 $p=1$ 时，该公式退化为严格的按块 $L_2$ 范数归一化；当 $p=0$ 时，该公式退化为按元素的符号更新。

## 2. 归一化操作并不改善局部方向精度

一个直观的推断是归一化修正了参数在高维空间中的更新方向。对于严格的行 $L_2$ 归一化，更新量为 $u_r = \frac{g_r}{\lVert g_r \rVert_2}$。计算其与真实梯度 $s_r$ 的方向余弦：

$$
\cos(u_r, s_r) = \frac{\langle u_r, s_r \rangle}{\lVert u_r \rVert_2 \lVert s_r \rVert_2} = \frac{\langle g_r, s_r \rangle}{\lVert g_r \rVert_2 \lVert s_r \rVert_2} = \cos(g_r, s_r)
$$

其方向与原始随机梯度 $g_r$ 产生的夹角完全一致。归一化操作本身在瞬时步上并未提升真实梯度的方向余弦。其在现代优化语境下的理论收益，必定来源于空间尺度的重新分配。

## 3. 零次齐次映射在高噪声区域的渐近展开

为了剖析更新公式在 $\rho_r \ll 1$ 时的行为，我们将上述各类归一化方法抽象为映射 $u_r = \phi_r(g_r)$。此类映射均满足两个核心数学性质：
1. 奇对称性：$\phi_r(-g) = -\phi_r(g)$。
2. 零次齐次性：对于任意标量 $c > 0$，有 $\phi_r(cg) = \phi_r(g)$。

### 3.1 雅可比矩阵的缩放律

零次齐次性必然导致其一阶响应随输入尺度的增大而严格衰减。对于任意非零向量 $g$ 与微小扰动方向 $h$：

$$
\phi_r(cg+\varepsilon h) = \phi_r\left(c\left(g+\frac{\varepsilon}{c}h\right)\right) = \phi_r\left(g+\frac{\varepsilon}{c}h\right)
$$

对 $\varepsilon$ 在 $0$ 处求方向导数，根据链式法则可得：

$$
J_{\phi_r}(cg)h = \frac{1}{c} J_{\phi_r}(g)h
$$

由于该等式对任意方向 $h$ 成立，故雅可比矩阵满足：

$$
J_{\phi_r}(cg) = \frac{1}{c} J_{\phi_r}(g)
$$

同理计算二阶导数可得 $\nabla^2\phi_r(cg) = \frac{1}{c^2}\nabla^2\phi_r(g)$。这表明在该映射下，信号产生的一阶响应将按 $1/\sigma$ 的比例被缩小，而二阶非线性误差将按 $1/\sigma^2$ 的比例被缩小。

### 3.2 极度高噪下的期望漂移与协方差

在 $s_r$ 趋于无穷小而 $\sigma_r z_r$ 主导时，对 $\phi_r(s_r + \sigma_r z_r)$ 在高方差噪声点 $\sigma_r z_r$ 处进行泰勒展开：

$$
u_r = \phi_r(\sigma_r z_r) + J_{\phi_r}(\sigma_r z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

应用零次齐次性以及雅可比矩阵的缩放律：

$$
u_r = \phi_r(z_r) + \frac{1}{\sigma_r} J_{\phi_r}(z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

对噪声分布取条件期望。由于 $z_r$ 分布关于原点对称，且 $\phi_r$ 为奇函数，故 $\mathbb{E}[\phi_r(z_r)] = 0$。因此期望漂移项为：

$$
\mathbb{E}[u_r \mid x] = \frac{1}{\sigma_r} A_r s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

其中 $A_r := \mathbb{E}[J_{\phi_r}(z_r)]$ 仅取决于基础噪声分布与归一化方式。

进一步考察协方差。在零阶近似下：

$$
\operatorname{Cov}(u_r \mid x) = \operatorname{Cov}(\phi_r(z_r)) + \mathcal{O}\left(\frac{\lVert s_r \rVert}{\sigma_r}\right)
$$

记 $B_r := \operatorname{Cov}(\phi_r(z_r))$。高噪声区归一化更新的动力学形式被严格重写为：

$$
u_r \approx \frac{1}{\sigma_r} A_r s_r + \zeta_r, \quad \mathbb{E}[\zeta_r \mid x] = 0, \quad \operatorname{Cov}(\zeta_r \mid x) \approx B_r
$$

由此可见，归一化操作将原本的随机梯度 $s_r + \sigma_r z_r$ 转换为一个被 $1/\sigma_r$ 缩小的有效漂移，并附加了一个幅值严格有界、不再随 $\sigma_r$ 线性放大的白噪声 $\zeta_r$。

### 3.3 特化到行 $L_2$ 归一化

考察 $L_2$ 归一化 $\phi_r(g) = \frac{g}{\lVert g \rVert_2}$。其雅可比矩阵为 $J_{\phi_r}(g) = \frac{1}{\lVert g \rVert_2} \left( I - \frac{g g^\top}{\lVert g \rVert_2^2} \right)$。

假设参数块内噪声近似各向同性高斯分布 $z_r \sim \mathcal{N}(0, I_{d_r})$。由旋转对称性可得 $A_r = a_{d_r} I_{d_r}$，其中：

$$
a_{d_r} = \frac{d_r-1}{d_r} \mathbb{E}\left[\frac{1}{\lVert z_r \rVert_2}\right] = \frac{d_r-1}{d_r\sqrt{2}} \frac{\Gamma\left(\frac{d_r-1}{2}\right)}{\Gamma\left(\frac{d_r}{2}\right)} \sim d_r^{-1/2}
$$

更新方向的期望漂移变为：

$$
\mathbb{E}[u_r \mid x] \approx \frac{a_{d_r}}{\sigma_r} s_r
$$

由于 $u_r$ 在单位球面上近似均匀分布，其协方差满足 $\mathbb{E}[u_r u_r^\top] \approx \frac{1}{d_r}I_{d_r}$，且有严格的二阶矩约束 $\mathbb{E}\lVert u_r \rVert_2^2 = 1$。

## 4. 全局与局部动力学稳定性的重构

归一化引入的 $1/\sigma$ 缩放律彻底改变了算法在全局与局部两个层面的稳定性条件。

### 4.1 全局单步损失下降的稳定性

假设目标函数 $F$ 满足常数为 $L$ 的利普希茨平滑条件。由下降引理，参数块更新 $x_r^+ = x_r - \eta u_r$ 满足：

$$
F(x^+) \le F(x) - \eta \sum_{r=1}^m \langle s_r, u_r \rangle + \frac{L}{2} \eta^2 \sum_{r=1}^m \lVert u_r \rVert_2^2
$$

对于非归一化的随机梯度下降 (SGD)，使用 $u_r = s_r + \sigma_r z_r$，取期望后得到：

$$
\mathbb{E}[F(x^+) \mid x] \le F(x) - \eta \sum_{r=1}^m \lVert s_r \rVert_2^2 + \frac{L}{2} \eta^2 \left( \sum_{r=1}^m \lVert s_r \rVert_2^2 + \sum_{r=1}^m d_r \sigma_r^2 \right)
$$

在噪声主导时，保证单步期望下降的最大允许步长严格受限于系统方差：

$$
\eta_{\text{SGD}} \lesssim \frac{2 \sum_{r} \lVert s_r \rVert_2^2}{L \sum_{r} d_r \sigma_r^2}
$$

对于块 $L_2$ 归一化方法，由于存在硬性约束 $\lVert u_r \rVert_2^2 = 1$，二次项总和变为确定性常数 $m$。代入高噪声区域的期望漂移展开：

$$
\mathbb{E}[F(x^+) \mid x] \le F(x) - \eta \sum_{r=1}^m \frac{a_{d_r}}{\sigma_r} \lVert s_r \rVert_2^2 + \frac{L}{2} \eta^2 m + \mathcal{O}\left(\eta \sum_{r=1}^m \frac{\lVert s_r \rVert_2^3}{\sigma_r^2}\right)
$$

忽略高阶小量，保持期望下降所需的步长上界变为：

$$
\eta_{\text{norm}} \lesssim \frac{2 \sum_{r} a_{d_r} \lVert s_r \rVert_2^2 / \sigma_r}{L m}
$$

归一化操作使得目标函数平滑界中的二次项不再随噪声幅值共同增长。算法最大稳定步长对噪声的依赖关系从 $\mathcal{O}(1/\sigma^2)$ 严格降阶为 $\mathcal{O}(1/\sigma)$，这是归一化方法在训练后期具有更强鲁棒性的核心原因。

### 4.2 局部均方收敛的稳态误差与收缩率

考察最优点附近的局部二次模型 $F(x) = \frac{1}{2} \sum_{r=1}^m \lambda_r \lVert x_r \rVert_2^2$，其中 $s_r = \lambda_r x_r$。

对于归一化更新，将其高噪近似形式代入参数演化方程：

$$
x_{r, t+1} = \left( 1 - \eta \frac{a_{d_r} \lambda_r}{\sigma_r} \right) x_{r, t} - \eta \zeta_{r, t}
$$

对均方误差求期望，利用 $\mathbb{E}[\zeta_{r,t}] = 0$ 消去交叉项，并忽略 $\mathcal{O}(\eta^2)$ 的收缩高阶项，得到线性递推：

$$
\mathbb{E}\lVert x_{r, t+1} \rVert_2^2 \approx \left( 1 - 2\eta \frac{a_{d_r} \lambda_r}{\sigma_r} \right) \mathbb{E}\lVert x_{r, t} \rVert_2^2 + \eta^2
$$

求解该过程的稳态方差界限：

$$
\mathbb{E}\lVert x_{r, \infty} \rVert_2^2 \approx \frac{\eta^2}{2\eta \frac{a_{d_r} \lambda_r}{\sigma_r}} = \frac{\eta \sigma_r}{2 a_{d_r} \lambda_r} \sim \mathcal{O}\left(\frac{\eta \sigma_r \sqrt{d_r}}{\lambda_r}\right)
$$

作为对照，SGD 的均方误差演化方程为 $x_{r, t+1} = (1 - \eta \lambda_r) x_{r, t} - \eta \xi_{r, t}$，其稳态方差为：

$$
\mathbb{E}\lVert x_{r, \infty} \rVert_2^2 \approx \frac{\eta d_r \sigma_r^2}{2\lambda_r} \sim \mathcal{O}\left(\frac{\eta d_r \sigma_r^2}{\lambda_r}\right)
$$

对比如上两组公式，归一化方法展现出截然不同的局部动力学权衡：
一方面，其局部线性收缩率从 $\eta \lambda_r$ 衰减为 $\eta a_{d_r} \lambda_r / \sigma_r$。在相同步长下，噪声越大，参数向最优点的拉回漂移越弱。
另一方面，其稳态噪声地板降低了整整一个 $\sigma_r \sqrt{d_r}$ 的因子。归一化以牺牲局部收缩速度为代价，换取了极限状态下严格更低的均方误差分布。

## 5. 行间异方差与隐式的逆方差加权

在现代网络架构中，不同层的参数或同一参数矩阵的不同行，往往存在显著的梯度异方差（Heteroscedasticity）。记所有参数块中的最大噪声方差为 $\sigma_{\max} := \max_r \sigma_r$。

在实际训练中，调度器通常只提供一个全局标量学习率 $\eta$。若要求每一行的稳态均方误差均不超过某个阈值 $\varepsilon$：

对于 SGD，必须满足对所有 $r$ 有 $\frac{\eta d_r \sigma_r^2}{2\lambda_r} \lesssim \varepsilon$。这迫使全局学习率必须适应噪声最大的块：

$$
\eta \lesssim \min_r \frac{2\lambda_r \varepsilon}{d_r \sigma_r^2} \propto \frac{1}{\sigma_{\max}^2}
$$

在此全局学习率下，所有块的线性收缩率均为恒定的 $\eta \lambda_r$。具有最大噪声方差的参数块将整个优化的收缩进程强行拖慢。

对于归一化方法，误差阈值要求 $\frac{\eta \sigma_r}{2 a_{d_r} \lambda_r} \lesssim \varepsilon$。全局学习率的约束被大幅放宽：

$$
\eta \lesssim \min_r \frac{2 a_{d_r} \lambda_r \varepsilon}{\sigma_r} \propto \frac{1}{\sigma_{\max}}
$$

更为关键的是，第 $r$ 个参数块的有效线性收缩率为：

$$
\text{rate}^{\text{norm}}_r = \eta \frac{a_{d_r} \lambda_r}{\sigma_r}
$$

对于噪声较小的参数块 $\sigma_r \ll \sigma_{\max}$，其有效收缩率将远大于极端高噪块。算法不仅没有因为最大噪声行而全局停滞，反而自动生成了正比于 $1/\sigma_r$ 的有效步长分配。这构成了数学上严格的隐式逆噪声加权（Inverse-noise weighting）。

综上推导，基于共轭范数的零次齐次更新机制，在严重噪声主导区域之所以能够加速预训练，并非由于其能够提取更准确的梯度方向，而是因为其截断了梯度幅值的范数维度。这阻断了行间方差的二次惩罚交叉干扰，降低了稳态误差下界的阶数，并通过自适应的逆方差加权分配了异质参数块间的有效演化步长。

## 参考文献

[1] Shazeer, N., & Stern, M. (2018). [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://proceedings.mlr.press/v80/shazeer18a.html). In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*, *Proceedings of Machine Learning Research*, 80, 4596–4604.

[2] Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., & Bernstein, J. (2024). [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/).

[3] Deng, S., Ouyang, Z., Pang, T., Liu, Z., Jin, R., Yu, S., & Yang, Y. (2026). [RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](https://arxiv.org/abs/2603.20527). *arXiv preprint* arXiv:2603.20527.

[4] Gu, Y., & Xie, Z. (2026). [Mano: Restriking Manifold Optimization for LLM Training](https://arxiv.org/abs/2601.23000). *arXiv preprint* arXiv:2601.23000.

[5] Wang, M., Wang, J., Zhang, J., Wang, W., Pei, P., Cai, X., E, W., & Wu, L. (2025). [GradPower: Powering Gradients for Faster Language Model Pre-Training](https://arxiv.org/abs/2505.24275). *arXiv preprint* arXiv:2505.24275.

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026noise-training-dynamics,
  title={在 LLM 语境下，梯度里的噪声会如何影响 training dynamics？},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/noise-training-dynamics/}
}
```