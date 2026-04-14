---
layout: post
title: "在 LLM 语境下，梯度里的噪声会如何影响 training dynamics？"
date: 2026-04-14 00:00:00
description: "从理论上分析现代大规模预训练后期的噪声主导现象，推导基于广义范数的零次齐次优化器如何本质地改变梯度下降的稳定性、稳态误差、收敛速度与超参数迁移缩放律。"
tags: [optimization, deep-learning, llm, scaling-law]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

近期社区中涌现出多项基于行归一化的优化器工作，在某些设定下展现出优于此前广泛使用的正交化优化器的性能。从传统的平滑优化视角来看，行归一化或参数的符号更新似乎只是一种舍弃了精确幅值信息的局部近似。然而，在现代大规模模型预训练的后期阶段，真实的梯度信号几乎被随机采样引入的巨大噪声所淹没，训练系统进入了严重的噪声主导区域。

在这种极端异方差与高噪的动力学系统中，传统机器学习基于信号主导区域的优化器设计直觉、训练动力学推导以及超参数缩放律往往不再适用。本文旨在通过严密的数学推导证明，这类基于共轭范数的归一化方法并非通过提升局部梯度方向的精度产生收益，而是通过其内在的零次齐次性质，阻断了高阶噪声的注入。这种机制将传统的“幅值噪声加方向噪声”动力学，严格转化为“有界步长、方向噪声与基于逆噪声方差加权的有效漂移”。我们将系统性地推导这一基础性质如何决定性地改变了训练稳定性、收敛速度以及超参数迁移法则。

## 1. 最小模型与广义范数更新的几何本质

考察一个矩阵参数块 $W \in \mathbb{R}^{m \times d}$。设第 $r$ 行的随机梯度为：

$$
g_r = s_r + \xi_r = s_r + \sigma_r z_r
$$

其中真实梯度定义为 $s_r := \nabla_r F(W)$，且假设随机变量 $z_r$ 满足零均值 $\mathbb{E}[z_r \mid W] = 0$ 及单位协方差。$\sigma_r$ 表征该行的噪声标准差。

目前常见的按块或按行归一化方法，以及涉及梯度的分数幂更新方法，其一般形式可以写为：

$$
u_t = \frac{\text{sign}(g_t) \odot \lvert g_t \rvert^p}{\lVert g_t \rVert_{p+1}^p}
$$

这一形式严格对应于共轭范数空间下的最速下降法。设定优化目标为在给定步长约束下，使得函数沿梯度方向的局部线性近似下降量最大。求解以下优化问题：

$$
v^* = \arg\min_{\lVert v \rVert_q \le 1} \langle g_t, v \rangle
$$

根据 Hölder 不等式，对于任意满足 $\frac{1}{p+1} + \frac{1}{q} = 1$ 的共轭指数，存在：

$$
\lvert\langle g_t, v \rangle\rvert \le \lVert g_t \rVert_{p+1} \lVert v \rVert_q
$$

目标是使得内积达到负的最大值，不等式必须取等号。Hölder 不等式取等号的条件是存在常数 $c > 0$ 使得：

$$
\lvert v_i \rvert^q = c \lvert g_{t,i} \rvert^{p+1}
$$

由共轭条件可得 $q = \frac{p+1}{p}$。代入上式解出分量的绝对值：

$$
\lvert v_i \rvert = c^{\frac{p}{p+1}} \lvert g_{t,i} \rvert^p
$$

将此关系代入范数约束 $\lVert v \rVert_q = 1$ 求解常数项：

$$
\sum_i \lvert v_i \rvert^q = c \sum_i \lvert g_{t,i} \rvert^{pq} = c \sum_i \lvert g_{t,i} \rvert^{p+1} = c \lVert g_t \rVert_{p+1}^{p+1} = 1
$$

解得 $c = \lVert g_t \rVert_{p+1}^{-(p+1)}$。将其代回并结合内积取负值所需的符号条件 $\text{sign}(v_i) = -\text{sign}(g_{t,i})$，剔除负号后即严格导出了前述的更新公式。当 $p=1$ 时，退化为严格的按行 $L_2$ 范数归一化；当 $p=0$ 时，退化为按元素的符号更新。

## 2. 零次齐次更新与高噪声展开的统一结构

上述各类广义归一化更新可以抽象为映射 $u_r = \phi_r(g_r)$。映射 $\phi_r$ 满足两个核心性质。其一为奇对称性 $\phi_r(-g) = -\phi_r(g)$；其二为零次齐次性：对于任意常数 $c > 0$，存在 $\phi_r(cg) = \phi_r(g)$。

定义局部信噪比 $\rho_r := \frac{\lVert s_r \rVert}{\sigma_r \sqrt{d}}$。我们关注噪声主导区域，即 $\rho_r \ll 1$。

首先分析零次齐次映射的雅可比矩阵性质。由定义可知：

$$
\phi_r(cg + \varepsilon h) = \phi_r\left(c\left(g + \frac{\varepsilon}{c}h\right)\right) = \phi_r\left(g + \frac{\varepsilon}{c}h\right)
$$

对 $\varepsilon$ 在零点处求导可得：

$$
J_{\phi_r}(cg)h = \frac{1}{c} J_{\phi_r}(g)h
$$

由于该等式对任意向量 $h$ 成立，故导出关键关系：

$$
J_{\phi_r}(cg) = \frac{1}{c} J_{\phi_r}(g)
$$

同理可得海森矩阵满足 $\nabla^2\phi_r(cg) = \frac{1}{c^2}\nabla^2\phi_r(g)$。这一代数性质表明，一阶响应被高噪声按 $1/\sigma$ 严格压制，二阶误差则按 $1/\sigma^2$ 压制。

对 $\phi_r(s_r + \sigma_r z_r)$ 在高噪声处进行泰勒展开：

$$
\phi_r(s_r + \sigma_r z_r) = \phi_r(\sigma_r z_r) + J_{\phi_r}(\sigma_r z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

代入齐次性带来的缩放关系：

$$
u_r = \phi_r(z_r) + \frac{1}{\sigma_r}J_{\phi_r}(z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

对给定的参数状态求条件期望。由于 $z_r$ 分布关于零点对称，且 $\phi_r$ 为奇函数，零阶项的期望 $\mathbb{E}[\phi_r(z_r)] = 0$。定义雅可比期望矩阵 $A_r := \mathbb{E}[J_{\phi_r}(z_r)]$，更新量的一阶期望漂移严格写为：

$$
\mathbb{E}[u_r \mid W] = \frac{1}{\sigma_r}A_r s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

同时考察协方差。定义 $B_r := \text{Cov}(\phi_r(z_r))$。在零阶近似下：

$$
\text{Cov}(u_r \mid W) = B_r + \mathcal{O}\left(\frac{\lVert s_r \rVert}{\sigma_r}\right)
$$

上述推导确立了所有零次齐次优化器在噪声主导区的基本动力学方程：更新量被重构为一个基于 $1/\sigma_r$ 缩放的有效漂移，叠加一个协方差由 $B_r$ 给定的有界扩散项。由于 $\phi_r(z_r)$ 有界，扩散项的幅值不再随原始梯度噪声 $\sigma_r$ 发生线性放大。

特化至 $L_2$ 行归一化 $\phi_r(g_r) = \frac{g_r}{\lVert g_r \rVert_2}$，若假设 $z_r \sim \mathcal{N}(0, I_{d_r})$，利用各向同性旋转对称可得 $A_r = a_{d_r} I_{d_r}$。常数 $a_{d_r}$ 的解析解为：

$$
a_{d_r} = \frac{d_r - 1}{d_r} \mathbb{E}\left[\frac{1}{\lVert z_r \rVert_2}\right] = \frac{d_r - 1}{d_r\sqrt{2}} \frac{\Gamma(\frac{d_r - 1}{2})}{\Gamma(\frac{d_r}{2})} \sim d_r^{-1/2}
$$

同时零阶协方差趋近于 $\frac{1}{d_r}I_{d_r}$。需要强调，对于纯行范数归一化，$\cos(u_r, s_r) = \cos(g_r, s_r)$，其并未提升行内的方向估计精度，其优化动力学改变完全来源于行间异方差条件下的有效漂移重构。

## 3. 训练稳定性分析

动力学系统的稳定性需要从单步下降边界与局部均方收敛两个独立的层面进行解构。

### 3.1 单步损失稳定性

假设目标函数满足利普希茨平滑系数为 $L$ 的条件。利用平滑不等式进行按块展开：

$$
F(x_{t+1}) \le F(x_t) - \eta \sum_r \langle s_r, u_r \rangle + \frac{L}{2}\eta^2 \sum_r \lVert u_r \rVert^2
$$

对于传统的随机梯度下降更新，$\mathbb{E}[\lVert g_r \rVert^2] \approx \lVert s_r \rVert^2 + d_r \sigma_r^2$。保证期望单步下降的步长窗口上限严格受制于系统的总噪声方差：

$$
\eta_{SGD} \lesssim \frac{2 \sum_r \lVert s_r \rVert^2}{L \sum_r d_r \sigma_r^2}
$$

对于行归一化方法，每行的更新范数严格为常数 $\lVert u_r \rVert^2 = 1$。总的二次罚项收敛为确定性常数 $\frac{L}{2}\eta^2 m$，与参数梯度中的噪声幅值绝对解耦。代入高噪展开中的漂移期望项，其步长窗口上限变为：

$$
\eta_{norm} \lesssim \frac{2 \sum_r \frac{a_{d_r}}{\sigma_r}\lVert s_r \rVert^2}{L m}
$$

对比可见，归一化操作将平滑边界对噪声方差的依赖关系从 $\mathcal{O}(1/\sigma^2)$ 严格降阶为 $\mathcal{O}(1/\sigma)$。在深度网络的训练后期，这直接切断了单次采样随机游走导致的二次项发散路径，提供了数学上更为稳健的下降区间。

### 3.2 局部均方动力学

构造局部二次平滑模型，设目标函数具有各向同性曲率 $F(x) = \frac{1}{2} \sum_r \lambda_r \lVert x_r \rVert^2$，使得真实梯度 $s_r = \lambda_r x_r$。

对于归一化更新，将其高噪形式代入迭代方程：

$$
x_{r,t+1} \approx \left(1 - \eta \frac{a_{d_r} \lambda_r}{\sigma_r}\right)x_{r,t} - \eta \zeta_{r,t}
$$

其中扩散项满足零均值且 $\mathbb{E}\lVert\zeta_{r,t}\rVert^2 \approx 1$。对范数取期望并忽略 $\mathcal{O}(\eta^2)$ 的收缩交叉项：

$$
\mathbb{E}\lVert x_{r,t+1} \rVert^2 \approx \left(1 - 2\eta \frac{a_{d_r} \lambda_r}{\sigma_r}\right)\mathbb{E}\lVert x_{r,t} \rVert^2 + \eta^2
$$

求解该线性差分方程，得到其演化上界：

$$
\mathbb{E}\lVert x_{r,t} \rVert^2 \lesssim \exp\left(-2\eta \frac{a_{d_r}\lambda_r}{\sigma_r} t\right)\lVert x_{r,0} \rVert^2 + \frac{\eta \sigma_r}{2 a_{d_r} \lambda_r}
$$

对比未归一化的传统更新过程：

$$
\mathbb{E}\lVert x_{r,t} \rVert^2 \lesssim \exp(-2\eta \lambda_r t)\lVert x_{r,0} \rVert^2 + \frac{\eta d_r \sigma_r^2}{2 \lambda_r}
$$

两相对比揭示了一个反直觉结论：同一基础步长 $\eta$ 下，归一化方法由于有效漂移下降，其局部线性收缩率实则更慢。然而，其稳态方差下界（噪声地板）从 $\mathcal{O}(\frac{\eta d_r \sigma_r^2}{\lambda_r})$ 显著下降为 $\mathcal{O}(\frac{\eta \sigma_r \sqrt{d_r}}{\lambda_r})$。归一化以牺牲局部恢复力为代价，换取了更低的极限稳态误差。

## 4. 收敛速度与异方差优势

前述推导引申出一个核心问题：在需要更低稳态误差的训练后期，归一化方法是否必然带来更快的全局收敛速度？

在完全同方差的理想模型中，设全局 $d_r=d, \sigma_r=\sigma, \lambda_r=\lambda$。为使稳态误差达到容忍阈值 $\varepsilon$，优化系统需要调整最优步长。对于传统更新，令 $\frac{\eta d \sigma^2}{2\lambda} \lesssim \varepsilon$，可得允许的最大步长 $\eta_{SGD}^* \propto \frac{\varepsilon}{\sigma^2}$。代入收缩率计算收敛至目标所需的迭代步数 $T_{SGD}(\varepsilon) = \Theta\left(\frac{d\sigma^2}{\lambda^2\varepsilon}\log\frac{1}{\varepsilon}\right)$。对于归一化更新，令 $\frac{\eta \sigma}{2 a_d \lambda} \lesssim \varepsilon$，最大允许步长 $\eta_{norm}^* \propto \frac{\varepsilon}{\sigma}$。计算其迭代步数：

$$
T_{norm}(\varepsilon) = \Theta\left(\frac{\sigma^2}{a_d^2\lambda^2\varepsilon}\log\frac{1}{\varepsilon}\right)
$$

由于 $a_d \sim d^{-1/2}$，存在渐进等价关系 $T_{norm}(\varepsilon) = \Theta(T_{SGD}(\varepsilon))$。这证明了在完全同构的噪声环境中，零次齐次映射无法在收敛阶数上产生严格的理论优势。

真实的加速效应来源于现代架构固有的行间异方差特性。考虑各行参数共享全局标量学习率 $\eta$。对于未归一化更新，必须满足所有行的稳定条件，故全局步长受限于最高噪声行：

$$
\eta_{SGD}^* \lesssim \min_r \frac{2 \lambda_r \varepsilon}{d_r \sigma_r^2} \approx \frac{2 \lambda \varepsilon}{d \sigma_{\max}^2}
$$

所有参数行被迫以相同且被严重压制的速率 $\eta_{SGD}^* \lambda$ 收敛。

相反，对于归一化更新，步长限制为 $\eta_{norm}^* \lesssim \frac{2 a_d \lambda \varepsilon}{\sigma_{\max}}$。代入各行的有效收缩率公式，第 $r$ 行的动态收缩率成为：

$$
\text{rate}_r = \eta_{norm}^* \frac{a_d \lambda}{\sigma_r} \propto \frac{1}{\sigma_r}
$$

归一化在共享学习率下，自发地实现了各行步长的逆噪声加权分配。对于方差较小的相对平稳通道，其收缩率严格高于被全局最高方差所主导的非归一化基线。这是其在高度复杂的网络参数空间中表现出更优经验速度的根本数学来源。

## 5. 超参数迁移与缩放律的重构

零次齐次更新所导出的 $1/\sigma$ 缩放与有界方差特性，重构了深层网络对批次大小、动量以及宽度的超参数迁移法则。

### 5.1 梯度尺度不变性

对于任意全局正标量缩放 $g_r' = c g_r$，由 $\phi_r(c g_r) = \phi_r(g_r)$ 可知，网络各层的更新完全保持不变。这使得优化过程对损失函数的全局乘性因子、混合精度训练中的动态缩放系数以及部分同构参数重参数化具备理论上的精确鲁棒性。需指出，非零次齐次的正则项（如权重衰减）将破坏这一精确等价性，在迁移时必须被单独剥离分析。

### 5.2 批次大小 (Batch Size) 迁移

考察批量缩放对最优学习率的影响。对于独立同分布采样，梯度噪声与批次大小满足 $\sigma(B) = \sigma_0 / \sqrt{B}$。
保持期望稳态误差不变，对于非归一化更新：

$$
\text{Var}(x_{\infty}^{SGD}) \propto \eta_{SGD} \sigma^2(B) \propto \frac{\eta_{SGD}}{B} = \text{const} \implies \eta_{SGD}^* \propto B
$$

对于归一化更新，由于雅可比矩阵带来的期望漂移重构：

$$
\text{Var}(x_{\infty}^{norm}) \propto \eta_{norm} \sigma(B) \propto \frac{\eta_{norm}}{\sqrt{B}} = \text{const} \implies \eta_{norm}^* \propto \sqrt{B}
$$

这一推导证明，归一化操作将最优学习率对批次大小的敏感度依赖从线性降低至平方根级别，使得跨批次超参数搜索域更加收敛。

### 5.3 动量 (Momentum) 迁移

考虑指数移动平均动量 $m_t = \beta m_{t-1} + (1-\beta)g_t$，再对其执行归一化更新 $u_t = \phi(m_t)$。稳态下动量估计的协方差阵满足：

$$
\text{Cov}(m_t) = \frac{1-\beta}{1+\beta} \text{Cov}(g_t)
$$

等效的局部噪声标准差变为 $\sigma_{eff} = \sigma \sqrt{\frac{1-\beta}{1+\beta}}$。
代入此前的动力学稳态方程，归一化更新的极限稳态方差正比于：

$$
\mathcal{O}\left(\frac{\eta \sigma}{\lambda}\sqrt{\frac{1-\beta}{1+\beta}}\right)
$$

对比未归一化优化的 $\mathcal{O}\left(\frac{\eta \sigma^2}{\lambda}\frac{1-\beta}{1+\beta}\right)$，动量系数 $\beta$ 在归一化系统中同样仅以平方根形式介入误差下界。动量因子与批次大小构成了一种更平缓的对偶关系，这解释了为什么在高噪预训练后期，动量参数微调的系统敏感度被大幅削弱。

### 5.4 最优范数指数与漂移扩散比

将上述理论推广至带有可调超参数 $p$ 的一般分数幂归一化 $\phi_p$。此时雅可比期望矩阵为 $A_p$，高噪扩散协方差为 $B_p$。动力学方程主导项可抽象为：

$$
x_{t+1} \approx \left(I - \frac{\eta}{\sigma} A_p H\right)x_t - \eta \zeta_t
$$

定义衡量优化效率的广义漂移扩散比：

$$
\Gamma_p := \frac{\lambda_{\min}(\text{sym}(A_p))}{\lambda_{\max}(B_p)}
$$

给定任务设定时，最优的 $p$ 值等价于寻找使得 $\Gamma_p$ 最大化的算子。当系统的信噪比条件因为序列长度或模态切换发生有限偏移时，目标分布的高阶矩可能发生微小扭曲。此时，依赖特定的漂移扩散比往往可以提供比纯 $L_2$（$p=1$）或纯符号算子（$p=0$）更为精细的噪声截断边界。这从算子理论的层面解释了某些非整数的经验幂次为何能够在特定的大型语料训练中提供额外增益。

综上所述，零次齐次优化器在现代大语言模型的训练体系中，并不旨在恢复更为精确的确定性高维几何方向。其本质作用是阻断高次误差项进入网络前向状态，利用内在的 $1/\sigma$ 雅可比收缩机制实现隐式的异方差加权，进而在保证训练系统不发生灾难性突变的约束下，将参数推至具有更低泛化误差的扁平区域。

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
  url={[https://jiaxuanzou0714.github.io/blog/2026/noise-training-dynamics/](https://jiaxuanzou0714.github.io/blog/2026/noise-training-dynamics/)}
}
```