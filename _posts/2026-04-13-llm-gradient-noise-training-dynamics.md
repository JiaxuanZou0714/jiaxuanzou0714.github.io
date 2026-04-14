---
layout: post
title: "在 LLM 语境下，梯度里的噪声会如何影响 training dynamics？"
date: 2026-04-14 00:00:00
description: "当梯度信号被噪声淹没时，行归一化优化器为何能有效工作？本文通过数学推导揭示：它的核心优势并非方向修正，而是零次齐次映射带来的噪声阶数压缩与隐式逆方差加权。"
tags: [optimization, deep-learning, llm, scaling-law]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

## 引言

大规模语言模型的预训练后期存在一个普遍现象：模型已经学到了数据中的大部分结构，剩余可学习的梯度信号变得微弱，而 mini-batch 采样引入的方差却依然很大。换言之，训练系统进入了噪声主导区域（noise-dominated regime）。

在这种环境下，近期涌现出了多种基于行归一化或参数分块归一化的优化器，它们在某些设定下表现优于传统方法。但从经典优化理论的视角看，归一化似乎是一种"粗糙的近似"——传统的算法设计追求的是更精确地估计梯度方向与曲率，而归一化直接丢弃了梯度的幅值信息。这引出了本文的核心问题：

> 在噪声主导的 setting 下，归一化更新究竟如何改变了训练的动力学？它对优化器设计有什么指导意义？

## 1. 问题建模

### 1.1 随机优化的最小模型

为了精确地讨论上述现象，我们首先建立一个最小化的数学模型。将待优化参数划分为 $m$ 个块，第 $r$ 个块记为 $x_r \in \mathbb{R}^{d_r}$。对应的随机梯度为：

$$
g_r = s_r + \xi_r = s_r + \sigma_r z_r
$$

其中 $s_r := \nabla_r F(x)$ 是真实梯度，$z_r$ 是零均值噪声（$\mathbb{E}[z_r \mid x] = 0$），$\sigma_r$ 控制噪声幅度。定义每个参数块的局部信噪比：

$$
\rho_r := \frac{\lVert s_r \rVert_2}{\sigma_r \sqrt{d_r}}
$$

本文关注的核心场景是 $\rho_r \ll 1$——即噪声远大于信号的训练后期。

### 1.2 归一化更新的统一形式：共轭范数下的最速下降

目前常见的按块归一化、按行归一化以及梯度分数幂更新方法，都可以写成如下统一形式：

$$
u_r = \frac{\operatorname{sign}(g_r) \odot \lvert g_r \rvert^p}{\lVert g_r \rVert_{p+1}^p}
$$

这个公式并非凭空构造，它恰好对应于共轭范数空间下的最速下降法。具体来说，考虑在给定范数步长约束下最大化线性下降量的优化问题：

$$
v^* = \arg\min_{\lVert v \rVert_q \le 1} \langle g_r, v \rangle
$$

根据 Hölder 不等式，对满足 $\frac{1}{p+1} + \frac{1}{q} = 1$ 的共轭指数：

$$
\lvert\langle g_r, v \rangle\rvert \le \lVert g_r \rVert_{p+1} \lVert v \rVert_q
$$

当不等式取等时（即 $|v_{r,i}|^q \propto |g_{r,i}|^{p+1}$），结合范数约束 $\lVert v \rVert_q = 1$ 和共轭条件 $q = \frac{p+1}{p}$，可以唯一解出各分量的绝对值：

$$
\lvert v_{r,i} \rvert = \frac{\lvert g_{r,i} \rvert^p}{\lVert g_r \rVert_{p+1}^p}
$$

加上使内积为负所需的符号条件 $\operatorname{sign}(v_{r,i}) = -\operatorname{sign}(g_{r,i})$，剔除负号后即严格导出了前述的更新公式。两个重要的特例：

- $p = 1$：退化为按块 $L_2$ 范数归一化
- $p = 0$：退化为按元素符号更新（Sign SGD）

## 2. 一个错误的直觉：归一化修正了方向吗？

建立了统一框架之后，一个自然的猜测是：归一化通过"校正梯度方向"来提升性能。直觉上，将噪声梯度投影到单位球面似乎去除了幅值的干扰，让方向更"纯净"。

但这是错的。

以按块 $L_2$ 归一化为例，更新量 $u_r = \frac{g_r}{\lVert g_r \rVert_2}$。计算它与真实梯度 $s_r$ 的方向余弦：

$$
\cos(u_r, s_r) = \frac{\langle u_r, s_r \rangle}{\lVert u_r \rVert_2 \lVert s_r \rVert_2} = \frac{\langle g_r, s_r \rangle}{\lVert g_r \rVert_2 \lVert s_r \rVert_2} = \cos(g_r, s_r)
$$

> 归一化前后，与真实梯度的夹角完全相同。归一化并不能在单步上提升方向精度。

既然不是方向修正，那归一化方法的优势究竟从何而来？答案在于：它改变了优化过程中噪声对系统动力学的影响方式。要理解这一点，我们需要深入分析零次齐次映射的数学性质。

## 3. 核心机制：零次齐次映射的噪声压缩效应

### 3.1 两个关键性质

上述各类归一化方法可以统一抽象为映射 $u_r = \phi_r(g_r)$，它们都满足两条核心性质：

1. 奇对称性：$\phi_r(-g) = -\phi_r(g)$
2. 零次齐次性：对任意 $c > 0$，$\phi_r(cg) = \phi_r(g)$

第二条性质看似简单，但它蕴含了极其深刻的后果：输入的整体缩放不影响输出。这意味着无论噪声 $\sigma_r$ 有多大，输出 $\phi_r(g_r)$ 的幅值始终不变——噪声的增长被"截断"了。

### 3.2 从齐次性到噪声压缩：一阶响应的衰减

有了这两条性质，下一个自然的问题是：当输入 $g_r = s_r + \sigma_r z_r$ 中的噪声项 $\sigma_r z_r$ 远大于信号 $s_r$ 时，映射的输出会如何响应信号的微弱变化？回答这个问题需要分析 $\phi_r$ 对扰动的一阶敏感度，即雅可比矩阵的缩放行为。

对任意非零向量 $g$ 和微小扰动方向 $h$，考虑缩放 $c$ 倍后的输入：

$$
\phi_r(cg+\varepsilon h) = \phi_r\left(c\left(g+\frac{\varepsilon}{c}h\right)\right) = \phi_r\left(g+\frac{\varepsilon}{c}h\right)
$$

对 $\varepsilon$ 在 $0$ 处求方向导数，得到：

$$
J_{\phi_r}(cg)h = \frac{1}{c} J_{\phi_r}(g)h
$$

由于对任意方向 $h$ 都成立，雅可比矩阵满足如下缩放律：

$$
J_{\phi_r}(cg) = \frac{1}{c} J_{\phi_r}(g)
$$

类似地，二阶导数满足 $\nabla^2\phi_r(cg) = \frac{1}{c^2}\nabla^2\phi_r(g)$。

这条缩放律直接回答了前面的问题。现在我们可以利用它来精确计算 $\rho_r \ll 1$ 时归一化更新 $u_r$ 的统计行为。由于噪声项 $\sigma_r z_r$ 远大于信号 $s_r$，自然的做法是在噪声点 $\sigma_r z_r$ 处对 $\phi_r(s_r + \sigma_r z_r)$ 进行泰勒展开：

$$
u_r = \phi_r(\sigma_r z_r) + J_{\phi_r}(\sigma_r z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

应用零次齐次性和雅可比缩放律，将 $\sigma_r$ 因子从映射及其雅可比中提出：

$$
u_r = \phi_r(z_r) + \frac{1}{\sigma_r} J_{\phi_r}(z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

对噪声取期望。由于 $z_r$ 关于原点对称，$\phi_r$ 为奇函数，所以 $\mathbb{E}[\phi_r(z_r)] = 0$。期望漂移为：

$$
\mathbb{E}[u_r \mid x] = \frac{1}{\sigma_r} A_r s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

其中 $A_r := \mathbb{E}[J_{\phi_r}(z_r)]$ 仅取决于噪声分布与归一化方式。协方差的零阶近似为：

$$
\operatorname{Cov}(u_r \mid x) = \operatorname{Cov}(\phi_r(z_r)) + \mathcal{O}\left(\frac{\lVert s_r \rVert}{\sigma_r}\right)
$$

记 $B_r := \operatorname{Cov}(\phi_r(z_r))$。最终，高噪声区归一化更新的动力学形式可以紧凑地写为：

$$
u_r \approx \frac{1}{\sigma_r} A_r s_r + \zeta_r, \quad \mathbb{E}[\zeta_r \mid x] = 0, \quad \operatorname{Cov}(\zeta_r \mid x) \approx B_r
$$

> 归一化将原始的随机梯度 $s_r + \sigma_r z_r$ 改造为两部分——一个被 $1/\sigma_r$ 缩小的有效漂移项，加上一个幅值严格有界、不随 $\sigma_r$ 增长的残余噪声 $\zeta_r$。这与 SGD 中噪声随 $\sigma_r$ 线性放大形成了鲜明对比。

以最常用的 $L_2$ 归一化为例做一个 sanity check。假设噪声近似各向同性高斯 $z_r \sim \mathcal{N}(0, I_{d_r})$，由旋转对称性可得 $A_r = a_{d_r} I_{d_r}$，其中 $a_{d_r} \sim d_r^{-1/2}$，期望漂移化简为 $\mathbb{E}[u_r \mid x] \approx \frac{a_{d_r}}{\sigma_r} s_r$，且有严格的二阶矩约束 $\mathbb{E}\lVert u_r \rVert_2^2 = 1$。这些都与直觉一致。

## 4. 动力学后果：稳定性与误差的重构

前面我们证明了归一化通过齐次映射的缩放律将噪声"管住"了。这对训练的实际动力学有什么影响？我们从全局和局部两个层面来分析。

### 4.1 全局单步下降：最大稳定步长的改善

假设目标函数 $F$ 满足 $L$-利普希茨平滑条件。由下降引理，更新 $x_r^+ = x_r - \eta u_r$ 满足：

$$
F(x^+) \le F(x) - \eta \sum_{r=1}^m \langle s_r, u_r \rangle + \frac{L}{2} \eta^2 \sum_{r=1}^m \lVert u_r \rVert_2^2
$$

SGD 的情形。 令 $u_r = s_r + \sigma_r z_r$，取期望后：

$$
\mathbb{E}[F(x^+) \mid x] \le F(x) - \eta \sum_{r=1}^m \lVert s_r \rVert_2^2 + \frac{L}{2} \eta^2 \left( \sum_{r=1}^m \lVert s_r \rVert_2^2 + \sum_{r=1}^m d_r \sigma_r^2 \right)
$$

在噪声主导时，保证单步期望下降的最大步长受限于：

$$
\eta_{\text{SGD}} \lesssim \frac{2 \sum_{r} \lVert s_r \rVert_2^2}{L \sum_{r} d_r \sigma_r^2}
$$

二次项中 $d_r \sigma_r^2$ 直接体现了噪声方差的平方级惩罚。

归一化方法的情形。 由于 $\lVert u_r \rVert_2^2 = 1$ 是硬约束，二次项总和变为确定性常数 $m$。代入高噪声区的期望漂移结果：

$$
\mathbb{E}[F(x^+) \mid x] \le F(x) - \eta \sum_{r=1}^m \frac{a_{d_r}}{\sigma_r} \lVert s_r \rVert_2^2 + \frac{L}{2} \eta^2 m + \mathcal{O}\left(\eta \sum_{r=1}^m \frac{\lVert s_r \rVert_2^3}{\sigma_r^2}\right)
$$

忽略高阶项，保持期望下降所需的步长上界为：

$$
\eta_{\text{norm}} \lesssim \frac{2 \sum_{r} a_{d_r} \lVert s_r \rVert_2^2 / \sigma_r}{L m}
$$

> 核心差异一目了然：SGD 的最大稳定步长以 $\mathcal{O}(1/\sigma^2)$ 的速率收缩，而归一化方法仅以 $\mathcal{O}(1/\sigma)$ 收缩。这意味着在同等噪声水平下，归一化允许使用显著更大的学习率，这是它在训练后期更为鲁棒的核心原因。

### 4.2 局部收敛：稳态误差地板的降低

上节分析了"步长能取多大"的问题。但更根本的问题是：即便选择了稳定的步长，训练最终能收敛到多好？在随机优化中，噪声的存在使得参数不会精确收敛到最优点，而是在其附近波动，形成一个稳态误差地板。下面通过局部二次模型来定量分析这个地板。

设 $F(x) = \frac{1}{2} \sum_{r=1}^m \lambda_r \lVert x_r \rVert_2^2$，其中 $s_r = \lambda_r x_r$。

归一化方法的参数演化为：

$$
x_{r, t+1} = \left( 1 - \eta \frac{a_{d_r} \lambda_r}{\sigma_r} \right) x_{r, t} - \eta \zeta_{r, t}
$$

对均方误差取期望，利用 $\mathbb{E}[\zeta_{r,t}] = 0$ 消去交叉项，得到线性递推：

$$
\mathbb{E}\lVert x_{r, t+1} \rVert_2^2 \approx \left( 1 - 2\eta \frac{a_{d_r} \lambda_r}{\sigma_r} \right) \mathbb{E}\lVert x_{r, t} \rVert_2^2 + \eta^2
$$

稳态时左右两侧相等，解出稳态方差：

$$
\mathbb{E}\lVert x_{r, \infty} \rVert_2^2 \approx \frac{\eta^2}{2\eta \frac{a_{d_r} \lambda_r}{\sigma_r}} = \frac{\eta \sigma_r}{2 a_{d_r} \lambda_r} \sim \mathcal{O}\left(\frac{\eta \sigma_r \sqrt{d_r}}{\lambda_r}\right)
$$

SGD 的参数演化为 $x_{r, t+1} = (1 - \eta \lambda_r) x_{r, t} - \eta \xi_{r, t}$，稳态方差为：

$$
\mathbb{E}\lVert x_{r, \infty} \rVert_2^2 \approx \frac{\eta d_r \sigma_r^2}{2\lambda_r} \sim \mathcal{O}\left(\frac{\eta d_r \sigma_r^2}{\lambda_r}\right)
$$

对比这两个结果，归一化展现了一个清晰的权衡：

> - 代价：局部收缩率从 $\eta \lambda_r$ 减弱为 $\eta a_{d_r} \lambda_r / \sigma_r$。噪声越大，参数向最优点的拉回越慢。
> - 收益：稳态误差地板降低了整整 $\sigma_r \sqrt{d_r}$ 倍。归一化以牺牲收缩速度为代价，换取了严格更低的极限误差。

## 5. 从理论到实践：异方差结构与隐式逆噪声加权

### 5.1 动机：为什么必须考虑参数块间的噪声差异？

前面的分析逐块展开，隐含的假设是每个参数块可以独立调控学习率。但在实际的 LLM 训练中，情况远非如此理想，有两个关键的现实约束：

第一，不同参数块的噪声水平差异悬殊——例如 embedding 层因梯度稀疏而方差极大，中间层则获得较稠密的信号，$\sigma_r$ 可跨越数个量级。这种系统性差异即异方差（heteroscedasticity）。

第二，学习率调度器通常只提供一个全局标量 $\eta$。尽管存在分层学习率（layer-wise learning rate）等技巧，主流训练管线仍然以单一学习率为主。这意味着全局 $\eta$ 必须同时满足所有参数块的稳定性要求。

> 这两个约束叠加在一起，产生了一个尖锐的问题：当噪声水平参差不齐时，全局学习率的选择会如何影响整体收敛？SGD 和归一化方法在这个问题上的表现截然不同。

### 5.2 分析：全局学习率受限于最差参数块

设所有参数块中的最大噪声标准差为 $\sigma_{\max} := \max_r \sigma_r$。要求每一块的稳态均方误差 $\mathbb{E}\lVert x_{r,\infty} \rVert_2^2$ 都不超过给定阈值 $\varepsilon$，全局学习率必须满足最严格的那个约束。

对于 SGD，利用第 4.2 节的稳态公式，要求 $\frac{\eta d_r \sigma_r^2}{2\lambda_r} \lesssim \varepsilon$ 对所有块 $r$ 成立。这迫使全局学习率被最高噪声的块卡住：

$$
\eta \lesssim \min_r \frac{2\lambda_r \varepsilon}{d_r \sigma_r^2} \propto \frac{1}{\sigma_{\max}^2}
$$

在这个被压低的学习率下，每个参数块的收缩率都是 $\eta \lambda_r$——一个不依赖于该块自身噪声水平的常数。换句话说，那些原本噪声很小、本可以快速收敛的参数块，被最高噪声的块拖慢到了同一速度。这是一种典型的"木桶效应"：最差的块决定了整个系统的速度。

对于归一化方法，稳态公式变为 $\frac{\eta \sigma_r}{2 a_{d_r} \lambda_r} \lesssim \varepsilon$。全局学习率的约束放宽为：

$$
\eta \lesssim \min_r \frac{2 a_{d_r} \lambda_r \varepsilon}{\sigma_r} \propto \frac{1}{\sigma_{\max}}
$$

仅这一点就已经是显著的改善：允许的最大步长从 $\propto 1/\sigma_{\max}^2$ 提升到了 $\propto 1/\sigma_{\max}$。

但更关键的差异在于收缩率的异质性。第 $r$ 个参数块的有效收缩率为：

$$
\text{rate}^{\text{norm}}_r = \eta \frac{a_{d_r} \lambda_r}{\sigma_r}
$$

这个收缩率与 $1/\sigma_r$ 成正比。噪声小的参数块收缩快，噪声大的参数块收缩慢——归一化自动为每个块分配了与其噪声水平成反比的有效步长。这正是统计学中经典的逆方差加权（inverse-variance weighting）原则，而归一化在不引入任何额外超参数的情况下天然实现了它。

### 5.3 对 LLM 训练的实际意义

这个隐式逆噪声加权的结论直接解释了几个实际训练中的现象：

1. 在预训练后期使用行归一化优化器时，即使不对 embedding 层和主干层设置不同的学习率，训练也往往保持稳定。这是因为归一化已经自动"压制"了 embedding 层的高梯度方差，无需人工干预。

2. SGD 在大规模训练后期需要极其保守的学习率衰减，否则容易因为个别高噪声参数块的发散而导致 loss spike。归一化方法对此天然免疫：最大稳定步长仅以 $1/\sigma$ 而非 $1/\sigma^2$ 收缩。

3. 在混合精度训练中，低精度带来的量化噪声进一步加剧了参数块间的异方差。归一化方法在此场景下的鲁棒性优势会被进一步放大。

## 6. 总结

本文的推导揭示了一个清晰的图景：在 LLM 预训练后期的噪声主导区域，基于共轭范数的零次齐次更新机制之所以优于经典 SGD，并非因为它能提取更精确的梯度方向。其真正的作用机制是三重的：

1. 截断噪声幅值：零次齐次性使输出幅值不随噪声增长，阻断了二次惩罚项的膨胀；
2. 降低误差阶数：稳态误差从 $\mathcal{O}(\sigma^2)$ 降至 $\mathcal{O}(\sigma)$，最大稳定步长从 $\mathcal{O}(1/\sigma^2)$ 升至 $\mathcal{O}(1/\sigma)$；
3. 隐式逆方差加权：在异方差参数块间自适应分配有效步长，避免高噪声块拖垮全局收敛。

这三个效果共同构成了归一化优化器在训练后期的理论优势基础。对于优化器设计，本文的分析给出的核心指导是：在噪声主导的 setting 下，更新规则的零次齐次性——而非对曲率的更精细估计——才是决定训练稳定性的关键结构性质。任何保持这一性质的归一化方案都能自动获得上述三重优势。

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