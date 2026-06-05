---
layout: post
title: "在 LLM 语境下，梯度里的噪声会如何影响 training dynamics？"
date: 2026-04-14 00:00:00
description: "讨论 LLM 预训练后期的梯度噪声，以及块归一化更新为什么更像是在限制更新幅值，而不是在修正梯度方向。"
tags: [optimization, deep-learning, llm, scaling-law]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

## 引言

LLM 预训练到后期时，随机梯度里的有效信号会变弱，但 mini-batch 采样带来的方差还在。于是很多更新看起来更像是 noise 在主导，而不是 signal 在主导。下面把这种情况叫做 noise-dominated regime。

这也是我想重新看行归一化、块归一化优化器的原因。按传统优化的直觉，优化器最好能更准确地估计梯度方向、曲率或者二阶结构；但归一化更新很粗暴，它直接丢掉梯度幅值。

所以可以换一个问法：

> 当梯度主要由噪声控制时，归一化更新到底改变了哪一部分 training dynamics？

我的结论比较保守：至少对 blockwise $$L_2$$ 归一化来说，它不改进单步方向精度，也没有从噪声里提取出额外信息。它做的是限制更新幅值，让原始梯度中的噪声尺度不能直接传到参数更新里。在多块异方差的 LLM 场景中，这会表现成一种隐式的块级学习率调节。

## 1. 问题建模

### 1.1 随机优化的最小模型

先从一个尽量小的模型开始。把参数分成 $$m$$ 个块，第 $$r$$ 个块记为 $$x_r \in \mathbb{R}^{d_r}$$。这一块上的随机梯度写成

$$
g_r = s_r + \xi_r = s_r + \sigma_r z_r
$$

其中 $$s_r := \nabla_r F(x)$$ 是真实梯度，$$z_r$$ 是零均值噪声（$$\mathbb{E}[z_r \mid x] = 0$$），$$\sigma_r$$ 控制这一块的噪声尺度。对应的局部信噪比定义为

$$
\rho_r := \frac{\lVert s_r \rVert_2}{\sigma_r \sqrt{d_r}}
$$

后面主要看 $$\rho_r \ll 1$$ 的情况，也就是噪声比信号大很多的时候。

### 1.2 归一化更新的统一形式：共轭范数下的最速下降

很多按块归一化、按行归一化和梯度分数幂更新，都可以写成下面这个形式：

$$
u_r = \frac{\operatorname{sign}(g_r) \odot \lvert g_r \rvert^p}{\lVert g_r \rVert_{p+1}^p}
$$

这个式子也可以从最速下降的角度推出来。给定一个范数步长约束，问线性项最多能降多少：

$$
v^* = \arg\min_{\lVert v \rVert_q \le 1} \langle g_r, v \rangle
$$

由 Hölder 不等式，对满足 $$\frac{1}{p+1} + \frac{1}{q} = 1$$ 的共轭指数，有

$$
\lvert\langle g_r, v \rangle\rvert \le \lVert g_r \rVert_{p+1} \lVert v \rVert_q
$$

等号成立时，$$\lvert v_{r,i} \rvert^q \propto \lvert g_{r,i} \rvert^{p+1}$$。再结合 $$\lVert v \rVert_q = 1$$ 和 $$q = \frac{p+1}{p}$$，可以得到

$$
\lvert v_{r,i} \rvert = \frac{\lvert g_{r,i} \rvert^p}{\lVert g_r \rVert_{p+1}^p}
$$

符号取负号是为了让内积下降；如果把更新方向的负号放到优化步骤里，就得到上面的 $$u_r$$。两个常见特例是：

- $$p = 1$$：blockwise $$L_2$$ 归一化；
- $$p = 0$$：按元素符号更新，也就是 Sign SGD。

## 2. 一个需要澄清的直觉：归一化是在修正方向吗？

看到归一化，一个很自然的解释是：它把梯度投到单位球面上，去掉了幅值噪声，所以方向更准。

这个说法至少对 $$L_2$$ 归一化不对。

以 blockwise $$L_2$$ 归一化为例，更新量为 $$u_r = \frac{g_r}{\lVert g_r \rVert_2}$$。它和真实梯度 $$s_r$$ 的方向余弦是

$$
\cos(u_r, s_r) = \frac{\langle u_r, s_r \rangle}{\lVert u_r \rVert_2 \lVert s_r \rVert_2} = \frac{\langle g_r, s_r \rangle}{\lVert g_r \rVert_2 \lVert s_r \rVert_2} = \cos(g_r, s_r)
$$

也就是说，blockwise $$L_2$$ 归一化前后，它和真实梯度的夹角完全一样。它只是把同一个方向缩放到单位长度，并没有让方向更接近 $$s_r$$。

当然，这个结论只适用于 $$p=1$$。如果是 Sign SGD 或一般 power update，各坐标会经过非线性变形，方向通常会改变。此时更新可以理解为在另一种几何下做最速下降；方向是否更好，要看梯度、噪声和曲率的关系，不能直接说它更接近真实梯度。

所以，如果连方向不变的 $$L_2$$ 归一化也能在某些训练里有用，解释就不该停在“方向更准”。更该问的是：噪声进入更新以后，它的幅值怎么影响动力学？

## 3. 零次齐次性会限制更新幅值

### 3.1 两个性质

把归一化更新抽象成 $$u_r = \phi_r(g_r)$$。这里关心的几类映射通常满足两条性质：

1. 奇对称性：$$\phi_r(-g) = -\phi_r(g)$$；
2. 零次齐次性：对任意 $$c > 0$$，$$\phi_r(cg) = \phi_r(g)$$。

第二条很重要。输入整体乘上一个正数，输出不变。于是噪声尺度 $$\sigma_r$$ 再大，只要它只是整体幅值上的放大，归一化后的更新幅值也不会跟着线性变大。

这不是 denoising。归一化没有从噪声里恢复出信号，也没有提高信噪比；它还丢弃了幅值信息。它只是把输出更新限制在一个固定尺度内。最简单的一维例子是

$$
u = \operatorname{sign}(s + \sigma z)
$$

这个 $$u$$ 的方差总是有界的，而原始梯度 $$g = s + \sigma z$$ 的噪声方差会随 $$\sigma^2$$ 增长。

### 3.2 一阶响应为什么会随噪声变弱

接下来要看的是：如果 $$g_r = s_r + \sigma_r z_r$$，并且 $$\sigma_r z_r$$ 比 $$s_r$$ 大很多，那么 $$\phi_r(g_r)$$ 对信号 $$s_r$$ 还剩多少响应？

先说明一个技术限制。下面用的是雅可比和 Taylor 展开，所以严格来说适用于光滑的归一化映射，比如 $$L_2$$ 归一化。Sign 这类不光滑映射需要换成次梯度或分布意义下的导数。这里先看光滑情形，因为它已经能解释主要的尺度关系。

对任意非零向量 $$g$$ 和微小扰动 $$h$$，利用零次齐次性：

$$
\phi_r(cg+\varepsilon h) = \phi_r\left(c\left(g+\frac{\varepsilon}{c}h\right)\right) = \phi_r\left(g+\frac{\varepsilon}{c}h\right)
$$

对 $$\varepsilon$$ 在 0 处求方向导数，得到

$$
J_{\phi_r}(cg)h = \frac{1}{c} J_{\phi_r}(g)h
$$

因此

$$
J_{\phi_r}(cg) = \frac{1}{c} J_{\phi_r}(g)
$$

二阶导数也类似：$$\nabla^2\phi_r(cg) = \frac{1}{c^2}\nabla^2\phi_r(g)$$。

现在在噪声点 $$\sigma_r z_r$$ 附近展开：

$$
u_r = \phi_r(\sigma_r z_r) + J_{\phi_r}(\sigma_r z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

再把齐次性和雅可比缩放律代进去：

$$
u_r = \phi_r(z_r) + \frac{1}{\sigma_r} J_{\phi_r}(z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

对噪声取期望。若 $$z_r$$ 关于原点对称，且 $$\phi_r$$ 是奇函数，则 $$\mathbb{E}[\phi_r(z_r)] = 0$$。于是

$$
\mathbb{E}[u_r \mid x] = \frac{1}{\sigma_r} A_r s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

其中 $$A_r := \mathbb{E}[J_{\phi_r}(z_r)]$$，只由噪声分布和归一化方式决定。协方差的零阶项为

$$
\operatorname{Cov}(u_r \mid x) = \operatorname{Cov}(\phi_r(z_r)) + \mathcal{O}\left(\frac{\lVert s_r \rVert}{\sigma_r}\right)
$$

记 $$B_r := \operatorname{Cov}(\phi_r(z_r))$$，高噪声区的更新可以写成

$$
u_r \approx \frac{1}{\sigma_r} A_r s_r + \zeta_r, \quad \mathbb{E}[\zeta_r \mid x] = 0, \quad \operatorname{Cov}(\zeta_r \mid x) \approx B_r
$$

这个式子后面会一直用。归一化后的更新有两部分：一部分是被 $$1/\sigma_r$$ 缩小的有效漂移，另一部分是幅值不随 $$\sigma_r$$ 增长的残余噪声。需要注意，信号项也被 $$1/\sigma_r$$ 缩小了。归一化没有增强信号，它只是限制输出噪声幅值。

以 $$L_2$$ 归一化作一致性检查。若 $$z_r \sim \mathcal{N}(0, I_{d_r})$$ 且噪声各向同性，旋转对称性给出 $$A_r = a_{d_r} I_{d_r}$$，其中 $$a_{d_r} \sim d_r^{-1/2}$$。于是

$$
\mathbb{E}[u_r \mid x] \approx \frac{a_{d_r}}{\sigma_r} s_r
$$

同时 $$\mathbb{E}\lVert u_r \rVert_2^2 = 1$$。漂移随 $$1/\sigma_r$$ 变小，二阶矩则被固定住。

## 4. 动力学后果：稳定性与误差地板

前面得到的是更新本身的统计形式。现在把它放回优化动力学里看。先看单步下降允许多大的学习率，再看局部二次模型下的稳态误差。

### 4.1 全局单步下降：最大稳定步长

设目标函数 $$F$$ 是 $$L$$-smooth。由下降引理，对更新 $$x_r^+ = x_r - \eta u_r$$ 有

$$
F(x^+) \le F(x) - \eta \sum_{r=1}^m \langle s_r, u_r \rangle + \frac{L}{2} \eta^2 \sum_{r=1}^m \lVert u_r \rVert_2^2
$$

先看 SGD，也就是 $$u_r = s_r + \sigma_r z_r$$。取期望后：

$$
\mathbb{E}[F(x^+) \mid x] \le F(x) - \eta \sum_{r=1}^m \lVert s_r \rVert_2^2 + \frac{L}{2} \eta^2 \left( \sum_{r=1}^m \lVert s_r \rVert_2^2 + \sum_{r=1}^m d_r \sigma_r^2 \right)
$$

在噪声主导时，保证期望下降的步长大致要满足

$$
\eta_{\text{SGD}} \lesssim \frac{2 \sum_{r} \lVert s_r \rVert_2^2}{L \sum_{r} d_r \sigma_r^2}
$$

这个限制来自二次项里的 $$d_r \sigma_r^2$$。

再看 blockwise $$L_2$$ 归一化。此时 $$\lVert u_r \rVert_2^2 = 1$$，所以二次项变成常数级的 $$m$$。把第 3 节的漂移近似代入：

$$
\mathbb{E}[F(x^+) \mid x] \le F(x) - \eta \sum_{r=1}^m \frac{a_{d_r}}{\sigma_r} \lVert s_r \rVert_2^2 + \frac{L}{2} \eta^2 m + \mathcal{O}\left(\eta \sum_{r=1}^m \frac{\lVert s_r \rVert_2^3}{\sigma_r^2}\right)
$$

忽略高阶项，步长上界约为

$$
\eta_{\text{norm}} \lesssim \frac{2 \sum_{r} a_{d_r} \lVert s_r \rVert_2^2 / \sigma_r}{L m}
$$

这里出现了一个很直接的差别：SGD 的稳定步长随 $$1/\sigma^2$$ 收缩，而归一化更新大致随 $$1/\sigma$$ 收缩。噪声越大，差别越明显。

不过这里不能过度解读。在单块、同方差模型里，更大的稳定步长不等于更大的单步下降量。对归一化方法，最优步长下的单步下降量约为 $$a_d^2 \lVert s \rVert^4 / (L\sigma^2)$$；对 SGD，则约为 $$\lVert s \rVert^4 / (Ld\sigma^2)$$。由于 $$a_d^2 \sim 1/d$$，两者同阶。

因此在这个最简单的模型里，归一化主要改变的是稳定性权衡：可用学习率区间更宽，但不会自动带来更大的单步进展。真正拉开差距的地方，是后面的多块异方差。

### 4.2 局部收敛：稳态误差地板

再看局部二次模型：

$$
F(x) = \frac{1}{2} \sum_{r=1}^m \lambda_r \lVert x_r \rVert_2^2
$$

此时 $$s_r = \lambda_r x_r$$。

对归一化更新，第 $$r$$ 块的近似动力学是

$$
x_{r, t+1} = \left( 1 - \eta \frac{a_{d_r} \lambda_r}{\sigma_r} \right) x_{r, t} - \eta \zeta_{r, t}
$$

对均方误差取期望，并用 $$\mathbb{E}[\zeta_{r,t}] = 0$$ 消去交叉项：

$$
\mathbb{E}\lVert x_{r, t+1} \rVert_2^2 \approx \left( 1 - 2\eta \frac{a_{d_r} \lambda_r}{\sigma_r} \right) \mathbb{E}\lVert x_{r, t} \rVert_2^2 + \eta^2
$$

稳态时解得

$$
\mathbb{E}\lVert x_{r, \infty} \rVert_2^2 \approx \frac{\eta^2}{2\eta \frac{a_{d_r} \lambda_r}{\sigma_r}} = \frac{\eta \sigma_r}{2 a_{d_r} \lambda_r} \sim \mathcal{O}\left(\frac{\eta \sigma_r \sqrt{d_r}}{\lambda_r}\right)
$$

SGD 的递推为 $$x_{r, t+1} = (1 - \eta \lambda_r) x_{r, t} - \eta \xi_{r, t}$$，对应稳态方差

$$
\mathbb{E}\lVert x_{r, \infty} \rVert_2^2 \approx \frac{\eta d_r \sigma_r^2}{2\lambda_r} \sim \mathcal{O}\left(\frac{\eta d_r \sigma_r^2}{\lambda_r}\right)
$$

这里的取舍很清楚。局部收缩变慢了，因为收缩率从 $$\eta \lambda_r$$ 变成 $$\eta a_{d_r} \lambda_r / \sigma_r$$。噪声越大，拉回越慢。另一方面，稳态误差从 $$\sigma_r^2$$ 级别降到 $$\sigma_r$$ 级别，维度因子也变弱了。

这仍然不是方向估计的改进。它只是控制随机更新的幅值，因此长期波动不会像 SGD 那样随原始噪声方差膨胀。

## 5. 从理论到实践：异方差与隐式块自适应学习率

### 5.1 为什么必须看块间噪声差异

如果只看单块同方差模型，归一化和 SGD 的最优单步下降同阶。那它在 LLM 训练里为什么还会有价值？

因为真实模型不是一个块，也不是同方差。不同参数块的噪声水平可能差很多。例如 embedding 层因为 token 访问稀疏，梯度方差可能很大；一些中间层的信号更密，噪声尺度又是另一种状态。用前面的记号说，$$\sigma_r$$ 可以在不同块之间差很多。

另一方面，训练管线通常仍然使用一个全局学习率 $$\eta$$。当然可以做 layer-wise learning rate，但主流程里最常见的仍然是一个共享 schedule。于是全局学习率必须同时照顾所有参数块。

这两个事实放在一起，问题就变得具体了：如果不同块的噪声水平差异很大，一个全局 $$\eta$$ 应该怎么选？SGD 的学习率会受最高噪声块约束；归一化则会自动给不同块不同的有效步长。

### 5.2 归一化等价于随机块学习率

对 blockwise $$L_2$$ 归一化，第 $$r$$ 块更新可以直接改写为

$$
x_r^+ = x_r - \eta \frac{g_r}{\lVert g_r \rVert_2} = x_r - \left(\frac{\eta}{\lVert g_r \rVert_2}\right) g_r
$$

也就是说，它等价于用随机有效步长

$$
\eta_r^{\text{eff}} = \frac{\eta}{\lVert g_r \rVert_2}
$$

来做 SGD。在高噪声区，$$\lVert g_r \rVert_2 \approx \sigma_r \sqrt{d_r}$$，所以

$$
\eta_r^{\text{eff}} \approx \frac{\eta}{\sigma_r \sqrt{d_r}}
$$

这就是隐式的逆噪声尺度加权。高噪声块得到更小的有效步长；低噪声块得到更大的有效步长。

这和第 3 节的幅值饱和是同一现象的两种写法。幅值饱和看输出更新的尺度；随机块学习率把归一化更新重新写回 SGD 的形式。

### 5.3 全局学习率为什么受最高噪声块约束

设最大噪声标准差为 $$\sigma_{\max} := \max_r \sigma_r$$。如果希望每个参数块的稳态均方误差 $$\mathbb{E}\lVert x_{r,\infty} \rVert_2^2$$ 都不超过阈值 $$\varepsilon$$，全局学习率必须满足最严格的块约束。

对 SGD，根据第 4.2 节的稳态公式，需要

$$
\frac{\eta d_r \sigma_r^2}{2\lambda_r} \lesssim \varepsilon
$$

对所有块成立。因此

$$
\eta \lesssim \min_r \frac{2\lambda_r \varepsilon}{d_r \sigma_r^2} \propto \frac{1}{\sigma_{\max}^2}
$$

一旦学习率受最高噪声块约束，所有块的收缩率都变成 $$\eta \lambda_r$$。低噪声块本来可以用更大的步长，但共享学习率会让它们一起变慢。

对归一化更新，稳态条件变成

$$
\frac{\eta \sigma_r}{2 a_{d_r} \lambda_r} \lesssim \varepsilon
$$

因此

$$
\eta \lesssim \min_r \frac{2 a_{d_r} \lambda_r \varepsilon}{\sigma_r} \propto \frac{1}{\sigma_{\max}}
$$

只看全局上界，最坏噪声块造成的压制已经从 $$1/\sigma_{\max}^2$$ 缓和到 $$1/\sigma_{\max}$$。

还要注意，各块的有效收缩率不是同一个数。第 $$r$$ 块为

$$
\text{rate}^{\text{norm}}_r = \eta \frac{a_{d_r} \lambda_r}{\sigma_r}
$$

它随 $$1/\sigma_r$$ 变化。低噪声块收缩快，高噪声块收缩慢。更准确地说，因为 $$\sigma_r$$ 是标准差，所以这是 inverse-noise-scale weighting，而不是严格的 inverse-variance weighting。

### 5.4 对 LLM 训练的实际意义

把上面的模型翻译回 LLM 训练，我会这样理解：

1. 如果某些层或参数块的梯度方差特别高，行归一化/块归一化会自动降低它们的有效步长。比如 embedding 相关参数不一定完全依赖人工调一个更小的 learning rate。

2. 在噪声主导阶段，SGD 类更新的稳定学习率会受高噪声块限制，否则容易出现 loss spike。归一化更新的稳定条件随 $$1/\sigma$$ 而不是 $$1/\sigma^2$$ 收缩，所以可用区间更宽。

3. 混合精度训练会引入额外量化噪声，也可能放大不同参数块之间的噪声差异。归一化方法对这类异方差更不敏感。

这些说法都依赖本文的简化模型，不能替代真实大模型实验。但它至少给了一个解释方向：归一化优化器有价值的地方，可能不在于更精确的方向估计，而在于它改变了不同噪声尺度参数块的有效学习率。

## 6. 小结

在块内噪声近似各向同性的模型下，blockwise $$L_2$$ 归一化可以理解为

$$
\text{SGD with } \eta_r^{\text{eff}} = \frac{\eta}{\lVert g_r \rVert}
$$

它不提高单步方向精度，也不增加信息量。它做的是把更新幅值限制在固定尺度内，因此原始梯度里的噪声幅值不会直接传到参数更新里。

零次齐次性让输出幅值不随输入噪声尺度增长。代价也很明确：有效漂移会随 $$1/\sigma_r$$ 变弱。

在单块同方差模型里，稳态误差从 $$\mathcal{O}(\sigma^2)$$ 降到 $$\mathcal{O}(\sigma)$$，最大稳定步长从 $$\mathcal{O}(1/\sigma^2)$$ 放宽到 $$\mathcal{O}(1/\sigma)$$。不过最优步长下的单步下降仍然同阶，不能把它说成免费加速。

到了多块异方差模型里，归一化会给每个块自动分配 $$\eta_r^{\text{eff}} \propto 1/\sigma_r$$ 的有效步长，避免全局学习率完全被高噪声块支配。

所以，在噪声主导的 training dynamics 里，我更愿意把归一化更新理解成一种幅值控制。它没有让方向估计更准，但让更新幅值更稳定。这个性质在单块模型里表现为稳定性权衡，在 LLM 的多块异方差结构里则更像一种可用的自适应机制。

## 7. 数值验证

下面用 5 组 Monte Carlo 实验检查前面的尺度关系。

记号如下：

- 噪声标准差：$$\sigma$$；
- 更新向量：SGD 取 $$u=g$$，归一化取 $$u=g/\lVert g \rVert_2$$；
- 漂移强度：$$\lvert \mathbb{E}\langle u,s \rangle \rvert$$；
- 噪声强度：$$\operatorname{tr}(\operatorname{Cov}(u))$$；
- 最大稳定学习率：在局部二次模型 $$F(x)=\frac{1}{2}\lVert x \rVert_2^2$$ 下，由单步期望下降条件估计得到的 $$\eta_{\max}$$。

### 7.1 方向不变性（归一化不修正单步方向）

{% include figure.liquid
  path='assets/img/post-04-14/01_direction_invariance.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='方向不变性验证'
  zoomable=true
  alt='方向不变性验证'
%}

图里有两个子图。

左图（Direction Cosine Is Preserved）把 $$\cos(g,s)$$ 放在横轴，把 $$\cos(g/\lVert g \rVert,s)$$ 放在纵轴。如果归一化不改变单步方向，散点应该贴着对角线 $$y=x$$。

右图（Numerical Difference Distribution）画的是 $$\Delta:=\lvert \cos(g,s)-\cos(g/\lVert g \rVert,s) \rvert$$ 的分布，纵轴用对数坐标。如果方向保持，$$\Delta$$ 应该集中在 0 附近。

结果里散点基本贴在 $$y=x$$ 上，且 $$\max \Delta=1.11\times10^{-16}$$。这和第 2 节一致：对 $$L_2$$ 归一化，方向没有被修正。

### 7.2 幅值饱和与漂移缩放

{% include figure.liquid
  path='assets/img/post-04-14/02_noise_compression.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='幅值饱和与漂移缩放'
  zoomable=true
  alt='幅值饱和与漂移缩放'
%}

这张图也有两个子图。

左图（Drift Scaling in Noise-Dominated Regime）看漂移项。横轴是 $$\sigma$$，纵轴是 $$\lvert \mathbb{E}\langle u,s \rangle \rvert$$，都是对数坐标。这里主要检查曲线是否接近 $$\sigma^{-1}$$。

右图（Noise Amplification vs Noise Compression）看噪声强度。横轴是 $$\sigma$$，纵轴是 $$\operatorname{tr}(\operatorname{Cov}(u))$$。SGD 应该接近 $$\sigma^2$$，归一化则应该接近水平线。

结果比较干净。SGD 的噪声协方差迹随 $$\sigma^2$$ 增长，斜率约为 $$2.00$$；归一化更新的协方差迹近似常数，斜率约为 $$0.00$$。同时，归一化漂移项随 $$1/\sigma$$ 衰减，斜率约为 $$-0.99$$。这和第 3 节的说法一致：信号响应变弱，但输出噪声幅值也被限制住。

### 7.3 最大稳定步长缩放

{% include figure.liquid
  path='assets/img/post-04-14/03_eta_scaling.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='最大稳定步长缩放'
  zoomable=true
  alt='最大稳定步长缩放'
%}

这张图检查第 4.1 节的学习率尺度。

横轴是 $$\sigma$$，纵轴是估计的 $$\eta_{\max}$$，两者都是对数坐标。圆点是经验估计，虚线和点线是理论曲线与参考幂律（$$\sigma^{-2}$$、$$\sigma^{-1}$$）。

读斜率即可。SGD 约为 $$-1.99$$，对应 $$\eta_{\max}\propto 1/\sigma^2$$；归一化约为 $$-0.99$$，对应 $$\eta_{\max}\propto 1/\sigma$$。

### 7.4 稳态误差地板缩放

{% include figure.liquid
  path='assets/img/post-04-14/04_steady_state_scaling.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='稳态误差地板缩放'
  zoomable=true
  alt='稳态误差地板缩放'
%}

这张图对应第 4.2 节的稳态误差地板。

横轴是 $$\sigma$$，纵轴是稳态 $$\mathbb{E}\lVert x \rVert_2^2$$，两者都是对数坐标。两条参考虚线分别对应 $$\sigma^2$$ 和 $$\sigma$$ 标度。

曲线与参考线的平行关系符合推导：SGD 约为 $$\mathcal{O}(\sigma^2)$$，斜率约 $$2.00$$；归一化约为 $$\mathcal{O}(\sigma)$$，斜率约 $$1.00$$。

### 7.5 异方差与隐式块自适应学习率

{% include figure.liquid
  path='assets/img/post-04-14/05_heteroscedastic_blocks.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='异方差与隐式块自适应学习率'
  zoomable=true
  alt='异方差与隐式块自适应学习率'
%}

这张图检查异方差场景下的块自适应效果。

左图（Global eta Under Heteroscedastic Noise）画块级误差 $$\mathbb{E}\lVert x_r \rVert_2^2$$ 随迭代步数的变化，纵轴是对数坐标。四条曲线分别对应 SGD/归一化在低噪声块与高噪声块上的轨迹。

右图（Implicit Inverse-Variance Weighting）画有效收缩系数 $$\kappa_r:=\mathbb{E}\langle u_r,s_r \rangle/\lVert s_r \rVert_2^2$$。横轴是块类别，分成低噪声和高噪声。

实验设定为 $$\sigma_{\text{low}}=0.5,\ \sigma_{\text{high}}=4.0$$。归一化下，低噪声块的有效收缩系数约为高噪声块的 $$7.49$$ 倍，接近标准差之比 $$\sigma_{\text{high}}/\sigma_{\text{low}} = 8$$。这支持第 5 节的解释：这里看到的是逆噪声尺度加权，不是严格的逆方差加权。

这 5 组实验分别检查了方向不变性、漂移缩放、协方差饱和、最大稳定步长、稳态误差地板以及异方差下的块自适应。它们没有证明真实 LLM 训练一定如此，但至少说明上面的简化模型在数值上是自洽的。

## 参考文献

[1] Shazeer, N., & Stern, M. (2018). [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://proceedings.mlr.press/v80/shazeer18a.html). In _Proceedings of the 35th International Conference on Machine Learning (ICML 2018)_, _Proceedings of Machine Learning Research_, 80, 4596-4604.

[2] Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., & Bernstein, J. (2024). [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/).

[3] Deng, S., Ouyang, Z., Pang, T., Liu, Z., Jin, R., Yu, S., & Yang, Y. (2026). [RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](https://arxiv.org/abs/2603.20527). _arXiv preprint_ arXiv:2603.20527.

[4] Gu, Y., & Xie, Z. (2026). [Mano: Restriking Manifold Optimization for LLM Training](https://arxiv.org/abs/2601.23000). _arXiv preprint_ arXiv:2601.23000.

[5] Wang, M., Wang, J., Zhang, J., Wang, W., Pei, P., Cai, X., E, W., & Wu, L. (2025). [GradPower: Powering Gradients for Faster Language Model Pre-Training](https://arxiv.org/abs/2505.24275). _arXiv preprint_ arXiv:2505.24275.

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
