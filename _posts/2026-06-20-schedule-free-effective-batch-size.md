---
layout: post
title: "DASF：一种闭环的 batch size schedule-free 方法"
date: 2026-06-20 12:00:00
description: "本文提出 DASF（Drift-Aware Schedule-Free）：基于 Schedule-Free 的对偶（迭代平均↔学习率调度、梯度平均↔batch size 调度），用就地测得的梯度统计在线设定有效 batch size，无 schedule、无调参，省去为标定 batch 而训练代理模型、拟合 scaling law 的开销。真实 transformer 上匹配或超过经调参的基线，并给出「compute-optimal 下最优有效 batch 近似恒定、非 √t 增长」的可证伪负结果。"
tags: [optimization, deep-learning, llm, schedule-free, batch-size]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

## 动机

在 LLM 预训练中设定 batch size 及其 schedule，通常依赖大范围超参搜索，或训练一系列小规模代理模型、拟合 critical batch size（CBS）的 scaling law 后外推到目标规模。这类做法成本高，且外推未必可靠：[Merrill 等](https://arxiv.org/abs/2505.23971)发现，由小模型外推的梯度噪声尺度在 7B 规模上与 CBS 的实际趋势并不一致。

因此，降低上述成本的关键不在于方法是否"自适应"，而在于直接利用目标训练自身就地测量的梯度统计量在线设定 batch size，避免跨规模外推。这也是方法可即插即用的前提。

上一篇 [《为什么 LLM pretrain 过程中途要把 batch size 翻倍》](https://jiaxuanzou0714.github.io/blog/2026/why-double-batch-size-llm-pretraining/) 调节的是**物理 batch size**：用变分法导出截断幂律形式的最优调度，但其最优切换点依赖一个实践中难以获得的指数。本文转而固定物理 batch size，利用 [Schedule-Free](https://arxiv.org/abs/2405.15682)（SF）内在的平均机制在线设定**有效 batch size**。

采用 SF 的依据是如下对偶关系：SF 对迭代点做平均，由此获得隐式的学习率衰减；与之对偶，对梯度做平均可获得隐式的有效 batch size 增大，因为对梯度的方差缩减等价于增大有效 batch size。

> ##### 核心对偶
> **迭代平均 ↔ 学习率调度**<br>**梯度平均 ↔ batch size 调度**
{: .block-tip}

二者都无需预设调度，因为平均窗口随 $$t$$ 自动变宽（$$c_t=1/t$$），无需预先知道训练总步数。本文据此提出 **DASF（Drift-Aware Schedule-Free）**：把这条对偶落到算法上——用就地测得的梯度漂移与噪声闭环设定 SF 的插值系数 $$\beta_t$$（即隐式有效 batch size），使有效 batch size 全程 schedule-free、无调参。下文沿这一对偶关系展开。

## 1. SF 与其动量形式

时变 SF 由三条序列定义（$$\Delta_t$$ 为评估点 $$y_t$$ 处的随机梯度，$$c_t=1/t$$）：

$$
\begin{aligned}
x_t&=(1-c_t)\,x_{t-1}+c_t\,z_t,\\
y_t&=(1-\beta_t)\,z_t+\beta_t\,x_t,\\
z_{t+1}&=z_t-\eta_t\,\Delta_t,
\end{aligned}
$$

其中 $$z$$ 为基础优化器序列，$$x$$ 为 $$z$$ 的 Polyak–Ruppert 平均、即推理/部署时使用的模型参数，$$y$$ 为唯一计算梯度的点（训练时前向所在的点）。将其改写为动量形式（[Through the River](https://arxiv.org/abs/2507.09846) §4.4），令 $$m_t:=(x_t-z_{t+1})/\gamma$$：

$$
\begin{aligned}
m_t&=(1-c_t)\,m_{t-1}+\Delta_t,\\
y_{t+1}&=y_t-\gamma\big(\beta\,c_{t+1}\,m_t+(1-\beta)\,\Delta_t\big).
\end{aligned}
$$

$$m_t$$ 是一个窗口随 $$t$$ 变宽的梯度累积量，即"动机"中对偶关系里"梯度平均"所对应的对象，它已内含于 SF，无需额外引入。每步实际使用的梯度是累积量 $$m_t$$（陈旧、低方差）与新鲜梯度 $$\Delta_t$$（满方差）的凸组合，$$\beta$$ 为组合权重。下一节在 NQM 中分析 $$m_t$$，以确定 $$\beta$$ 究竟提供多大的有效 batch size。

## 2. NQM 分析：四个结论

取噪声二次模型（NQM），沿 Hessian 的特征方向逐坐标分析（标量）：曲率 $$h$$，真梯度 $$g(\theta)=h\theta$$，单样本随机梯度 $$\Delta=h\theta+\epsilon$$，$$\epsilon\sim\mathcal N(0,\sigma^2)$$。物理 batch size $$B$$ 将噪声方差缩小为 $$\sigma^2/B$$，据此给出有效 batch size 的操作定义：噪声方差被压低的倍数。分析对象是 §1 的累积量

$$
m_t=(1-c_t)\,m_{t-1}+\Delta_t=\sum_{s\le t}w_{t,s}\,\Delta_s,\qquad w_{t,s}=\prod_{r=s+1}^{t}(1-c_r).
$$

**结论 1（读出端的有效 batch size 线性增长）.** 当 $$c_t=1/t$$，权重逐项相消，化简为 $$w_{t,s}=\prod_{r=s+1}^t\frac{r-1}{r}=\frac{s}{t}$$。归一化累积梯度（对信号无偏）的有效样本量由 Kish 公式给出

$$
B_{\text{eff}}(t)=\frac{\big(\sum_s w_{t,s}\big)^2}{\sum_s w_{t,s}^2}=\frac{(t/2)^2}{t/3}=\frac{3}{4}\,t,
$$

即用于推理的平均序列 $$x$$ 所聚合的梯度有效样本量约为 $$3t/4$$，随 $$t$$ 线性增长。就"SF 是否隐式进行 batch warmup"而言，读出端的回答是肯定的。

**结论 2（优化端的有效 batch size 被 $$\beta$$ 限制为常数）.** 基础步 $$z_{t+1}=z_t-\eta\Delta_t$$ 使用单个新鲜梯度，$$z$$ 本身无方差缩减，方差缩减只存在于平均序列 $$x$$。但真正决定损失下降的既非 $$z$$ 也非 $$x$$，而是计算梯度的评估点序列 $$y$$。其等价更新为 $$y_{t+1}=y_t-\gamma\,G^{\text{eff}}_t$$，每步实际使用的有效梯度是累积量 $$m_t$$ 与新鲜梯度 $$\Delta_t$$ 的凸组合：

$$
G^{\text{eff}}_t=\beta\,c_{t+1}\,m_t+(1-\beta)\,\Delta_t.
$$

下面计算 $$G^{\text{eff}}_t$$ 的噪声。设单样本梯度 $$\Delta_s=h\theta_s+\epsilon_s$$，其中噪声 $$\epsilon_s$$ 相互独立、方差均为 $$\sigma^2$$；累积量按结论 1 展开为 $$m_t=\sum_{s\le t}w_{t,s}\Delta_s$$、$$w_{t,s}=s/t$$。代入后，$$G^{\text{eff}}_t$$ 的噪声部分 $$\xi_t$$ 是这些独立噪声 $$\{\epsilon_s\}$$ 的线性组合：

$$
\xi_t=\beta c_{t+1}\sum_{s\le t}\tfrac{s}{t}\,\epsilon_s\;+\;(1-\beta)\,\epsilon_t\;=\;\sum_{s\le t}a_s\,\epsilon_s,
$$

两项分别来自累积量 $$m_t$$ 与新鲜梯度 $$\Delta_t$$。这里 $$a_s$$ 指噪声 $$\epsilon_s$$ 进入 $$\xi_t$$ 时所乘的权重，它分两种情形：当前噪声 $$\epsilon_t$$ 同时出现在两项中（在 $$m_t$$ 内权重 $$w_{t,t}=1$$，在新鲜项内权重 $$1-\beta$$），而过去噪声 $$\epsilon_s\,(s<t)$$ 只经由 $$m_t$$ 进入，故

$$
a_t=(1-\beta)+\beta c_{t+1},\qquad a_s=\beta c_{t+1}\,\tfrac{s}{t}\ \ (s<t).
$$

由独立性，方差等于系数平方之和 $$\mathrm{Var}(\xi_t)=\sigma^2\sum_{s}a_s^2$$，即

$$
\frac{\mathrm{Var}(\xi_t)}{\sigma^2}=\big[(1-\beta)+\beta c_{t+1}\big]^2+\beta^2 c_{t+1}^2\sum_{s<t}\Big(\frac{s}{t}\Big)^2\ \xrightarrow{\,t\to\infty\,}\ (1-\beta)^2.
$$

取极限的依据：$$c_{t+1}\sim1/t\to0$$，累积量部分 $$\big(\sim\beta^2c_{t+1}^2\cdot t/3\big)$$ 趋于零，但新鲜项每步注入一份比例为 $$(1-\beta)$$、未经缩减的单样本噪声，构成不随 $$t$$ 衰减的方差下界。因此 $$y$$ 轨迹的实际有效 batch size $$B_{\text{eff}}^{\text{realized}}=\sigma^2/\mathrm{Var}(\xi_t)\to 1/(1-\beta)^2$$ 为常数（$$\beta=0.9$$ 对应 $$100$$，$$\beta=0.98$$ 对应 $$2500$$）。即恒定 $$\beta$$ 的 vanilla SF 不向优化器提供增长的 batch size：结论 1 的方差缩减作用于仅用于推理的 $$x$$，而优化轨迹 $$y$$ 被 $$\beta$$ 限制为常数。该关系在 SGD 下成立；AdamW 预条件下 $$B_{\text{eff}}$$ 为由 $$\beta$$ 导出的*名义*有效 batch size，下文报告的均为此值。

> ##### 时变 β 下的修正
> §1 的动量重写取 $$\beta$$ 恒定。对一般时变 $$\beta_t$$，直接展开 $$y_{t+1}=(1-\beta_{t+1})z_{t+1}+\beta_{t+1}x_{t+1}$$（代入 $$x_{t+1}=(1-c_{t+1})x_t+c_{t+1}z_{t+1}$$、$$z_{t+1}=z_t-\gamma\Delta_t$$）得
>
> $$
> y_{t+1}=y_t-\gamma\big[(1-\beta_t)\Delta_t+(\beta_t-\beta_{t+1}+\beta_{t+1}c_{t+1})\,m_t\big],
> $$
>
> 仅当 $$\beta_{t+1}=\beta_t=\beta$$ 时才退化为前式 $$G^{\text{eff}}_t=(1-\beta)\Delta_t+\beta c_{t+1}m_t$$。因此本文 $$B_{\text{eff}}=1/(1-\beta_t)^2$$ 在时变 $$\beta_t$$ 下是 quasi-static 近似，要求 $$\lvert\beta_{t+1}-\beta_t\rvert\ll c_{t+1}$$，即 $$\beta_t$$ 的变化慢于平均速率。DASF 的 $$\beta_t$$ 由 EMA 平滑的统计量驱动、变化平缓，基本满足此条件；若 $$\beta_t$$ 快速跳变，多出的 $$(\beta_t-\beta_{t+1})\,m_t$$ 项会改变累积梯度项的符号与方差结构，须另行处理。
{: .block-tip}

由结论 2，要使优化轨迹也获得 batch size 增长，必须令 $$\beta_t\to1$$；问题因此归结为 $$\beta$$ 的调度。同时 $$\beta$$ 承担双重作用：它既决定有效 batch size，又决定稳定阈值（[Through the River](https://arxiv.org/abs/2507.09846) 为解耦二者引入参数 $$C$$）。

**结论 3（$$\beta_t\to1$$ 解锁增长，$$\rho$$ 即增长指数）.** 将实际有效 batch size 对准目标 $$B^\ast(t)$$，得 $$\beta_t=1-1/\sqrt{B^\ast(t)}$$。[AMUSE](https://arxiv.org/abs/2605.22432)（Kim et al. 2026）令 $$1-\beta_t$$ 渐近按 $$t^{-\rho}$$ 衰减（其 Eq.5 实现含 warmup-hold，仅在大 $$t$$ 下为此幂律，见实验一节），故大 $$t$$ 下 $$B_{\text{eff}}^{\text{realized}}(t)\propto t^{2\rho}$$。即 AMUSE 用于保证稳定性的指数 $$\rho$$ 就是隐式 batch size 的增长指数；$$\rho=1/4$$ 给出 $$B_{\text{eff}}\propto\sqrt t$$，与 [Merrill 等](https://arxiv.org/abs/2505.23971)测得的 CBS 标度一致。AMUSE 将 $$\beta_t\uparrow$$ 解释为评估点由快序列 $$z$$ 移向平均序列 $$x$$ 以抑制振荡，而 batch size 视角下同一操作即逐步关闭新鲜噪声注入、使累积量带来的增长得以体现；二者等价，并赋予 $$\rho$$ 可证伪的含义。

**结论 4（陈旧性给出立方根上限）.** $$y$$ 轨迹用一个对过去约 $$W=B_{\text{eff}}$$ 个梯度的窗口平均，来估计当前真梯度 $$G(y_t)$$。但它不是同尺寸的 i.i.d. batch，而是沿移动轨迹的时间平均，故加宽窗口需权衡两个方向相反的误差。

其一，方差。平均 $$W$$ 个梯度样本，噪声按大数定律降低 $$W$$ 倍：

$$
V(W)\approx\frac{\sigma^2}{W}\qquad(\text{窗越宽越小}).
$$

其二，陈旧偏置。窗口内的梯度算在过去的点 $$y_s\,(s<t)$$ 上，而迭代持续移动。设梯度随步数以速率 $$\dot g$$ 漂移（NQM 中 $$\dot g=h\dot y$$，$$h$$ 为曲率、$$\dot y$$ 为迭代速度），窗口的权重质心滞后当前步约 $$W/2$$ 步（$$c_t=1/t$$ 时质心在 $$2t/3$$、滞后 $$t/3$$，同阶），故平均梯度系统偏离当前真梯度：

$$
\text{bias}\approx\dot g\cdot\frac{W}{2}\ \Longrightarrow\ \text{bias}^2\approx\frac{\dot g^2 W^2}{4}\qquad(\text{窗越宽越大}).
$$

两项相加为均方误差；对 $$W$$ 求导并令其为零，即得最优窗 $$W^\ast$$：

$$
\begin{aligned}
\mathrm{MSE}(W)&=\frac{\sigma^2}{W}+\frac{\dot g^2 W^2}{4},\\
\frac{d\,\mathrm{MSE}}{dW}&=-\frac{\sigma^2}{W^2}+\frac{\dot g^2 W}{2}=0
\ \Longrightarrow\ W^3=\frac{2\sigma^2}{\dot g^2}
\ \Longrightarrow\ W^\ast=\Big(\frac{2\sigma^2}{\dot g^2}\Big)^{1/3}.
\end{aligned}
$$

有用累积窗宽（即有用有效 batch size 的上限）由梯度噪声 $$\sigma^2$$ 与梯度漂移率平方 $$\dot g^2$$ 之比的立方根决定。其结构类似噪声尺度 $$\sigma^2/g^2$$，但分母是梯度的变化率 $$\dot g$$ 而非梯度本身，因为陈旧性取决于梯度变化的快慢而非其大小；两个量都可就地测量。而 $$c_t=1/t$$ 给出线性 $$W\sim t$$，远超此立方根最优，故 vanilla SF 在读出端过度累积，$$B_{\text{eff}}\propto t$$ 在后期将超过 CBS。

综合四条结论：SF 的 $$1/t$$ 平均在读出端自带线性增长的有效 batch size（结论 1），但优化端被 $$\beta$$ 限制为常数（结论 2）；解放优化端只能令 $$\beta_t\to1$$，这正是 AMUSE 所做，其 $$\rho$$ 即增长指数、$$\rho=1/4$$ 复现 Merrill 的 $$\sqrt t$$（结论 3）；而该增长存在由陈旧性决定的立方根上限（结论 4）。因此无需引入新的优化器：合适的 schedule-free batch size 方法是保留 AMUSE 的结构，仅将 $$\beta_t$$ 由开环的 $$t^{-\rho}$$ 改为由就地测量的梯度统计量闭环驱动——以结论 4 的 $$W^\ast$$ 为目标、经结论 3 的换算设定 $$\beta_t$$，即下节的 DASF。

> ##### 分析所用近似
> 以上分析基于若干近似：NQM（二次、加性常方差噪声）、逐特征方向（跨谱聚合需按谱加权）、漂移的局部线性化，以及读取 $$B_{\text{eff}}$$ 时对 $$c_t$$ 的准静态处理。这些只影响常数，不影响标度（$$t^{2\rho}$$、立方根等指数稳定）。
{: .block-tip}

## 3. DASF：drift-aware 闭环控制器

由结论 3–4，控制器的形式即确定：以结论 4 的 $$W^\ast$$ 为目标有效 batch size，经结论 3 的 $$\beta_t=1-1/\sqrt{B^\ast}$$ 换算。将立方根定律由逐坐标形式聚合（$$\sigma^2\to\operatorname{tr}\Sigma$$，$$\dot g^2\to\lVert\dot G\rVert^2$$）并代入，得设定点

$$
1-\beta_t=\Big(\frac{\lVert\dot G\rVert^2}{2\,\operatorname{tr}\Sigma}\Big)^{1/6}.
$$

指数 $$1/6=\tfrac13\times\tfrac12$$：$$\tfrac13$$ 来自结论 4 的 bias–variance 最优窗口，$$\tfrac12$$ 来自结论 2–3 的 $$B_{\text{eff}}=1/(1-\beta)^2$$ 换算。其中 $$\operatorname{tr}\Sigma$$、$$\lVert G\rVert^2$$、$$\lVert\dot G\rVert^2$$ 三个量全部在训练中就地估计、不碰数据管线（方法见下）。与 AMUSE 相比，本控制器省去了三处人工选择——启发式的 warmup-hold、经验指数 $$\rho$$、初始 $$\beta_1$$：DASF 的 $$\beta_t$$ 整条轨迹（含早期的小 batch 段）由就地测得的 $$\dot G$$ 与 $$\operatorname{tr}\Sigma$$ 逐点定出，而非预设曲线——这正是它相对 AMUSE 的关键简化。

设定点取决于梯度漂移 $$\dot G$$ 而非梯度大小 $$G$$，这是本设计的核心。若直接以噪声尺度 $$\operatorname{tr}\Sigma/\lVert G\rVert^2$$ 作为设定点，则随信号收敛、$$\lVert G\rVert\to0$$，噪声尺度发散，$$\beta$$ 被推高至 $$0.99$$ 以上、有效 batch size 增至上万，导致过度累积乃至发散。梯度变小并不意味着进入噪声主导区。drift-aware 设定点以梯度漂移率为依据：信号快速下降时 $$\dot G$$ 较大，故 batch size 保持较小，避免过度滞后。这与 [Merrill](https://arxiv.org/abs/2505.23971) 关于噪声尺度不可靠的结论在机制上一致。

**就地估计.** DASF 只需两个量：噪声 $$\operatorname{tr}\Sigma$$ 与漂移 $$\lVert\dot G\rVert^2$$——它并不需要梯度大小 $$\lVert G\rVert^2$$。$$\operatorname{tr}\Sigma$$ 用 [McCandlish](https://arxiv.org/abs/1812.06162) 两点法：把一个物理 batch 拆成大小 $$B_{\text{small}}$$ 的 micro-batch，由单 micro 范数平方均值 $$\overline{\lVert g_{\text{small}}\rVert^2}$$ 与整批范数平方 $$\lVert G_{\text{big}}\rVert^2$$、据 $$\mathbb E\lVert g_B\rVert^2=\lVert G\rVert^2+\operatorname{tr}\Sigma/B$$ 消去 $$\lVert G\rVert^2$$，得

$$
\operatorname{tr}\Sigma=\frac{\overline{\lVert g_{\text{small}}\rVert^2}-\lVert G_{\text{big}}\rVert^2}{1/B_{\text{small}}-1/B_{\text{big}}}.
$$

漂移 $$\lVert\dot G\rVert^2$$ 取相邻 $$k$$ 步整批梯度之差 $$\lVert G_t-G_{t-k}\rVert^2$$，再减去噪声地板 $$2\operatorname{tr}\Sigma/B_{\text{big}}$$（两个独立含噪梯度之差的噪声方差）而得。两量各自做 EMA 以压噪；探针只把同一物理 batch 拆小来读，不额外消耗 token。

需强调：要估计 $$\lVert G\rVert^2$$ 的是作为对比的噪声尺度设定点 $$\operatorname{tr}\Sigma/\lVert G\rVert^2$$，而非 DASF。且 $$\lVert G\rVert^2$$ 必须取两点无偏解 $$\tfrac{B_{\text{big}}\lVert G_{\text{big}}\rVert^2-B_{\text{small}}\overline{\lVert g_{\text{small}}\rVert^2}}{B_{\text{big}}-B_{\text{small}}}$$——直接用 $$\lVert G_{\text{big}}\rVert^2$$ 会偏高 $$\operatorname{tr}\Sigma/B_{\text{big}}$$，该偏差在低信号后期（正是噪声尺度方案工作的区间）最大。DASF 绕开了这个最难估、最不可靠的量。

> ##### 公式中的常数
> 指数 $$1/6$$ 由理论确定；公式中的常数（系数 2 及略去的窗形、物理 batch size 等归一化因子）只平移 $$\beta$$ 的绝对水平、不改变函数形式。又因 $$1/6$$ 次幂对尺度高度不敏感，直接代入实测量即可使 $$\beta$$ 落入合适区间。
{: .block-tip}

## 4. 实验

设置与上一篇严格对齐：标准 GPT（FineWeb，GPT-2 tokenizer），SF-AdamW，learning rate 全程恒定，相同的初始化与数据顺序。唯一变量为 $$\beta_t$$。对比四种 $$\beta_t$$ 调度（均为固定物理 batch、vanilla SF），各自的 $$1-\beta_t$$ 与（由结论 2 换算的）隐含有效 batch size $$B_{\text{eff}}=1/(1-\beta_t)^2$$ 为：

- 常数 $$\beta$$：$$1-\beta_t$$ 与 $$B_{\text{eff}}$$ 均为常数。
- AMUSE（开环）：$$1-\beta_t\propto t^{-\rho}$$，故 $$B_{\text{eff}}\propto t^{2\rho}$$。
- 闭环噪声尺度：$$1-\beta_t=\sqrt{\lVert G\rVert^2/\operatorname{tr}\Sigma}$$，故 $$B_{\text{eff}}=\operatorname{tr}\Sigma/\lVert G\rVert^2$$（即噪声尺度本身）。
- 闭环 drift-aware：$$1-\beta_t=(\lVert\dot G\rVert^2/2\operatorname{tr}\Sigma)^{1/6}$$，故 $$B_{\text{eff}}=(2\operatorname{tr}\Sigma/\lVert\dot G\rVert^2)^{1/3}$$（即结论 4 的 $$W^\ast$$）。

前两者为基线（常数、开环幂律），后两者为本文的闭环设定点。其中 AMUSE 按其 Eq.5 实现——warmup 内持 $$\beta_1$$，其后 $$1-\beta_t=(1-\beta_1)\big(\tfrac{T_0-1}{t-1}\big)^\rho$$（$$T_0$$ 为 warmup 步数，渐近 $$\propto t^{-\rho}$$）——且为充分调参的强基线：$$\rho$$ 与 $$\beta_1$$ 经网格搜索取最优（两规模均为 $$\rho=0.6$$、$$\beta_1=0.4$$），故对比的是其最优调参版本，而 drift-aware 全程无调参。两个规模：45M / 600M 与 117M / 2.5B tokens。

{% include figure.liquid
  path='assets/img/post-06-20/bsf_compare.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='45M / 600M tokens。四面板：$B_\text{eff}$（左上）、val loss（右上）、$\beta_t$（左下）、实测噪声尺度（右下）。drift-aware（红，即 DASF，无调参）val loss 最低，略优于手调 const $\beta=0.9$，优于 AMUSE 与 const $\beta=0.98$。'
  zoomable=true
  alt='45M transformer comparison of beta schedules on FineWeb'
%}

右下子图说明了 drift-aware 设定点的必要性：面对同一条发散的噪声尺度，噪声尺度方案过度累积并发散，drift-aware 方案改用 $$\dot G$$，保持稳定并取得最低 val loss。这与 Merrill 关于噪声尺度不可靠的论述一致。

{% include figure.liquid
  path='assets/img/post-06-20/bsf_big_compare.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='117M / 2.5B tokens（modded-nanogpt）。drift-aware 复现优势（val loss 最低）；其 $B_\text{eff}$（右上红线）触底 25 后仅回升至约 43、后半程在 27–52 间波动，未进入 batch 放大阶段；AMUSE 在约 1750M 出现一次不稳定的 val loss 回升。'
  zoomable=true
  alt='117M transformer comparison showing effective batch stays small'
%}

两个规模的结果一致：

| | 45M / 600M | 117M / 2.5B |
| :--- | :---: | :---: |
| drift-aware（无调参）val | **4.235** | **3.643** |
| const $$\beta=0.9$$ | 4.275 | 3.659 |
| AMUSE 式开环 | 显著落后 | 3.990 |
| drift-aware 的 $$B_{\text{eff}}$$ | ~14–37 | ~27–52 |
{: .table .table-striped .table-sm}

DASF 在两个规模上均匹配或超过经过调参的 AMUSE 与最优手调常数，达成即插即用、免代理模型标定的目标。但其优势并非来自更大的隐式 batch——DASF 的 $$B_{\text{eff}}$$（约 40）小于 const $$\beta=0.9$$（100）；在此体制下较小的有效 batch 反而更优，DASF 的价值在于自动校准到合适的（小）值、不过度累积。

## 5. 结论与方法定位

作为方法，本文将 SF 的 $$\beta$$ 用作有效 batch size 的在线控制量、以 drift-aware 立方根定律闭环，得到 DASF，一个无需调参的有效 batch size 设定器：在 45M 与 117M 上均匹配或略优于手调常数、明显优于开环 AMUSE，并规避了噪声尺度方案的发散。

作为一项可证伪的科学发现，batch size 增长的收益始终未出现：即使在 117M / 2.5B / compute-optimal 下，$$B_{\text{eff}}$$ 也仅在数十量级、领先不随训练扩大。更确切地说，compute-optimal 体制下最优有效 batch size 小且近似恒定，而非按 $$\sqrt t$$ 增长——按 [Chinchilla](https://arxiv.org/abs/2203.15556) 比例训练时梯度持续高速变化，结论 4 的立方根公式始终给出较小的 batch size。这也解释了恒定 $$\beta$$ 的 SF 在该体制下为何已经够用。

关于定位：固定物理 batch size 下，$$\beta$$ 只改变每步梯度的方差、不改变优化步数；而 [Merrill](https://arxiv.org/abs/2505.23971) 式 batch warmup 收益的主要来源，是早期较小的物理 batch 在同 token 预算下提供的更多优化步（step-count 通道），$$\beta$$ 在结构上无法触及。故 DASF 是有效 batch size（方差）的设定器，而非物理 batch size 的调度器——即 Route A（动物理 batch、有 step-count 收益但有系统开销）与 Route B（动有效 batch、即插即用）之分中，Route B 的上界。要兑现增长收益，须转入 $$B^\ast$$ 实质变化的体制：重度过训练、LR cooldown，或令物理 batch 本身变化。

## 参考文献

1. A. Defazio, X. Yang, H. Mehta, K. Mishchenko, A. Khaled, and A. Cutkosky. "The Road Less Scheduled." *NeurIPS*, 2024. [arXiv:2405.15682](https://arxiv.org/abs/2405.15682).
2. S. McCandlish, J. Kaplan, D. Amodei, and OpenAI Dota Team. "An Empirical Model of Large-Batch Training." [arXiv:1812.06162](https://arxiv.org/abs/1812.06162), 2018.
3. W. Merrill, S. Arora, D. Groeneveld, and H. Hajishirzi. "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training." *NeurIPS*, 2025. [arXiv:2505.23971](https://arxiv.org/abs/2505.23971).
4. M. Song, B. Baek, K. Ahn, and C. Yun. "Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training." *NeurIPS*, 2025. [arXiv:2507.09846](https://arxiv.org/abs/2507.09846).
5. J. Kim, B. Shin, J. Yun, B. Baek, M. Song, and C. Yun. "AMUSE: Anytime Muon with Stable Gradient Evaluation." [arXiv:2605.22432](https://arxiv.org/abs/2605.22432), 2026.
6. D. Morwani et al. "Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants." [arXiv:2502.02431](https://arxiv.org/abs/2502.02431), 2025.
7. J. Hoffmann et al. "Training Compute-Optimal Large Language Models." [arXiv:2203.15556](https://arxiv.org/abs/2203.15556), 2022.

*姊妹篇：[为什么 LLM pretrain 过程中途要把 batch size 翻倍](https://jiaxuanzou0714.github.io/blog/2026/why-double-batch-size-llm-pretraining/)（物理 batch size 调度的变分解）。*

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026sfbatch,
  title={DASF：一种闭环的 batch size schedule-free 方法},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/schedule-free-effective-batch-size/}
}
```
