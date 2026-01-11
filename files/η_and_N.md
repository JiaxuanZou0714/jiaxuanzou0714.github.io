---
layout: single
title: "基于 Modes--Zipf--隐式偏置框架的学习率随模型规模缩放规律推导"
permalink: /files/eta-and-n/
---

*Jiaxuan Zou*
*日期：2026-01-11*

## 摘要 {#abstract}

本文在一种可解析的 “Modes--Zipf--隐式偏置” 抽象框架下，给出学习率 $\eta$ 随模型规模 $N$ 的缩放规律推导。框架的核心是：数据由一族可数的 “模式”（modes）组成，出现频率满足 Zipf 幂律；训练过程在每个模式上的剩余误差 $q_k$ 呈乘法收缩；并允许 “频率调制” 的有效学习强度 $\lambda_k$。在该框架中，训练动力学只通过内禀时间尺度 $s=c\tau$ 进入，其中 $c=\eta\lambda_0$、$\tau$ 为 token-step。要把学习率与模型规模关联起来，必须补充一条把 $\lambda_0$ 与模型可承载模式数 $M(N)$ 联系的结构性假设。本文给出一个最小且可证伪的 “资源密度” 假设 $\lambda_0(N)\propto N/M(N)$，从而推出

$$
\eta(N)\ \propto\ \frac{M(N)}{N}.
$$

若进一步假设容量映射 $M(N)\propto N^\gamma$，则得到幂律缩放 $\eta(N)\propto N^{\gamma-1}$；在常见情形 $\gamma=\tfrac12$ 时，得到 $\eta\propto N^{-1/2}$，等价于宽度倍率下的 $\eta\propto 1/\text{width}$ 级别缩放，与 $\mu$P 类方法在形式上相容。

---

## 目录 {#toc}

- [引言](#intro)
- [Modes、Zipf 与 excess loss 分解](#modes-zipf-excess)
- [乘法收缩动力学与内禀时间](#dynamics)
- [容量映射与模型规模误差](#capacity)
- [学习率随模型规模的缩放律](#lr-scaling)
- [结论](#conclusion)

---

## 引言 {#intro}

经验上，大模型训练往往需要随规模调整学习率等超参数；而 $\mu$P 一类工作强调：存在一套参数化与超参数缩放，使得学习率等可在不同宽度间近似稳定（超参数迁移）。本文的目标是：在你提出的 “Modes--Zipf--隐式偏置” 理论中，推导一个**学习率 $\eta$ 随模型规模 $N$ 的缩放律**，并明确指出推导所需的最小额外假设与可检验预测。

本文结构如下：
- 第 2 节给出 modes 与 Zipf 假设以及 excess loss 分解；
- 第 3 节给出 $q_k$ 的乘法收缩动力学并导出 “内禀时间”；
- 第 4 节给出容量映射 $M(N)$；
- 第 5 节在框架内推导学习率缩放律，并讨论与 $\mu$P 的形式对齐与可证伪点。

---

## Modes、Zipf 与 excess loss 分解 {#modes-zipf-excess}

> **定义（模式与频率分布）**
>
> 设存在可数的模式集合 $\mathcal{K}=\{1,2,3,\dots\}$。在数据分布下，模式 $k$ 出现的概率为 $p_k$，满足 $\sum_{k\ge 1}p_k=1$。

> **假设（Zipf 幂律分布）**
>
> 存在指数 $\alpha>1$ 与缓变函数 $L(\cdot)$，使得当 $k$ 足够大时
>
> $$
> p_k \;=\; k^{-\alpha}L(k).
> $$
>
> 为便于显式推导常数项，后文常采用纯幂律特例
>
> $$
> p_k \;=\; \frac{1}{Z}k^{-\alpha},\qquad Z=\sum_{k=1}^\infty k^{-\alpha}=\zeta(\alpha).
> $$

> **定义（excess loss 的模式分解）**
>
> 设总体风险（或损失）为 $L$，不可约误差（Bayes 风险或熵项等）为 $E$，定义 excess loss
>
> $$
> \Delta L := L-E.
> $$
>
> 假设 $\Delta L$ 可按模式加和分解为
>
> $$
> \Delta L \;\approx\; \sum_{k=1}^\infty p_k\,q_k,
> $$
>
> 其中 $q_k\in[0,1]$ 表示模型在模式 $k$ 上的 “剩余误差”（未学习/未拟合程度）。

> **Remark**
>
> 上式的含义是：模式越常见（$p_k$ 越大），其剩余误差对总损失贡献越大；而隐式偏置将驱动训练优先压低高频模式的 $q_k$（见后文动力学部分）。

---

## 乘法收缩动力学与内禀时间 {#dynamics}

### 有序学习与有效前沿

> **假设（有序学习与有效前沿）**
>
> 训练过程中存在一族随训练进度变化的剩余误差序列 $\{q_k(\tau)\}_{k\ge 1}$（$\tau$ 为 token-step），满足：
>
> 1. （单调性）$0\le q_1(\tau)\le q_2(\tau)\le \cdots \le 1$；
> 2. （有效前沿）存在 $k_\star(\tau)$ 使得当 $k\ll k_\star(\tau)$ 时 $q_k(\tau)\to 0$，当 $k\gg k_\star(\tau)$ 时 $q_k(\tau)\to 1$。

> **Remark**
>
> 这是对 “隐式偏置 + feature learning” 的抽象刻画：训练更倾向先学习高频/低复杂度模式，形成由 $k_\star$ 划分的 “学会” 与 “未学会” 区域。

### i.i.d. 采样与乘法更新

> **假设（token-step 下的 i.i.d. 模式采样）**
>
> 每个 token-step $t=0,1,2,\dots$ 观测到的模式 $K_t\in\mathbb{N}$ 独立同分布，且
>
> $$
> \Pr[K_t=k]=p_k.
> $$
>
> 记指示变量 $I_{t,k}:=\mathbf{1}\{K_t=k\}$，则 $\mathbb{E}[I_{t,k}]=p_k$。

> **假设（乘法收缩更新）**
>
> 对任意模式 $k$，其剩余误差 $q_k(t)\in[0,1]$ 在离散 token-step 下按如下方式更新：
>
> $$
> q_k(t+1)=
> \begin{cases}
> (1-\eta\lambda_k)\,q_k(t), & I_{t,k}=1,\\
> q_k(t), & I_{t,k}=0,
> \end{cases}
> $$
>
> 其中 $\eta>0$ 为（全局）学习率，$\lambda_k>0$ 表示模式 $k$ 的有效学习强度。假设初始 $q_k(0)=1$。

> **假设（频率调制的有效学习强度）**
>
> 存在常数 $\lambda_0>0$ 与指数 $\beta>0$，使得
>
> $$
> \lambda_k \;=\; \lambda_0\,p_k^{\beta-1}.
> $$
>
> 当 $\beta=1$ 时，各模式的 $\lambda_k$ 近似同阶；当 $\beta>1$ 时，高频模式具有更强的有效收缩强度；当 $0<\beta<1$ 时则相反。

### 连续近似与显式解

记截至 token-step $\tau$，模式 $k$ 被观测到的次数为

$$
n_k(\tau):=\sum_{t=0}^{\tau-1}I_{t,k}.
$$

由乘法更新立即得到

$$
q_k(\tau) \;=\; (1-\eta\lambda_k)^{n_k(\tau)} q_k(0).
$$

> **命题（指数形式与内禀时间）**
>
> 在 “小步长” 条件 $\eta\lambda_k\ll 1$ 下，结合大数定律 $n_k(\tau)=\tau p_k+o(\tau)$，有近似
>
> $$
> q_k(\tau)\ \approx\ \exp\!\big(-\eta\lambda_k\,\tau p_k\big)
> \ =\ \exp\!\big(-c\,\tau\,p_k^\beta\big),
> \qquad c:=\eta\lambda_0.
> $$
>
> 等价地，存在连续时间动力学
>
> $$
> \frac{d}{d\tau}q_k(\tau)=-c\,p_k^\beta q_k(\tau),\qquad q_k(0)=1.
> $$

**证明（略写）**：由

$$
\log(1-x)=-x+o(x)\quad (x\to 0)
$$

得

$$
(1-\eta\lambda_k)^{n_k(\tau)}
=\exp\!\big(n_k(\tau)\log(1-\eta\lambda_k)\big)
\approx \exp\!\big(-\eta\lambda_k\,n_k(\tau)\big).
$$

代入 $n_k(\tau)=\tau p_k+o(\tau)$ 并结合 $\lambda_k=\lambda_0 p_k^{\beta-1}$ 即得。

> **定义（内禀时间尺度）**
>
> 定义内禀时间（或 “有效训练时间”）
>
> $$
> s \;:=\; c\,\tau \;=\; \eta\lambda_0\,\tau.
> $$
>
> 在上述近似下，$\{q_k\}$ 以及由其诱导的 $\Delta L$ 的训练动力学仅通过 $s$ 进入。

> **Remark（学习率在该抽象层面的地位）**
>
> 在内禀时间 $s$ 下，学习率 $\eta$ 的作用等价于对横轴（token-step）做线性重标尺：$\tau \mapsto s=\eta\lambda_0\,\tau$。因此，要讨论 “$\eta$ 随模型规模 $N$ 如何缩放”，本质上是在讨论：当 $N$ 改变时，$\lambda_0$ 是否保持常数；若不保持，则应如何选择 $\eta(N)$ 使 $s$ 的标定保持可比。

### 训练前沿与时间缩放（复述框架中的结论）

从

$$
q_k(\tau)\approx \exp\big(-c\tau p_k^\beta\big)
$$

可见，模式 $k$ 何时 “被学会” 由量级条件 $c\tau p_k^\beta\gtrsim 1$ 控制。

> **定义（训练前沿）**
>
> 定义训练前沿 $k_\tau$（忽略常数）为满足
>
> $$
> c\tau\,p_{k_\tau}^\beta \asymp 1
> $$
>
> 的模式指标。

在纯幂律 $p_k=\frac{1}{Z}k^{-\alpha}$ 下，得到

$$
k_\tau \;\asymp\; Z^{-1/\alpha}\,(c\tau)^{1/(\alpha\beta)}.
$$

此外，将 $q_k$ 代入 excess loss 分解，可得到时间/计算缩放的幂律：

> **引理（时间缩放律与 $c$ 的幂次进入）**
>
> 在纯幂律 $p_k=\frac{1}{Z}k^{-\alpha}$ 下，令
>
> $$
> \Delta L(\tau) \approx \sum_{k=1}^\infty p_k\exp(-c\tau p_k^\beta),
> $$
>
> 则当 $\tau\to\infty$ 时有渐近
>
> $$
> \Delta L(\tau)\ \sim\
> \underbrace{\frac{1}{\alpha\beta}\,Z^{-1/\alpha}\,
> \Gamma\!\Big(\frac{\alpha-1}{\alpha\beta}\Big)}_{=:C_{\alpha,\beta,Z}}
> \cdot (c\tau)^{-(\alpha-1)/(\alpha\beta)}.
> $$

**证明（积分近似与变量替换）**：

设 $p(x)=\frac{1}{Z}x^{-\alpha}$，则

$$
\Delta L(\tau)\approx \int_{1}^{\infty}\frac{1}{Z}x^{-\alpha}
\exp\!\Big(-c\tau\Big(\frac{1}{Z}x^{-\alpha}\Big)^\beta\Big)\,dx
=\int_1^\infty \frac{1}{Z}x^{-\alpha}\exp(-a\tau x^{-\alpha\beta})dx,
$$

其中 $a:=cZ^{-\beta}$。令 $u=a\tau x^{-\alpha\beta}$，整理幂次并取极限得到结论。

> **Remark**
>
> 该引理直接显示：$c=\eta\lambda_0$ 以幂次形式进入时间缩放律。若希望不同设置下训练曲线的时间尺度可对齐，核心是控制 $c(N)$ 的变化。

---

## 容量映射与模型规模误差 {#capacity}

> **假设（容量映射，capacity map）**
>
> 存在函数 $M:\mathbb{N}\to\mathbb{N}$（随模型规模 $N$ 单调增长），使得当模型规模为 $N$ 时，有效前沿满足
>
> $$
> k_\star(N)\ \asymp\ M(N).
> $$
>
> 并假设存在 $\gamma>0$ 使得幂律容量映射成立：
>
> $$
> M(N)\ \propto\ N^\gamma.
> $$

> **命题（模型规模误差的幂律）**
>
> 在 Zipf 与容量映射假设下，若训练足够充分使得 $k\lesssim M(N)$ 的模式被学习（$q_k\approx 0$），则模型规模瓶颈下的剩余误差满足
>
> $$
> \varepsilon_N(N)
> \ :=\ \sum_{k>M(N)}p_k
> \ \asymp\ M(N)^{-(\alpha-1)}
> \ \asymp\ N^{-\gamma(\alpha-1)}.
> $$

**证明**：由 Zipf 幂律尾和（积分比较）

$$
\sum_{k>M}k^{-\alpha}\asymp \int_M^\infty x^{-\alpha}dx \asymp M^{-(\alpha-1)}.
$$

代入 $M=M(N)\propto N^\gamma$ 即得。

> **Remark**
>
> 该命题是 “模型规模 scaling law” 的一条直接来源：只要 tail（未覆盖模式）主导误差，就得到幂律指数 $\gamma(\alpha-1)$。

---

## 学习率随模型规模的缩放律 {#lr-scaling}

### 为什么仅靠原框架无法推出 $\eta(N)$

到目前为止，学习率 $\eta$ 只通过 $c=\eta\lambda_0$ 进入训练动力学；而模型规模 $N$ 仅通过 $M(N)$ 进入模型瓶颈误差。若不说明 $\lambda_0$ 与 $N$ 的关系，则 **$N$ 与 $\eta$ 在理论上彼此独立**：任何 $\eta(N)$ 都与容量映射不发生代数耦合，因此无法推出唯一缩放律。

因此，要从该理论推出 “$\eta$ 应如何随 $N$ 缩放”，必须补充一条把 $\lambda_0$ 连接到 $N$（或 $M(N)$）的结构性假设。

### 内禀时间不变性：缩放律的第一原则

> **定理（尺度不变性原则：固定 $c(N)$）**
>
> 在采样、乘法收缩与频率调制假设下，对任意给定的模式频率 $\{p_k\}$，训练动力学由
>
> $$
> q_k(\tau)\approx \exp\big(-c(N)\tau p_k^\beta\big),\qquad c(N):=\eta(N)\lambda_0(N),
> $$
>
> 决定。特别地，若希望不同模型规模 $N$ 下的训练曲线在 “内禀时间” 坐标 $s=c(N)\tau$ 上可直接对齐（即实现零样本超参数迁移意义下的动力学相似），充分条件是
>
> $$
> c(N)\equiv c_\mathrm{ref}\quad\text{为常数}.
> $$

> **Remark（与稳定性条件的一致性）**
>
> 小步长与稳定性通常要求 $\eta\lambda_k\ll 1$，而当 $\beta\ge 1$ 且 $p_k\le 1$ 时 $\lambda_k\le \lambda_0$，固定 $c=\eta\lambda_0$ 也等价于固定最坏情形的步长尺度，从而稳定性不随 $N$ 系统性恶化。

### 资源密度假设：把 $\lambda_0(N)$ 接到 $M(N)$

> **假设（模式资源密度 $\Rightarrow$ 基准学习强度）**
>
> 将模型规模 $N$ 理解为可调自由度总量。容量映射 $M(N)$ 描述模型能稳定承载/表征的有效模式数量。假设在已覆盖模式集合上，**单个模式的平均资源密度**与 $N/M(N)$ 同阶，且基准有效学习强度 $\lambda_0(N)$ 与该资源密度成正比：
>
> $$
> \lambda_0(N)\ \propto\ \frac{N}{M(N)}.
> $$

> **Remark（可证伪性）**
>
> 这是本文唯一新增的 “结构性” 假设。它可通过实验估计 $\lambda_0(N)$ 的相对变化（例如从早期训练曲线拟合 $c(N)$，或从 $q_k$ 的衰减速度估计）来检验；若观测到 $\lambda_0(N)$ 近似常数，则学习率无需按本文方式缩放。

### 主结论：$\eta(N)\propto M(N)/N$

> **定理（学习率--模型规模缩放律）**
>
> 在“固定 $c(N)$”与“资源密度”假设下，若希望不同模型规模 $N$ 的训练动力学在内禀时间上保持相似（即 $c(N)$ 保持常数），则学习率应满足
>
> $$
> \eta(N)\ \propto\ \frac{1}{\lambda_0(N)}\ \propto\ \frac{M(N)}{N}.
> $$
>
> 进一步地，若 $M(N)\propto N^\gamma$，则得到幂律缩放
>
> $$
> \eta(N)\ \propto\ N^{\gamma-1}.
> $$

> **推论（宽度倍率形式与 $\mu$P 形式对齐）**
>
> 设某类架构的参数量与宽度 $d$ 近似满足 $N\propto d^2$。若容量映射满足 $M(N)\propto \sqrt{N}\propto d$（即 $\gamma=\tfrac12$），则
>
> $$
> \eta(N)\propto N^{-1/2}\ \propto\ d^{-1}.
> $$
>
> 即：宽度放大 $m$ 倍（$d\mapsto md$）时，学习率按 $1/m$ 级别缩放。

> **Remark**
>
> 这解释了为何在一些经验/工程规则中会出现 “宽度放大 $m$ 倍，学习率缩小 $\sim 1/m$” 的规律：在本文框架中，这对应于 “可承载模式数与宽度线性增长” 的容量映射，以及 “基准学习强度与单模式资源密度正比” 的结构假设。

### 进一步讨论：若 $\lambda_0(N)$ 的指数不同会怎样？

> **Remark（一般化：$\lambda_0(N)\propto (N/M(N))^\delta$）**
>
> 若把资源密度假设推广为 $\lambda_0(N)\propto (N/M(N))^\delta$（$\delta>0$），则在保持 $c(N)$ 常数的原则下有
>
> $$
> \eta(N)\propto \Big(\frac{M(N)}{N}\Big)^\delta.
> $$
>
> 当 $M(N)\propto N^\gamma$ 时，得到 $\eta(N)\propto N^{\delta(\gamma-1)}$。
>
> 本文取 $\delta=1$ 是最小的 “线性资源密度” 情形。

---

## 结论 {#conclusion}

本文在 Modes--Zipf--隐式偏置的抽象框架中，指出学习率只通过 $c=\eta\lambda_0$ 进入训练动力学，从而 “学习率随模型规模缩放” 的问题等价于：厘清 $\lambda_0$ 随 $N$ 的变化。通过补充一个最小的资源密度假设 $\lambda_0(N)\propto N/M(N)$，我们得到主结论

$$
\eta(N)\propto \frac{M(N)}{N}\propto N^{\gamma-1}.
$$

该缩放律的关键优势是可证伪：只要能从训练曲线或模式级残差衰减估计 $\lambda_0(N)$ 的相对变化，就可以直接验证/否定该结论，并据此修正 $\delta$ 或容量映射 $M(N)$ 的形式。
