---
layout: post
title: "并行性与表达能力的权衡：从电路复杂度看 Linear Attention 的理论边界"
date: 2026-03-23 12:00:00
description: "固定深度 Transformer 与一类基本线性递归架构的表达能力均被限制在 TC⁰ 内。本文从电路复杂度出发，阐明为何保持最强 token 维度并行性的架构无法无代价地获得更高表达能力，以及为何放松并行性约束（如引入对数深度或更强的状态更新）是突破该上界的必要条件。"
tags: [llm, reasoning, transformer, linear-attention, complexity-theory]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

## 1. 问题

本文讨论以下问题：是否存在一种 linear attention 或线性递归架构，它在 token 维度上保持与最基本线性递归模型相同程度的完全并行性，同时在表达能力上显著超越后者？

我们将论证，在标准复杂度猜想下，答案是否定的。具体而言，我们将建立如下逻辑链：

1. 固定深度 Transformer 与一类基本线性递归架构（包括 S4、Mamba）的表达能力均被限制在复杂度类 $\mathsf{TC}^0$ 内。
2. 存在与语言模型能力直接相关的计算问题（如一般正则语言识别），其复杂度为 $\mathsf{NC}^1$-complete。在标准猜想 $\mathsf{TC}^0 \neq \mathsf{NC}^1$ 下，这些问题不可被上述架构表达。
3. 已知的突破 $\mathsf{TC}^0$ 上界的方法——无论是 chain of thought、对数深度 looping，还是引入更强的输入依赖状态转移——均需要向计算图中引入随输入长度增长的有效深度，从而不可避免地削弱 token 维度上的并行性。

因此，token 维度上的并行性与对顺序计算的表达能力之间存在系统性的权衡：前者对应浅层计算图，后者要求计算图沿序列方向加深。

本文的所有结论均基于已有的严格理论结果。我们的贡献仅在于将这些分散的结果组织在一个统一的框架下，以回答上述架构设计问题。

## 2. 预备知识：电路复杂度

本节回顾所需的电路复杂度基本概念。更详细的处理参见 [1, Chapter 2] 和 [AB09]。

### 2.1 电路与电路族

电路是有向无环计算图。每个叶节点对应一个输入变量，每个内部节点对应一个门（gate），计算某个函数。对于布尔电路，门的类型包括 AND、OR、NOT（有界扇入）以及 MAJ（多数门，无界扇入）。一个具有单一输出节点的布尔电路定义了一个函数 $\{0,1\}^n \to \{0,1\}$，其中 $n$ 为该电路的输入数。

电路族是一族电路 $\mathcal{C} = \{C_n\}_{n=0}^{\infty}$，其中 $C_n$ 处理长度为 $n$ 的输入。这使得电路族可以定义函数 $\Sigma^* \to \{0,1\}$，即定义形式语言。

复杂度度量.电路的深度（depth）是从任意输入节点到输出节点的最长路径长度。电路的规模（size）是电路中连线的数量。对电路族而言，深度和规模均为输入长度 $n$ 的函数。

### 2.2 标准电路复杂度类

以下复杂度类由对深度和规模的不同约束定义 [1, Section 2.2]：

- $\mathsf{NC}^k$：有界扇入（每个门至多两个输入，门类型为 AND/OR/NOT）、$O(\log^k n)$ 深度、$\mathrm{poly}(n)$ 规模的电路族所识别的语言类。
- $\mathsf{AC}^k$：与 $\mathsf{NC}^k$ 相同，但允许无界扇入。
- $\mathsf{TC}^k$：与 $\mathsf{AC}^k$ 相同，但额外允许 MAJ 门（多数门，对输入位取多数投票）。

这些类满足如下包含关系 [1]：对任意 $k$，

$$
\mathsf{NC}^k \subseteq \mathsf{AC}^k \subseteq \mathsf{TC}^k \subseteq \mathsf{NC}^{k+1}.
$$

定义 $\mathsf{NC} = \bigcup_k \mathsf{NC}^k$。本文最关键的类是 $\mathsf{AC}^0$、$\mathsf{TC}^0$、$\mathsf{NC}^1$ 和 $\mathsf{NC}$。它们之间的关系可总结为：

$$
\mathsf{AC}^0 \subseteq \mathsf{TC}^0 \subseteq \mathsf{NC}^1 \subseteq \mathsf{NC} \subseteq \mathsf{P}.
$$

$\mathsf{TC}^0$ 对应常数深度、多项式规模的阈值电路，代表可高度并行化的计算。$\mathsf{NC}^1$ 对应 $O(\log n)$ 深度的有界扇入电路。$\mathsf{NC}$ 对应 $\mathrm{polylog}(n)$ 深度。$\mathsf{P}$ 对应多项式时间可计算问题。

### 2.3 一致性

未加约束的电路族是非一致（non-uniform）计算模型，其中每个 $C_n$ 可以被任意指定，甚至可以识别不可判定的语言。为排除这种退化情形，需要施加一致性（uniformity）条件：要求存在某个受限计算过程，能够从输入 $1^n$ 生成 $C_n$ 的描述。标准选择包括对数空间（$\mathsf{L}$）一致性和更强的一阶（$\mathsf{FO}$）一致性 [MIS90]。本文中，所有涉及的复杂度类均取其一致版本。

### 2.4 完备性与分离

若语言 $L$ 是某复杂度类 $\mathsf{X}$ 的完备问题（$\mathsf{X}$-complete），则 $L \in \mathsf{X}$ 且 $\mathsf{X}$ 中的所有语言均可通过某种简单归约映射到 $L$。若归约本身属于更小的类（如 $\mathsf{AC}^0$ 或 $\mathsf{FO}$ 归约），则：$L$ 可被更小的类识别，当且仅当两个类相等。

例如：若 $L$ 是 $\mathsf{NC}^1$-complete 的（在 $\mathsf{AC}^0$ 归约下），且 $L \in \mathsf{TC}^0$，则 $\mathsf{TC}^0 = \mathsf{NC}^1$。因此，在标准猜想 $\mathsf{TC}^0 \neq \mathsf{NC}^1$ 下，$\mathsf{NC}^1$-complete 问题不属于 $\mathsf{TC}^0$。

## 3. 固定深度 Transformer 的表达能力上界

### 3.1 Transformer 的定义

我们采用 [1, Definition 2.2] 中的标准定义。一个 $p$-精度、$d$ 层、$h$ 头、模型维度 $m$、前馈宽度 $w$ 的 decoder-only transformer $\mathcal{T}$ 计算函数 $\Sigma^* \to \Sigma$，其计算过程包括：

1. 嵌入层：$\mathbf{h}_i^0 = \mathbf{E}[w_i] + \pi(i)$。
2. 自注意力子层：在每一层 $\ell$，对每个头 $k$，计算查询 $$\mathbf{q}_{i, k}^\ell$$、键 $$\mathbf{k}_{i, k}^\ell$$、值 $$\mathbf{v}_{i, k}^\ell$$，然后

$$
\mathbf{a}_{i,k}^\ell = \sum_{j=1}^{m} \frac{\exp\bigl(\tfrac{1}{\tau} (\mathbf{q}_{i,k}^\ell)^\top \mathbf{k}_{j,k}^\ell\bigr)}{Z_{i,k}^\ell} \cdot \mathbf{v}_{j,k}^\ell, \quad Z_{i,k}^\ell = \sum_{j=1}^{m} \exp\bigl(\tfrac{1}{\tau} (\mathbf{q}_{i,k}^\ell)^\top \mathbf{k}_{j,k}^\ell\bigr).
$$

3. 前馈子层：逐位置的非线性变换。
4. 输出层：$\arg\max\, \mathbf{h}_n^d \mathbf{U}$。

关键参数约束：层数 $d$ 固定（不随输入长度 $n$ 增长），精度 $p$ 至多为 $\mathrm{poly}(n)$。

### 3.2 主定理：Transformer $\subseteq$ $\mathsf{TC}^0$

定理 3.1（[MSS22; MS23a; MS23b]）. 设 $\mathcal{T}$ 为一个（固定深度、至多多项式精度的）transformer。则 $\mathcal{T}$ 所计算的函数可被 $\mathsf{FO}$-uniform $\mathsf{TC}^0$ 电路族计算。

证明思路.证明的核心是逐一验证 transformer 计算图的每个组件均可在 $\mathsf{FO}$-uniform $\mathsf{TC}^0$ 中实现，然后利用 $\mathsf{TC}^0$ 在组合下的封闭性。

首先，我们需要以下组合引理：

引理 3.2（[MS23a, Corollary 3.2]）. 设 $\mathcal{G}$ 为 $\mathsf{FO}$-uniform、多项式规模、常数深度的计算图族，其节点类型集为有限集 $\mathfrak{F}$，其中每个 $\mathcal{F} \in \mathfrak{F}$ 由一个 $\mathsf{FO}$-uniform $\mathsf{TC}^0$ 族指定。则 $\mathcal{G}$ 可被一个 $\mathsf{FO}$-uniform $\mathsf{TC}^0$ 族模拟。

Transformer 恰好是这样一个常数深度计算图族：其节点类型包括嵌入层、层归一化、自注意力子层、前馈子层和输出投影。因此，只需证明每个组件均在 $\mathsf{FO}$-uniform $\mathsf{TC}^0$ 中：

- 嵌入层（引理 3.3 [1]）：$\mathbf{E}[w_i] + \pi(i)$ 的每一位可在 $\mathsf{FO}[\mathsf{M}]$ 中定义。$\pi(i)$ 是解析函数，可在 $\mathsf{TC}^0$ 中用多项式精度近似 [RT92]。加法在 $\mathsf{FO}$ 中可实现。

- 层归一化（引理 3.4 [1]）：涉及加法、乘法、除法和平方根。二元加法和乘法在 $\mathsf{TC}^0$ 中是平凡的。精确除法在 uniform $\mathsf{TC}^0$ 中 [Hes01]。平方根可在 uniform $\mathsf{TC}^0$ 中近似到多项式精度 [RT92; Chi24]。

- 自注意力子层（引理 3.5 [1]）：对单个头 $$\mathbf{a}_{i,k}$$，查询-键内积在 $\mathsf{TC}^0$ 中可计算，除以 $\tau$ 用 [Hes01]，$\exp$ 可近似到多项式精度 [RT92; Chi24]。归一化常数 $Z_{i,k}$ 是对 $j$ 的求和，迭代加法在 $\mathsf{FO}$-uniform $\mathsf{TC}^0$ 中 [Imm98]。加权求和同理。

- 前馈子层（引理 3.6 [1]）：仅涉及二元加法、乘法和逐元素非线性，均在 $\mathsf{TC}^0$ 中。

由引理 3.2，整个 transformer 可被 $\mathsf{FO}$-uniform $\mathsf{TC}^0$ 模拟。$\square$

### 3.3 该上界的含义

定理 3.1 的含义不限于给 transformer 贴一个复杂度标签。它精确刻画了固定深度 transformer 的计算结构：无论 self-attention 如何全局混合信息，只要整体深度不随输入长度增长，模型就被限制在常数深度阈值电路的表达范围内。

结合复杂度理论中的已知结果，可以立即推导出以下不可表达性结论（均假设相应的复杂度分离成立）：

| 问题 | 复杂度 | Transformer 可表达？ |
|------|--------|-------------------|
| 一般正则语言识别 | $\mathsf{NC}^1$-complete [IL95] | 否（除非 $\mathsf{TC}^0 = \mathsf{NC}^1$） |
| 布尔公式求值 | $\mathsf{NC}^1$-complete [Bus87] | 否（除非 $\mathsf{TC}^0 = \mathsf{NC}^1$） |
| 有向图连通性 | $\mathsf{NL}$-complete [Imm98] | 否（除非 $\mathsf{TC}^0 = \mathsf{NL}$） |
| 电路求值 | $\mathsf{P}$-complete [GHR91] | 否（除非 $\mathsf{TC}^0 = \mathsf{P}$） |
{: .table .table-striped .table-sm .w-auto .mx-auto style="font-size: 0.8em;"}


其中，正则语言识别与状态追踪（state tracking）直接相关。正则语言可由确定性有限自动机（DFA）识别，其核心操作是沿输入序列传播离散状态。该问题的电路复杂度取决于其转移幺半群（transition monoid）的代数结构：若幺半群可解（solvable），则识别在 $\mathsf{TC}^0$ 中 [KMR67; IL95]；若不可解，则识别是 $\mathsf{NC}^1$-complete 的。典型的不可解正则语言是置换群 $S_5$ 的字问题 [1, Section 3.4.1]。

## 4. 基本线性递归架构的表达能力上界

### 4.1 线性 RNN 的定义

定义 4.16（[MPS24]）. 给定输入序列 $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^k$，线性 RNN 层的递归形式定义状态序列 $\mathbf{h}_1, \ldots, \mathbf{h}_n \in \mathbb{R}^d$：

$$
\mathbf{h}_i = \mathbf{A}_i \mathbf{h}_{i-1} + \mathbf{B}_i \mathbf{x}_i,
$$

其中 $\mathbf{A}_i \in \mathbb{R}^{d \times d}$，$\mathbf{B}_i \in \mathbb{R}^{d \times k}$，可以依赖于 $\mathbf{x}_i$。层输出为 $\mathbf{y}_i = \mathbf{C}_i \mathbf{h}_i + \mathbf{D}_i \mathbf{x}_i$。

该递推可等价地写成卷积形式：

$$
\mathbf{h}_i = \sum_{j=1}^{i} \left(\prod_{k=j+1}^{i} \mathbf{A}_k\right) \mathbf{B}_j \mathbf{x}_j.
$$

卷积形式中，状态计算可归结为前缀积（prefix product）与前缀和（prefix sum），两者均存在高效的并行算法 [Ble90]。这正是此类架构可在 token 维度上并行化的原因。

该定义足以涵盖 S4 [GGR22]、Mamba [GD24] 等模型。

### 4.2 S4 与 Mamba 的 $\mathsf{TC}^0$ 上界

定理 4.17（[MPS24, Theorem 4.2]）. 考虑一个线性 RNN（如 S4），其每层具有形式 $$\mathbf{h}_i = \mathbf{A}\mathbf{h}_{i-1} + \mathbf{B}\mathbf{x}_i$$（$\mathbf{A}$ 不依赖于输入）。则该网络的输出可在 $\mathsf{L}$-uniform $\mathsf{TC}^0$ 中计算。

定理 4.18（[MPS24, Theorem 4.3]）. 考虑一个线性 RNN（如 Mamba），其中 $\mathbf{A}_i$ 对角且依赖于输入，且映射 $\mathbf{x}_i \mapsto \mathbf{A}_i$ 可在 $\mathsf{L}$-uniform $\mathsf{TC}^0$ 中计算。则该网络的输出可在 $\mathsf{L}$-uniform $\mathsf{TC}^0$ 中计算。

### 4.3 $\mathsf{TC}^0$ 上界的统一含义

定理 3.1、4.17 和 4.18 共同表明：固定深度 Transformer、S4 和 Mamba 三类架构的表达能力均被限制在 $\mathsf{TC}^0$ 中。

这一结论有两个直接推论：

推论 1.在标准猜想 $\mathsf{TC}^0 \neq \mathsf{NC}^1$ 下，上述三类架构均不能识别 $\mathsf{NC}^1$-complete 的正则语言。

推论 2.对于线性递归架构而言，"具有递归形式"与"具有超越 $\mathsf{TC}^0$ 的表达能力"是两个不同的性质。当递推的结构足够规则（如 $\mathbf{A}_i$ 不依赖输入，或 $\mathbf{A}_i$ 对角），递推可以被高效地并行化为前缀积运算，整个计算仍在 $\mathsf{TC}^0$ 内完成。递归形式本身不蕴含更强的顺序计算能力。

## 5. 突破 $\mathsf{TC}^0$：已知方法与共同特征

本节回顾已知的将模型表达能力提升到 $\mathsf{TC}^0$ 之外的方法。我们将论证，这些方法共享一个共同特征：向计算图中引入随输入长度 $n$ 增长的有效深度。

### 5.1 Chain of thought

给定生成模型 $f : \Sigma^* \to \Sigma$，chain of thought（CoT）将输入 $x \in \Sigma^*$ 映射到输出 $y \in \Sigma$ 的过程定义为 [1, Section 4.1.1]：

$$
z_0 = x, \qquad z_{i+1} = z_i \cdot f(z_i), \qquad y = f(z_{t(|x|)}).
$$

定义 4.1（[MS24b]）. $\mathrm{CoT}[T]$ 为使用 $O(t(n))$ 步 CoT 的 AHAT（averaging hard attention transformer with masked pre-norm）可识别的语言类。

CoT 的计算图结构与固定深度 transformer 有本质区别：每一步生成的 token 都被送回模型作为下一步的输入，从而在计算图中引入了长度为 $t(n)$ 的顺序依赖链。

定理 4.2（[MS24b, Theorem 2]）. $\mathrm{TIME}[t] \subseteq \mathrm{CoT}[t]$。

该定理表明，$t$ 步 CoT 至少与 $t$ 步 Turing 机等价。其证明构造了一个 transformer decoder，每一步解码模拟一步 Turing 机转移。

推论 4.3（[MS24b]）. $\mathrm{CoT}[n]$ 可识别任意正则语言。由于一般正则语言识别是 $\mathsf{NC}^1$-complete 的 [IL95]，在 $\mathsf{TC}^0 \neq \mathsf{NC}^1$ 下，线性 CoT 严格扩展了 transformer 的表达能力。

推论 4.7（[MS24b]）. $\bigcup_{c \geq 1} \mathrm{CoT}[n^c] = \mathsf{P}$。

另一方面，CoT 的上界为：

定理 4.6（[MS24b, Theorem 3]）. $\mathrm{CoT}[t] \subseteq \widetilde{\mathrm{TIME}}[t^2 + n^2]$。

定理 4.8（[MS24b, Theorem 4]）. $\mathrm{CoT}[t] \subseteq \mathrm{SPACE}[t + \log n]$。

综合上下界：

$$
\mathrm{TIME}[t] \subseteq \mathrm{CoT}[t] \subseteq \begin{cases} \widetilde{\mathrm{TIME}}[t^2 + n^2], \\ \mathrm{SPACE}[t + \log n]. \end{cases}
$$

值得注意的是，对数步 CoT 并不扩展表达能力：$\mathrm{CoT}[\log n] \subseteq \mathsf{TC}^0$ [Li+24b; MS24a]。因此，获得超越 $\mathsf{TC}^0$ 的表达能力至少需要接近线性量级的 CoT 步数。

### 5.2 Padding 与 looping

CoT 是完全顺序的扩展方式。[MS25; MS24a] 研究了两种保留更多并行性的扩展：

- Padding：在 transformer 输入末尾追加空白 token，增加电路宽度但不引入顺序依赖。
- Looping：将一组固定层重复执行 $d(n)$ 次，增加电路深度但层参数不随 $n$ 变化。

定理 4.11（[MS25]）. 使用 $n^k$ 个 padding token 和固定深度 $d$ 的 masked pre-norm AHAT 可计算任何具有 $k$ 个不同变量和嵌套深度 $d$ 的 $\mathsf{FO}[\mathsf{M}^2]$ 公式。

推论 4.12（[MS25]）. 使用多项式 padding 的 masked pre-norm AHAT 恰好识别 $\mathsf{FO}$-uniform $\mathsf{TC}^0$。

因此，仅增加 padding（宽度）不能突破 $\mathsf{TC}^0$。

定理 4.13（[MS24a, Theorem 4.1]）. 对任意正则语言 $$L$$ over $$ \Sigma $$，存在对数深度的 looped AHAT with masked pre-norm，当展开到 $$ \lceil \log_2 \rvert w\rvert \rceil$$ 层深度时，可识别 $$w \in L$$。

推论 4.14.若 $\mathsf{TC}^0 \neq \mathsf{NC}^1$，则对数深度 looped AHAT 严格强于固定深度 AHAT。

定理 4.15（[MS25]）. 对 $k \geq 1$，$\log^k$-深度 looped、$\mathrm{poly}(n)$-padded 的 masked pre-norm AHAT 恰好识别 $\mathsf{L}$-uniform $\mathsf{TC}^k$。

特别地，取 $k=1$ 得 $\mathsf{TC}^1$；取任意多对数深度 looping 得 $\mathsf{NC}$。

### 5.3 更强的线性递归：IDS4 与 RWKV-7

第三条路线是修改线性 RNN 的状态转移结构，使其更具表达能力。

IDS4（input-dependent S4）[MPS24]：将 S4 中固定的转移矩阵 $$\mathbf{A}$$ 替换为 $$\mathbf{h}_{i-1}$$ 的线性投影，即 $$\mathbf{A}_i = \mathbf{A}(\mathbf{h}_{i-1})$$。

定理 4.19（[MPS24, Theorem 5.2]）. 对任意正则语言，存在 1 层 IDS4 模型可识别之。

RWKV-7[Pen+25]：使用对角加秩一（diagonal-plus-rank-1）的转移矩阵参数化。

定理 4.20（[Pen+25, Theorem 3]）. 对任意正则语言，存在 4 层 RWKV-7 模型可识别之。

由于一般正则语言识别是 $\mathsf{NC}^1$-complete 的，定理 4.19 和 4.20 在 $\mathsf{TC}^0 \neq \mathsf{NC}^1$ 下意味着 IDS4 和 RWKV-7 的表达能力严格超越 S4、Mamba 和固定深度 Transformer。

### 5.4 共同特征：有效深度的增长

上述三类方法的共同点可以精确表述如下。定义架构族 $\mathfrak{A}$ 在输入长度 $n$ 下的有效深度$D_{\mathfrak{A}}(n)$ 为其计算图中从输入到输出的最长依赖路径长度。则：

| 方法 | 有效深度 $D(n)$ | 表达能力 |
|------|-------------|--------|
| 固定深度 Transformer / S4 / Mamba | $O(1)$ | $\subseteq \mathsf{TC}^0$ |
| $\log$-depth looping | $O(\log n)$ | $\supseteq$ 所有正则语言（$\mathsf{NC}^1$-complete） |
| $\mathrm{polylog}$-depth looping + poly padding | $O(\log^k n)$ | $= \mathsf{TC}^k$；$\bigcup_k = \mathsf{NC}$ |
| 线性 CoT | $O(n)$ | $\supseteq \mathsf{NC}^1$，$\subseteq$ 上下文有关语言 |
| 多项式 CoT | $O(n^c)$ | $= \mathsf{P}$ |
| IDS4（1 层，顺序推理） | $O(n)$ | $\supseteq$ 所有正则语言 |
| RWKV-7（4 层，顺序推理） | $O(n)$ | $\supseteq$ 所有正则语言 |
{: .table .table-striped .table-sm .w-auto .mx-auto style="font-size: 0.8em;"}

每一种突破 $\mathsf{TC}^0$ 的方法，都以某种方式使 $D(n)$ 从 $O(1)$ 增长到 $\Omega(\log n)$ 或更大。这不是巧合，而是 $\mathsf{TC}^0$ 类定义本身的必然结果：$\mathsf{TC}^0$ 恰好由常数深度电路定义，任何超越该类的计算必然需要超常数的深度。

对于 IDS4 和 RWKV-7，虽然它们使用并行 scan 算法在训练时实现了高效并行，但关键区别在于：其状态更新 $$\mathbf{A}_i = \mathbf{A}(\mathbf{h}_{i-1})$$ 引入了对前一步隐状态的依赖，使得计算图中出现了长度为 $$n$$ 的顺序依赖链。定理 4.17 和 4.18 的证明之所以对 S4 和 Mamba 成立，正是因为在这些模型中 $$\mathbf{A}_i$$ 要么不依赖输入，要么对角，使得前缀积可在 $$\mathsf{TC}^0$$ 中完成。一旦 $$\mathbf{A}_i$$ 以一般方式依赖于 $$\mathbf{h}_{i-1}$$，该论证不再适用。

## 6. 并行性-表达能力权衡的形式表述

基于前述结果，我们可以给出并行性-表达能力权衡的精确表述。

### 6.1 有效深度作为并行性的度量

对给定架构族 $\mathfrak{A}$，其有效深度 $D_{\mathfrak{A}}(n)$ 直接对应 token 维度上的并行时间（parallel time）。$D_{\mathfrak{A}}(n)$ 越小，模型在 token 维度上的并行性越强：

- $D_{\mathfrak{A}}(n) = O(1)$：最强并行性，对应 $\mathsf{TC}^0$。
- $D_{\mathfrak{A}}(n) = O(\log n)$：对应 $\mathsf{NC}^1$。
- $D_{\mathfrak{A}}(n) = O(\mathrm{polylog}(n))$：对应 $\mathsf{NC}$。
- $D_{\mathfrak{A}}(n) = O(\mathrm{poly}(n))$：对应 $\mathsf{P}$。

### 6.2 权衡的形式化

设 $\mathfrak{A}(g)$ 表示架构族 $\mathfrak{A}$ 中有效深度至多为 $O(g(n))$ 的子类。对目标语言 $L$，定义

$$
d_{\mathfrak{A},L}(n) := \inf\bigl\{g(n) \mid \exists\, \mathcal{M} \in \mathfrak{A}(g),\; \mathcal{M} \text{ 识别 } L\bigr\}.
$$

则以下结论成立：

命题 6.1.若 $L \notin \mathsf{TC}^0$，则对任何有效深度为 $O(1)$ 的架构族 $\mathfrak{A}$，$d_{\mathfrak{A},L}(n) \neq O(1)$。

这直接由 $\mathsf{TC}^0$ 的定义得出：$\mathsf{TC}^0$ 恰好是常数深度阈值电路可识别的语言类。

命题 6.2.若 $L$ 是 $\mathsf{P}$-complete 的，则 $d_{\mathfrak{A},L}(n) \notin O(\mathrm{polylog}(n))$，除非 $\mathsf{NC} = \mathsf{P}$。

这直接由 $\mathsf{NC}$ 的定义和 $\mathsf{P}$-completeness 得出。

推论 6.3（权衡的核心表述）.对固定深度 Transformer 族、S4 族或 Mamba 族 $\mathfrak{A}$：

$$
D_{\mathfrak{A}}(n) = O(1) \quad \Longrightarrow \quad \mathcal{E}(\mathfrak{A}) \subseteq \mathsf{TC}^0,
$$

其中 $\mathcal{E}(\mathfrak{A})$ 为 $\mathfrak{A}$ 可识别的语言类。因此，若 $L \notin \mathsf{TC}^0$，则识别 $L$ 的任何架构必须具有 $D(n) = \omega(1)$ 的有效深度。

这就是并行性-表达能力权衡的精确含义：

- 更小的 $D_{\mathfrak{A}}(n)$意味着更强的 token 维度并行性。
- 更大的表达能力需求（即需要识别 $\mathsf{TC}^0$ 之外的语言）意味着需要更大的 $d_{\mathfrak{A},L}(n)$。
- 这两者不可同时无代价地满足。

### 6.3 对最初问题的回答

现在可以精确回答第 1 节的问题。

问题.是否存在一种 linear attention，保持与基本线性递归模型相同程度的完全并行性（$D(n) = O(1)$），同时在表达能力上显著超越后者？

回答.在标准猜想 $\mathsf{TC}^0 \neq \mathsf{NC}^1$ 下，不存在。原因如下：

1. 基本线性递归模型（S4、Mamba）的表达能力已被证明在 $\mathsf{TC}^0$ 内（定理 4.17、4.18）。
2. "显著超越"的自然标准是：能够识别 $\mathsf{NC}^1$-complete 问题（如一般正则语言识别），因为这些问题与状态追踪等基本能力直接相关。
3. 若一种 linear attention 的有效深度仍为 $O(1)$，则其表达能力仍在 $\mathsf{TC}^0$ 内，无法识别 $\mathsf{NC}^1$-complete 问题。
4. 因此，在保持 $D(n) = O(1)$ 的约束下，表达能力不可能显著超越基本线性递归模型。

反之，IDS4、RWKV-7 等架构之所以能够识别任意正则语言（定理 4.19、4.20），正是因为它们引入了对前一步隐状态的一般性依赖，使得计算图的有效深度增长到 $\Omega(n)$。

## 7. 理论边界与实际并行性的关系

需要指出一个重要的区分：$\mathsf{TC}^0$ 意义下的"完全并行"与 GPU 上的"高效并行"并非同一概念。

如 [1, Section 5.2] 所指出，当今 GPU 的底层计算单元是 AND/OR 电路，缺少 MAJ 门。因此，在 GPU 上实现一个 $\mathsf{TC}^0$ 电路（如 softmax attention 中的求和与除法），其计算图深度实际上已经是 $O(\log n)$。并行 scan 算法 [Ble90] 计算一层 SSM 的深度同样为 $O(\log n)$。

这意味着：从 $\mathsf{TC}^0$ 放宽到 $\mathsf{NC}^1$（即有效深度从 $O(1)$ 增加到 $O(\log n)$），在理论上是放松了并行性约束，但在当前硬件上未必对应显著的额外 wall-clock 开销。换言之，$\mathsf{TC}^0$ 约束对当前硬件而言可能是过度限制的（overly restrictive），存在设计出表达能力超越 $\mathsf{TC}^0$ 但在实际硬件上仍保持高效并行的架构的空间。

IDS4 和 RWKV-7 的实践就是这一观察的具体实例：它们使用并行 scan 算法训练，在硬件上的并行效率与 S4/Mamba 接近，但在表达能力上严格更强。

## 8. 总结

本文的核心结论可总结为以下三点：

1. $\mathsf{TC}^0$ 上界的普遍性.固定深度 Transformer（定理 3.1）与基本线性递归模型 S4/Mamba（定理 4.17/4.18）的表达能力均被限制在 $\mathsf{TC}^0$ 内。这一限制的根源不在于 attention 或 recurrence 的具体形式，而在于计算图的有效深度为 $O(1)$。

2. 突破 $\mathsf{TC}^0$ 的必要条件.所有已知的将表达能力提升到 $\mathsf{TC}^0$ 之外的方法——CoT（定理 4.2）、对数深度 looping（定理 4.13）、IDS4（定理 4.19）、RWKV-7（定理 4.20）——均需要有效深度随 $n$ 增长。这不是某种特定方法的局限，而是 $\mathsf{TC}^0$ 类定义本身的结构性约束。

3. 权衡的精确含义.在标准复杂度猜想下，保持最强 token 维度并行性（$D(n) = O(1)$）与获得超越 $\mathsf{TC}^0$ 的表达能力之间存在不可调和的矛盾。但这并不排除在当前硬件约束下，设计出既具有更高表达能力又保持实际并行效率的架构——因为当前硬件本身并不能充分利用 $\mathsf{TC}^0$ 相对于 $\mathsf{NC}^1$ 的额外并行性。

## 参考文献

[1] William Merrill. *A theory of the computational power and limitations of language modeling architectures.* PhD thesis, 2025.

[AB09] Sanjeev Arora and Boaz Barak. *Computational Complexity: A Modern Approach.* Cambridge University Press, 2009.

[Ble90] Guy E. Blelloch. *Prefix Sums and Their Applications.* Tech. rep. CMU-CS-90-190, Carnegie Mellon University, 1990.

[Bus87] S. R. Buss. "The Boolean formula value problem is in ALOGTIME." STOC '87, 1987.

[Chi24] David Chiang. *Transformers in Uniform TC⁰.* 2024.

[GD24] Albert Gu and Tri Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." First Conference on Language Modeling, 2024.

[GGR22] Albert Gu, Karan Goel, and Christopher Re. "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR, 2022.

[GHR91] Raymond Greenlaw, H. James Hoover, and Walter L. Ruzzo. *A compendium of problems complete for P.* 1991.

[Hes01] William Hesse. "Division Is In Uniform TC⁰." ICALP, 2001.

[IL95] Neil Immerman and Susan Landau. "The complexity of iterated multiplication." *Inf. Comput.* 116(1), 1995.

[Imm98] Neil Immerman. *Descriptive Complexity.* Springer Verlag, 1998.

[KMR67] Kenneth Krohn, Richard Mateosian, and John Rhodes. "Methods of the algebraic theory of machines." *JCSS* 1(1), 1967.

[Li+24b] Zhiyuan Li et al. "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems." ICLR, 2024.

[MIS90] David A. Mix Barrington, Neil Immerman, and Howard Straubing. "On uniformity within NC¹." *JCSS* 41(3), 1990.

[MPS24] William Merrill, Jackson Petty, and Ashish Sabharwal. "The Illusion of State in State-Space Models." ICML, 2024.

[MS23a] William Merrill and Ashish Sabharwal. "A Logic for Expressing Log-Precision Transformers." NeurIPS, 2023.

[MS23b] William Merrill and Ashish Sabharwal. "The Parallelism Tradeoff: Limitations of Log-Precision Transformers." *TACL* 11, 2023.

[MS24a] William Merrill and Ashish Sabharwal. *A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers.* 2024.

[MS24b] William Merrill and Ashish Sabharwal. "The Expressive Power of Transformers with Chain of Thought." ICLR, 2024.

[MS25] William Merrill and Ashish Sabharwal. *Exact Expressive Power of Transformers with Padding.* 2025.

[MSS22] William Merrill, Ashish Sabharwal, and Noah A. Smith. "Saturated Transformers are Constant-Depth Threshold Circuits." *TACL* 10, 2022.

[Pen+25] Bo Peng et al. "RWKV-7 'Goose' with Expressive Dynamic State Evolution." *CoRR* abs/2503.14456, 2025.

[RT92] John H. Reif and Stephen R. Tate. "On Threshold Circuits and Polynomial Computation." *SIAM J. Comput.* 21(5), 1992.

## 引用

```bibtex
@article{zou2026parallel_expressiveness_tradeoff_linear_attention,
  title={并行性与表达能力的权衡：从电路复杂度看 Linear Attention 的理论边界},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/parallel-expressiveness-tradeoff-linear-attention/}
}
```