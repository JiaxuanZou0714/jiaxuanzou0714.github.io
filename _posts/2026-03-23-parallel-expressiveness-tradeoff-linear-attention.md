---
layout: post
title: "并行性与表达能力的权衡：从 AC⁰/TC⁰ 到 Linear Attention 的理论边界"
date: 2026-03-23 12:00:00
description: "从电路复杂度的视角，统一解释为什么常数深度 Transformer 无法精确完成任意长度整数乘法，以及为什么更强的 linear attention 变体往往无法保持完全 token 并行性。"
tags: [llm, reasoning, transformer, linear-attention, complexity-theory]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

> 本文整理自半年前我的一篇笔记。最近有人重新翻出那篇笔记，我发现其中部分内容已过时，遂重新梳理后以blog的形式发出来。

## 1. 两个问题

本文围绕两个问题展开。

问题一。 考虑常数宽度、常数深度、常数精度的 Transformer。给定两个 $$n$$ 位二进制整数，这一架构能否精确计算它们的乘积？

二进制乘法的核心困难在于进位传播：低位的局部运算结果会影响高位，且影响链的长度随 $$n$$ 增长。若模型每层每个位置只能保存常数比特信息，且总计算深度不随 $$n$$ 增长，那么当 $$n$$ 足够大时，模型缺少足够的内部计算深度来精确传播这条进位链。相比之下，chain of thought（CoT）允许模型将中间结果写回上下文并继续生成新 token，由此使计算图的有效深度随输出步数增长，因而原则上可以逐步处理进位。

问题二。 在 linear attention 的各种变体中，为什么表达能力更强的模型往往无法保持与最基本线性递归模型同等级别的完全 token 并行性？

实践中反复出现这样的现象：最容易做 scan 的架构表达能力最弱；而引入更复杂 gate、更强状态更新、或状态转移显式依赖当前隐状态的变体，虽然表达能力确实增强，却往往只能做到 chunk-wise 并行，甚至退化为训练时并行、推理时顺序。

这两个问题指向同一个核心矛盾：
> 在序列建模中，更强 token 并行性与更强顺序表达能力之间是否存在结构性的 trade-off？

本文用电路复杂度给出肯定的回答。具体结论如下：

1. 固定深度 Transformer 的表达能力上界是 $$\mathsf{TC}^0$$。若进一步限制为常数比特精度并采用逐步舍入语义，上界收紧至 $$\mathsf{AC}^0$$。
2. S4、Mamba 等可完全 scan 的线性递归模型同样受限于 $$\mathsf{TC}^0$$。
3. 要显著超越这一层级——例如识别一般正则语言、完成长程进位传播或做复杂状态追踪——模型必须引入随输入长度增长的有效计算深度。无论这一增长以 CoT、looping 还是更强的状态依赖形式出现，都会削弱最强意义下的 token 并行性。


## 2. 电路复杂度基础

本文只需要少量背景知识。

布尔电路是有向无环图，输入为比特，内部节点为逻辑门。两个关键复杂度度量是深度（输入到输出的最长路径长度）和规模（门数或连线数，通常要求关于输入长度 $$n$$ 为多项式量级）。

本文涉及四个标准复杂度类：

$$
\mathsf{AC}^0 \subseteq \mathsf{TC}^0 \subseteq \mathsf{NC}^1 \subseteq \mathsf{NC}.
$$

- $$\mathsf{AC}^0$$：常数深度、多项式规模、无界扇入 AND/OR/NOT 电路。
- $$\mathsf{TC}^0$$：在 $$\mathsf{AC}^0$$ 基础上允许 threshold（majority）门，因此能完成纯 AND/OR 无法实现的全局计数与阈值判断。
- $$\mathsf{NC}^1$$：对数深度、多项式规模、有界扇入电路，允许长度为 $$O(\log n)$$ 的依赖链。
- $$\mathsf{NC}$$：polylogarithmic 深度、多项式规模电路。

这几个类的层级恰好对应并行计算能力的分层。$$\mathsf{AC}^0$$ 和 $$\mathsf{TC}^0$$ 代表极强的常数深度并行；$$\mathsf{NC}^1$$ 开始允许随输入长度对数增长的依赖链，能表达一般正则语言等状态追踪问题。
> 因此，若某个架构族的所有计算都可被常数深度电路模拟，它就无法表达真正需要随 $$n$$ 增长的顺序依赖的问题。


## 3. 乘法作为测试问题

$$n$$ 位二进制整数乘法之所以是好的测试问题，在于它集中暴露了全局进位传播的困难。二进制乘法可以分解为三步：生成部分积、对部分积求和、处理跨位进位。前两步是局部的，真正的困难在于第三步：进位链的长度随 $$n$$ 增长，常数层局部变换无法处理任意长的进位传播。

这一直觉可以用复杂度理论严格表述。标准结果表明 $$n$$ 位整数乘法（记为 $$\mathrm{MULT}$$）不属于 $$\mathsf{AC}^0$$ [8]。一种经典证明思路是将 PARITY 规约到 $$\mathrm{MULT}$$：若 $$\mathrm{MULT} \in \mathsf{AC}^0$$，则 $$\mathrm{PARITY} \in \mathsf{AC}^0$$，而后者违反已知下界。

由此得到一个直接推论：任何表达能力被 $$\mathsf{AC}^0$$ 限制的架构族，都不可能精确实现任意长度的二进制乘法。


## 4. Transformer 的电路复杂度上界

### 4.1 标准语义下的上界：$$\mathsf{TC}^0$$

对固定深度 Transformer，主流理论结果 [1] 表明其表达能力上界为 uniform $$\mathsf{TC}^0$$。直觉上，attention 的全局聚合涉及归一化、比较和加权求和，这些操作等价于 threshold 型并行聚合，超出了纯 AND/OR 门的能力，但仍可在常数深度 threshold 电路内实现。因此：

$$
\text{fixed-depth Transformer} \subseteq \mathsf{TC}^0.
$$

这意味着固定深度 Transformer 无法表达不在 $$\mathsf{TC}^0$$ 中的问题，如一般正则语言识别、复杂状态追踪、图可达性等。限制因素不是感受野——attention 已提供全局感受野——而是计算图深度为常数。

### 4.2 常数精度语义下的收紧：$$\mathsf{AC}^0$$

若对 Transformer 施加更强的限制——深度、宽度、精度均为常数，且每步算术操作后立即将结果 round 或 clip 回固定有限字母表——那么每层每个位置的隐藏状态只取有限集合中的值。此时整层计算不再是在随 $$n$$ 增长的数值域上做聚合，而是在固定有限状态空间上做计数后查表。

具体而言，对固定常数阈值 $$k$$，形如「某类 token 至少出现 $$k$$ 次」的谓词可以用深度 2、规模 $$n^k$$ 的 DNF/CNF 表示。因为 $$k$$ 为常数，这些聚合操作落在 $$\mathsf{AC}^0$$ 内。常数层复合后整体仍在 $$\mathsf{AC}^0$$ 中：

$$
\text{constant-depth, width, bit precision} + \text{immediate rounding} \subseteq \mathsf{AC}^0.
$$

### 4.3 两种语义的区别

这两个上界并不矛盾，因为它们对应不同的模型设定。讨论 Transformer 的理论表达能力时必须明确区分：

- 标准理论语义：允许 $$O(\log n)$$ 位精度或其他随 $$n$$ 增长的数值表示，上界为 $$\mathsf{TC}^0$$。
- 强受限的常数精度语义：精度和指数位数均为常数，每步立即舍入，上界收紧至 $$\mathsf{AC}^0$$。

对乘法问题而言这一区分至关重要。$$\mathrm{MULT} \notin \mathsf{AC}^0$$，因此在第二种语义下可以直接推出：常数精度的固定深度 Transformer 不能精确完成任意 $$n$$ 位整数乘法。


## 5. CoT 如何突破上界

CoT 不改变单步 Transformer 的局部算子，但改变了整个计算图的深度结构。

无 CoT 时，模型读入输入后经过常数层即输出，从输入到输出的最长依赖路径为 $$O(1)$$。有 CoT 时，模型生成一个中间 token，将其拼入上下文，再生成下一个 token。若共进行 $$t(n)$$ 步，计算图中就存在一条长度为 $$t(n)$$ 的顺序依赖链，有效深度从 $$O(1)$$ 增长为 $$O(t(n))$$。

从电路复杂度的角度看，CoT 将单次前向传播的固定浅层电路转化为可迭代的序列计算过程 [2]。一旦允许线性或多项式步数的 CoT，模型就可以模拟串行推理，状态传播、长程进位、有限自动机模拟等问题随之进入可表达范围。

因此 CoT 能处理乘法的原因在于：它提供了足够长的顺序计算轨迹，使进位可以在中间步骤中被逐步显式表示和传递。


## 6. 线性递归模型的表达能力上界

S4、Mamba 等模型的状态更新可写为

$$
\mathbf h_i = \mathbf A_i \mathbf h_{i-1} + \mathbf B_i \mathbf x_i.
$$

当 $$\mathbf A_i$$ 具有足够规则的结构时，这一递推可改写为前缀积与前缀和，进而通过 prefix scan 在 token 维度高效并行。Scan 的可行性依赖一个关键性质：局部摘要可以通过一个满足结合律的二元运算合并为全局摘要。

但这一并行性优势同时构成表达能力的约束。相关理论结果 [3] 表明，这类基本线性状态空间模型的表达能力上界同样落在 $$\mathsf{TC}^0$$ 内。虽然这些模型具有递归形式，由于递推的结构足够规则以至于整个计算可被完全并行化为 scan，计算图本质上仍是浅层的。

具有递归形式并不等价于具有更强的顺序计算能力。判断标准不在于形式上是否存在状态变量，而在于状态更新的依赖结构是否真正引入了随 $$n$$ 增长的计算深度。


## 7. 更强的 linear attention 为何丧失完全并行性

> 若某个 linear attention 或线性递归变体想显著超越 $$\mathsf{TC}^0$$，它必须打破支撑 scan 的结构约束。

完全并行的 prefix scan 要求状态更新可以表示为某种满足结合律的摘要组合：每个 chunk 的摘要独立计算，不同 chunk 的摘要通过固定的二元组合规则合并，整个序列用树形规约并行求值。

更强的 linear attention 变体通常会采取以下修改中的一种或多种：

1. 让转移矩阵以更一般的方式依赖输入；
2. 让状态更新显式依赖前一步隐状态本身，而非仅依赖固定形式的局部参数；
3. 引入不满足简单结合律的 gate 或状态选择规则。

这些修改中的任何一种都可能破坏 scan 所需的可结合摘要结构。一旦无法用固定大小的摘要、通过满足结合律的算子来精确概括任意长前缀对后缀的影响，模型就只能退回到以下方案之一：沿序列顺序展开；chunk 内并行、chunk 间顺序；或以某种近似 scan 代替精确 scan。

这不是实现层面的技术问题，而是结构性的限制：完全并行要求递推具有强可结合结构，更强的表达能力往往需要打破这一结构。 很多更强的 linear attention 变体最终采用 chunk-wise parallel，原因正在于此——状态更新本身已不再允许全局 scan。


## 8. 统一视角：有效深度

前述讨论可以通过一个统一的量来刻画：有效深度。

对架构族 $$\mathfrak A$$，定义 $$D_{\mathfrak A}(n)$$ 为输入长度为 $$n$$ 时从输入到输出的最长依赖路径长度。这个量对应序列方向上的并行时间下界：

| 有效深度 | 对应层级 | 示例 |
|---------|---------|------|
| $$O(1)$$ | $$\mathsf{AC}^0$$ 或 $$\mathsf{TC}^0$$ | fixed-depth Transformer，S4/Mamba |
| $$O(\log n)$$ | $$\mathsf{NC}^1$$ | log-depth looping [4] |
| $$O(\mathrm{polylog}(n))$$ | $$\mathsf{NC}$$ | — |
| $$O(\mathrm{poly}(n))$$ | $$\mathsf{P}$$ | CoT，强状态依赖递归模型 |
{: .table .table-striped .table-sm style="font-size: 0.5em;"}

对 constant-precision fixed-depth Transformer，虽然 $$D(n) = O(1)$$，但数值语义的进一步限制使其从 $$\mathsf{TC}^0$$ 收紧到 $$\mathsf{AC}^0$$。

> 核心观察是：要显著超越浅层并行架构的表达能力上界，有效深度必须随输入长度增长；而有效深度的增长不可避免地，反过来削弱 token 并行性。


## 9. 形式表述

将上述 trade-off 写为形式命题。设 $$\mathfrak A(g)$$ 为架构族 $$\mathfrak A$$ 中所有满足 $$D_{\mathfrak A}(n) = O(g(n))$$ 的模型子类。对目标函数 $$L$$，定义

$$
d_{\mathfrak A,L}(n) := \inf \left\{ g(n) \;\middle|\; \exists M \in \mathfrak A(g),\ M \text{ 能计算 } L \right\}.
$$

$$d_{\mathfrak A,L}(n)$$ 刻画了在架构族 $$\mathfrak A$$ 内表达 $$L$$ 所需的最小有效深度。由此可得以下推论，均直接从复杂度类的定义和已知分离结果导出：

- 若 $$L \notin \mathsf{TC}^0$$，则任何 $$D(n) = O(1)$$ 且可被 uniform 常数深度 threshold 电路模拟的架构族不能表达 $$L$$。
- 若 $$L \notin \mathsf{AC}^0$$，则任何被 uniform $$\mathsf{AC}^0$$ 电路模拟的架构族不能表达 $$L$$。
- 若 $$L$$ 是 $$\mathsf{NC}^1$$-hard 的，则表达 $$L$$ 的有效深度不能始终为 $$O(1)$$。
- 若 $$L$$ 是 $$\mathsf{P}$$-complete 的，则除非 $$\mathsf{NC} = \mathsf{P}$$，仅有 polylog 深度的并行模型不能表达 $$L$$。

这即并行性—表达能力 trade-off 的形式表述。


## 10. 回答两个问题

### 10.1 问题一

在常数精度、逐步舍入的语义下，设此类 Transformer 构成的架构族为 $$\mathcal T$$。由第 4.2 节，$$\mathcal T \subseteq \mathsf{AC}^0$$；由第 3 节，$$\mathrm{MULT} \notin \mathsf{AC}^0$$。因此 $$\mathrm{MULT} \notin \mathcal T$$。

这一不可能性的根源在于计算图的深度和数值状态空间均为常数，无法承载随 $$n$$ 增长的全局进位传播。CoT 通过引入中间生成步骤将有效深度扩展为 $$O(t(n))$$，从而使进位可被逐步传递。

### 10.2 问题二

完全 token 并行性依赖 prefix scan 的可行性，后者要求状态更新满足强可结合结构。满足这一结构的基本线性递归模型受限于 $$\mathsf{TC}^0$$。增强表达能力所需的修改——更一般的输入依赖、对前一步隐状态的显式依赖、不满足结合律的 gate——恰好破坏 scan 结构，导致模型退化为 chunk-wise 并行或顺序计算。

因此并行性与表达能力之间的 trade-off 是结构性的：保持最强 token 并行性将架构约束在 $$\mathsf{AC}^0$$ 或 $$\mathsf{TC}^0$$ 内；超越这一层级必然要求有效深度随 $$n$$ 增长，从而削弱并行性。




## 11. 总结

本文的核心论点是：更强的顺序表达能力必须以有效深度增长为代价。

固定深度 Transformer 和可完全 scan 的基本线性递归模型同属极浅层并行计算，表达能力上界为 $$\mathsf{TC}^0$$（常数精度逐步舍入语义下收紧至 $$\mathsf{AC}^0$$）。任意 $$n$$ 位整数乘法、一般正则语言识别、复杂状态追踪等任务要求处理随 $$n$$ 增长的依赖链，不在这些上界范围内。CoT、looping、更强的状态依赖递归能突破这些上界，原因统一在于它们引入了随 $$n$$ 增长的有效计算深度；而这一增长不可避免地削弱最强形式的 token 并行性。

> 两个问题的答案由此统一：常数精度固定深度 Transformer 不能精确做任意长度乘法，更强的 linear attention 不能保持完全并行——根源是同一个，即并行计算-表达能力 trade-off。


## 参考文献

[1] William Merrill, Ashish Sabharwal, and Noah A. Smith. *[Saturated Transformers are Constant-Depth Threshold Circuits](https://aclanthology.org/2022.tacl-1.49/).* TACL, 2022.

[2] Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. *[Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://openreview.net/forum?id=3EWTEy9MTM).* ICLR, 2024.

[3] William Merrill, Jackson Petty, and Ashish Sabharwal. *[The Illusion of State in State-Space Models](https://proceedings.mlr.press/v235/merrill24a.html).* ICML, 2024.

[4] William Merrill and Ashish Sabharwal. *[A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](https://openreview.net/forum?id=5pHfYe10iX).* 2025.

[5] William Merrill and Ashish Sabharwal. *[Exact Expressive Power of Transformers with Padding](https://openreview.net/forum?id=O1abxStFcy).* 2025.

[6] Bo Peng et al. *[RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://openreview.net/forum?id=ayB1PACN5j).* 2025.

[7] Yiding Hao, Dana Angluin, and Robert Frank. *[Formal Language Recognition by Hard Attention Transformers: Perspectives from Circuit Complexity](https://aclanthology.org/2022.tacl-1.46/).* TACL, 2022.

[8] Neil Immerman and Susan Landau. *[The Complexity of Iterated Multiplication](https://www.cs.umass.edu/~immerman/pub/mult.pdf).* Information and Computation, 1995.

[9] Sanjeev Arora and Boaz Barak. *[Computational Complexity: A Modern Approach](https://theory.cs.princeton.edu/complexity/).* 2009.


## 引用

```bibtex
@article{zou2026parallel_expressiveness_tradeoff_linear_attention,
  title   = {并行性与表达能力的权衡：从 AC0/TC0 到 Linear Attention 的理论边界},
  author  = {Zou, Jiaxuan},
  journal = {Jiaxuan's Blog},
  year    = {2026}
}
```