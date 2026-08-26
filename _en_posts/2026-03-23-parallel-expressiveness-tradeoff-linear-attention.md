---
source_sha: 1e027c4ab5c78a73
layout: post
title: "The Trade-off Between Parallelism and Expressiveness: Theoretical Boundaries from $AC^0$/$TC^0$ to Linear Attention"
date: 2026-03-23 12:00:00
description: "From the perspective of circuit complexity, this post gives a unified explanation of why constant-depth Transformers cannot exactly perform integer multiplication of arbitrary length, and why stronger linear attention variants often fail to maintain full token parallelism."
tags: [llm, reasoning, transformer, linear-attention, complexity-theory]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/parallel-expressiveness-tradeoff-linear-attention/
ref: parallel-expressiveness-tradeoff-linear-attention
related_posts: false
---

> This post is based on a note I wrote half a year ago, titled "When we talk about LLM reasoning, what are we talking about?" At that time, I had just joined Gaoling and was working on LLM latent reasoning (although now I think neither CoT nor latent reasoning touches the essence). Recently, someone revisited that note, and I found some parts outdated, so I reorganized it and published it as a blog post.

## 1. Two Questions

This post revolves around two questions.

Question 1. Consider a Transformer with constant width, constant depth, and constant precision. Given two $$n$$-bit binary integers, can this architecture compute their product exactly?

The core difficulty of binary multiplication lies in carry propagation: the results of local operations on the low bits affect the high bits, and the length of the chain of influence grows with $$n$$. If each position in each layer can only store a constant number of bits, and the total computational depth does not grow with $$n$$, then when $$n$$ is large enough, the model lacks sufficient internal computational depth to propagate this carry chain exactly. In contrast, chain of thought (CoT) allows the model to write intermediate results back to the context and continue generating new tokens, thereby making the effective depth of the computational graph grow with the number of output steps, so it can in principle process carries step by step.

Question 2. Among various linear attention variants, why do models with stronger expressiveness often fail to maintain the same level of full token parallelism as the most basic linear recurrent model?

The following phenomenon recurs in practice: the architectures that are easiest to scan have the weakest expressiveness, whereas variants that introduce more complex gates, stronger state updates, or state transitions that explicitly depend on the current hidden state do gain expressiveness, yet can often achieve only chunk-wise parallelism, or even degrade to parallel training with sequential inference.

These two questions point to the same core contradiction:
> In sequence modeling, is there a structural trade-off between stronger token parallelism and stronger sequential expressiveness?

This post uses circuit complexity to give an affirmative answer. The specific conclusions are as follows:

1. The expressiveness upper bound of fixed-depth Transformers is $$\mathsf{TC}^0$$. If further restricted to constant-bit precision with stepwise rounding semantics, the upper bound tightens to $$\mathsf{AC}^0$$.
2. Fully scannable linear recurrent models such as S4 and Mamba are also limited to $$\mathsf{TC}^0$$.
3. To significantly surpass this level—for example, to recognize general regular languages, complete long-range carry propagation, or perform complex state tracking—the model must introduce effective computational depth that grows with input length. Whether this growth takes the form of CoT, looping, or stronger state dependence, it will weaken token parallelism in the strongest sense.


## 2. Circuit Complexity Basics

This post only requires a small amount of background knowledge.

A Boolean circuit is a directed acyclic graph with bits as inputs and logic gates as internal nodes. The two key complexity measures are depth (the longest path length from input to output) and size (the number of gates or wires, usually required to be polynomial in the input length $$n$$).

This post involves four standard complexity classes:

$$
\mathsf{AC}^0 \subseteq \mathsf{TC}^0 \subseteq \mathsf{NC}^1 \subseteq \mathsf{NC}.
$$

- $$\mathsf{AC}^0$$: constant-depth, polynomial-size, unbounded fan-in AND/OR/NOT circuits.
- $$\mathsf{TC}^0$$: allows threshold (majority) gates on top of $$\mathsf{AC}^0$$, and can therefore perform global counting and threshold tests that pure AND/OR circuits cannot implement.
- $$\mathsf{NC}^1$$: logarithmic-depth, polynomial-size, bounded fan-in circuits, allowing dependency chains of length $$O(\log n)$$.
- $$\mathsf{NC}$$: polylogarithmic-depth, polynomial-size circuits.

The hierarchy of these classes exactly corresponds to the layering of parallel computational capability. $$\mathsf{AC}^0$$ and $$\mathsf{TC}^0$$ represent extremely strong constant-depth parallelism; $$\mathsf{NC}^1$$ begins to allow dependency chains that grow logarithmically with input length, and can therefore express state tracking problems such as general regular languages.
> Therefore, if all computations of an architecture family can be simulated by constant-depth circuits, it cannot express problems that truly require sequential dependencies growing with $$n$$.


## 3. Multiplication as a Test Problem

$$n$$-bit binary integer multiplication is a good test problem because it isolates the difficulty of global carry propagation. Binary multiplication can be decomposed into three steps: generating partial products, summing the partial products, and handling cross-bit carries. The first two steps are local; the real difficulty lies in the third: the carry chain length grows with $$n$$, and constant-depth local transformations cannot handle arbitrarily long carry propagation.

This intuition can be rigorously stated in complexity theory. Standard results show that $$n$$-bit integer multiplication (denoted $$\mathrm{MULT}$$) is not in $$\mathsf{AC}^0$$ [8]. One classical proof idea is to reduce PARITY to $$\mathrm{MULT}$$: if $$\mathrm{MULT} \in \mathsf{AC}^0$$, then $$\mathrm{PARITY} \in \mathsf{AC}^0$$, which violates known lower bounds.

This yields a direct corollary: any architecture family whose expressiveness is limited to $$\mathsf{AC}^0$$ cannot exactly implement binary multiplication of arbitrary length.


## 4. Upper Bounds on the Circuit Complexity of Transformers

### 4.1 Upper bound under standard semantics: $$\mathsf{TC}^0$$

For fixed-depth Transformers, the mainstream theoretical result [1] shows that their expressiveness is upper-bounded by uniform $$\mathsf{TC}^0$$. Intuitively, the global aggregation in attention involves normalization, comparison, and weighted summation, which are equivalent to threshold-type parallel aggregation, exceeding the capability of pure AND/OR gates, but still implementable within constant-depth threshold circuits. Therefore:

$$
\text{fixed-depth Transformer} \subseteq \mathsf{TC}^0.
$$

This means that fixed-depth Transformers cannot express problems not in $$\mathsf{TC}^0$$, such as general regular language recognition, complex state tracking, and graph reachability. The limiting factor is not the receptive field—attention already provides a global receptive field—but the constant depth of the computational graph.

### 4.2 Tightening under constant-precision semantics: $$\mathsf{AC}^0$$

If we impose stronger restrictions on Transformers—depth, width, and precision are all constant, and after each arithmetic operation the result is immediately rounded or clipped back to a fixed finite alphabet—then the hidden state at each layer and position takes values only in a finite set. At this point, the computation of an entire layer is no longer aggregation over a numerical domain that grows with $$n$$, but rather counting over a fixed finite state space followed by table lookup.

Specifically, for a fixed constant threshold $$k$$, predicates of the form “a certain type of token appears at least $$k$$ times” can be represented by a DNF/CNF of depth 2 and size $$n^k$$. Since $$k$$ is constant, these aggregation operations fall within $$\mathsf{AC}^0$$. After composing a constant number of layers, the whole remains in $$\mathsf{AC}^0$$:

$$
\text{constant-depth, width, bit precision} + \text{immediate rounding} \subseteq \mathsf{AC}^0.
$$

### 4.3 Difference between the two semantics

These two upper bounds are not contradictory because they correspond to different model settings. When discussing the theoretical expressiveness of Transformers, one must clearly distinguish:

- Standard theoretical semantics: allows $$O(\log n)$$-bit precision or other numerical representations that grow with $$n$$, with an upper bound of $$\mathsf{TC}^0$$.
- Strongly restricted constant-precision semantics: precision and exponent bit-width are constant, with immediate rounding at each step, tightening the upper bound to $$\mathsf{AC}^0$$.

This distinction is crucial for the multiplication problem. $$\mathrm{MULT} \notin \mathsf{AC}^0$$, so under the second semantics one can directly derive: a constant-precision fixed-depth Transformer cannot exactly perform arbitrary $$n$$-bit integer multiplication.


## 5. How CoT Breaks Through the Upper Bound

CoT does not change the local operator of a single Transformer step, but it changes the depth structure of the entire computational graph.

Without CoT, the model reads the input and outputs after a constant number of layers, so the longest dependency path from input to output is $$O(1)$$. With CoT, the model generates an intermediate token, appends it to the context, and then generates the next token. If a total of $$t(n)$$ steps are performed, there exists a sequential dependency chain of length $$t(n)$$ in the computational graph, and the effective depth grows from $$O(1)$$ to $$O(t(n))$$.

From the perspective of circuit complexity, CoT transforms the fixed shallow circuit of a single forward pass into an iterative sequential computation process [2]. Once a linear or polynomial number of CoT steps is allowed, the model can simulate serial reasoning, and problems such as state propagation, long-range carry propagation, and finite automaton simulation come within the expressible range.

Therefore, the reason CoT can handle multiplication is that it provides a sufficiently long sequential computation trajectory, allowing carries to be explicitly represented and propagated step by step in intermediate steps.


## 6. Expressiveness Upper Bounds for Linear Recurrent Models

The state updates of models such as S4 and Mamba can be written as

$$
\mathbf h_i = \mathbf A_i \mathbf h_{i-1} + \mathbf B_i \mathbf x_i.
$$

When $$\mathbf A_i$$ has sufficiently regular structure, this recurrence can be rewritten as prefix products and prefix sums, and then efficiently parallelized over the token dimension via prefix scan. The feasibility of scan relies on a key property: local summaries can be combined into a global summary via a binary operation that satisfies associativity.

But this parallelism advantage also constitutes a constraint on expressiveness. Related theoretical results [3] show that the expressiveness upper bound of such basic linear state space models also falls within $$\mathsf{TC}^0$$. Although these models have a recurrent form, the recurrence is regular enough that the entire computation can be fully parallelized into a scan, so the computational graph remains essentially shallow.

Having a recurrent form is not equivalent to having stronger sequential computational power. The criterion is not whether state variables exist formally, but whether the dependency structure of state updates truly introduces computational depth that grows with $$n$$.


## 7. Why Stronger Linear Attention Loses Full Parallelism

> If a linear attention or linear recurrent variant aims to significantly surpass $$\mathsf{TC}^0$$, it must break the structural constraints that support the scan.

Fully parallel prefix scan requires that state updates can be expressed as a combination of summaries satisfying associativity: each chunk's summary is computed independently, summaries of different chunks are merged via a fixed binary combination rule, and the entire sequence is evaluated in parallel via tree reduction.

Stronger linear attention variants typically adopt one or more of the following modifications:

1. Let the transition matrix depend on the input in a more general way;
2. Let the state update explicitly depend on the previous hidden state itself, rather than only on fixed-form local parameters;
3. Introduce gate or state selection rules that do not satisfy simple associativity.

Any of these modifications may break the associative summary structure required by scan. Once a fixed-size summary combined through an associative operator can no longer capture exactly the effect of an arbitrarily long prefix on the suffix, the model can only fall back to one of the following schemes: unfolding sequentially along the sequence; parallelizing within chunks and proceeding sequentially across chunks; or replacing the exact scan with some approximate scan.

This is not an implementation-level technical issue but a structural limitation: full parallelism requires the recurrence to have a strongly associative structure, and stronger expressiveness often requires breaking this structure. Many stronger linear attention variants ultimately adopt chunk-wise parallelism precisely because the state update itself no longer allows a global scan.


## 8. A Unified Perspective: Effective Depth

The above discussion can be characterized by a unified quantity: effective depth.

For an architecture family $$\mathfrak A$$, define $$D_{\mathfrak A}(n)$$ as the length of the longest dependency path from input to output when the input length is $$n$$. This quantity corresponds to the parallel time lower bound in the sequence direction:

| Effective Depth | Corresponding Level | Example |
|---------|---------|------|
| $$O(1)$$ | $$\mathsf{AC}^0$$ or $$\mathsf{TC}^0$$ | fixed-depth Transformer, S4/Mamba |
| $$O(\log n)$$ | $$\mathsf{NC}^1$$ | log-depth looping [4] |
| $$O(\mathrm{polylog}(n))$$ | $$\mathsf{NC}$$ | — |
| $$O(\mathrm{poly}(n))$$ | $$\mathsf{P}$$ | CoT, strongly state-dependent recurrent models |
{: .table .table-striped .table-sm .w-auto .mx-auto style="font-size: 0.8em;"}

For constant-precision fixed-depth Transformers, although $$D(n) = O(1)$$, further restrictions on numerical semantics tighten it from $$\mathsf{TC}^0$$ to $$\mathsf{AC}^0$$.

> The core observation is: to significantly surpass the expressiveness upper bound of shallow parallel architectures, the effective depth must grow with the input length; and growth in effective depth inevitably weakens token parallelism in turn.


## 9. Formal Statement

Write the above trade-off as a formal proposition. Let $$\mathfrak A(g)$$ be the subclass of all models in the architecture family $$\mathfrak A$$ that satisfy $$D_{\mathfrak A}(n) = O(g(n))$$. For a target function $$L$$, define

$$
d_{\mathfrak A,L}(n) := \inf \left\{ g(n) \;\middle|\; \exists M \in \mathfrak A(g),\ M \text{ computes } L \right\}.
$$

$$d_{\mathfrak A,L}(n)$$ characterizes the minimum effective depth required to express $$L$$ within the architecture family $$\mathfrak A$$. This yields the following corollaries, all of which follow directly from the definitions of the complexity classes and known separation results:

- If $$L \notin \mathsf{TC}^0$$, then any architecture family with $$D(n) = O(1)$$ that can be simulated by uniform constant-depth threshold circuits cannot express $$L$$.
- If $$L \notin \mathsf{AC}^0$$, then any architecture family that can be simulated by uniform $$\mathsf{AC}^0$$ circuits cannot express $$L$$.
- If $$L$$ is $$\mathsf{NC}^1$$-hard, then the effective depth required to express $$L$$ cannot always be $$O(1)$$.
- If $$L$$ is $$\mathsf{P}$$-complete, then unless $$\mathsf{NC} = \mathsf{P}$$, parallel models with only polylog depth cannot express $$L$$.

This is the formal statement of the parallelism–expressiveness trade-off.


## 10. Answering Two Questions

### 10.1 Question 1

Under the semantics of constant precision and stepwise rounding, let the architecture family formed by such Transformers be $$\mathcal T$$. By Section 4.2, $$\mathcal T \subseteq \mathsf{AC}^0$$; by Section 3, $$\mathrm{MULT} \notin \mathsf{AC}^0$$. Therefore $$\mathrm{MULT} \notin \mathcal T$$.

The root of this impossibility lies in the fact that both the depth of the computational graph and the numerical state space are constant, which cannot accommodate global carry propagation that grows with $$n$$. CoT expands the effective depth to $$O(t(n))$$ by introducing intermediate generation steps, allowing carries to be propagated step by step.

### 10.2 Question 2

Full token parallelism relies on the feasibility of prefix scan, which requires the state update to have a strongly associative structure. Basic linear recurrent models that have this structure are limited to $$\mathsf{TC}^0$$. The modifications needed to enhance expressiveness—more general input dependence, explicit dependence on the previous hidden state, and gates that do not satisfy associativity—are precisely what breaks the scan structure, causing the model to degenerate into chunk-wise parallelism or sequential computation.

Therefore, the trade-off between parallelism and expressiveness is structural: maintaining the strongest token parallelism constrains the architecture within $$\mathsf{AC}^0$$ or $$\mathsf{TC}^0$$; surpassing this level necessarily requires the effective depth to grow with $$n$$, thereby weakening parallelism.




## 11. Summary

The core argument of this post is that stronger sequential expressiveness must come at the cost of growth in effective depth.

Fixed-depth Transformers and basic linear recurrent models that can be fully scanned both fall under extremely shallow parallel computation, with an expressiveness upper bound of $$\mathsf{TC}^0$$ (tightened to $$\mathsf{AC}^0$$ under constant precision stepwise rounding semantics). Tasks such as arbitrary $$n$$-bit integer multiplication, general regular language recognition, and complex state tracking require handling dependency chains that grow with $$n$$, and lie outside these upper bounds. CoT, looping, and stronger state-dependent recurrence can break these upper bounds, and in each case the reason is that they introduce effective computational depth that grows with $$n$$; this growth inevitably weakens the strongest form of token parallelism.

> The answers to the two questions are thus unified: constant-precision fixed-depth Transformers cannot exactly perform arbitrary-length multiplication, and stronger linear attention cannot maintain full parallelism—the root cause is the same, namely the trade-off between parallel computation and expressiveness.


## References

[1] William Merrill, Ashish Sabharwal, and Noah A. Smith. *[Saturated Transformers are Constant-Depth Threshold Circuits](https://aclanthology.org/2022.tacl-1.49/).* TACL, 2022.

[2] Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. *[Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://openreview.net/forum?id=3EWTEy9MTM).* ICLR, 2024.

[3] William Merrill, Jackson Petty, and Ashish Sabharwal. *[The Illusion of State in State-Space Models](https://proceedings.mlr.press/v235/merrill24a.html).* ICML, 2024.

[4] William Merrill and Ashish Sabharwal. *[A Little Depth Goes a Long Way: The Expressive Power of Log-Depth Transformers](https://openreview.net/forum?id=5pHfYe10iX).* 2025.

[5] William Merrill and Ashish Sabharwal. *[Exact Expressive Power of Transformers with Padding](https://openreview.net/forum?id=O1abxStFcy).* 2025.

[6] Bo Peng et al. *[RWKV-7 "Goose" with Expressive Dynamic State Evolution](https://openreview.net/forum?id=ayB1PACN5j).* 2025.

[7] Yiding Hao, Dana Angluin, and Robert Frank. *[Formal Language Recognition by Hard Attention Transformers: Perspectives from Circuit Complexity](https://aclanthology.org/2022.tacl-1.46/).* TACL, 2022.

[8] Neil Immerman and Susan Landau. *[The Complexity of Iterated Multiplication](https://www.cs.umass.edu/~immerman/pub/mult.pdf).* Information and Computation, 1995.

[9] Sanjeev Arora and Boaz Barak. *[Computational Complexity: A Modern Approach](https://theory.cs.princeton.edu/complexity/).* 2009.


## Citation

```bibtex
@article{zou2026parallel_expressiveness_tradeoff_linear_attention,
  title   = {并行性与表达能力的权衡：从 AC0/TC0 到 Linear Attention 的理论边界},
  author  = {Zou, Jiaxuan},
  journal = {Jiaxuan's Blog},
  year    = {2026}
}
```