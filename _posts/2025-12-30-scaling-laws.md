---
title: "Can We Derive Scaling Law From First Principles?"
date: 2025-12-30
permalink: /posts/2025/12/scaling-laws/
tags:
  - scaling law
  - deep learning
  - theory
---

We are excited to share our latest work titled **"Can We Derive Scaling Law From First Principles?"**.

In this work, we explore the theoretical foundations of scaling laws in deep learning. We attempt to derive these laws from first principles, offering new insights into why neural networks scale the way they do.

我们很高兴分享我们的最新工作 **"Can We Derive Scaling Law From First Principles?"**。

在这项工作中，我们探索了深度学习中缩放定律（Scaling Laws）的理论基础。我们尝试从第一性原理推导这些定律，为神经网络为何如此缩放提供了新的见解。

You can read the full paper here / 您可以在此处阅读完整论文：

[Read PDF / 阅读 PDF](/files/Can_We_Derive_Scaling_Law_From_First_Principles.pdf)

### Abstract / 摘要

In LLM pre-training, the loss function exhibits stable scaling laws with respect to model size $N$, data size $D$, and compute $C$. Moreover, there exists an approximate power-law compute-optimal frontier given a compute budget; that is, the line connecting optimal compute points also follows a power law (covering four common scaling laws). Existing theories often rely on specific assumptions regarding kernel spectra or Teacher coefficients. This paper aims to propose a first-principles explanatory framework based on statistical properties of data, without assuming NTK linearization or specific spectral distributions. We demonstrate that scaling laws stem from the scale-free long-tail structure of language data and the implicit ordering preference of gradient descent.

We abstract language modeling as a stepwise mastery of modes following a Zipf distribution ($p_k \propto k^{-\alpha}$). Based on the "greedy coverage" nature of gradient descent to prioritize reducing dominant error terms, the total reducible loss is modeled as the tail integral of unmastered mode weights. In this framework, we analytically derive power-law scaling forms for model size $N$, data size $D$, and compute $C$, with exponents $-\gamma(\alpha-1)$, $-\frac{\alpha-1}{\alpha}$, and $-\frac{\alpha-1}{\alpha\beta}$ respectively. Furthermore, we reconstruct the compute-optimal frontier under a unified bottleneck geometry, proving the consistency between Kaplan's and Chinchilla's results.

LLM 预训练中损失函数随模型规模 $N$、数据规模 $D$ 与计算量 $C$ 呈现稳定的幂律（scaling laws），并且在给定算力预算下存在近似幂律的 compute-optimal 前沿，也就是说 optimal compute point 的连线也构成幂律。（一共四种常见 scaling law）已有理论常依赖对核谱或 Teacher 系数的特定假设。本文旨在不预设 NTK 线性化或特定谱分布的前提下，提出一种基于数据统计特性的第一性解释框架：证明幂律源于语言数据的无尺度长尾结构与梯度下降的隐式排序偏好。

我们将语言建模抽象为对服从 Zipf 分布（$p_k \propto k^{-\alpha}$）的模式（modes）的逐步掌握过程。基于梯度下降优先降低主导误差项的“贪婪覆盖”性质，总体可约损失被建模为未掌握模式权重的尾部积分。在此框架下，我们解析推导得到模型规模 $N$、数据规模 $D$ 与计算量 $C$ 的幂律缩放形式，其指数分别为 $-\gamma(\alpha-1)$、$-\frac{\alpha-1}{\alpha}$ 与 $-\frac{\alpha-1}{\alpha\beta}$。进一步地，我们在统一的瓶颈几何下重构了 compute-optimal 前沿，证明 Kaplan 与 Chinchilla 工作的一致性。

### Citation / 引用

If you find this work useful, please cite it as follows: / 如果您觉得这项工作有用，请按如下方式引用：

```bibtex
@article{zou2025scaling,
  title={Can We Derive Scaling Law From First Principles?},
  author={Zou, Jiaxuan},
  journal={Preprint},
  year={2025}
}
```
