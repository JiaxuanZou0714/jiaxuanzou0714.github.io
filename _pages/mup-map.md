---
layout: page
title: μP Map
permalink: /mup-map/
description: μP 相关博客的阅读导航与脉络梳理。
nav: true
nav_order: 2
---

这是一页专门整理 μP 相关文章脉络的导航页。与其把这些文章看成几篇彼此分散的 blog，我更希望把它们看成围绕同一个核心问题展开的几条互相补充的推导路线：为什么大模型在放宽宽度时仍然能够保持 feature learning，以及相应的初始化、学习率和优化器缩放规律到底应该如何理解。

## 核心问题

这些文章大体都围绕下面这个问题展开：

1. 当模型宽度变大时，怎样保持 feature learning，而不是退化到 lazy learning。
2. 如果想让小模型上调好的超参数迁移到大模型，缩放规律应该怎么推。
3. 对不同架构和不同优化器，哪些量才是真正应该对齐的对象。

## 总体脉络

如果把整组文章压缩成一条主线，那么它的结构大致如下：

1. 背景问题：先解释 scaling law、feature learning 与宽度扩展为什么值得研究。
2. Tensor Programs 路线：从谱条件、LLN/CLT 与参数化规律出发，推导出 μP。
3. 球面动力学路线：在 RMSNorm 架构下，把对齐问题重新解释为球面上的动力学对齐。
4. Hyperball 扩展：进一步把优化器与范数约束纳入同一个几何框架。
5. 细节补充：把 Hyperball 一文中用到但未展开的更新矩阵范数估计单独补齐。

## 推荐阅读顺序

### 1. 背景与现象

这部分现在更适合作为论文入口来读：先看 [Effective Frontiers: A Unification of Neural Scaling Laws](https://arxiv.org/abs/2602.02593)。

这篇 paper 基本承接了原来 [Can We Derive Scaling Law From First Principles?]({% post_url 2025-12-30-scaling-law %}) 那篇 blog 所对应的问题意识，也就是“为什么 scaling law 可以被第一性原理解释”这个大问题。放在这条阅读路径里，它更适合作为正式版本的背景入口，而旧 blog 可以看作一个更早期的网页落点。

接着可以看 [《如何对齐不同初始化大小下的 Data scaling 曲线》]({% post_url 2026-02-01-data-scaling-and-std %})。

这篇偏经验现象，讨论初始化标准差变化之后，data scaling 曲线为什么会偏移。它和后面的 μP 文章之间的关系是：前者展示现象，后者尝试解释什么样的缩放才会让不同宽度模型保持一致的学习动力学。

### 2. Tensor Programs 路线

这一条线适合想从 Greg Yang 的原始理论框架出发理解 μP 的读者。

第一篇是 [《Tensor Programs (一)：从Feature Learning 的谱条件到 μP》]({% post_url 2026-02-14-spectral-condition-feature-learning %})。

这篇的重点不是直接给出一套参数化规则，而是先回答一个更基本的问题：为什么我们需要 μP。它从 feature learning 的保持条件出发，用谱范数和一步更新的视角解释宽度扩展时哪些量必须保持稳定。

第二篇是 [《Tensor Programs (二)：从Tensor Programs到 μP》]({% post_url 2026-03-02-tensor-programs-mup-intuition %})。

这篇可以看成第一篇的推进版。它把 Tensor Programs 路线里最关键的统计学 insight 拆开说明：初始化为什么由 CLT 主导，而梯度更新为什么由 LLN 主导。也正是在这里，μP 的初始化缩放和学习率缩放被更系统地整理出来。

如果你希望先把 Tensor Programs 这条线读完整，再看几何路线，那么读完这两篇再往后走最自然。

### 3. 球面动力学路线

如果你更关心 RMSNorm 架构下“真正被对齐的对象是什么”，那么可以从 [《球面之上：从球面动力学到 μP》]({% post_url 2026-03-04-spherical-dynamics-mup %}) 开始。

这篇文章的重要性在于，它没有继续沿用 Tensor Programs 的概率极限定理语言，而是换了一个视角：在 RMSNorm 下，真正需要对齐的并不是参数模长本身，而是特征在超球面上的演化速率。换句话说，这篇文章提供了和 Tensor Programs 并行、但更几何化的一条 μP 推导路径。

从阅读关系上说，这篇既可以放在 Tensor Programs（二）之后作为对照阅读，也可以单独作为几何路线的起点。

### 4. Hyperball 与优化器扩展

在球面动力学路线的基础上，下一篇自然就是 [《球面之上：带有 Hyperball 机制的优化器的 μP 缩放》]({% post_url 2026-03-06-spherical-hyperball %})。

这篇文章把问题继续向前推进了一步：如果不仅讨论 SGD，而是把 Adam、Muon 以及 Hyperball 这样的范数约束机制一起放进来，跨尺度对齐条件会怎么变化。它的核心作用是把“球面动力学”从基础几何分析推进到“不同优化器如何缩放”的层面。

如果说 [《球面之上：从球面动力学到 μP》]({% post_url 2026-03-04-spherical-dynamics-mup %}) 主要解决的是基础框架问题，那么这篇解决的就是优化器层面的推广问题。

### 5. 范数估计的补充说明

最后读 [《Adam 与 Muon 优化器更新矩阵的 Frobenius 范数估计》]({% post_url 2026-03-08-optimizer-update-matrix-norm %})。

这篇文章的定位更像一篇“技术补注”。在 Hyperball 那篇里，AdamH 与 MuonH 的推导都用了 $\lVert u_t \rVert_F = \Theta(n)$ 这个量级结论；而这篇文章的作用，就是把这个结论单独拆出来，专门讨论 Adam 与 Muon 更新矩阵的 Frobenius 范数为什么会落在这个量级上。

因此，它最适合放在 Hyperball 一文之后阅读。这样读的时候，你会很清楚它不是另起炉灶的一篇独立博客，而是在补 Hyperball 推导中一个被暂时压缩掉的技术前提。

## 两条主线的关系

如果只保留最核心的结构，那么目前 μP 相关博客其实可以分成两条主线。

### 主线 A：Tensor Programs 视角

1. [《Tensor Programs (一)：从Feature Learning 的谱条件到 μP》]({% post_url 2026-02-14-spectral-condition-feature-learning %})
2. [《Tensor Programs (二)：从Tensor Programs到 μP》]({% post_url 2026-03-02-tensor-programs-mup-intuition %})

这条线强调的是：通过宽度极限、谱条件、LLN/CLT 与 Tensor Programs 的形式化工具，理解 μP 为什么成立。

### 主线 B：球面动力学视角

1. [《球面之上：从球面动力学到 μP》]({% post_url 2026-03-04-spherical-dynamics-mup %})
2. [《球面之上：带有 Hyperball 机制的优化器的 μP 缩放》]({% post_url 2026-03-06-spherical-hyperball %})
3. [《Adam 与 Muon 优化器更新矩阵的 Frobenius 范数估计》]({% post_url 2026-03-08-optimizer-update-matrix-norm %})

这条线强调的是：在 RMSNorm 与范数约束的设定下，把缩放问题重新解释为几何动力学和优化器更新结构的问题。

## 如何使用这页导航

如果你是第一次读这些文章，建议直接按“推荐阅读顺序”从前往后读。

如果你已经大致知道 μP 是什么，但想补理论细节，可以直接进入下面任一分支：

1. 想看 Greg Yang 那条原始理论路线：从 [《Tensor Programs (一)：从Feature Learning 的谱条件到 μP》]({% post_url 2026-02-14-spectral-condition-feature-learning %}) 开始。
2. 想看 RMSNorm 与几何动力学路线：从 [《球面之上：从球面动力学到 μP》]({% post_url 2026-03-04-spherical-dynamics-mup %}) 开始。
3. 想直接看优化器层面的推广：从 [《球面之上：带有 Hyperball 机制的优化器的 μP 缩放》]({% post_url 2026-03-06-spherical-hyperball %}) 开始。

后续如果 μP 相关博客继续增加，这一页也会继续更新。