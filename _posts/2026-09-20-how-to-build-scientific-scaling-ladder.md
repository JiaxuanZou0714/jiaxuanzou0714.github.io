---
layout: post
title: "如何搭建一个科学的 Scaling Ladder"
date: 2026-09-20 10:00:00
description: "系统整理 Scaling Ladder 的设计与搭建：从实验矩阵、模型尺寸与训练量选择、超参搜索、Loss Scaling Law 拟合，到下游任务预测与常见陷阱，综合 Chinchilla、DeepSeek、StepFun、Cerebras、River Valley WSD 等公开文献与工程实践。"
tags: [scaling-laws, pretraining, hyperparameter, optimization, llm, empirical-methodology]
categories: [research]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: zh-CN
---

本文综合整理自**<u>公开</u>**文献与工程实践经验，**<u>不包含任何涉密内容</u>**，系统介绍如何搭建一个科学的 Scaling Ladder——通过小模型实验拟合 Scaling Law 并外推大模型表现，从而在启动大模型训练前预知最优配置与预期性能。

---

## 1. 核心概念与目标

### 1.1 什么是 Scaling Ladder

Scaling Ladder 是一组不同模型规模（参数量 N）× 不同训练数据量（D）的实验点矩阵，用于拟合 Scaling Law 并外推大模型表现。每个"横档"是一个模型尺寸，每个"竖档"是一个训练量档位，交叉点就是一个实际训练实验。它的具体设计原则在 [§2.1](#21-实验矩阵结构) 展开。

### 1.2 为什么需要科学的 Ladder

Scaling Laws 的本质是训练配置、过程、结果的可预估。通过小模型上的大量实验得出规律形式，进一步外推预测大模型的表现，使得在启动训练之前就可以预估大模型的训练配置、Loss 甚至下游任务 Performance，降低大规模训练配置失误的风险。

如果没有 Scaling Law，对于大模型最终可以达到的能力和表现是非常模糊的，也无法抉择该使用多大的模型、训练多少 Token 量、如何配置训练参数。训练一次大模型成本和风险很大，如果训到最后发现效果不及预期，会造成非常严重的资源浪费。

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 1. Delphi 首次实验（Cautious AdamC 配方）。右图的大规模训练偏离预测并发散。<a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">来源：Marin, 2025</a>。'
  alt='Delphi 首次 scaling 实验：10 的 22 次方 FLOPs 运行的 loss 比预测高 2.5%，10 的 23 次方 FLOPs 运行发散。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

一个科学的 scaling ladder，应该能做到经过正式发版大模型验证，预估精度可达到：Training Loss diff ≤ 0.02。

### 1.3 Ladder 的核心产出

一个科学的 Scaling Ladder 应该产出以下可外推的规律：

| 产出 | 用途 |
|---|---|
| 超参 Scaling Law（LR/BSZ/WD） | 预测大模型最优训练超参 |
| Loss Scaling Law | 预测大模型最终 Training/Eval Loss |
| Loss Curve Scaling Law | 预测大模型完整训练曲线 |
| 退火比例 Scaling Law | 预测大模型最优 LR decay 比例 |
| 数据重复 Scaling Law | 预测多 epoch 训练的等效效果 |
| 下游任务 Scaling Law | 预测大模型 Benchmark 指标 |

## 2. Ladder 整体设计

### 2.1 实验矩阵结构

```text
训练量 D →    0.5×    1×     2×     4×
模  130M       ●      ●      ●      ●
型  520M       ●      ●      ●      ●
尺  2.3B       ●      ●      ●      ●
寸  8B         ○      ○      ○      ○  ← holdout（不参与拟合）
N   ↓
```

设计原则：

- 拟合点：较小尺寸的全部档位用于拟合 Scaling Law
- Holdout 点：最大尺寸不参与拟合，用于验证外推精度
- 训练量档位与尺寸跨度的具体建议见 [§2.2](#22-尺寸数量与跨度)

上述模型尺寸仅为示例。

### 2.2 尺寸数量与跨度

[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 首次系统验证了跨越 7 个数量级的幂律关系，为 Ladder 的尺寸跨度设计提供了理论基础。具体建议：

| 维度 | 建议 | 来源 |
|---|---|---|
| 模型尺寸数 | 拟合至少 3–4 个，并另设 holdout；增加尺寸可降低单点方差 | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840) |
| 尺寸跨度 | 覆盖目标外推区间；部分 family 上 34× 仍可用 | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840) |
| 相邻尺寸比 | 等比递增，常见 $\sqrt{2}$–4× | [Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) |
| 训练量档位数 | 覆盖 under-trained 到 over-trained，常用 4 档 | [Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) |
| 训练量跨度 | 0.5×–4× Chinchilla（包住 1× 最优配比） | [Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) |

拟合数据处理：中间 checkpoint 可纳入拟合以增加样本；训练初期 loss 尚未稳定下降的 checkpoint 应排除。截断点按固定规则事先定下，不放在 holdout 上挑选——[Choshen et al., 2025](https://arxiv.org/abs/2410.11840) 在 485 个公开模型上扫过该取值，排除约前 10% checkpoint 或前 10B tokens 可显著降低误差，截得更少噪声更大、截得更多无额外收益，该结论可直接沿用。若要在自身数据上标定，[Lourie et al., 2026](https://arxiv.org/abs/2608.11859) 的做法是把模型尺寸分为拟合、验证、测试三段，截断点等拟合选项在验证段上确定，测试段只在最后评估一次。随机种子方差不可忽略，小模型上增加独立运行有时比再加一个大尺寸更有效。

### 2.3 公开规模配置

公开文献中可供参考的模型规模与训练量配置：

| 来源 | 尺寸范围（去 embedding） | 训练量 | 文献 |
|---|---|---|---|
| OpenAI | 0.8M–1.5B | 22M–2.3B tokens | [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) |
| DeepMind Chinchilla | 70M–16B（400+ 模型） | 5B–500B tokens | [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) |
| StepFun | 多尺寸（3,700+ 模型） | 全部实验累计 100T tokens | [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) |
| Gadre et al. | 11M–6.9B（104 个模型） | 最高 32× Chinchilla 配比 | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540) |
| Fantastic Optimizers | 0.1B–1.2B（4 个尺寸） | 1×–8× Chinchilla 配比 | [Wen et al., 2025](https://arxiv.org/abs/2509.02046) |
| Delphi | 72M–6.9B（holdout 25B） | 拟合 3e18–3e20 FLOPs，holdout 1e23 FLOPs | [Marin, 2025](https://openathena.ai/blog/delphi/) |
| NVIDIA Nemotron | 15B, 340B | 8T, 9T tokens | [Nemotron-4, 2024](https://arxiv.org/abs/2402.16819) |
| OLMo | 1B, 7B, 13B, 32B | 2T–6.6T tokens | [OLMo, 2024](https://arxiv.org/abs/2402.00838); [OLMo 2, 2025](https://arxiv.org/abs/2501.00656) |

### 2.4 共同配置清单

所有 Ladder 点必须保持以下训练配置一致（架构一致性要求见 [§3.3](#33-架构一致性)）：

| 配置项 | 说明 | 示例 |
|---|---|---|
| seq_len | 序列长度 | 4,096（常见配置） |
| 数据版本 | 训练数据 | 同一数据版本 |
| 优化器 | Optimizer | AdamW（主流选择） |
| 评估集 | Validation | 同一 eval set |
| 评估频率 | Eval interval | 按训练进度等比例设置 |

评估频率说明：固定 step 间隔（如每 1000 step）会使不同训练长度的有效 eval 点数不同。按总步数 $T$ 的固定比例设置，可使各配置的 eval 点数接近。

### 2.5 分阶段 Ladder

预训练、mid-training 和长上下文扩展具有不同的数据分布、序列长度和 LR schedule，最优超参数及 scaling law 系数可能随阶段改变。[Qwen3, 2025](https://arxiv.org/abs/2505.09388) 分别针对通用预训练、推理强化和长上下文三个阶段预测 LR 与 batch size；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) 也为 pretraining、midtraining 和 long-context extension 分别设置训练量、batch size、序列长度与 LR schedule。

如果交付流程包含多个阶段，各阶段应单独建立 Ladder：

1. 预训练 Ladder：从随机初始化开始，固定基础数据配比，拟合 $N$、$D$、LR 与 batch size；
2. Mid-training Ladder：各尺寸从对应预训练 checkpoint 开始，固定阶段数据配比，拟合新增 token 量、LR schedule 与能力增量；
3. 长上下文 Ladder：各尺寸从对应 mid-training checkpoint 开始，单独搜索序列长度、长短样本比例、RoPE 配置、训练量与 LR。

数据分布、序列长度、优化器或架构变化后，不直接复用上一阶段的 scaling law 系数。阶段间应同时记录基础能力变化和目标能力增量，避免后续阶段在提高目标指标的同时掩盖基础能力下降。

## 3. 模型尺寸选择

### 3.1 参数量口径

拟合 $L(N,D)$ 与计算算力 $C$ 所需的参数量口径不同，需分开记录。

拟合 $L(N,D)$ 时，$N$ 取 Transformer 主干参数量 $N_{\text{body}}$，排除 input embedding 与 output head。Embedding 与 head 的参数量级为 $O(Vd)$，与主干的 $O(d^2 n)$ 随尺寸增长的幂次不同，在小模型上占比高，混入会扭曲跨尺寸的 scaling 关系。[Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972) Appendix G 做过消融：含 embedding 口径在拟合区间内可拟合，但无法准确外推到 25.1B。[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 基于同样理由排除 embedding。

计算算力 $C$ 时，必须加回 output head。Head 是每 token 一次 $d \times V$ 矩阵乘，属于真实算力开销。用 $N_{\text{body}}$ 直接代入 $C = 6ND$ 会系统性低估小模型的算力，低估幅度随尺寸减小单调增大（[Porian et al., 2024](https://arxiv.org/abs/2406.19146) Table 2）。正确做法是按各层矩阵乘累加，或将 $N_{\text{body}} + Vd$ 代入 $6ND$。

Input embedding 是按 token ID 查表，不产生矩阵乘，两个口径都不计入。权重绑定减少的是存储而非计算，head 算力照常计入。引用外部 TPP 数值时需确认口径，例如 Chinchilla 的 20 TPP 按含 embedding 的总参数量计。

### 3.2 计算量与宽深比

常用的 $C \approx 6ND$ 只计入矩阵乘，忽略了注意力的二次项。长序列训练下该修正项不可忽略：

$$C \approx \left(6 + \frac{L}{d}\right) \cdot N \cdot D$$

$L$ 为序列长度。$L/d$ 来自注意力计算，在 $d$ 小、$L$ 大时可以超过常数 6。

参数量近似为 $N \approx 12 d^2 n$。给定 $N$，$d$ 与 $n$ 之间有权衡。$d$ 越小则 $L/d$ 越大，参数量相同的模型实际算力开销可能显著不同。若 Ladder 各尺寸的宽深比不一致，用 $N \times D$ 作自变量会把算力差异误判为模型能力差异。建议用实际 $C$ 作自变量，或在 Ladder 内固定宽深比。

[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 报告：给定参数量，宽深比在较大范围内对 loss 影响很小——即宽深比主要影响实际算力消耗，而非模型能力。因此 Ladder 内保持宽深比一致即可，无需为每个尺寸单独调优。

### 3.3 架构一致性

Ladder 各实验点需保持以下架构属性一致：

| 属性 | 要求 |
|---|---|
| 架构类型 | 一致，如均为 decoder-only Transformer |
| 归一化 | 一致，如均为 RMSNorm |
| 位置编码 | 一致，如均为 RoPE |
| 激活函数 | 一致，如均为 SwiGLU |
| Embedding 共享 | tied / untied 在 Ladder 内保持一致 |
| 注意力类型 | 一致，如均为 GQA |
| 宽深比 | 固定，或与目标大模型对齐 |

### 3.4 MoE Ladder 的参数口径与稀疏度轴

MoE 模型不能只用单一参数量 $N$ 描述规模。每个实验点至少需要同时记录：

- 总参数量 $N_{total}$：包括全部专家，用于描述模型容量与存储开销；
- 激活参数量 $N_{active}$：单个 token 实际参与计算的参数量，用于近似训练与推理 FLOPs；
- 专家总数 $E_{total}$、每 token 激活专家数 $E_{active}$，以及稀疏率 $S=E_{total}/E_{active}$；
- 路由器、共享专家和负载均衡损失的配置。

[Kimi K2, 2025](https://arxiv.org/abs/2507.20534) 在固定 $N_{active}$ 和训练 FLOPs 的条件下改变 $E_{total}$，单独拟合 sparsity scaling law。结果显示，稀疏率从 8 提高到 48 时，相同目标 loss 所需 FLOPs 持续下降，但通信、存储和推理复杂度同时增加。因此，MoE Ladder 应包含两类实验：

1. 规模 Ladder：固定稀疏率和路由配置，改变 $N_{active}$ 与训练量 $D$；
2. 稀疏度 Ladder：固定 $N_{active}$、$D$ 和激活专家数，只改变专家总数。

两类实验的拟合变量不能混用。前者用于预测模型规模与训练量的关系，后者用于选择总容量与计算量的配比。

## 4. 训练量设计

### 4.1 Compute Optimal 配比

Chinchilla（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)）的 IsoFLOP 方法：固定每个算力预算 $C$，在不同尺寸的模型上训练，取 loss 曲线的最低点，得到该算力下的最优 $(N, D)$ 组合。

{% include figure.liquid
  path='assets/img/pretrain-scaling/chinchilla-isoflop.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 2. Chinchilla 的 IsoFLOP 结果。左：各算力预算下 loss 对模型尺寸呈抛物线，谷底为最优尺寸；中、右：最优参数量与 token 数随算力的幂律。<a href="https://arxiv.org/abs/2203.15556">来源：Hoffmann et al., 2022, Fig. 3</a>。'
  alt='Chinchilla IsoFLOP 曲线：左图为各 FLOP 预算下 loss 对参数量的抛物线，中图和右图为最优参数量、token 数对 FLOPs 的幂律拟合。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

在 400+ 个模型（70M–16B，5B–500B tokens）上得到：

$$N_{opt} \propto D_{opt} \propto C^{0.5}$$

即 $D_{opt}/N_{opt}$ 为常数，模型与数据等比例增长。参数化损失 $L(N,D) = E + A/N^\alpha + B/D^\beta$。

偏离最优配比时规律仍然成立：不同 $D/N$ 下 Loss vs $C$ 幂律的指数接近，对齐指数后只需建模系数随 $M=D/N$ 的变化，过训练到 32× 依然可靠（[Gadre et al., 2024](https://arxiv.org/abs/2403.08540)，104 个模型）。这为 §4.2 中超出 1× Chinchilla 的训练量档位提供了理论支撑。

### 4.2 Ladder 训练量档位

推荐每个模型尺寸跑 4 个训练量档位：0.5× / 1× / 2× / 4× Chinchilla 最优配比（$D/N \approx 20$ tokens/parameter，$N$ 按含 embedding 的总参数量计，为 Chinchilla 原文 Appendix F 口径；换算到 $N_{\text{body}}$ 时该比值会变大），覆盖 under-trained 到 over-trained（档位数量与跨度的详细依据见 [§2.2](#22-尺寸数量与跨度)）。

### 4.3 数据质量对 Scaling Law 的影响

[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) 的发现：

- 高质量数据使最优分配偏向更大模型：数据质量越高，增加的计算预算应更多分配给参数量而非数据量
- 不同数据集的 scaling law 参数差异显著，不能假设一个数据集上拟合的规律直接适用于另一个数据集
- 最优 $N/D$ 配比的差异可作为间接评估数据质量的方法

搭建 Ladder 时，如果训练数据版本发生变化，需要重新验证 scaling law 参数是否仍然适用，不能直接复用旧数据版本的拟合结果。

评估新数据源时无需重跑完整 $N \times D$ 矩阵。[OLMo 2, 2025](https://arxiv.org/abs/2501.00656) 的 Micro-annealing 范式：从主干训练的中间 checkpoint 出发，仅用候选领域数据做 5B–10B tokens 的线性退火至 0，即可在极低算力下独立验证该数据源对下游能力的增益，且结论可迁移到全量训练。

### 4.4 数据配比 Ladder

数据源比例需要通过独立实验确定。数据配比 Ladder 固定模型结构、参数量、训练量和优化器，仅改变数据混合向量 $\mathbf{w}=(w_1,\ldots,w_k)$。

[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) 的流程：以自然分布为中心从 Dirichlet 分布采样候选配比，每个配比训练一个 30M、3B tokens（约 5× Chinchilla）的代理模型，数量约为数据域数量的 5 倍；用按能力聚合的 answer BPB 评估各模型，对每个任务拟合"配比 → BPB"的广义线性模型，再在 token 预算与最大重复次数（约 4–7 次）约束下求解平均 BPB 最低的配比。数据源配比与源内质量分布分开处理：先优化域间比例，再按质量分位数设置上采样曲线。[Qwen3, 2025](https://arxiv.org/abs/2505.09388) 进一步将配比轴从数据源扩展到实例属性（教育价值、领域、语言、安全），通过小型代理模型消融确定过滤与混合策略。

数据配比 Ladder 的产出是训练分布，模型规模 Ladder 在该分布上拟合 $N$、$D$ 与 loss 的关系。两者串行：配比确定后需要重新验证规模 Ladder 的系数。

### 4.5 数据重复 Scaling Law

Chinchilla scaling law 假设每个 token 只训练一次，实际训练常使用多 epoch。重复数据的边际收益递减，需先折算为等效单 epoch 数据量。

等效数据量公式（[Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)）：重复数据的信息价值指数衰减，等效数据量为

$$D^{\prime} = U_D + U_D \cdot R_D^{\ast} \left(1 - e^{-R_D / R_D^{\ast}}\right)$$

其中 $U_D$ 为 unique token 数，$R_D$ 为重复次数（$R_D = 0$ 即单 epoch），$R_D^{\ast}$ 为信息半衰期，由 ladder 数据拟合。$R_D \to \infty$ 时 $D^{\prime} \to U_D(1+R_D^{\ast})$，重复收益有上限。

{% include figure.liquid
  path='assets/img/pretrain-scaling/muennighoff-epoch-returns.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 3. 重复数据的边际收益（4.2B 模型，84B token 预算）。4 epoch 内接近新数据，4–10 epoch 快速递减，40 epoch 后 loss 转升。<a href="https://arxiv.org/abs/2305.16264">来源：Muennighoff et al., 2023, Fig. 1 左</a>。'
  alt='重复数据收益曲线：横轴为 training tokens/epochs，纵轴为 final test loss；标注了约 4 epochs 内接近新数据、4-10 epochs 快速递减、40 epochs 后无收益。'
  avoid_scaling=true
  zoomable=true
%}

对 Ladder 设计的指导：

- epoch 上限随数据集大小：相变点 $K^{\ast} \propto \log N$（[Yan et al., 2025](https://arxiv.org/abs/2511.13421)）。参数量大、数据量小、重复次数多三者叠加时过拟合损失加速增长（[Lovelace et al., 2026](https://arxiv.org/abs/2605.01640)）。
- 参数量而非 FLOPs 决定过拟合（[Xue et al., 2023](https://arxiv.org/abs/2305.13230)）：更大的数据集可缓解，数据质量高不能。
- 缓解手段：dropout 是唯一对重复过拟合有效的标准正则化项，需训练中期加入，且在大模型上效果减弱（[Xue et al., 2023](https://arxiv.org/abs/2305.13230), Table 4）。

## 5. 超参数搜索策略

本章目标：为每个 ladder 点标定 Fully-Tuned 的 $(\eta, B)$，并将其写成可随 $N, D$（或 $C$）外推的函数。以下按搭建 ladder 时的决策顺序展开。

### 5.1 搜参目标：Fully-Tuned Frontier 与宽平台

[Lourie et al., 2026](https://arxiv.org/abs/2608.11859) 与 [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) 揭示了小模型搜参的核心难点：
- 小模型对超参极度敏感：小尺寸模型在次优超参下的性能劣化极其陡峭，其真实的 Scaling 规律只有在"完全调优前沿"（Fully-Tuned Frontier）上才会显现；
- 搜参不充分的后果：如果在小模型上仅凭经验粗搜并停留在局部极小值，拟合出的幂律曲线会严重扭曲，导致大模型预测失真。

大模型的行为与此相反：最优值处于较宽的平底区域（[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954)；[Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715)；[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320)），LR 在 $\sqrt{2}$ 范围内、BSZ 在 $\pm 25\%$ 内波动对最终泛化误差影响 $<0.25\%$。

由此得出搜参的资源分配原则：算力集中在小模型的穷举搜索，确保拟合点落在 Fully-Tuned Frontier 上；大模型只需在外推值附近做 $\pm \sqrt{2}$ 的局部确认。

### 5.2 搜索流程

参数敏感度排序：$\text{Learning Rate} > \text{Batch Size} > \text{Schedule (Warmup/Decay)} > \text{Weight Decay}$。$\epsilon$ 等稳定性超参在基线确定后固定即可。

在资源受限下，搜索应遵循坐标下降（Coordinate Descent）与阶段收敛：

1. 小模型全局网格搜索：在最小尺寸（如 ~130M）上对敏感度最高的维度穷尽搜索，确定核心区间；
2. 中模型验证与外推：在中等尺寸（如 ~500M）上检验幂律趋势，修正斜率；
3. 大模型局部确认：LR 在外推值附近做 $\pm\sqrt{2}$ 微调（[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)）；BSZ 在外推 $B_{opt}$ 附近扫 $[B/2,\,2B]$ 即可（[§5.3](#53-batch-size-与-learning-rate-的迁移规律)）。

其中，Batch Size 的搜索维度可以进一步削减——见下一节。此外，[Power Lines (Bergsma et al., 2025)](https://arxiv.org/abs/2505.13738) 提出的 timescale $\tau = B/(\eta\lambda D)$ 可将 $(LR, BSZ, WD)$ 三维网格压缩为一维搜索，在步骤 1 中可显著降低算力开销。

### 5.3 Batch Size 与 Learning Rate 的迁移规律

[Power Lines (Bergsma et al., 2025)](https://arxiv.org/abs/2505.13738) 在多组 $N, D$ 网格上测得：
$$B_{opt} \propto D^{0.4}, \quad B_{crit} \propto D^{0.5}$$
两者均与模型参数量 $N$ 无关。该指数来自该工作在其优化器与数据设置下的拟合，可作为 Ladder 的初始化先验，仍需在自身数据上验证。按此先验，不同尺寸、相同 token 量的实验可先共用同一 BSZ，$B_{opt}$ 随 $D$ 迁移，不必对每个尺寸从零搜索。

$B_{crit}$ 是数据并行效率的上限：$B = B_{crit}$ 时达到目标 loss 所需 tokens 已是 $B_{opt}$ 下的 2 倍，继续增大 $B$ 只增加算力消耗而不降低 loss。

按幂律外推 $B_{opt}$ 之后，局部 sweep 不需要大步长或宽区间。[Schaipp, 2026](https://arxiv.org/abs/2607.01487) 定义浪费至多 5% 算力的次优批次区间 $[b_{\min}, b_{\max}]$（等价于用 $0.95D$ 达到同一 loss），经验宽度约为 $2^{2}$，即 $[B_{opt}/2,\,2B_{opt}]$。该区间与 [§5.1](#51-搜参目标fully-tuned-frontier-与宽平台) 的宽平台一致：DeepSeek 报告的 $\pm 25\%$ 是近零误差平台，Schaipp 的 $\times 2$/$\div 2$ 是可接受算力浪费的外沿。Ladder 上中、大尺寸确认时，在外推值两侧各取一倍即可。再缩小到 $B_{opt}/4$ 已超出该区间，训练常不稳定甚至发散（公开讨论中苏剑林亦有相近观察）。$B_{crit}$ 约束的是过大 batch；过小 batch 的主要风险是噪声与发散。

在 $B < B_{crit}$ 范围内，跨配置迁移时 $B/\eta$ 应保持恒定，即 $B$ 增大时 $\eta$ 同比例增大。由此推出：固定 $N$ 时，$\eta_{opt}$ 也按 $D^{0.4}$ 迁移——BSZ 的幂律同时给出了 LR 的迁移方向。

{% include figure.liquid
  path='assets/img/pretrain-scaling/cerebras-bcrit-hyperbola.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 4. 左：达到目标 loss 所需 token 数与步数呈双曲线，转折点即 B_crit；右：不同 batch size 下的 loss–step 曲线。<a href="https://arxiv.org/abs/2505.13738">来源：Bergsma et al., 2025, Fig. 4</a>。'
  alt='左图为 tokens 对 steps 的双曲线，标注 B_crit 转折点；右图为不同 batch size 下 loss 随步数下降的曲线。'
  avoid_scaling=true
  zoomable=true
%}

### 5.4 超参 Scaling Law 公式选用

根据 ladder 覆盖的 $D/N$ 档位数量，选择不同的外推公式。典型 ladder（多尺寸 × 多训练量档位）建议使用 Step Law 或 Power Lines；如果 ladder 只跑单一 $D/N$ 配比，$C$ 幂律即可。

固定 $D/N$ 配比时，可按总算力 $C$ 拟合超参幂律：

$$\eta_{opt} = a \cdot C^{-\alpha}, \quad B_{opt} = b \cdot C^{\beta}$$

在多个小尺寸上搜索最优 $(\eta, B)$，分别拟合出 $(a, \alpha)$ 与 $(b, \beta)$。[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) 提供了拟合系数，可作为先验参考。

Ladder 覆盖多档 $D/N$ 时，$C$ 幂律把 $N$ 和 $D$ 耦合进单一变量，精度不足，需将两者解耦：

| 方案 | $\eta_{opt}$ | $B_{opt}$ | 适用条件与局限 |
|---|---|---|---|
| Step Law ([Li et al., 2025](https://arxiv.org/abs/2503.04715)) | $c \cdot N^{-\alpha} D^{\beta}$ | $d \cdot D^{\gamma}$ | 支持任意 $D/N$ 比；3,700+ 模型验证；$B_{opt}$ 与 $N$ 无关 |
| Power Lines ([Bergsma et al., 2025](https://arxiv.org/abs/2505.13738)) | 由 $B/\eta$ 恒定推出 | $d \cdot D^{\gamma}$（与 $N$ 无关） | 通过 timescale $\tau$ 统一 LR/WD/BSZ 三者关系 |
| Token Horizons ([Bjorck et al., 2024](https://arxiv.org/abs/2409.19913)) | $c \cdot N^{-\alpha} D^{-\beta}$ | — | 固定模型尺寸下，peak LR 随训练长度衰减 |

Token Horizons 与 Step Law 都在固定 $N$ 的条件下研究 $\eta_{opt}$ 随 $D$ 的变化，但 $D$ 的指数符号相反（前者 $D^{-\beta}$，后者 $D^{+\beta}$）。原因未查明。搭建 Ladder 时应根据自身的 schedule 选择对应公式，并在 holdout 上验证。

$\mu$-Transfer ([Yang et al., 2022](https://arxiv.org/abs/2203.03466)) 通过参数化使最优 LR 与模型宽度无关，可从小模型零样本迁移到大模型。但限制较严格：不能开 WD、BSZ / depth / token 量变化时无法迁移、最优 BSZ 无法跨规模迁移。实际 ladder 搭建中通常用幂律拟合而非 $\mu$P。

### 5.5 对比新优化器时的注意事项

[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) 提出了两个与 ladder 搜参相关的结论：
- 同等调参预算原则：对比新算法/新优化器（如 Muon/Soap 与 AdamW）时，若直接沿用 AdamW 的默认超参，会因基线欠调制造虚假优势。在公平网格搜索下，新优化器的真实收益会被大幅修正；
- 收益是否随尺寸衰减：矩阵优化器相对 AdamW 的加速比从 0.1B 的 $1.4\times$ 衰减到 1.2B 的 $1.1\times$。任何在小模型上的方案优势都必须在 Ladder 的尺寸轴上观察其衰减趋势，不能单点外推。

### 5.6 Weight Decay 处理

WD 是 [§5.2](#52-搜索流程) 敏感度排序中最低的一项，发版 recipe 普遍固定 $\lambda=0.1$ 且不随 $N$、$D$ 调整。[Kimi K2, 2025](https://arxiv.org/abs/2507.20534) 在 1T MoE、15.5T tokens 上全程使用 0.1（MuonClip 优化器）；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) 沿用 AdamW 并将 embedding 排除在 decay 之外；[Qwen3, 2025](https://arxiv.org/abs/2505.09388) 对 LR 与 batch size 做了 scaling law 预测，但未将 WD 纳入预测范围。

最优 $\lambda$ 随 TPP 变化：TPP 越高，$\lambda_{opt}$ 越小（[Han et al., 2026](https://arxiv.org/abs/2602.11137)；方向与 [Power Lines (Bergsma et al., 2025)](https://arxiv.org/abs/2505.13738) 的 timescale $\tau = B/(\eta\lambda D)$ 幂律一致）。Ladder 的 0.5×–4× Chinchilla 档位正落在这一变化区间内。复现发版 recipe 时可全 Ladder 固定 0.1；研究 $B_{opt}$、$B_{crit}$ 或跨 TPP 外推时，固定 WD 会扭曲拟合，需按 $\tau$ 联合校准。

## 6. LR Schedule 与训练收尾

### 6.1 LR Schedule

主流采用以下两种调度策略：

- Cosine 调度：单阶段训练常用，包含短步数 warmup 与连续平滑衰减至 0（或小残余值）。在多阶段训练中，不同阶段的最小学习率可能不同。
- WSD 调度（Warmup-Stable-Decay）：Scaling Law 研究的理想选择。训练初期 loss 尚未稳定，拟合时排除该段；Stable 阶段 Loss Curve 具有清晰幂律；Decay 阶段可按任意训练预算切出线性/阶梯退火。River Valley 理论（[Wen et al., 2024](https://arxiv.org/abs/2410.05192)）从物理上解释了其解耦机制。

退火比例：[Tissue et al., 2024](https://arxiv.org/abs/2408.11029) 给出的最优退火比例区间为总步数的 10%–20%，并可按总步数直接预测。[Wang et al., 2025](https://arxiv.org/abs/2512.13705) 进一步研究了退火策略的 scaling 与跨规模迁移。

评估时点约束：评估必须在每个训练预算的退火终点进行。[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) 发现 LR 退火过程中两个优化器的 loss 排名可能翻转——前期领先的方案在退火结束后被反超。因此 Ladder 上比较方案时，不能使用 stable 阶段中途的 checkpoint 作为比较依据。

### 6.2 收尾流程

以下两类收尾步骤对最终质量的收益稳定，建议纳入 Ladder 的标准训练流程：

- Continued Training / 数据课程：主训练结束后，切换到更高质量的数据分布（上采样高质量数据源、加入少量 benchmark 风格样本或合成数学数据），并用更陡峭的 LR 衰减 schedule 继续训练少量 tokens（通常占总训练量 5–10%）。[Nemotron-4, 2024](https://arxiv.org/abs/2402.16819) 和 [OLMo 2, 2025](https://arxiv.org/abs/2501.00656) 的 mid-training 阶段都采用了类似做法。Ladder 中每个尺寸的最后一阶段都应包含这个收尾，否则拟合出的 loss 与最终交付模型的 loss 会有系统性偏差。如果 mid-training 或长上下文扩展在交付流程中属于独立阶段而非收尾，应按 [§2.5](#25-分阶段-ladder) 分别建立 Ladder。
- Weight Averaging：对训练轨迹上的多个 checkpoint 做权重平均。两种做法：(1) 多次独立训练后合并（model soup，[Wortsman et al., 2022](https://arxiv.org/abs/2203.05482)），OLMo 2 在 mid-training 阶段采用此方案；(2) 单条轨迹的滑动窗口平均（LAWA，[Kaddour, 2022](https://arxiv.org/abs/2209.14981)；[Sanyal et al., 2024](https://arxiv.org/abs/2306.03241)），成本更低。[Model Merging (ByteDance Seed, 2025)](https://arxiv.org/abs/2505.12082) 在 Dense 411M–70B 与 MoE 0.7B/7B–20B/200B 上验证了 WSD stable 阶段的 checkpoint 合并（PMA）可近似退火终点的下游表现。建议作为 Ladder 训练的标准收尾步骤。

## 7. Loss Scaling Law 拟合方法

拟合前需确认三项口径问题已处理，否则不同实验的 scaling law 指数无法对齐（[Porian et al., 2024](https://arxiv.org/abs/2406.19146)）：

1. $C = 6ND$ 中的 $N$ 需计入 output head（[§3.1](#31-参数量口径)）
2. Warmup token 数需随模型尺寸调整，不取固定值（[§10.4](#104-训练初期-checkpoint-纳入拟合)）
3. 各尺寸档位的超参需逐一搜索至 Fully-Tuned Frontier（[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)）

同一工作还表明 LR 衰减并非 scaling law 成立的必要条件。

### 7.1 函数形式

[§4.1](#41-compute-optimal-配比) 给出的 Chinchilla 加性形式隐含 $\partial^2 L / \partial N \partial D \equiv 0$。[Skaling (Videau et al., 2026)](https://arxiv.org/abs/2608.07222) 对实测 loss 曲面做梯度分析，发现该混合偏导数持续为负。修改方式是给两项之和加外指数 $k$：

$$L(N,D)=\left(\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}\right)^{k}+E$$

$k=1$ 退化为 Chinchilla；$0<k<1$ 时混合偏导数为负。实测 $k \approx 0.31$–$0.45$。仅多一个参数，闭式解的代数形式与 Chinchilla 相同。

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-vs-chinchilla-error.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 5. 两种形式在同一 (N, D) 网格上的预测残差，每点为一次训练。左、中：带符号百分比误差，共用色标；Chinchilla 的残差呈鞍形并向四角增大，Skaling 全网格接近零。右：两者误差之比，Skaling 在 76% 的配置上更准，中位数 2.2 倍。<a href="https://arxiv.org/abs/2608.07222">来源：Videau et al., 2026, Fig. 1</a>。'
  alt='三张 (N, D) 网格散点图：左为 Chinchilla 的带符号百分比误差，中为 Skaling 的同类误差，右为两者误差之比。'
  avoid_scaling=true
  zoomable=true
%}

替代方案：[Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972) 令数据侧系数与指数显式依赖 $N$，共九参数。[Skaling (Videau et al., 2026)](https://arxiv.org/abs/2608.07222) Table 1 在 Farseer 与 SK-Grid 两套数据上比较了三种形式；Skaling 使用六个参数，在插值和单轴外推中误差最低，远端外推也更稳定。

推荐 Skaling 作为默认选择。该函数族在 $k=1$ 时包含 Chinchilla 形式，实际外推精度仍需通过 holdout 验证。耦合项会改变大规模下的 compute-optimal 配比，需在自己的数据上确定方向。

固定 $D/N$ 配比时可简化为一维幂律 $L = G(M)/C^\gamma + E$（$M=D/N$），适用于单一配比外推，但无法用于跨配比的资源分配。

实验点布局与函数形式相关。[§2.1](#21-实验矩阵结构) 的全网格是通用做法；采用 Skaling 时，可退化为 L-shape（最小若干尺寸扫完整 $D$ + 其余尺寸各跑一个小 $D$）。Skaling 论文的实验中，L-shape 使用约 1/10 的全网格算力，外推误差与全网格结果接近。Chinchilla 加性形式在 L-shape 下精度大幅退化，仍需全网格覆盖。

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-sampling-strategies.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 6. 三种实验点布局。Random：全网格随机划分训练/验证；L-shape：只跑左列（小 N 扫完整 D）和底行（各 N 跑小 D），其余为验证区；D-Band：只跑底部若干行。L-shape 配合 Skaling 可用约 1/10 全网格算力达到相近外推精度。<a href="https://arxiv.org/abs/2608.07222">来源：Videau et al., 2026, Fig. 4</a>。'
  alt='三种 (N, D) 网格布局示意图：Random 为全网格随机采样，L-shape 为左列加底行，D-Band 为底部若干行。'
  avoid_scaling=true
  zoomable=true
%}

### 7.2 LR 退火 Scaling Law

[Tissue et al., 2024](https://arxiv.org/abs/2408.11029) 将 loss 表示为训练 step 的函数：

$$\hat L(s) = L_0 + A \cdot S_1(s)^{-\alpha} - C \cdot S_2(s)$$

$S_1(s) = \sum_{i \leq s} \eta_i$ 为累计学习率面积，$S_2(s)$ 为带遗忘核的累计退火量。优势在于可拟合整条 loss 曲线而非仅终点 loss，且一条训练曲线上的全部 eval 点均可作为拟合样本，无需每个预算跑完整条曲线。局限：公式以 LR schedule 为输入，peak LR 固定在训练时的取值，无法预测更换 peak LR 后的曲线——LR 的跨规模外推仍需依赖 [§5.4](#54-超参-scaling-law-公式选用) 的超参 scaling law。

个人经验：遗忘核选择幂律核 $K(\Delta) = (1+\Delta)^{-p}$ 通常优于原论文的单指数核。

## 8. 验证方法

### 8.1 Holdout 验证

用较小尺寸拟合，最大尺寸做 holdout 验证。Holdout 不应只放一个最大尺寸，建议按外推倍数拉开多档（如 10×、100×、300×），以定位 recipe 开始失效的外推倍数。[Delphi (Marin, 2025)](https://openathena.ai/blog/delphi/) 在约 2 个数量级上拟合，holdout 拉开到 3×–300×，第一版 recipe 在 30× 处开始发散，而拟合区间内未显示异常。

架构或优化器变更后，旧 scaling law 系数不再适用，需重新拟合。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 切换架构 + Muon 后最优 LR 和 batch size 均偏移，直接沿用旧系数落在次优区域。变更后的验证策略：LR 在大模型上确认，BSZ 在小模型上确认，利用宽平台特性（[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)）做局部确认即可。

Holdout 误差阈值需事先确定，依据为目标方案的最小可区分差异与随机种子方差。[Choshen et al., 2025](https://arxiv.org/abs/2410.11840) 报告：文献中推动建模改动的最小相对差异约为 4%，同架构随机重启的波动可到约 3.5%，通常能获得的最佳相对误差约 4%。Delphi 的 0.2%–0.5% 是特定实验结果，不能作为通用合格线。同时报告外推置信区间。

### 8.2 稳定性压力测试

小规模训练无法充分暴露大规模训练中的 loss spike 和梯度异常。架构、优化器或归一化变更时，需在外推前增加压力测试。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 使用中等规模模型、恒定学习率以及 2×/4× 预测最优 LR 提高优化压力，在相同压力下比较新旧 recipe。通过条件：新 recipe 稳定性不低于基线，预测最优 LR 与 batch size 位于稳定区域。

### 8.3 小规模评估指标校准

搭建 Ladder 前需确认各评估指标在拟合规模上具有可分辨信号。小模型在数学、代码等任务上可能接近随机水平，部分指标在大规模上饱和，两种情况都无法支持实验决策。对小规模 accuracy 无法区分的任务，使用 answer BPB 等连续代理指标（[OLMo 3, 2025](https://arxiv.org/abs/2512.13961)）。

## 9. 下游任务预测

Ladder 不仅要预测 Training Loss，还要预测下游任务指标，才能指导训练决策。核心难点：评估集内不同样本的 scaling 规律不一致，单一 Acc vs Compute 曲线无法涵盖全部情况。

### 9.1 两类方法

| 方法路线 | 核心思路 | 代表工作 | 局限 |
|---|---|---|---|
| Loss → Performance | 先预测 loss，再映射到下游指标（幂律或 sigmoid） | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540)（幂律）；[Delphi (Marin, 2025)](https://openathena.ai/blog/delphi/)（sigmoid） | 映射关系跨任务不一致 |
| End-to-End | 直接对 Acc vs Compute 建模，按难度聚类分组拟合 | [COD (Xu et al., 2025, v2)](https://arxiv.org/abs/2502.17262v2)；[GPT-4 Technical Report (OpenAI, 2023)](https://arxiv.org/abs/2303.08774) | 需要足够评估样本量；主要适用于 Freeform 任务 |

Loss → Performance 路线实现更简单：[Delphi (Marin, 2025)](https://openathena.ai/blog/delphi/) 用 IsoFLOP 预测 loss，再 sigmoid 映射到 MMLU/HumanEval/GSM8K，适合作为 Ladder 的初始方案。End-to-End 路线精度更高但流程更重。

### 9.2 COD 框架概述

[COD (Xu et al., 2025, v2)](https://arxiv.org/abs/2502.17262v2) 的四阶段流程：

1. 聚类：用多个小模型对评估集每道题多次采样，取平均正确率作为难度特征，按难度聚类分组；
2. 拟合：对每个聚类分别拟合 $E[\mathrm{Acc}(C)] = g + (1-g) \cdot e^{-aC^{-b}-c}$；
3. 外推：筛选 accuracy 随 $C$ 单调递增的聚类（有 scaling 规律），代入目标算力得到预测值，按聚类样本数加权平均；
4. 映射：用已有模型的评估数据校准"可外推子集指标 → 全评估集指标"的映射曲线。

[COD v2](https://arxiv.org/abs/2502.17262v2) 报告 70B 模型 8 个 benchmark 的平均预测误差为 1.36%。

{% include figure.liquid
  path='assets/img/pretrain-scaling/cod-prediction-accuracy.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 7. 三种方法在 MATH、MMLU-pro 上的外推结果。红点为用于拟合的小模型，蓝点为 70B 目标模型实测值，虚线为其算力。COD 曲线最接近实测点，End-to-End 高估，Loss-Intermediate 低估。<a href="https://arxiv.org/abs/2502.17262v2">来源：Xu et al., 2025, v2, Fig. 4 中、右</a>。'
  alt='两幅 Accuracy 对 Compute 的外推曲线图，分别为 MATH 和 MMLU-pro，比较 End-to-End、Loss-Intermediate 与 COD 三种方法在 70B 目标模型处的预测值与实测值。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

### 9.3 适用条件与局限

- 评估样本量需足够，样本过少时聚类指标不稳定（[COD (Xu et al., 2025, v2)](https://arxiv.org/abs/2502.17262v2)）
- 聚类基于评估集的内在属性，可跨模型复用。v2 的逐簇外推实验使用相同数据分布和架构的小模型；子集到全集的映射可引入不同架构或数据分布的外部模型作为 anchor
- Continued Training / mid-training 退火后数据分布变化，需对小模型同样执行退火后再评估（[COD (Xu et al., 2025, v2)](https://arxiv.org/abs/2502.17262v2) Appendix D）
- 论文实验采用 constant LR，MoE 外推尚未充分验证。Multiple-Choice 仅比较选项概率的评估方式与 passrate 定义不一致；MMLU-Pro 已纳入实验，但未覆盖仅依赖选项排序的设置。CoT 推理能力的预测尚未充分验证

## 10. 常见陷阱与教训

### 10.1 小模型搜参不充分

最常见也最严重的问题。以历史最优为中心搜索，遇到局部极小就停止，导致外推大模型最优超参显著偏离合理先验。解决：扩大搜索范围，在 [Power Lines (Bergsma et al., 2025)](https://arxiv.org/abs/2505.13738) 的 timescale $\tau$ 空间搜索可降低维度（见 [§5.2](#52-搜索流程)）。

### 10.2 参数量口径不一致

混用总参数量与主干参数量，或把拟合用的 $N_{\text{body}}$ 直接代入 $C = 6ND$。典型表现是不同文档的 TPP 数值对不上。解决：两个口径分开记录，引用外部数值前先确认口径（见 [§3.1](#31-参数量口径)）。

### 10.3 外推跨度过大

Loss curve scaling law（[§7.2](#72-lr-退火-scaling-law)）的可靠外推范围有限，跨度过大时出现系统性偏差。解决：控制外推跨度在已验证范围内，更大跨度需补充实验点。

### 10.4 训练初期 checkpoint 纳入拟合

使用中间 checkpoint 拟合时，训练初期 loss 尚未进入稳定下降区间，纳入会引入噪声。该区间与 LR warmup 可能重合，但二者不是同一概念。解决：排除初期 checkpoint，截断点按固定规则事先定下（约前 10%，见 [§2.2](#22-尺寸数量与跨度)），不在 holdout 上挑选；仅用 endpoint loss 拟合时无此步骤。Warmup token 数本身仍需随模型尺寸调整（见 [§7](#7-loss-scaling-law-拟合方法) 开头第 2 点）。

以下问题已在前文对应章节详细讨论，此处仅列为提醒：

- 计算量近似：长序列下 $6ND$ 不准确，需用含 $L/d$ 修正项的公式（[§3.2](#32-计算量与宽深比)）
- 评估间隔：固定 step 间隔会引入统计伪影，按训练进度等比例设置（[§2.4](#24-共同配置清单)）
- 拟合公式：Chinchilla 加性形式的耦合假设与实测不符，推荐 Skaling（[§7.1](#71-函数形式)）

## 11. Scaling Ladder 配置参考

公开文献中有两套较完整的 Ladder 配置可供参考，侧重点不同：

### 11.1 Fantastic Optimizers Ladder（Dense，超参搜索导向）

来源：[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) Table 2–3。Llama 2 架构，四个尺寸均固定 32 层、MHA、seq_len 4096。目标是公平对比优化器，因此超参搜索流程最为严格。

| 尺寸 | hidden_dim | inter_dim | heads | 数据比 |
|---|---|---|---|---|
| 130M | 512 | 2,048 | 8 | 1×–8× Chinchilla |
| 300M | 768 | 3,072 | 12 | 同上 |
| 520M | 1,024 | 4,096 | 16 | 同上 |
| 1.2B | 1,536 | 6,144 | 24 | 同上 |

论文比较十种优化器，并对各优化器分别搜索超参数。Table 3 展示的是 130M、1× Chinchilla 下 AdamW 的坐标下降示例，其中最终配置为 Peak LR 8e-3、WD 0.1、warmup 2000 steps、BSZ 128；这些数值不构成所有尺寸的共同设置。例如 520M、1× Chinchilla 的 AdamW 最优配置使用 WD 0.2、warmup 1000 steps、BSZ 256。数据由 DCLM-baseline、StarCoder V2 Data 与 ProofPile 2 混合而成。

### 11.2 Delphi Ladder（端到端 Scaling Law 导向）

来源：[Delphi (Marin, 2025)](https://openathena.ai/blog/delphi/)。Qwen 3 架构，MLP ratio 4，seq_len 4096，Dense decoder-only。目标是拟合 IsoFLOP scaling law 并外推到 1e23 FLOPs（25B 参数）。Marin 团队后续已将 recipe 扩展到 MoE（[535B-A23B hero run](https://openathena.ai/blog/)），MoE 版本的 scaling ladder 设计可关注其后续发布。

共同设置：AdamH、WSD schedule（10% linear warmup、20% linear decay to 0）、f32 参数 + bf16 计算、FSDP。数据为 Nemotron-CC + StarCoderData + ProofPile 2。

Ladder 结构：在 3e18–3e20 FLOPs（约 2 个数量级）上做 IsoFLOP 扫描，每个算力预算跑多个尺寸-token 组合取最优点，共 7 个 IsoFLOP 最优点用于拟合。Holdout 设在 1e21、1e22、1e23 FLOPs（3×–300× 外推）。超参由 recipe 公式从算力预算自动推导（LR、BSZ、初始化均为 $H$ 和 $T$ 的函数），不做逐点手动搜索。

### 11.3 选择建议

- 需要搜索最优超参并拟合超参 scaling law → Fantastic Optimizers 的网格设计
- 需要端到端 loss 预测并外推到大规模 → Delphi 的 IsoFLOP + recipe 公式
- 实际 Ladder 的架构应与目标大模型对齐（GQA / head_dim / 宽深比等），上述配置仅为起点

## 12. 搭建 Checklist

### 12.1 设计阶段

- [ ] 拟合至少 3–4 个尺寸，覆盖目标外推区间，并另设 holdout（[§2.2](#22-尺寸数量与跨度)）
- [ ] 每个尺寸覆盖 under-trained 到 over-trained 的多档训练量（[§4.2](#42-ladder-训练量档位)）
- [ ] Holdout 按多档外推倍数拉开（[§8.1](#81-holdout-验证)）
- [ ] 参数量 $N_{\text{body}}$ 与算力 $C$ 分开记录（[§3.1](#31-参数量口径)）
- [ ] 算力用含 $L/d$ 修正项的公式计算（[§3.2](#32-计算量与宽深比)）
- [ ] MoE 需同时记录总参、激活参、专家数与稀疏率，规模与稀疏度 Ladder 分开（[§3.4](#34-moe-ladder-的参数口径与稀疏度轴)）
- [ ] 数据配比未确定时先运行数据配比 Ladder（[§4.4](#44-数据配比-ladder)）
- [ ] 多阶段交付时各阶段分别设计 Ladder（[§2.5](#25-分阶段-ladder)）
- [ ] 算力受限时评估 L-shape 布局（[§7.1](#71-函数形式)）

### 12.2 配置阶段

- [ ] 架构属性在 Ladder 内一致：归一化、位置编码、激活函数、注意力类型、宽深比（[§3.3](#33-架构一致性)）
- [ ] 训练数据版本、validation set、优化器、LR schedule 在 Ladder 内一致（[§2.4](#24-共同配置清单)）
- [ ] eval 间隔按训练进度等比例设置，使各配置 eval 点数接近（[§2.4](#24-共同配置清单)）
- [ ] 各评估指标在 Ladder 规模上有可分辨信号（[§8.3](#83-小规模评估指标校准)）

### 12.3 搜参阶段

- [ ] 小模型穷举搜索 LR / BSZ，确保落在 Fully-Tuned Frontier（[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)）
- [ ] 中模型验证幂律趋势；大模型 LR 做 $\pm\sqrt{2}$ 确认，BSZ 在 $[B_{opt}/2,\,2B_{opt}]$ 内确认（[§5.2](#52-搜索流程)、[§5.3](#53-batch-size-与-learning-rate-的迁移规律)）
- [ ] BSZ 用 $B_{opt} \propto D^{0.4}$ 作先验迁移，不超过 $B_{crit}$，并在自身数据上验证（[§5.3](#53-batch-size-与-learning-rate-的迁移规律)）
- [ ] 复现发版 recipe 时 WD 可固定 0.1；拟合 $B_{opt}$、$B_{crit}$ 时按 timescale 联合校准（[§5.6](#56-weight-decay-处理)）

### 12.4 拟合阶段

- [ ] 拟合前核对三项口径：FLOPs 含 output head、warmup 随 N 调整、超参已搜至 Frontier（[§7](#7-loss-scaling-law-拟合方法)）
- [ ] 推荐 Skaling 形式，$k \to 1$ 时退化为 Chinchilla（[§7.1](#71-函数形式)）
- [ ] 多 epoch 训练需折算等效无重复 token 量（[§4.5](#45-数据重复-scaling-law)）
- [ ] 用中间 checkpoint 拟合时排除训练初期不稳定段，截断点按固定规则事先定下（[§10.4](#104-训练初期-checkpoint-纳入拟合)）

### 12.5 验证阶段

- [ ] 预先确定 Holdout 误差阈值，按多档外推倍数拉开（[§8.1](#81-holdout-验证)）
- [ ] 架构或优化器变更后重拟合并做稳定性压力测试（[§8.2](#82-稳定性压力测试)）
- [ ] 下游任务预测误差在可接受范围内（[§9](#9-下游任务预测)）

## 13. 参考文献

按主题分组。标注章节号的条目在正文中有对应讨论。

### Scaling Law 基础与函数形式

1. Scaling Laws for Neural Language Models — Kaplan et al., OpenAI, 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)  
   幂律关系跨越 7 个数量级；给定参数量时宽深比对 loss 影响很小。见 [§3.1](#31-参数量口径)、[§3.2](#32-计算量与宽深比)
2. Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., DeepMind, 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)  
   IsoFLOP 方法与 $N_{opt} \propto D_{opt} \propto C^{0.5}$；加性函数形式 $L=E+A/N^{\alpha}+B/D^{\beta}$。见 [§4.1](#41-compute-optimal-配比)
3. Language models scale reliably with over-training and on downstream tasks — Gadre et al., 2024. [arXiv:2403.08540](https://arxiv.org/abs/2403.08540)  
   104 个模型验证过训练区域的 scaling law 仍可靠；不同 $D/N$ 下幂律指数接近。见 [§4.1](#41-compute-optimal-配比)、[§9.1](#91-两类方法)
4. Skaling: Chinchilla's Exponents Meet Kaplan's Coupling — Videau et al., FAIR at Meta, 2026. [arXiv:2608.07222](https://arxiv.org/abs/2608.07222)  
   实测 $\partial^2 L/\partial N\partial D$ 持续为负，加性形式强制其为零；加外指数 $k$ 修正，并提出 L-shape 布局。见 [§7.1](#71-函数形式)
5. Predictable Scale: Part II, Farseer — Li et al., StepFun & Fudan, NeurIPS 2025. [arXiv:2506.10972](https://arxiv.org/abs/2506.10972)  
   数据侧系数与指数显式依赖 $N$ 的九参数形式；参数量口径消融。见 [§3.1](#31-参数量口径)、[§7.1](#71-函数形式)
6. Scaling Law with Learning Rate Annealing — Tissue et al., 2024. [arXiv:2408.11029](https://arxiv.org/abs/2408.11029)  
   将 loss 表示为累计学习率面积与退火量的函数，可拟合整条曲线。见 [§7.2](#72-lr-退火-scaling-law)
7. A Hitchhiker's Guide to Scaling Law Estimation — Choshen et al., MIT/IBM, ICML 2025. [arXiv:2410.11840](https://arxiv.org/abs/2410.11840)  
   中间 checkpoint 可纳入拟合，需排除训练初期不稳定段；尺寸数量与外推跨度见正文。见 [§2.2](#22-尺寸数量与跨度)、[§8.1](#81-holdout-验证)、[§10.4](#104-训练初期-checkpoint-纳入拟合)
8. Resolving Discrepancies in Compute-Optimal Scaling of Language Models — Porian et al., NeurIPS 2024. [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)  
   Kaplan 与 Chinchilla 指数不一致的三个口径问题及修正。见 [§3.1](#31-参数量口径)、[§7](#7-loss-scaling-law-拟合方法)

### 超参 Scaling Law

{:start="9"}
9. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism — DeepSeek, 2024. [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)  
   按算力 $C$ 的超参幂律；数据质量影响最优 $N/D$ 配比；最优区为宽平台。见 [§4.3](#43-数据质量对-scaling-law-的影响)、[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)、[§5.4](#54-超参-scaling-law-公式选用)
10. Predictable Scale: Optimal Hyperparameter Scaling Law (Step Law) — Li et al., StepFun, 2025. [arXiv:2503.04715](https://arxiv.org/abs/2503.04715)  
    3,700+ 模型验证的 $\eta_{opt}=c\,N^{-\alpha}D^{\beta}$、$B_{opt}=d\,D^{\gamma}$。见 [§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)、[§5.4](#54-超参-scaling-law-公式选用)
11. Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training — Bergsma et al., Cerebras, 2025. [arXiv:2505.13738](https://arxiv.org/abs/2505.13738)  
    $B_{opt} \propto D^{0.4}$、$B_{crit} \propto D^{0.5}$ 且与 $N$ 无关；timescale $\tau=B/(\eta\lambda D)$ 将三维搜索降为一维。见 [§5.2](#52-搜索流程)、[§5.3](#53-batch-size-与-learning-rate-的迁移规律)、[§5.6](#56-weight-decay-处理)
12. Scaling Optimal LR Across Token Horizons — Bjorck et al., Microsoft, 2024. [arXiv:2409.19913](https://arxiv.org/abs/2409.19913)  
    固定模型尺寸时 peak LR 随训练长度衰减。见 [§5.4](#54-超参-scaling-law-公式选用)
13. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer ($\mu$-Transfer) — Yang et al., Microsoft, 2022. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)  
    $\mu$P 参数化下最优 LR 与模型宽度无关；但不能开 WD，且 BSZ、深度、token 量变化时无法迁移。见 [§5.4](#54-超参-scaling-law-公式选用)
14. Fantastic Pretraining Optimizers and Where to Find Them — Wen et al., Stanford, 2025. [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)  
    4 个模型尺寸、1×–8× Chinchilla 数据配比范围内的公平调参基准；优化器加速比随尺寸衰减；退火阶段排名翻转。见 [§5.5](#55-对比新优化器时的注意事项)、[§6.1](#61-lr-schedule)、[§11.1](#111-fantastic-optimizers-ladderdense超参搜索导向)
15. Weight Decay Improves Language Model Plasticity — Han et al., 2026. [arXiv:2602.11137](https://arxiv.org/abs/2602.11137)  
    最优 WD 随 TPP 增大而降低。见 [§5.6](#56-weight-decay-处理)
16. Small-Scale Experiments: Are We There Yet? — Lourie et al., NYU & Meta, 2026. [arXiv:2608.11859](https://arxiv.org/abs/2608.11859)  
    小模型对超参的敏感度与 Fully-Tuned Frontier；尺寸划分为拟合、验证、测试三段，拟合选项在验证段上确定。见 [§2.2](#22-尺寸数量与跨度)、[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)
17. How to Allocate Your Tokens? Scaling Laws with Training Steps and Batch Size — Schaipp, Inria, 2026. [arXiv:2607.01487](https://arxiv.org/abs/2607.01487)  
    浪费至多 5% 算力的次优 batch size 区间宽度约为 $2^{2}$。见 [§5.3](#53-batch-size-与-learning-rate-的迁移规律)

### 数据与训练量

{:start="18"}
18. Scaling Data-Constrained Language Models — Muennighoff et al., 2023. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)  
    重复数据的等效 token 量公式与边际收益衰减曲线。见 [§4.5](#45-数据重复-scaling-law)
19. To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis — Xue et al., 2023. [arXiv:2305.13230](https://arxiv.org/abs/2305.13230)  
    参数量而非 FLOPs 决定重复过拟合；dropout 是唯一有效的标准正则化项。见 [§4.5](#45-数据重复-scaling-law)
20. Prescriptive Scaling Laws for Data Constrained Training — Lovelace et al., 2026. [arXiv:2605.01640](https://arxiv.org/abs/2605.01640)  
    参数量大、数据量小、重复次数多三者叠加时过拟合加速。见 [§4.5](#45-数据重复-scaling-law)
21. Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression — Yan et al., 2025. [arXiv:2511.13421](https://arxiv.org/abs/2511.13421)  
    重复 epoch 上限的相变点随数据集大小变化。见 [§4.5](#45-数据重复-scaling-law)

### LR Schedule 与训练收尾

{:start="22"}
22. Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al., 2024. [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)  
    WSD 的 stable 与 decay 阶段解耦机制。见 [§6.1](#61-lr-schedule)
23. Scaling and Transferability of Annealing Strategies in Large Language Model Training — Wang et al., 2025. [arXiv:2512.13705](https://arxiv.org/abs/2512.13705)  
    退火策略的 scaling 与跨规模迁移。见 [§6.1](#61-lr-schedule)
24. Model Merging in Pre-training of Large Language Models — ByteDance Seed, 2025. [arXiv:2505.12082](https://arxiv.org/abs/2505.12082)  
    WSD stable 阶段的 checkpoint 合并可近似退火终点表现。见 [§6.2](#62-收尾流程)
25. Model Soups: Averaging Weights of Multiple Fine-tuned Models — Wortsman et al., 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)  
    多次独立训练后的权重平均。见 [§6.2](#62-收尾流程)
26. Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging (LAWA) — Kaddour, 2022. [arXiv:2209.14981](https://arxiv.org/abs/2209.14981)  
    单条轨迹的滑动窗口权重平均。见 [§6.2](#62-收尾流程)
27. Early Weight Averaging meets High Learning Rates for LLM Pre-training — Sanyal et al., COLM 2024. [arXiv:2306.03241](https://arxiv.org/abs/2306.03241)  
    高学习率下的早期权重平均。见 [§6.2](#62-收尾流程)

### 下游任务预测

{:start="28"}
28. Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective (COD) — Xu et al., 2025. [arXiv:2502.17262v2](https://arxiv.org/abs/2502.17262v2)  
    按难度聚类构建可预测子集，分组拟合后映射到全集。见 [§9](#9-下游任务预测)
29. GPT-4 Technical Report — OpenAI, 2023. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)  
    用评估集子集拟合外推。见 [§9.1](#91-两类方法)

### 模型技术报告与 Ladder 实例

{:start="30"}
30. Delphi: An Open Scaling Suite from 3e18 to 1e23 FLOPs — Marin Team, 2025. [openathena.ai/blog/delphi](https://openathena.ai/blog/delphi/)  
    IsoFLOP 拟合 + recipe 公式驱动超参；holdout 多档外推。见 [§1.2](#12-为什么需要科学的-ladder)、[§8.1](#81-holdout-验证)、[§11.2](#112-delphi-ladder端到端-scaling-law-导向)
31. Nemotron-4 15B Technical Report — NVIDIA, 2024. [arXiv:2402.16819](https://arxiv.org/abs/2402.16819)  
    batch size 递增策略；训练末期 continued training。见 [§6.2](#62-收尾流程)
32. OLMo: Accelerating the Science of Language Models — Groeneveld et al., Allen Institute, 2024. [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)  
    完全开放的训练数据、代码与中间 checkpoint。见 [§2.3](#23-公开规模配置)
33. OLMo 2: The Next Generation of Fully Open Language Models — OLMo Team, Allen Institute, 2025. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)  
    两阶段训练；Micro-annealing 低成本验证数据源；model souping。见 [§4.3](#43-数据质量对-scaling-law-的影响)、[§6.2](#62-收尾流程)
34. OLMo 3 — Allen Institute, 2025. [arXiv:2512.13961](https://arxiv.org/abs/2512.13961)  
    分阶段 Ladder；Dirichlet 采样的数据配比流程；评估指标有效规模范围校准。见 [§2.5](#25-分阶段-ladder)、[§4.4](#44-数据配比-ladder)、[§8.3](#83-小规模评估指标校准)
35. Qwen3 Technical Report — Qwen Team, 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)  
    分阶段预测 LR 与 batch size；实例属性维度的数据配比。见 [§2.5](#25-分阶段-ladder)、[§4.4](#44-数据配比-ladder)
36. On the Design of Qwen3.8-Next Architecture — Qwen Team, 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)  
    架构与优化器变更后超参偏移；稳定性压力测试方法。见 [§8.1](#81-holdout-验证)、[§8.2](#82-稳定性压力测试)
37. Kimi K2: Open Agentic Intelligence — Moonshot AI, 2025. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)  
    MoE 稀疏度 scaling law。见 [§3.4](#34-moe-ladder-的参数口径与稀疏度轴)、[§5.6](#56-weight-decay-处理)

### 延伸阅读

以下文献与 Ladder 设计相关，正文未展开：

{:start="38"}
38. Chinchilla Scaling: A Replication Attempt — Besiroglu et al., Epoch AI, 2024. [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)
39. (Mis)Fitting: A Survey of Scaling Laws — Li et al., 2025. [arXiv:2502.18969](https://arxiv.org/abs/2502.18969)
40. Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws — Sardana et al., 2024. [arXiv:2401.00448](https://arxiv.org/abs/2401.00448)
41. Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training — Merrill et al., 2025. [arXiv:2505.23971](https://arxiv.org/abs/2505.23971)
42. An Empirical Model of Large-Batch Training — McCandlish et al., 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
43. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al., Google, 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
44. Cost-Optimal Grouped-Query Attention for Long-Context Modeling — Chen et al., 2025. [arXiv:2503.09579](https://arxiv.org/abs/2503.09579)
45. Nemotron-4 340B Technical Report — NVIDIA, 2024. [arXiv:2406.11704](https://arxiv.org/abs/2406.11704)
