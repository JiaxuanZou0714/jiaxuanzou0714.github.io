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

本文综合整理自**<u>公开</u>**文献与工程实践经验，**<u>不包含任何涉密内容</u>**，系统介绍如何搭建一个科学的 Scaling Ladder：通过小模型实验拟合 Scaling Law 并外推大模型表现，在启动大模型训练前预测配置与性能。

---

## 1. 核心概念与目标

### 1.1 什么是 Scaling Ladder

Scaling Ladder 是一组不同模型规模（参数量 N）与训练数据量（D）的实验点，用于拟合 Scaling Law 并外推大模型表现。实验点的布局及拟合、验证用途在 [§4.1](#41-实验矩阵结构) 展开。

### 1.2 为什么需要科学的 Ladder

Scaling Laws 的本质是训练配置、过程、结果的可预估。通过小模型上的大量实验得出规律形式，进一步外推预测大模型的表现，使得在启动训练之前就可以预估大模型的训练配置、Loss 甚至下游任务 Performance，降低大规模训练配置失误的风险。

Scaling Law 可为模型尺寸、训练 Token 量和训练参数的选择提供定量依据。大模型训练成本高，提前验证预测精度与配置合理性，有助于降低结果不及预期的风险。

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 1. Delphi 首次实验（Cautious AdamC 配方）。右图的大规模训练偏离预测并发散。<a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">来源：Marin, 2026</a>。'
  alt='Delphi 首次 scaling 实验：10 的 22 次方 FLOPs 运行的 loss 比预测高 2.5%，10 的 23 次方 FLOPs 运行发散。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

一个科学的 scaling ladder，应该能在正式发版大模型上验证外推精度。合格线取决于目标方案间的最小可区分差异与随机种子方差（见 [§7.1](#71-holdout-验证)）；Training Loss 相对误差 ≤ 0.02 可作为参考的合格线。

### 1.3 Ladder 的核心产出

Scaling Ladder 可选择以下产出：

| 产出 | 用途 |
|---|---|
| 超参 Scaling Law（LR/BSZ/WD） | 预测大模型最优训练超参 |
| Loss Scaling Law | 预测大模型最终 Training/Eval Loss |
| Loss Curve Scaling Law | 预测大模型完整训练曲线 |
| 退火比例 Scaling Law | 预测大模型最优 LR decay 比例 |
| 数据重复 Scaling Law | 预测多 epoch 训练的等效效果 |
| 下游任务 Scaling Law | 预测大模型 Benchmark 指标 |

### 1.4 决策目标

Scaling Ladder 可支持四类决策，采样需求和验收条件各不相同：

1. 既定配方预测：预测固定配方在目标规模或训练阶段的表现。配方可具有可预测性但仍存在欠优化；有效的 scaling law 不要求每个实验点穷尽调优。
2. 最优资源分配：在预算约束下选择参数量 $N$ 与训练量 $D$ 的分配。
3. 候选方案比较：比较不同架构、优化器或数据方案。
4. 超参数预测：预测跨规模或跨训练量的超参数。

搭建 Ladder 前应确定本次采用哪一种或哪几种目标，并明确目标模型、训练阶段、目标能力、预算约束、结果用途以及推理部署条件（如适用：输入/输出长度、延迟/吞吐目标、显存约束），以及何种预测误差或性能差异会改变最终决策。[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) 分别考察了 competitive（性能竞争力）与 predictable（可预测性），二者需分别验证。[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台) 的 Fully-Tuned Frontier 要求主要适用于目标 2 和 3；目标 1 在既定配方上拟合即可，无需穷尽调优。

## 2. 评价协议与验收指标

搭建 Ladder 前需确定评价协议：

- 代理指标与最终指标：开发阶段的代理指标（training loss、answer BPB）与最终验收指标（下游 benchmark accuracy）应分别明确。小模型上 loss 的改善不自动保证目标规模下游指标的改善（[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) §1 报告了 loss 与下游表现不一致的情况）。
- 信号可分辨性：各评估指标在拟合规模上须有可分辨信号。小模型在数学、代码等任务上可能接近随机水平，部分指标在大规模上饱和，两种情况都无法支持实验决策。对小规模 accuracy 无法区分的任务，使用 answer BPB 等连续代理指标（[OLMo 3, 2025](https://arxiv.org/abs/2512.13961)）。
- 跨规模排序相关性：小模型上可分辨的信号，只能支持该规模下的比较。代理指标用于方案选择时，还需与目标模型和目标能力指标建立排序相关性。[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3 和 Appendix A.4 检查了小规模信号的信噪比和跨规模排序一致性。
- 评价版本控制：记录提示模板、生成设置（temperature、top-p、max tokens）、评分方法及实现版本，并检查训练数据与评估数据是否存在重叠。
- 目标能力代表性：说明所选评估任务覆盖了哪些交付目标能力，聚合方式是什么，以及关键单项退化的容忍条件。高噪声任务可增加采样或单独报告。对代表性有疑问的能力，可增加独立新题或不同形式的评估，并明确其开发/验收用途（[Phi-4, 2024](https://arxiv.org/abs/2412.08905) §5；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3.3–3.3.4）。

{% include figure.liquid
  path='assets/img/pretrain-scaling/olmo3-evaluation.png'
  id='fig-olmo3-evaluation'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 2. OLMo 3 的数学评估：左、中分别展示 Easy suite 的 bits-per-byte（BPB）和 Main suite 的 pass@1 随算力的变化；小规模模型在 BPB 上已有可分辨差异，同时期 pass@1 接近零。右图展示两类指标在所考察模型中的关系；跨规模排序仍需针对目标任务验证。<a href="https://arxiv.org/pdf/2512.13961v1#page=12">来源：Olmo 3, 2025, v1, Fig. 6（PDF 第 12 页）</a>。'
  alt='OLMo 3 原论文 Figure 6 的三个子图：数学 Easy suite 的 BPB 随算力变化、Main suite 的 pass@1 随算力变化，以及 BPB 与 pass@1 的关系。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

代理结论的迁移证据不足时，应标明适用范围与仍需在目标规模上验证的内容。

若最终交付对象为后训练模型（SFT/RL），进入最终选择的代表性候选应经过条件可比的后训练验证。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 报告 RoPE 与 NoPE 预训练差异很小，但 NoPE 后训练后出现无法终止的生成；稀疏读取残差分支在预训练 loss 和 benchmark 上影响很小，后训练后质量退化。[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.5.1 对候选 midtraining 配比在完整退火后再做快速 instruction tuning，用后训练评估集验收。后训练验收除能力分数外，还应检查终止行为、输出长度和格式等实际交付指标。先证明简化后训练协议对目标选择有区分度，再将它用于大规模筛选。

## 3. 基线配方与模型尺寸

### 3.1 参数量口径

拟合 $L(N,D)$ 与计算算力 $C$ 所需的参数量口径不同，需分开记录。

本文拟合 $L(N,D)$ 时采用 Transformer 主干参数量 $N_{\text{body}}$，排除 input embedding 与 output head。Embedding 与 head 为 $O(Vd)$，主干为 $O(d^2 n)$，两者在小模型中的占比不同。[Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972v3) Appendix G 的消融中，含 embedding 的口径外推到 25.1B 时误差更高；[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 也采用排除 embedding 的口径。引用其他研究时需保留并注明其参数量定义。

计算算力 $C$ 时需计入 output head。密集输出层每个 token 执行一次 $d\times V$ 矩阵乘；将 $N_{\text{body}}$ 直接代入 $6ND$ 会漏计这部分开销。在固定词表和结构缩放规则下，小模型的 head 占比通常更高（[Porian et al., 2024](https://arxiv.org/abs/2406.19146) Table 2）。可按各层计算累加，或以 $6(N_{\text{body}}+Vd)D$ 近似参数相关的矩阵乘开销，再计入注意力计算。

Input embedding 主要按 token ID 查表，在本文口径中单独记录其参数与访存开销。权重绑定减少存储，输出 head 的计算仍需计入。外部研究的口径可能不同：Porian 的 $N$ 含 output head、排除 input embedding；Chinchilla 的 20 TPP 按含 embedding 的总参数量计。

### 3.2 计算量与宽深比

常用的 $C\approx6ND$ 未显式计入注意力矩阵计算。以标准 MHA、隐藏宽度为 $4d$ 的两层 FFN，并按完整注意力矩阵计算为例：

$$C \approx \left(6 + \frac{L}{d}\right) N_{\text{body}}D + 6VdD$$

$L$ 为序列长度，$d$ 为隐藏维度，$V$ 为词表大小。$L/d$ 项来自注意力计算，最后一项为输出 head 开销。SwiGLU、GQA、MoE 或不同注意力计算约定会改变系数，应按实际架构统计。

上述架构下 $N_{\text{body}}\approx12d^2n$，$n$ 为层数。给定参数量，宽度与深度仍会影响实际算力开销。Ladder 应预先确定并记录结构缩放规则，比较计算效率时使用实际 $C$。

[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 报告：给定参数量，宽深比在较大范围内对 training loss 影响很小。但后续受控实验（[Gemstones, 2025](https://arxiv.org/abs/2502.06857) §4.4）表明，不同宽深配置在 benchmark 和实际训练时间上可能产生差异；[GLM-4.5, 2025](https://arxiv.org/abs/2508.06471) 也报告深度和注意力头配置影响推理能力。因此，Ladder 内应在既定模型族中使用一致的宽度、深度缩放规则，并验证该结构对目标能力的代理有效性。当目标决策涉及深度或注意力结构选择时，应增加少量代表性结构对照。

### 3.3 架构一致性

同一模型族内，除明确研究的架构变量外，各实验点保持以下属性或缩放规则一致：

| 属性 | 要求 |
|---|---|
| 架构类型 | 一致，如均为 decoder-only Transformer |
| 归一化 | 一致，如均为 RMSNorm |
| 位置编码 | 一致，如均为 RoPE |
| 激活函数 | 一致，如均为 SwiGLU |
| Embedding 共享 | tied / untied 在 Ladder 内保持一致 |
| 注意力类型 | 一致，如均为 GQA |
| 宽深配置 | 按预定规则缩放；涉及结构选择时增加代表性对照（[§3.2](#32-计算量与宽深比)） |

### 3.4 共同配置与变量分类

同一配方内，除明确研究的变量外，以下训练配置保持一致（架构规则见 [§3.3](#33-架构一致性)）：

| 配置项 | 说明 | 示例 |
|---|---|---|
| seq_len | 序列长度 | 4,096（常见配置） |
| 数据版本 | 训练数据 | 同一数据版本 |
| 优化器 | Optimizer | AdamW（主流选择） |
| 评估集 | Validation | 同一 eval set |
| 评估频率 | Eval interval | 按训练进度等比例设置 |

评估频率说明：固定 step 间隔（如每 1000 step）会使不同训练长度的有效 eval 点数不同。按总步数 $T$ 的固定比例设置，可使各配置的 eval 点数接近。

上表的固定项与评估规则在同一配方内保持一致，明确研究的变量除外。Ladder 中还需区分以下变量类别：

| 变量类别 | 含义 | 示例 |
|---|---|---|
| 固定量 | 所有 Ladder 点取同一值 | 架构、数据版本、优化器类型、seq_len |
| 缩放规则 | 随 $N$ 或 $D$ 按预定规则变化 | LR（按幂律或 $\mu$P 缩放）、BSZ（以 $D^{0.4}$ 为待验证先验）、warmup（按预算与规模设置） |
| 实验自变量 | 待拟合或待搜索的量 | $N$、$D$、最终的最优 LR 和 BSZ |

同一配方内的缩放变量与模型族之间的差异需分别处理。架构、优化器、数据处理或数值精度变更后，先在小规模上重新验证结论可迁移性，再根据结果决定局部校准或完整重拟合。不预设每次变更都需重跑完整矩阵。此外，tokenizer 与词表、初始化与参数化、loss 统计口径等也应在 Ladder 内保持一致。

## 4. 实验矩阵与训练量设计

### 4.1 实验矩阵结构

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
- 训练量档位与尺寸跨度的具体建议见 [§4.2](#42-尺寸数量与跨度)

上述模型尺寸、训练量档位和完整网格布局均为示例；实际布局应结合决策目标、拟合形式与预算选择，无须统一采用完整网格。

实验点布局、拟合形式和外推方向应联合设计。设计矩阵时需写明 $N$、$D$、数据重复程度及训练阶段的拟合范围与目标范围，区分沿 $N$、沿 $D$ 和二者联合外推。根据 [§1.4](#14-决策目标) 的目标选择全网格、IsoFLOP 或稀疏布局（采样布局讨论见 [§6.1](#61-函数形式)）。预先留出覆盖目标方向的验证点，并约定何种结果触发补充实验。搜索、随机种子、评估、验证及追加实验均应计入总预算。

### 4.2 尺寸数量与跨度

[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 系统报告了跨越 7 个数量级的经验幂律关系。尺寸数量和跨度需结合拟合形式、外推目标与预算确定：

| 维度 | 建议 | 来源 |
|---|---|---|
| 模型尺寸数 | 覆盖目标外推方向，并另设 holdout；增加尺寸可检验拟合稳定性 | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) |
| 尺寸跨度 | 覆盖目标外推区间；部分 family 上 34× 仍可用 | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840) |
| 相邻尺寸比 | 可等比递增，具体间距按预算选择 | 本文设计建议 |
| 训练量档位数 | 按目标覆盖不同训练长度；Fantastic Optimizers 使用 4 档 | [Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046v2) |
| 训练量跨度 | Fantastic Optimizers 使用 1×、2×、4×、8× Chinchilla 配比 | [Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046v2) |

中间 checkpoint 可纳入拟合，但需检查训练初期点的影响。[Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) 在公开模型分析中发现，排除初期 checkpoint 可降低预测误差；其约前 10% 或前 10B tokens 的截断设置不宜直接用于所有训练预算。截断规则应预先确定，或像 [Lourie et al., 2026](https://arxiv.org/abs/2608.11859) 一样，将模型尺寸划分为拟合、开发验证和最终测试三段，在开发验证段选择拟合选项，最终测试段只用于验收。随机种子方差也需纳入实验设计。

### 4.3 公开规模配置

公开文献中可供参考的模型规模与训练量配置：

| 来源 | 模型参数量（按各来源口径） | 训练量 | 文献 |
|---|---|---|---|
| OpenAI | 多尺寸，最大 1.5B（非 embedding 参数） | 22M–23B tokens | [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) |
| DeepMind Chinchilla | 70M–16B（400+ 模型） | 5B–500B tokens | [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) |
| StepFun | 多尺寸（3,700+ 模型） | 全部实验累计 100T tokens | [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) |
| Gadre et al. | 11M–6.9B（104 个模型） | 最高 32× Chinchilla 配比 | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540) |
| Fantastic Optimizers | 0.1B–1.2B（4 个尺寸） | 1×–8× Chinchilla 配比 | [Wen et al., 2025](https://arxiv.org/abs/2509.02046) |
| Delphi | 按算力预算扫描尺寸与训练量，最大 holdout 为 25B | 拟合 3e18–3e20 FLOPs，holdout 至 1e23 FLOPs | [Marin, 2026](https://openathena.ai/blog/delphi/) |
| NVIDIA Nemotron | 15B, 340B | 8T, 9T tokens | [15B 报告](https://arxiv.org/abs/2402.16819)；[340B 报告](https://arxiv.org/abs/2406.11704) |
| OLMo | 1B, 7B, 13B, 32B（发布模型） | OLMo 1：2T–2.46T；OLMo 2：按模型设置多阶段预算 | [OLMo, 2024](https://arxiv.org/abs/2402.00838v4)；[OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3) |

### 4.4 Compute Optimal 配比

Chinchilla（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)）的 IsoFLOP 方法：固定每个算力预算 $C$，在不同尺寸的模型上训练，取 loss 曲线的最低点，得到该算力下的最优 $(N, D)$ 组合。

{% include figure.liquid
  path='assets/img/pretrain-scaling/chinchilla-isoflop.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 3. Chinchilla 的 IsoFLOP 结果。左：在各算力预算下对 loss 与 log N 作近似二次拟合，最低点对应最优尺寸；中、右：最优参数量与 token 数随算力的幂律拟合。<a href="https://arxiv.org/abs/2203.15556">来源：Hoffmann et al., 2022, Fig. 3</a>。'
  alt='Chinchilla IsoFLOP 曲线：左图为 loss 对对数参数量的近似二次拟合，中图和右图为最优参数量、token 数对 FLOPs 的幂律拟合。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

在 400+ 个模型（70M–16B，5B–500B tokens）上得到：

$$N_{opt}\propto C^a,\qquad D_{opt}\propto C^b,\qquad a\approx b\approx0.5$$

即在该实验范围内，模型与数据近似等比例增长，$D_{opt}/N_{opt}$ 近似为常数；不同拟合方法的指数略有差异。参数化损失采用 $L(N,D)=E+A/N^\alpha+B/D^\beta$。

[Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2) 在 104 个模型中发现，不同固定 $D/N$ 下 Loss vs $C$ 的幂律指数接近，可用随 $M=D/N$ 变化的系数近似描述；所测范围包括最高 32× Chinchilla 配比。这为过训练区域的外推提供了经验证据。

### 4.5 Ladder 训练量档位

可从 0.5× / 1× / 2× / 4× Chinchilla 配比的档位开始设计，覆盖 under-trained 到 over-trained。这里的档位为示例，实际范围应覆盖目标训练量，稀疏布局无需每个尺寸遍历全部档位。作为参考，Chinchilla 的 $D/N\approx20$ tokens/parameter 使用含 embedding 的总参数量（原文 Appendix F）；换算到 $N_{\text{body}}$ 时该比值会变大，最优配比仍需在自己的配方上验证。

### 4.6 数据质量对 Scaling Law 的影响

[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) 在所比较的训练语料中观察到：

- 质量较高的语料对应更偏向模型参数量的最优算力分配
- 不同数据集的 scaling law 参数差异显著，不能假设一个数据集上拟合的规律直接适用于另一个数据集
- 在控制配方和评估分布时，最优 $N/D$ 的差异可作为数据质量的辅助观察

搭建 Ladder 时，如果训练数据版本发生变化，需要重新验证 scaling law 参数是否仍然适用，不能直接复用旧数据版本的拟合结果。

评估新数据源时无需重跑完整 $N \times D$ 矩阵。[OLMo 2, 2025](https://arxiv.org/abs/2501.00656) 的 Micro-annealing 范式：从主干训练的指定 checkpoint 出发，将候选领域数据与通用数据混合做短期线性退火，即可在较低算力下判断该数据候选对当前阶段下游能力的增益。注意该方法的迁移范围：基线为退火前的 checkpoint，结论适用于该起点和阶段；不能直接推断从随机初始化开始的全程数据排序。

### 4.7 数据配比 Ladder

数据源比例需要通过独立实验确定。数据配比 Ladder 固定模型结构、参数量、训练量和优化器，仅改变数据混合向量 $\mathbf{w}=(w_1,\ldots,w_k)$。

[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) 的流程：以自然分布为中心从 Dirichlet 分布采样候选配比，每个配比训练一个 30M、3B tokens（约 5× Chinchilla）的代理模型，数量约为数据域数量的 5 倍；用按能力聚合的 answer BPB 评估各模型，对每个任务拟合"配比 → BPB"的广义线性模型，再在 token 预算与最大重复次数（约 4–7 次）约束下求解平均 BPB 最低的配比。数据源配比与源内质量分布分开处理：先优化域间比例，再按质量分位数设置上采样曲线。[Qwen3, 2025](https://arxiv.org/abs/2505.09388) 进一步将配比轴从数据源扩展到实例属性（教育价值、领域、语言、安全），通过小型代理模型消融确定过滤与混合策略。

数据配比实验用于选择训练分布，模型规模 Ladder 在该分布上拟合 $N$、$D$ 与 loss 的关系。可先确定配比，再验证规模规律；若配比随目标训练量调整，两类实验需相应迭代。

当代理实验旨在近似保留目标训练的逐桶重复次数时，可在缩短训练量的同时按比例缩小各桶数据池（[Marin 数据流程, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/) §3；该文脚注 13 明确指出拟合响应向更大模型迁移仍待测试）。此外，从头预训练的数据选择与已有 checkpoint 的继续训练数据选择应分开处理，各自标明迁移范围——继续训练的代理实验受起点 checkpoint 的已有能力约束，不能直接推断全程数据排序。

### 4.8 数据重复 Scaling Law

多 epoch 训练需区分累计训练 tokens、unique tokens 与重复次数。建模时可用等效数据量描述重复收益，并检查其是否覆盖目标重复范围。

[Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264v5) 假设重复数据的边际收益指数衰减，给出等效数据量：

$$D^{\prime} = U_D + U_D \cdot R_D^{\ast} \left(1 - e^{-R_D / R_D^{\ast}}\right)$$

其中 $U_D$ 为 unique token 数，$R_D$ 为额外重复次数（$R_D=0$ 即单 epoch），$R_D^\ast$ 为由数据拟合的衰减尺度。$R_D\to\infty$ 时，$D^\prime\to U_D(1+R_D^\ast)$。该形式描述收益饱和；若 loss 随重复次数增加而回升，还需建模过拟合项。

{% include figure.liquid
  path='assets/img/pretrain-scaling/muennighoff-epoch-returns.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 4. 重复数据的边际收益（4.2B 模型，12B unique tokens）。前几轮重复接近新数据的收益，随后边际收益下降，约 40 epoch 后趋于饱和。<a href="https://arxiv.org/pdf/2305.16264v5#page=1">来源：Muennighoff et al., 2023, v5, Fig. 1 左</a>。'
  alt='4.2B 模型在 12B unique tokens 上的重复训练曲线：横轴为累计训练 tokens 和 epochs，纵轴为 final test loss；约 40 epochs 后收益趋于饱和。'
  avoid_scaling=true
  zoomable=true
%}

对 Ladder 设计的指导：

- 重复次数上限取决于数据与配方。[Yan et al., 2025](https://arxiv.org/abs/2511.13421v2) 在线性回归的特定假设下得到最优重复次数随样本数对数增长的结论，并在所测 LLM 设置中观察到相关趋势；不能直接作为通用 epoch 上限。[Lovelace et al., 2026](https://arxiv.org/abs/2605.01640) 在所测设置中发现，较大参数量、较少 unique tokens 和较多重复会共同加剧过拟合。
- [Xue et al., 2023](https://arxiv.org/abs/2305.13230v2) 的受控实验中，重复过拟合主要随参数量变化；增加数据量可缓解该问题，同等数据量下提高质量未带来同样改善。该结果限于论文测试的模型和数据设置。
- 在 Xue 等测试的正则化方法中，dropout 对重复过拟合改善较明显（v2 Table 4）；可延迟启用以改善早期学习，较大模型需重新调节 dropout rate。

## 5. 超参数搜索与训练执行

本章重点是为最优表现预测标定 Fully-Tuned 的 $(\eta,B)$，并将其写成可随 $N,D$（或 $C$）外推的函数；既定配方预测（§1.4 目标 1）按配方缩放规则运行。以下按搭建 Ladder 时的决策顺序展开。

### 5.1 搜参目标：Fully-Tuned Frontier 与宽平台

[Lourie et al., 2026](https://arxiv.org/abs/2608.11859) 与 [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) 揭示了小模型搜参的核心难点：
- 小模型对超参极度敏感：小尺寸模型在次优超参下的性能劣化极其陡峭，其真实的 Scaling 规律只有在"完全调优前沿"（Fully-Tuned Frontier）上才会显现；
- 搜参不充分的后果：如果在小模型上仅凭经验粗搜并停留在局部极小值，拟合出的幂律曲线会严重扭曲，导致大模型预测失真。

部分大模型实验观察到较宽的近优区间。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320v1) 在其 156B-A7B 实验中测试了 LR 乘以或除以 $\sqrt{2}$、BSZ 增加 25% 的配置，最终 training loss 差不超过 $7\times10^{-4}$。这一范围可作为局部确认的参考，仍需检查自己的目标指标。

由此得出搜参的资源分配原则（适用于 [§1.4](#14-决策目标) 中目标 2–4，即最优资源分配、方案比较和超参预测）：算力集中在小模型的穷举搜索，确保拟合点落在 Fully-Tuned Frontier 上；大模型可先在外推 LR 乘以或除以 $\sqrt{2}$ 的范围内局部确认，必要时扩大搜索。若仅用于目标 1（既定配方预测），按配方的固定缩放规则运行即可，无需逐点穷尽调优。

### 5.2 Weight Decay 处理

若干公开配方固定 WD：[Kimi K2, 2025](https://arxiv.org/abs/2507.20534v2) 在 1T MoE、15.5T tokens 上使用 $\lambda=0.1$（MuonClip 优化器）；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961v1) 使用 AdamW，并将 embedding 排除在 decay 之外；[Qwen3, 2025](https://arxiv.org/abs/2505.09388) 报告了 LR 与 batch size 的 scaling law，未报告 WD 的 scaling 预测。

最优 WD 还取决于训练量和评价目标。[Han et al., 2026](https://arxiv.org/abs/2602.11137v2) 在所测 TPP 下发现，预训练 loss 偏好的 WD 随 TPP 增大而降低，但较强 WD 可能有利于后训练可塑性。复现配方时可沿用其 WD；研究充分调优后的 $B_{opt}$、$B_{crit}$ 或跨 TPP 关系时，应检查 WD 的影响。在 [Power Lines (Bergsma et al., 2025)](https://arxiv.org/abs/2505.13738v2) 的 AdamW 设置下，可利用 timescale $\tau=B/(\eta\lambda D)$ 联合校准。

### 5.3 Batch Size 与 Learning Rate 的迁移规律

[Power Lines (Bergsma et al., 2025)](https://arxiv.org/abs/2505.13738) 在多组 $N, D$ 网格上测得：
$$B_{opt}\propto D^{0.4},\qquad B_{crit}\propto D_{min}^{0.5}$$
$D$ 为训练预算，$D_{min}$ 为达到目标 loss 所需的最少 tokens。上述指数为该文拟合结果的近似值；在其优化器、参数化和数据设置中，两者对参数量 $N$ 的依赖较弱。可据此初始化 BSZ 的迁移规则，再在自身数据上验证，无需默认对每个尺寸从零搜索。

$B_{crit}$ 表示 token 效率与步数之间的权衡转折。在该文的双曲线模型中，$B=B_{crit}$ 时达到同一 loss 需要约 $2D_{min}$ tokens，训练步数相应减少。继续增大 $B$ 后，步数仍可下降，但所需 tokens 和理论算力开销增加。实际训练时间还取决于硬件利用率与并行效率，需对应系统实测。

[Schaipp, 2026](https://arxiv.org/abs/2607.01487v1) 以“使用 $D$ tokens 的次优配置，loss 不超过最优配置使用 $0.95D$ tokens 的结果”定义约 5% 算力损失的近优 batch 区间。其实验中区间宽度约为 4 倍；在该文对数对称的拟合模型下，对应 $[B_{opt}/2,2B_{opt}]$。可将这一范围作为局部搜索起点，验证预测中心和区间边界，并同时检查训练稳定性。

在 $B < B_{crit}$ 范围内，固定 $N$、$D$、改变 $B$ 时，Power Lines 通过调整 $\lambda$ 维持最优 timescale $\tau = B/(\eta\lambda D)$；LR 还受最大稳定学习率限制（[Power Lines §2.4](https://arxiv.org/abs/2505.13738)）。将 $B_{opt} \propto D^{0.4}$ 外推到不同训练量时，$\lambda$ 与最优 $\tau$ 随 $D$ 的变化也参与约束，仅凭 $B$ 的幂律不足以确定 $\eta$ 的幂律。LR 的跨训练量外推需在 timescale 框架下联合校准 $(\eta, \lambda, B)$。

{% include figure.liquid
  path='assets/img/pretrain-scaling/cerebras-bcrit-hyperbola.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 5. 达到相同目标 loss 所需训练 tokens 与 steps 的双曲线关系，左、右分别为 610M 和 1.7B 模型，颜色表示 loss。B_crit 表示 token 效率与步数之间的权衡转折。<a href="https://arxiv.org/pdf/2505.13738v2#page=6">来源：Bergsma et al., 2025, v2, Fig. 4</a>。'
  alt='610M 和 1.7B 模型的两幅 tokens 对 steps 曲线图：不同目标 loss 对应不同双曲线，颜色表示 loss，曲线标出 B_crit。'
  avoid_scaling=true
  zoomable=true
%}

### 5.4 超参 Scaling Law 公式选用

根据 ladder 覆盖的 $D/N$ 档位数量，选择不同的外推公式。典型 ladder（多尺寸 × 多训练量档位）建议使用 Step Law 或 Power Lines；如果 ladder 只跑单一 $D/N$ 配比，$C$ 幂律即可。

固定 $D/N$ 配比时，可按总算力 $C$ 拟合超参幂律：

$$\eta_{opt} = a \cdot C^{-\alpha}, \quad B_{opt} = b \cdot C^{\beta}$$

在多个小尺寸上搜索最优 $(\eta, B)$，分别拟合出 $(a, \alpha)$ 与 $(b, \beta)$。[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) 提供了拟合系数，可作为先验参考。

Ladder 覆盖多档 $D/N$ 时，仅用 $C$ 无法区分相同算力下的不同模型尺寸与训练量，需将 $N$ 和 $D$ 分开建模：

| 方案 | $\eta_{opt}$ | $B_{opt}$ | 适用条件与局限 |
|---|---|---|---|
| Step Law ([Li et al., 2025](https://arxiv.org/abs/2503.04715v3)) | $c\cdot N^{-\alpha}D^{\beta}$ | $d\cdot D^{\gamma}$ | 可跨 $D/N$ 建模；3,700+ 模型验证；所测设置下 $B_{opt}$ 主要随 $D$ 变化，目标范围仍需验证 |
| Power Lines ([Bergsma et al., 2025](https://arxiv.org/abs/2505.13738v2)) | 由 timescale $\tau$ 联合约束 | $d\cdot D^{\gamma}$（所测范围内对 $N$ 的依赖较弱） | $\tau=B/(\eta\lambda D)$ 提供条件约束；仍需检查 LR 稳定性与 BSZ 效率范围 |
| Token Horizons ([Bjorck et al., 2024](https://arxiv.org/abs/2409.19913)) | $c \cdot N^{-\alpha} D^{-\beta}$ | — | 固定模型尺寸下，peak LR 随训练长度衰减 |

Token Horizons 与 Step Law 都研究了固定 $N$ 时 $\eta_{opt}$ 随 $D$ 的变化，但 $D$ 的指数符号相反（前者 $D^{-\beta}$，后者 $D^{+\beta}$）。原因未查明。应结合自身的参数化、优化器、batch size、数据和 schedule 选择候选形式，并在 holdout 上验证。

{% include figure.liquid
  path='assets/img/pretrain-scaling/step-law-hyperparameter-validation.png'
  id='fig-step-law-hyperparameter-validation'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 6. Step Law 在 N = 1B、D = 100B 的测试条件下，对照超参数公式的预测配置与实验确定的最优配置。等高线依据 120 次不同学习率与 batch size 组合的训练实验得到；该条件超出论文的拟合范围。图示比较限于论文采用的训练配方与学习率调度。<a href="https://arxiv.org/pdf/2503.04715v3#page=1">来源：Predictable Scale: Part I, 2025, v3, Fig. 1（PDF 第 1 页）</a>。'
  alt='Step Law 原论文 Figure 1：学习率与 batch size 对应的 loss 等高线，以及 Step Law、其他超参数公式和实验确定的最优配置。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

$\mu$-Transfer ([Yang et al., 2022](https://arxiv.org/abs/2203.03466v2)) 通过参数化支持宽度方向的最优 LR 迁移；含非零 WD 的情况也有实验支持（原论文 Appendix G.1.2；[Power Lines](https://arxiv.org/abs/2505.13738v2) 使用 $\mu$P + AdamW + 非零 WD）。原论文对 BSZ、深度和训练长度方向也有有限实验，迁移到新配方仍需验证。$\mu$P 可与超参幂律拟合结合使用。

### 5.5 搜索流程

可优先搜索 Learning Rate、Batch Size，再检查 Schedule 与 Weight Decay；具体顺序按敏感性诊断调整。$\epsilon$、$\beta_2$ 等稳定性超参可在基线和代表性尺度验证后固定。

资源受限时，可采用坐标下降（Coordinate Descent），逐步确定搜索范围：

1. 小模型网格搜索：在最小尺寸（如 ~130M）的预定范围内充分搜索，确定核心区间；
2. 中模型验证与外推：在中等尺寸（如 ~500M）上检验幂律趋势，修正斜率；
3. 大模型局部确认：LR 可先在外推值乘以或除以 $\sqrt{2}$ 的范围内确认；BSZ 可先搜索 $[B/2,2B]$，检查边界后决定是否扩展（[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)、[§5.3](#53-batch-size-与-learning-rate-的迁移规律)）。

其中，Batch Size 的搜索维度可以进一步削减（见 [§5.3](#53-batch-size-与-learning-rate-的迁移规律)）。此外，[Power Lines (Bergsma et al., 2025)](https://arxiv.org/abs/2505.13738) 提出的 timescale $\tau = B/(\eta\lambda D)$，在已选定或可迁移 LR 的条件下，可通过调整 WD 搜索最优 $\tau$，减少联合搜索成本。使用时仍需检查最大学习率的稳定性限制和 batch size 的效率范围（[§5.3](#53-batch-size-与-learning-rate-的迁移规律)）；等 $\tau$ 本身不保证不同 $(\eta, \lambda, B)$ 三元组性能等价。

### 5.6 对比新优化器时的注意事项

[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) 提出了两个与 ladder 搜参相关的结论：
- 公平调参：各优化器需分别调优，并报告搜索范围和预算。AdamW 基线调优不足可能夸大新优化器的收益。
- 检查跨规模收益：在该文的 8× Chinchilla 对照中，矩阵优化器相对 AdamW 达到同一 loss 的 token 效率改善从约 0.1B 模型的 $1.4\times$ 降至 1.2B 的 $1.1\times$。这不等同于训练时间加速比；方案优势需沿尺寸和训练量两个方向验证。

### 5.7 LR Schedule

可采用以下两类调度策略：

- Cosine 调度：单阶段训练常用，包含短步数 warmup 与连续平滑衰减至 0（或小残余值）。在多阶段训练中，不同阶段的最小学习率可能不同。
- WSD 调度（Warmup-Stable-Decay）：便于复用 stable 训练轨迹，并在选定预算开始退火。Stable 阶段的曲线拟合与退火后的收益需在目标配方下验证。[River Valley 理论 (Wen et al., 2024)](https://arxiv.org/abs/2410.05192v3) 在特定损失几何与优化动力学假设下解释了 stable 和 decay 阶段的作用。

退火比例：[Tissue et al., 2024](https://arxiv.org/abs/2408.11029v2) 在所分析设置中得到约 10%–20% 的最优退火比例，可作为初始参考，目标预算仍需验证。[Wang et al., 2025](https://arxiv.org/abs/2512.13705) 进一步研究了退火策略的 scaling 与跨规模迁移。

若比较最终退火后的交付性能，应在可比预算下评估退火终点。[Fantastic Optimizers](https://arxiv.org/abs/2509.02046v2) 观察到优化器的 loss 排名在退火期间翻转，因此 stable 中途排名不能直接替代最终排名。中间 checkpoint 可用于阶段分析或经过验证的代理筛选。

### 5.8 收尾流程

以下两类步骤见于公开训练配方。若目标交付流程包含这些步骤，Ladder 对应阶段也应执行或使用经验证的代理，以保持拟合目标与交付目标一致。

- Continued Training / 数据课程：主训练后切换数据配比，并配合 LR 衰减继续训练。[Nemotron-4, 2024](https://arxiv.org/abs/2402.16819) 和 [OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3) 均报告了这类做法；具体预算取决于模型和阶段。预测最终交付性能时需纳入这些阶段的影响，阶段实验的设计见 [§8.1](#81-分阶段-ladder)。
- Weight Averaging：对共享初始化的多个 checkpoint 做权重平均。一类做法是从共同预训练 checkpoint 出发，独立微调后合并（[Model Soups](https://arxiv.org/abs/2203.05482v3)；OLMo 2 在 mid-training 中采用类似流程）；另一类是同一训练轨迹的滑动窗口平均（[LAWA](https://arxiv.org/abs/2209.14981)；[Sanyal et al., 2024](https://arxiv.org/abs/2306.03241)）。[Model Merging (ByteDance Seed, 2025)](https://arxiv.org/abs/2505.12082v3) 在 1.3B 和 13B 的直接对照中发现，WSD stable 阶段的 checkpoint 合并（PMA）可接近退火终点的下游表现。合并窗口和起点仍需验证，再决定是否纳入流程。

## 6. Loss Scaling Law 与下游任务拟合

拟合前需明确以下三项设置。[Porian et al., 2024](https://arxiv.org/abs/2406.19146) 表明，它们会影响 compute-optimal 指数的估计：

1. 算力统计需计入 output head，并与拟合参数量口径分开记录（[§3.1](#31-参数量口径)）
2. 明确 warmup 的缩放规则，避免固定 warmup 在小预算中占比过高（[§7.5](#75-常见陷阱与教训)）
3. 追求最优表现或公平比较时（[§1.4](#14-决策目标) 目标 2–4），各尺寸档位的超参需逐一搜索至 Fully-Tuned Frontier（[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)）；既定配方预测（目标 1）按配方缩放规则运行

同一工作还在无 LR 衰减的设置下恢复了近似的 compute-optimal 规律；这一结果限于其研究目标和实验条件。

### 6.1 函数形式

[§4.4](#44-compute-optimal-配比) 给出的 Chinchilla 加性形式隐含 $\partial^2L/\partial N\partial D\equiv0$。[Skaling (Videau et al., 2026)](https://arxiv.org/abs/2608.07222v1) 在所分析的 loss 曲面上发现负的混合偏导数，并给两项之和增加外指数 $k$：

$$L(N,D)=\left(\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}\right)^{k}+E$$

$k=1$ 时退化为 Chinchilla；$0<k<1$ 时混合偏导数为负。该文拟合得到 $k\approx0.31$–$0.45$。仅多一个参数，compute-optimal 闭式解的代数形式与 Chinchilla 相同。

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-vs-chinchilla-error.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 7. 两种形式在同一 (N, D) 网格上的预测残差，每点为一次训练。左、中：带符号百分比误差，共用色标；Chinchilla 的残差呈鞍形并向四角增大，Skaling 全网格接近零。右：两者误差之比，Skaling 在 76% 的配置上更准，中位数 2.2 倍。<a href="https://arxiv.org/abs/2608.07222">来源：Videau et al., 2026, Fig. 1</a>。'
  alt='三张 (N, D) 网格散点图：左为 Chinchilla 的带符号百分比误差，中为 Skaling 的同类误差，右为两者误差之比。'
  avoid_scaling=true
  zoomable=true
%}

替代方案：[Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972) 令数据侧系数与指数显式依赖 $N$，共九参数。[Skaling (Videau et al., 2026)](https://arxiv.org/abs/2608.07222) Table 1 在 Farseer 与 SK-Grid 两套数据上比较了三种形式；Skaling 使用六个参数，在插值和单轴外推中误差最低，远端外推也更稳定。

可将 Skaling 作为默认候选，在自身数据上与 Chinchilla 等形式比较。该函数族在 $k=1$ 时包含 Chinchilla；重新拟合得到的参数可能改变预测的 compute-optimal 配比，方向和外推精度均需由 holdout 验证。

固定 $D/N$ 时，可尝试一维近似 $L=G(M)/C^\gamma+E$（$M=D/N$），用于该配比下的外推；跨配比资源分配仍需建模 $N,D$。

实验点布局需与函数形式共同验证。Skaling 论文采用 L-shape：较小尺寸覆盖不同 $D$，其余尺寸使用较小 $D$。在两套实验中，算力约降至完整网格的 1/5–1/10，精度随数据集和外推方向变化。Chinchilla 加性形式在这些稀疏实验中的误差明显增加，采用稀疏布局时应单独验证。

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-v1-sampling-evaluation.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 8. Skaling 的采样策略与评价范围。左：Random 随机留出验证点；L-shape 在小 D 下覆盖不同 N，并在小 N 下覆盖不同 D。右：插值、沿 N 或 D 外推及联合外推的区域。横轴为 D，纵轴为 N。<a href="https://arxiv.org/pdf/2608.07222v1#page=5">来源：Videau et al., 2026, v1, Fig. 4（PDF 第 5 页）</a>。'
  alt='Skaling v1 Figure 4 原图：左侧为 Random 与 L-shape 采样，右侧为插值及沿模型尺寸、训练量和二者联合外推的评价区域；横轴 D，纵轴 N。'
  avoid_scaling=true
  zoomable=true
%}

### 6.2 LR 退火 Scaling Law

[Tissue et al., 2024](https://arxiv.org/abs/2408.11029) 将 loss 表示为训练 step 的函数：

$$\hat L(s) = L_0 + A \cdot S_1(s)^{-\alpha} - C \cdot S_2(s)$$

$S_1(s)=\sum_{i\leq s}\eta_i$ 为累计学习率面积，$S_2(s)$ 为带遗忘核的累计退火量，warmup 按原文约定另行处理。该式以 schedule 为输入，可利用同一轨迹的多个 eval 点拟合整条 loss 曲线。原文 §4.7 还预测了 continued training 中不同 re-warmup 最大 LR 的曲线；跨模型规模或数据分布复用系数仍需验证，最优 LR 的选择见 [§5.4](#54-超参-scaling-law-公式选用)。

个人经验：遗忘核选择幂律核 $K(\Delta) = (1+\Delta)^{-p}$ 通常优于原论文的单指数核。

### 6.3 拟合诊断与失效处理

总体拟合误差可能掩盖系统性残差。拟合完成后需检查以下几项：

- 残差结构：检查残差是否随模型规模 $N$、训练量 $D$ 或训练阶段呈系统性变化。结论是否依赖少数实验点或特定函数形式。
- Checkpoint 相关性：同一训练轨迹的多个 checkpoint 存在序列相关性，不能仅凭点数判断独立信息量。[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) 按 IsoFLOP bucket 的最优点进行 bootstrap，重采样单位需考虑数据相关性。
- 不确定性分离：分别报告随机种子方差、评估方差和拟合不确定性。
- 截断与验证划分：截断规则和拟合选项应预先确定或用独立的开发验证集选择；最终 holdout 保持验收用途，避免兼作反复调参数据（见 [§4.2](#42-尺寸数量与跨度)）。
- 失效处理：误差过大时，明确需追加什么实验（如补充模型尺寸、延长训练或重新搜索超参数）。哪些资源分配、方案排序或目标性能决策暂不能据此作出，应明确记录。没有证据的失败解释记为"原因未查明"。
- 产物清单：保存配方版本、实验记录、拟合方法、预测区间、验证范围和未解决问题。

若决策涉及下游能力，还需验证相应任务指标的预测。评估集内不同样本的 scaling 规律可能不同，单一 Acc vs Compute 曲线未必能准确描述整体变化。

### 6.4 下游预测方法

| 方法路线 | 核心思路 | 代表工作 | 局限 |
|---|---|---|---|
| Loss → Performance | 先预测 loss 或 perplexity，再映射到下游指标 | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2)（错误率对 perplexity 的幂律，等价于对交叉熵 loss 的指数关系）；[Delphi](https://openathena.ai/blog/delphi/)（sigmoid） | 映射依赖任务与评估协议 |
| $(N,D)$ → Task Loss → Acc | 两阶段：先由 $N,D$ 预测 task-specific loss，再拟合 loss → accuracy | [Bhagia et al., 2024](https://arxiv.org/abs/2412.04403)（OLMo Task Ladder） | 任务间噪声与预测误差差异大 |
| End-to-End | 直接对任务指标随算力的变化建模；可按难度分组 | [COD (Xu et al., 2026, v4)](https://arxiv.org/pdf/2502.17262v4)（难度特征聚类）；[GPT-4 Technical Report](https://arxiv.org/abs/2303.08774v6)（HumanEval 难度分桶） | 需要可分辨的评估信号；具体范围取决于方法与评分协议 |

[Delphi](https://openathena.ai/blog/delphi/) 用 IsoFLOP 预测 loss，再以 sigmoid 映射到 MMLU/HumanEval/GSM8K，可作为初始方案。COD 在其测试任务中提高了预测精度，同时增加了采样、聚类与映射步骤；不同路线的优劣需在相同任务和协议下比较。

### 6.5 COD 框架概述

[COD (Xu et al., 2026, v4)](https://arxiv.org/pdf/2502.17262v4) 的四阶段流程：

1. 聚类：用多个小模型对评估集每道题多次采样，取平均正确率作为难度特征，按难度聚类分组；
2. 拟合：对每个聚类分别拟合 $E[\mathrm{Acc}(C)] = g + (1-g) \cdot e^{-aC^{-b}-c}$；
3. 外推：筛选增长趋势和拟合可靠性满足条件的聚类，代入目标算力得到预测值，按聚类样本数加权平均；
4. 映射：用已有模型的评估数据校准"可外推子集指标 → 全评估集指标"的映射曲线。

COD v4 在 70B 模型、8 个 benchmark 上的平均绝对预测误差为 1.55 个百分点（Table 1）。

{% include figure.liquid
  path='assets/img/pretrain-scaling/cod-v4-prediction-accuracy.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 9. COD v4 在 MATH、MMLU-pro 上的预测曲线与 70B 实测值，包含 COD、Loss-Intermediate、End-to-End(exp) 和 End-to-end(BNSL)。红点为小模型结果，蓝点为目标模型实测值。<a href="https://arxiv.org/pdf/2502.17262v4#page=9">来源：Xu et al., 2026, v4, Fig. 4 的 MATH 与 MMLU-pro 子图（PDF 第 9 页）</a>。'
  alt='COD v4 Figure 4 的 MATH 与 MMLU-pro 原始子图：横轴为算力，纵轴为准确率，比较 COD、Loss-Intermediate、End-to-End(exp) 与 End-to-end(BNSL) 的拟合和 70B 目标预测。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

### 6.6 适用条件与局限

- 评估样本量需足够，样本过少时聚类指标不稳定（[COD v4](https://arxiv.org/pdf/2502.17262v4)）。
- 论文在若干跨模型设置中复用了聚类，并用外部模型校准子集到全集的映射。新架构或数据分布下仍需验证难度特征和映射的稳定性。
- 主预训练实验采用 warmup 后恒定 LR。Continued training 实验另含数据变化与退火；小模型匹配两阶段数据分布和 TPP，再在对应终点评估（v4 Appendix D–E）。
- v4 使用 dense 模型的聚类预测激活参数为 32B 的 MoE 目标，平均与最大绝对误差分别为 3.11、8.11 个百分点；该结果提供了有限的跨架构证据（§5.3.1、Table 2）。
- MMLU-Pro 的生成答案评估已纳入实验；仅按选项概率排序的协议与论文 passrate 定义不同。CoT 情况已有经验预测结果，但理论尚未充分覆盖非唯一答案和推理路径（Appendix H）。

## 7. 外推验证与失效处理

### 7.1 Holdout 验证

用较小尺寸拟合，在目标外推方向设置 holdout。可按预算安排多个外推倍数，以检验预测开始偏离的范围。[Delphi](https://openathena.ai/blog/delphi/) 在 3e18–3e20 FLOPs 上拟合，holdout 设在 1e21、1e22、1e23 FLOPs。首次 recipe 在约 33× 外推（1e22）时 loss 比预测高 2.5%，约 333×（1e23）的运行发散。

架构或优化器变更后，先在小规模上验证旧系数的迁移性，再根据结果决定局部校准或完整重拟合（见 [§3.4](#34-共同配置与变量分类)）。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 切换架构 + Muon 后最优 LR 和 batch size 均偏移，直接沿用旧系数落在次优区域。变更后的验证策略：LR 在大模型上确认，BSZ 在小模型上确认，利用宽平台特性（[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)）先做局部确认，必要时扩大搜索。

Holdout 误差阈值需事先确定，依据为目标方案的最小可区分差异与随机种子方差。[Choshen et al., 2025](https://arxiv.org/abs/2410.11840) 报告：文献中推动建模改动的最小相对差异约为 4%，同架构随机重启的波动可到约 3.5%，通常能获得的最佳相对误差约 4%。Delphi 的 0.2%–0.5% 是特定实验结果，不能作为通用合格线。同时报告外推置信区间。

### 7.2 稳定性压力测试

小规模训练无法充分暴露大规模训练中的 loss spike 和梯度异常。架构、优化器或归一化变更时，需在外推前增加压力测试。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 使用中等规模模型、恒定学习率以及 2×/4× 预测最优 LR 提高优化压力，在相同压力下比较新旧 recipe。通过条件：新 recipe 稳定性不低于基线，预测最优 LR 与 batch size 位于稳定区域。

短期高学习率下稳定与在目标训练量附近稳定是不同条件。对关键候选应保留覆盖代表性训练量和阶段转换的训练轨迹，除总 loss 外按模型结构记录梯度范数、激活范围和路由统计。[Nemotron 3 Ultra (NVIDIA, 2026)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) §2.7 记录了训练后期两次发散——第一次与输出层梯度精度有关，恢复 FP32 路径后稳定；第二次原因未查明，通过提前退火缓解。[Marin 公开设计记录](https://github.com/marin-community/marin/issues/8435)也将随 token horizon 检查梯度范数和评估轨迹、判断大模型异常是否需要干预列为 Ladder 用途。

将这些轨迹与阶段配置一同交付，目标训练持续偏离时启动调查。若随后更改 schedule、训练量或数据配比，应更新预测条件与验证范围。

### 7.3 实际效率与部署约束

理论 FLOPs 的改善未必带来相同幅度的训练时间改善。在报告方案收益时，应在明确硬件、并行方式和数值精度的条件下同时报告实际训练时间；这些条件变化后需重新检查收益。[Marin 后续](https://openathena.ai/blog/pretraining-speedup/)区分了 theoretical 与 realized efficiency。[Nemotron 3 Ultra (NVIDIA, 2026)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) 在训练过程中设置了高精度分支对照，说明精度方案本身需要验证，而非直接假设等价。

方案采用判断应同时考虑预测精度和训练稳定性。存在推理部署约束时，将推理成本纳入资源分配与架构选择。

### 7.4 组合验证

架构、优化器、数据和精度等单项改动通过后，应再验证最终组合。独立实验中各项收益可能存在相互影响，不能直接相加。[Marin 后续](https://openathena.ai/blog/pretraining-speedup/)在候选单项测试后另行测试了组合；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) Appendix A.2.5 对数据混合与上采样分别进行了独立及联合验证。

验证要点：

- 将数据配比与目标训练量、数据重复程度和最终配置联合检查。
- 对各阶段起始 checkpoint 明确其训练历史和选择规则。
- Micro-annealing 等代理实验的结论应标明迁移验证范围。
- 组合验证若仅在小规模完成，应保留这一范围限制，不能表述为目标最大规模已验证。

### 7.5 常见陷阱与教训

**小模型搜参不充分**

小模型搜参不充分可能导致最优超参外推偏差。应检查搜索边界与局部最优；在适用条件下，[Power Lines](https://arxiv.org/abs/2505.13738v2) 的 timescale $\tau$ 可帮助组织联合搜索（见 [§5.5](#55-搜索流程)）。

**参数量口径不一致**

混用总参数量与主干参数量，或把拟合用的 $N_{\text{body}}$ 直接代入 $C = 6ND$。典型表现是不同文档的 TPP 数值对不上。解决：两个口径分开记录，引用外部数值前先确认口径（见 [§3.1](#31-参数量口径)）。

**外推跨度过大**

Loss curve scaling law（[§6.2](#62-lr-退火-scaling-law)）的可靠外推范围有限，跨度过大时出现系统性偏差。解决：控制外推跨度在已验证范围内，更大跨度需补充实验点。

**训练初期 checkpoint 纳入拟合**

使用中间 checkpoint 拟合时，需检查训练初期点是否造成系统偏差，按预定规则或独立开发集选择截断位置（见 [§4.2](#42-尺寸数量与跨度)），不在最终 holdout 上挑选。仅用 endpoint loss 拟合时无此步骤。截断规则与 warmup 规则需分别确定，并检查 warmup 在不同预算中的占比。

以下问题已在前文对应章节详细讨论，此处仅列为提醒：

- 计算量近似：按实际架构计入输出 head 与注意力计算，明确 $6ND$ 的适用条件（[§3.2](#32-计算量与宽深比)）
- 评估间隔：固定 step 间隔可能使较长轨迹在拟合中占更大权重；明确采样和加权方式，并处理轨迹内相关性（[§3.4](#34-共同配置与变量分类)、[§6.3](#63-拟合诊断与失效处理)）
- 拟合公式：检查加性假设是否符合自身数据，将 Skaling 与其他候选形式一起验证（[§6.1](#61-函数形式)）

## 8. 特定场景

### 8.1 分阶段 Ladder

预训练、mid-training 和长上下文扩展具有不同的数据分布、序列长度和 LR schedule，最优超参数及 scaling law 系数可能随阶段改变。[Qwen3, 2025](https://arxiv.org/abs/2505.09388) 分别针对通用预训练、推理强化和长上下文三个阶段预测 LR 与 batch size；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) 也为 pretraining、midtraining 和 long-context extension 分别设置训练量、batch size、序列长度与 LR schedule。

如果交付流程包含多个阶段，应针对需要预测或选择配置的阶段设计实验，必要时分别建立 Ladder：

1. 预训练 Ladder：从随机初始化开始，固定基础数据配比，拟合 $N$、$D$、LR 与 batch size；
2. Mid-training Ladder：各尺寸从对应预训练 checkpoint 开始，固定阶段数据配比，拟合新增 token 量、LR schedule 与能力增量；
3. 长上下文 Ladder：各尺寸从对应 mid-training checkpoint 开始，单独搜索序列长度、长短样本比例、RoPE 配置、训练量与 LR。

数据分布、序列长度、优化器或架构变化后，不直接复用上一阶段的 scaling law 系数。阶段间应同时记录基础能力变化和目标能力增量，避免后续阶段在提高目标指标的同时掩盖基础能力下降。

### 8.2 MoE Ladder 的参数口径与稀疏度轴

MoE 模型不能只用单一参数量 $N$ 描述规模。每个实验点至少需要同时记录：

- 总参数量 $N_{total}$：包括全部专家，用于描述模型容量与存储开销；
- 激活参数量 $N_{active}$：单个 token 实际参与计算的参数量，用于近似训练与推理 FLOPs；
- 路由专家总数 $E_{total}$、每 token 激活的路由专家数 $E_{active}$，以及稀疏率 $S=E_{total}/E_{active}$；共享专家单独记录；
- 路由器、共享专家和负载均衡损失的配置。

[Kimi K2, 2025](https://arxiv.org/abs/2507.20534) 在固定 $N_{active}$ 和训练 FLOPs 的条件下改变 $E_{total}$，单独拟合 sparsity scaling law。结果显示，稀疏率从 8 提高到 48 时，相同目标 loss 所需 FLOPs 持续下降，但通信、存储和推理复杂度同时增加。因此，MoE Ladder 应包含两类实验：

1. 规模 Ladder：固定稀疏率和路由配置，改变 $N_{active}$ 与训练量 $D$；
2. 稀疏度 Ladder：固定 $N_{active}$、$D$、激活专家数、专家宽度及其余结构，只改变路由专家总数。

两类实验的拟合变量不能混用。前者用于预测模型规模与训练量的关系，后者用于选择总容量与计算量的配比。

### 8.3 Scaling Ladder 配置参考

公开文献中有两套较完整的 Ladder 配置可供参考，侧重点不同：

### 8.3.1 Fantastic Optimizers Ladder（Dense，超参搜索导向）

来源：[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046v2) Table 2–3。Llama 2 架构，四个尺寸均固定 32 层、MHA、seq_len 4096，宽度随尺寸变化。目标是公平对比优化器，并为各优化器分别搜索超参数。

| 尺寸 | hidden_dim | inter_dim | heads | 数据比 |
|---|---|---|---|---|
| 130M | 512 | 2,048 | 8 | 1×–8× Chinchilla |
| 300M | 768 | 3,072 | 12 | 同上 |
| 520M | 1,024 | 4,096 | 16 | 同上 |
| 1.2B | 1,536 | 6,144 | 24 | 同上 |

论文比较十种优化器。Table 3 给出 AdamW 的坐标下降示例，最终配置为 Peak LR 8e-3、WD 0.1、warmup 2000 steps、BSZ 128；这些数值不构成所有尺寸的共同设置。例如 520M、1× Chinchilla 的 AdamW 配置使用 WD 0.2、warmup 1000 steps、BSZ 256。数据由 DCLM-baseline、StarCoder V2 Data 与 ProofPile 2 混合而成。

### 8.3.2 Delphi Ladder（端到端 Scaling Law 导向）

来源：[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/)。Qwen 3 架构，MLP ratio 4，seq_len 4096，Dense decoder-only。目标是拟合 IsoFLOP scaling law 并外推到 1e23 FLOPs（25B 参数）。Marin 团队后续已将 recipe 扩展到 MoE（[535B-A23B hero run](https://openathena.ai/blog/pretraining-speedup/)），该后续文章包含多尺寸方案比较、预注册预测、理论与实际效率的区分以及组合实验设计经验，其中小规模的组合测试规模范围需注意。

共同设置：AdamH、WSD schedule（10% linear warmup、20% linear decay to 0）、f32 参数 + bf16 计算、FSDP。数据为 Nemotron-CC + StarCoderData + ProofPile 2。

Ladder 结构：在 3e18–3e20 FLOPs（约 2 个数量级）上做 IsoFLOP 扫描，每个算力预算跑多个尺寸-token 组合取最优点，共 7 个 IsoFLOP 最优点用于拟合。Holdout 设在 1e21、1e22、1e23 FLOPs（相对最大拟合预算约 3×–333× 外推）。LR、BSZ 和初始化按 recipe 中的模型宽度、训练长度及预算规则设置，不做逐点手动搜索。

### 8.3.3 选择建议

- 需要搜索最优超参并拟合超参 scaling law → Fantastic Optimizers 的网格设计
- 需要端到端 loss 预测并外推到大规模 → Delphi 的 IsoFLOP + recipe 公式
- 两套配置均为起点；模型族与结构缩放规则应对目标大模型具有代理有效性（GQA、head_dim、宽深配置等）

## 9. 搭建 Checklist

### 9.1 设计阶段

- [ ] 明确决策目标：配方预测、资源分配、方案比较还是超参预测（[§1.4](#14-决策目标)）
- [ ] 确定评价协议：代理指标、最终指标、跨规模排序相关性、评价版本控制（[§2](#2-评价协议与验收指标)）
- [ ] 区分固定量、缩放规则与实验自变量（[§3.4](#34-共同配置与变量分类)）
- [ ] 按拟合形式、目标外推范围与预算确定尺寸数量，并另设 holdout（[§4.2](#42-尺寸数量与跨度)）
- [ ] 训练量档位覆盖目标范围，并按所选布局分配实验点（[§4.5](#45-ladder-训练量档位)）
- [ ] Holdout 按多档外推倍数拉开（[§7.1](#71-holdout-验证)）
- [ ] 参数量 $N_{\text{body}}$ 与算力 $C$ 分开记录（[§3.1](#31-参数量口径)）
- [ ] 按实际架构统计 FLOPs，计入输出 head 与注意力计算（[§3.2](#32-计算量与宽深比)）
- [ ] MoE 同时记录总参、激活参、路由及共享专家配置；规模与稀疏度实验分别设计（[§8.2](#82-moe-ladder-的参数口径与稀疏度轴)）
- [ ] 数据配比未确定时安排配比实验，并验证其与目标训练量的匹配（[§4.7](#47-数据配比-ladder)）
- [ ] 对需要预测或选择配置的训练阶段分别设计实验与验证（[§8.1](#81-分阶段-ladder)）
- [ ] 算力受限时评估 L-shape 布局（[§6.1](#61-函数形式)）

### 9.2 配置阶段

- [ ] 同一模型族保持约定的架构属性及宽深缩放规则，明确结构实验自变量（[§3.3](#33-架构一致性)）
- [ ] 同一配方保持约定的数据、评估、优化器及 schedule 规则，明确实验自变量（[§3.4](#34-共同配置与变量分类)）
- [ ] 明确 eval 采样与加权规则，并考虑同一训练轨迹的相关性（[§3.4](#34-共同配置与变量分类)、[§6.3](#63-拟合诊断与失效处理)）
- [ ] 各评估指标在 Ladder 规模上有可分辨信号，说明目标能力代表性与关键单项退化容忍条件（[§2](#2-评价协议与验收指标)）
- [ ] 若交付后训练模型，代表性候选需经过条件可比的 SFT/RL 验收（[§2](#2-评价协议与验收指标)）

### 9.3 搜参阶段

- [ ] 目标 2–4 时小模型穷举搜索 LR / BSZ 至 Fully-Tuned Frontier；目标 1 按配方缩放规则运行（[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)）
- [ ] 中模型验证幂律趋势；大模型 LR 可先乘以或除以 $\sqrt{2}$，BSZ 可先搜索 $[B_{opt}/2,2B_{opt}]$，验证边界后决定是否扩展（[§5.5](#55-搜索流程)、[§5.3](#53-batch-size-与-learning-rate-的迁移规律)）
- [ ] BSZ 用 $B_{opt} \propto D^{0.4}$ 作先验迁移，超过 $B_{crit}$ 后 token 效率显著下降，需权衡步数收益与算力开销，并在自身数据上验证（[§5.3](#53-batch-size-与-learning-rate-的迁移规律)）
- [ ] 复现配方时沿用其 WD；研究充分调优后的跨 TPP 关系时检查 WD 的影响，必要时按适用的 timescale 模型联合校准（[§5.2](#52-weight-decay-处理)）

### 9.4 拟合阶段

- [ ] 拟合前核对 FLOPs 与参数量口径、warmup 规则，以及目标 2–4 的 Fully-Tuned Frontier / 目标 1 的配方缩放规则（[§6](#6-loss-scaling-law-与下游任务拟合)）
- [ ] 在自身数据上比较候选函数形式；Skaling 在 $k=1$ 时退化为 Chinchilla（[§6.1](#61-函数形式)）
- [ ] 多 epoch 训练明确 unique tokens 与重复次数，并验证重复数据模型（[§4.8](#48-数据重复-scaling-law)）
- [ ] 使用中间 checkpoint 时检查初期点的影响，截断规则预定或由独立开发集选择（[§7.5](#75-常见陷阱与教训)）
- [ ] 检查残差结构、checkpoint 相关性与不确定性分离（[§6.3](#63-拟合诊断与失效处理)）
- [ ] 误差过大时记录待补充实验和暂不能作出的决策（[§6.3](#63-拟合诊断与失效处理)）

### 9.5 验证阶段

- [ ] 预先确定 Holdout 误差阈值，按多档外推倍数拉开（[§7.1](#71-holdout-验证)）
- [ ] 架构或优化器变更后先验证迁移性，再决定局部校准或完整重拟合，并做稳定性压力测试（[§7.1](#71-holdout-验证)、[§7.2](#72-稳定性压力测试)）
- [ ] 决策涉及下游能力时，验证相应任务指标的预测误差（[§6.4](#64-下游预测方法)）
- [ ] 单项改动通过后验证最终组合（[§7.4](#74-组合验证)）
- [ ] 同时报告理论 FLOPs 与实际训练时间；精度方案变化后重新验证（[§7.3](#73-实际效率与部署约束)）
- [ ] 对关键候选保留覆盖代表性训练量的轨迹（梯度、激活、路由），目标训练偏离时启动调查（[§7.2](#72-稳定性压力测试)）
- [ ] 保存配方版本、实验记录、拟合方法、预测区间与未解决问题（[§6.3](#63-拟合诊断与失效处理)）

## 10. 参考文献

按主题分组。标注章节号的条目在正文中有对应讨论。

### Scaling Law 基础与函数形式

1. Scaling Laws for Neural Language Models — Kaplan et al., OpenAI, 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)  
   在所测范围内观察到跨越 7 个数量级的经验幂律；给定参数量时宽深比对 loss 影响较小。见 [§3.1](#31-参数量口径)、[§3.2](#32-计算量与宽深比)
2. Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., DeepMind, 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)  
   IsoFLOP 方法与模型、数据近似等比例扩展的结果；加性形式 $L=E+A/N^\alpha+B/D^\beta$。见 [§4.4](#44-compute-optimal-配比)
3. Language models scale reliably with over-training and on downstream tasks — Gadre et al., 2024. [arXiv:2403.08540](https://arxiv.org/abs/2403.08540)  
   104 个模型验证过训练区域的 scaling law 仍可靠；不同 $D/N$ 下幂律指数接近。见 [§4.4](#44-compute-optimal-配比)、[§6.4](#64-下游预测方法)
4. Skaling: Chinchilla's Exponents Meet Kaplan's Coupling — Videau et al., FAIR at Meta, 2026. [arXiv:2608.07222](https://arxiv.org/abs/2608.07222)  
   在所分析数据上观察到负混合偏导，提出外指数 $k$ 和 L-shape 采样，并验证其外推表现。见 [§6.1](#61-函数形式)
5. Predictable Scale: Part II, Farseer — Li et al., StepFun & Fudan, NeurIPS 2025. [arXiv:2506.10972](https://arxiv.org/abs/2506.10972)  
   数据侧系数与指数显式依赖 $N$ 的九参数形式；参数量口径消融。见 [§3.1](#31-参数量口径)、[§6.1](#61-函数形式)
6. Scaling Law with Learning Rate Annealing — Tissue et al., 2024. [arXiv:2408.11029](https://arxiv.org/abs/2408.11029)  
   将 loss 表示为累计学习率面积与退火量的函数，可拟合整条曲线。见 [§6.2](#62-lr-退火-scaling-law)
7. A Hitchhiker's Guide to Scaling Law Estimation — Choshen et al., MIT/IBM, ICML 2025. [arXiv:2410.11840](https://arxiv.org/abs/2410.11840)  
   中间 checkpoint 可纳入拟合，需排除训练初期不稳定段；尺寸数量与外推跨度见正文。见 [§4.2](#42-尺寸数量与跨度)、[§7.1](#71-holdout-验证)、[§7.5](#75-常见陷阱与教训)
8. Resolving Discrepancies in Compute-Optimal Scaling of Language Models — Porian et al., NeurIPS 2024. [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)  
   分析 output head、warmup 和超参调优对 compute-optimal 指数差异的影响。见 [§3.1](#31-参数量口径)、[§6](#6-loss-scaling-law-与下游任务拟合)

### 超参 Scaling Law

{:start="9"}
9. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism — DeepSeek, 2024. [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)  
   按算力 $C$ 拟合超参幂律；所测语料的最优资源分配存在差异；观察到近优超参区间。见 [§4.6](#46-数据质量对-scaling-law-的影响)、[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)、[§5.4](#54-超参-scaling-law-公式选用)
10. Predictable Scale: Part I — Optimal Hyperparameter Scaling Law in Large Language Model Pretraining (Step Law) — Li et al., StepFun, 2025. [arXiv:2503.04715v3](https://arxiv.org/abs/2503.04715v3)
    3,700+ 模型验证的 $\eta_{opt}=c\,N^{-\alpha}D^{\beta}$、$B_{opt}=d\,D^{\gamma}$。见 [§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)、[§5.4](#54-超参-scaling-law-公式选用)
11. Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training — Bergsma et al., Cerebras, 2025. [arXiv:2505.13738](https://arxiv.org/abs/2505.13738)  
    在所测配方下得到 $B_{opt}\propto D^{0.4}$、$B_{crit}\propto D_{min}^{0.5}$ 的近似关系，对 $N$ 的依赖较弱；timescale $\tau=B/(\eta\lambda D)$ 可在已验证的 LR/BSZ 规则与稳定性约束下帮助联合搜索。见 [§5.5](#55-搜索流程)、[§5.3](#53-batch-size-与-learning-rate-的迁移规律)、[§5.2](#52-weight-decay-处理)
12. Scaling Optimal LR Across Token Horizons — Bjorck et al., Microsoft, 2024. [arXiv:2409.19913](https://arxiv.org/abs/2409.19913)  
    固定模型尺寸时 peak LR 随训练长度衰减。见 [§5.4](#54-超参-scaling-law-公式选用)
13. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer ($\mu$-Transfer) — Yang et al., Microsoft, 2022. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)  
    $\mu$P 下的宽度方向超参迁移；其他维度有有限实验，迁移到新配方需验证。见 [§5.4](#54-超参-scaling-law-公式选用)
14. Fantastic Pretraining Optimizers and Where to Find Them — Wen et al., Stanford, 2025. [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)  
    四尺寸、1×–8× Chinchilla 范围内的公平调参基准；token 效率收益随尺寸和训练量变化，退火期间可能出现排名翻转。见 [§5.6](#56-对比新优化器时的注意事项)、[§5.7](#57-lr-schedule)、[§8.3.1](#831-fantastic-optimizers-ladderdense超参搜索导向)
15. Weight Decay Improves Language Model Plasticity — Han et al., 2026. [arXiv:2602.11137](https://arxiv.org/abs/2602.11137)  
    所测设置中，预训练 loss 偏好的 WD 随 TPP 增大而降低；后训练可塑性还可能受益于更强 WD。见 [§5.2](#52-weight-decay-处理)
16. Small-Scale Experiments: Are We There Yet? — Lourie et al., NYU & Meta, 2026. [arXiv:2608.11859](https://arxiv.org/abs/2608.11859)  
    小模型对超参的敏感度与 Fully-Tuned Frontier；尺寸划分为拟合、验证、测试三段，拟合选项在验证段上确定。见 [§4.2](#42-尺寸数量与跨度)、[§5.1](#51-搜参目标fully-tuned-frontier-与宽平台)
17. How to Allocate Your Tokens? Scaling Laws with Training Steps and Batch Size — Schaipp, Inria, 2026. [arXiv:2607.01487](https://arxiv.org/abs/2607.01487)  
    按 loss 等价的算力损失定义近优 batch 区间，所测区间宽度约四倍。见 [§5.3](#53-batch-size-与-learning-rate-的迁移规律)

### 数据与训练量

{:start="18"}
18. Scaling Data-Constrained Language Models — Muennighoff et al., 2023. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)  
    重复数据的等效 token 量公式与边际收益衰减曲线。见 [§4.8](#48-数据重复-scaling-law)
19. To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis — Xue et al., 2023. [arXiv:2305.13230](https://arxiv.org/abs/2305.13230)  
    在受控实验中分析参数量、数据量和质量对重复过拟合的影响；dropout 有效性及 rate 需结合规模验证。见 [§4.8](#48-数据重复-scaling-law)
20. Prescriptive Scaling Laws for Data Constrained Training — Lovelace et al., 2026. [arXiv:2605.01640](https://arxiv.org/abs/2605.01640)  
    在所测范围内建模参数量、unique tokens 和重复次数共同影响的过拟合。见 [§4.8](#48-数据重复-scaling-law)
21. Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression — Yan et al., 2025. [arXiv:2511.13421](https://arxiv.org/abs/2511.13421)  
    在线性回归的特定假设下分析重复次数与数据集样本数的关系，并提供 LLM 实验。见 [§4.8](#48-数据重复-scaling-law)

### LR Schedule 与训练收尾

{:start="22"}
22. Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al., 2024. [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)  
    在特定损失几何与优化动力学假设下解释 WSD 的 stable 和 decay 阶段。见 [§5.7](#57-lr-schedule)
23. Scaling and Transferability of Annealing Strategies in Large Language Model Training — Wang et al., 2025. [arXiv:2512.13705](https://arxiv.org/abs/2512.13705)  
    退火策略的 scaling 与跨规模迁移。见 [§5.7](#57-lr-schedule)
24. Model Merging in Pre-training of Large Language Models — ByteDance Seed, 2025. [arXiv:2505.12082](https://arxiv.org/abs/2505.12082)  
    研究预训练中的 checkpoint 合并，并在特定对照中比较 stable 阶段 PMA 与退火终点表现。见 [§5.8](#58-收尾流程)
25. Model Soups: Averaging Weights of Multiple Fine-tuned Models — Wortsman et al., 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)  
    对共享预训练起点、独立微调所得模型进行权重平均。见 [§5.8](#58-收尾流程)
26. Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging (LAWA) — Kaddour, 2022. [arXiv:2209.14981](https://arxiv.org/abs/2209.14981)  
    单条轨迹的滑动窗口权重平均。见 [§5.8](#58-收尾流程)
27. Early Weight Averaging meets High Learning Rates for LLM Pre-training — Sanyal et al., COLM 2024. [arXiv:2306.03241](https://arxiv.org/abs/2306.03241)  
    高学习率下的早期权重平均。见 [§5.8](#58-收尾流程)

### 下游任务预测

{:start="28"}
28. Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective (COD) — Xu et al., ICLR 2026, v4（2026-03-09）. [arXiv:2502.17262v4](https://arxiv.org/pdf/2502.17262v4)
    按难度特征聚类、筛选可预测簇并映射到全集；包含 dense 与 MoE 目标预测及 continued training 实验。见 [§6.4](#64-下游预测方法)–[§6.6](#66-适用条件与局限)
29. GPT-4 Technical Report — OpenAI, 2023. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)  
    按小模型表现对 HumanEval 难度分桶，并在子集上拟合外推。见 [§6.4](#64-下游预测方法)
30. Establishing Task Scaling Laws via Compute-Efficient Model Ladders (OLMo Task Ladder) — Bhagia et al., Allen Institute, 2024. [arXiv:2412.04403](https://arxiv.org/abs/2412.04403)  
    两阶段下游预测：先由 $N,D$ 拟合 task-specific loss，再拟合 loss → accuracy；任务间噪声差异显著。见 [§6.4](#64-下游预测方法)

### 模型技术报告与 Ladder 实例

{:start="31"}
31. Delphi: An Open Scaling Suite from 3e18 to 1e23 FLOPs — Marin Team, 2026. [openathena.ai/blog/delphi](https://openathena.ai/blog/delphi/)  
    IsoFLOP 拟合 + recipe 公式驱动超参；holdout 多档外推。见 [§1.2](#12-为什么需要科学的-ladder)、[§7.1](#71-holdout-验证)、[§8.3.2](#832-delphi-ladder端到端-scaling-law-导向)
32. Nemotron-4 15B Technical Report — NVIDIA, 2024. [arXiv:2402.16819](https://arxiv.org/abs/2402.16819)  
    batch size 递增策略；训练末期 continued training。见 [§5.8](#58-收尾流程)
33. OLMo: Accelerating the Science of Language Models — Groeneveld et al., Allen Institute, 2024. [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)  
    完全开放的训练数据、代码与中间 checkpoint。见 [§4.3](#43-公开规模配置)
34. OLMo 2: The Next Generation of Fully Open Language Models — OLMo Team, Allen Institute, 2025. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)  
    两阶段训练；Micro-annealing 低成本验证数据源；model souping。见 [§4.6](#46-数据质量对-scaling-law-的影响)、[§5.8](#58-收尾流程)
35. OLMo 3 — Allen Institute, 2025. [arXiv:2512.13961](https://arxiv.org/abs/2512.13961)  
    分阶段训练配置与集成验证；Dirichlet 数据配比流程；评价指标的有效规模范围校准。见 [§8.1](#81-分阶段-ladder)、[§4.7](#47-数据配比-ladder)、[§2](#2-评价协议与验收指标)
36. Qwen3 Technical Report — Qwen Team, 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)  
    分阶段预测 LR 与 batch size；实例属性维度的数据配比。见 [§8.1](#81-分阶段-ladder)、[§4.7](#47-数据配比-ladder)
37. On the Design of Qwen3.8-Next Architecture — Qwen Team, 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)  
    架构与优化器变更后超参偏移；稳定性压力测试方法。见 [§7.1](#71-holdout-验证)、[§7.2](#72-稳定性压力测试)
38. Kimi K2: Open Agentic Intelligence — Moonshot AI, 2025. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)  
    在固定激活规模下研究路由专家稀疏率与计算效率。见 [§8.2](#82-moe-ladder-的参数口径与稀疏度轴)、[§5.2](#52-weight-decay-处理)
39. Nemotron 3 Ultra Technical Report — NVIDIA, 2026. [PDF](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)  
    训练中高精度分支对照，说明精度方案需要验证。见 [§7.3](#73-实际效率与部署约束)
40. Marin：MoE 与训练效率后续 — Marin Team, 2026. [openathena.ai/blog/pretraining-speedup](https://openathena.ai/blog/pretraining-speedup/)  
    区分 theoretical 与 realized efficiency；多尺度方案比较、预注册预测与组合实验设计。见 [§7.3](#73-实际效率与部署约束)、[§7.4](#74-组合验证)、[§8.3.2](#832-delphi-ladder端到端-scaling-law-导向)

### 延伸阅读

以下文献与 Ladder 设计相关，正文未展开：

{:start="41"}
41. Chinchilla Scaling: A Replication Attempt — Besiroglu et al., Epoch AI, 2024. [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)
42. (Mis)Fitting: A Survey of Scaling Laws — Li et al., 2025. [arXiv:2502.18969](https://arxiv.org/abs/2502.18969)
43. Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws — Sardana et al., 2024. [arXiv:2401.00448](https://arxiv.org/abs/2401.00448)
44. Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training — Merrill et al., 2025. [arXiv:2505.23971](https://arxiv.org/abs/2505.23971)
45. An Empirical Model of Large-Batch Training — McCandlish et al., 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
46. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al., Google, 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
47. Cost-Optimal Grouped-Query Attention for Long-Context Modeling — Chen et al., 2025. [arXiv:2503.09579](https://arxiv.org/abs/2503.09579)
48. Nemotron-4 340B Technical Report — NVIDIA, 2024. [arXiv:2406.11704](https://arxiv.org/abs/2406.11704)
