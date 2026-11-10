---
layout: post
title: "如何搭建一个科学的 Scaling Ladder"
date: 2026-09-20 10:00:00
description: "整理 LLM 预训练 Scaling Ladder 的设计与搭建方法：涵盖参数与算力统计、Dense 与 MoE 实验设计、超参搜索与调度、数据配比与多轮重复、Loss Scaling Law 拟合、下游任务预测及外推验证。"
tags: [scaling-laws, pretraining, hyperparameter, optimization, llm, empirical-methodology]
categories: [research]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: zh-CN
---

本文基于**<u>公开文献与个人实践经验</u>**整理，**<u>不包含任何涉密内容</u>**。在上一篇博客 [《预训练与 scaling：一种通用方法论》]({% post_url 2026-09-07-pretrain-scaling-methodology-scientific-perspective %}) 中，我讨论了为什么预训练与 scaling 是一套通用方法论；本文则具体梳理在 LLM 预训练中，如何从零搭建一套可靠的 Scaling Ladder。

---

## 1. 基本概念与范围

### 1.1 什么是 Scaling Ladder

Scaling Ladder 是一组跨不同模型尺寸 $N$ 与训练数据量 $D$ 的小规模预训练实验矩阵。在工业界与各家 Frontier Lab 的预训练研发中，它不只是一个确定超参的工具，而是贯穿整个研发流程的基础实验框架，主要有三方面作用：

1. **探索超参数与模型配置的缩放规律**：在不同 $(N, D)$ 规模上研究 LR、BSZ 等训练超参，以及宽深比、MoE 稀疏比、数据配比与重复轮数等设置随规模变化的迁移规律。
2. **充当检验新技术的金标准（Optimal Baseline）**：它相当于一把统一的尺子。无论是新优化器、架构改进还是新一批数据，想要并入下一代基座模型的主线，都需要在已有的 Ladder 上验证出稳定的收益。学界不少工作容易在单一小规模或欠调基线上高估（overclaim）某项新技术的效果；放到充分调优的 Ladder 上跨规模对比，就能筛掉无效方案，看清某项改动到底有没有真实收益，以及收益随规模放大是保持、扩大还是逐渐消失。
3. **预测旗舰训练（Hero Run）的效果**：在超参数规律、模型架构与数据配置都敲定之后，反过来利用 Ladder 拟合出的曲线，在正式投入海量算力前预测旗舰训练的 loss 曲线、最终 loss 乃至下游 benchmark 表现。

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 1. Delphi 首次实验（Cautious AdamC）。右图的大规模训练偏离预测并发散。<a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">来源：Marin, 2026</a>。'
  alt='Delphi 首次 scaling 实验：10 的 22 次方 FLOPs 运行的 loss 比预测高 2.5%，10 的 23 次方 FLOPs 运行发散。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

要让这把尺子真正可靠，并不是跑几个小模型、拟一条曲线就够了。如果 baseline 的小模型超参没搜透、参数量与 FLOPs 统计不准、数据重复轮数没对齐，或者函数形式选错，不仅会误判新技术的收益，外推预测也会失真。例如图 1 中 [Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) 的第一版实验，在小规模区间（$3\times 10^{18}$–$3\times 10^{20}$ FLOPs）拟合良好，但到了 $10^{22}$ FLOPs 实测 loss 比预测偏高 2.5%，到 $10^{23}$ FLOPs 训练直接发散。因此，搭一套科学的 Scaling Ladder，既要把小规模实验与曲线拟合的细节做扎实，也需要留出多档 holdout 实验和稳定性测试来验证外推是否可靠（[§10](#10-外推验证)）。

### 1.2 典型场景

本文包含三类预训练场景。第 2–10 章介绍通用流程，MoE 与数据受限的相关问题在各章对应小节单独说明。

| 场景 | 主要内容 | 对应章节 |
|---|---|---|
| Dense | 如何设置宽深比与分配算力；既用于训练 dense 模型，也作为 MoE 的对照基线 | [§4.3](#43-宽深配置) |
| MoE | 总参数量与激活参数量分开统计；如何设置稀疏比、专家大小、路由与负载均衡，以及评估专家并行的实际训练速度 | [§3.1](#31-参数量计算)、[§4.4](#44-moe-结构缩放规则)、[§5.4](#54-moe-实验设计)、[§6.3](#63-lr-与-bsz-的-scaling-law)、[§10.2](#102-稳定性压力测试)、[§10.4](#104-实际训练速度与推理成本) |
| 数据受限 | 可用 unique tokens 少于目标训练量时，如何确定数据重复轮数与各领域配比 | [§5.3](#53-训练量档位)、[§7.3](#73-数据受限训练与重复) |

实际场景中常采用 MoE 架构，同时可能面临数据不足（例如可用数据不到 $1\times$ Chinchilla 配比），或因推理成本考虑而增加训练量。因此目标 TPP（tokens per parameter）可能低于或高于 Chinchilla 最优配比，实验的训练量档位需要根据目标 TPP 来设置（[§5.3](#53-训练量档位)）。

本文聚焦 LLM 预训练，不讨论 SFT/RL、原生多模态以及蒸馏和合成数据的 scaling law。

### 1.3 Ladder 的产出

| 产出 | 预测目标 | 章节 |
|---|---|---|
| 超参 scaling law（LR、BSZ、WD） | 最优训练超参 | [§6](#6-超参数搜索与训练配置) |
| Loss scaling law | 最终 training / eval loss | [§8.1](#81-函数形式) |
| Loss 曲线 scaling law | 完整训练曲线与退火收益 | [§8.2](#82-loss-曲线与退火-scaling-law) |
| LR 调度与收尾 | 终点 LR 设置、调度选择与权重平均 | [§6.5](#65-lr-schedule-与训练收尾) |
| 数据配比与重复 scaling law | 最优数据配比；多 epoch 训练的等效 token 数 | [§7](#7-数据-ladder) |
| MoE 稀疏度 scaling law | 固定激活参数下的专家数量与大小 | [§5.4](#54-moe-实验设计) |
| 下游任务 scaling law | Benchmark 指标 | [§9](#9-下游任务预测) |

### 1.4 术语与符号

| 术语 | 定义 |
|---|---|
| $N_{\text{body}}$ | Transformer 主干参数量，不含 input embedding 与 output head（[§3.1](#31-参数量计算)） |
| $N_{total}$、$N_{active}$ | MoE 主干总参数量与单 token 激活参数量 |
| $D$ | 累计训练 tokens（按实际参与 loss 计算的 token 计，见 [§3.3](#33-loss-与-token-统计)） |
| $U$ | 去重后的可用 unique tokens |
| $C$ | 训练总 FLOPs（[§3.2](#32-计算量)） |
| TPP | Tokens per parameter（$D/N$）；对比不同文献时需注意 $N$ 是否含 embedding |
| LR（$\eta$）、BSZ（$B$）、WD（$\lambda$） | 学习率、batch size（按 tokens 计）、weight decay |
| Holdout | 不参与拟合、仅用于验证外推精度的实验点 |
| Fully-Tuned Frontier | 各实验点充分调优超参后达到的最低 loss（[§6.1](#61-搜参目标与搜索流程)） |
| 等效算力倍数 | 达到相同 loss 所需的算力比值（[§2.3](#23-验收标准)） |

## 2. 决策目标与验收标准

### 2.1 决策目标

搭建 Ladder 前先明确实验目的，常见有四类：

1. 预测固定 recipe 的表现：给定一套固定的超参缩放规则，预测它在大模型上的效果，不需要对每个实验点单独搜参。
2. 最优算力分配：在给定算力预算 $C$ 下，求最优的 $(N, D)$ 组合。
3. 方案对比：比较不同架构、优化器或数据配比在大模型上的优劣。
4. 超参预测：拟合最优超参 $(\eta, B, \lambda)$ 随 $N$ 和 $D$ 的变化规律。

[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) 区分了训练方案的绝对效果（competitive，loss 是否更低）与可预测性（predictable，小模型规律能否准确外推到大模型），这两点需要分开验证。做第 2–4 类实验时，每个拟合点都需要调参到 Fully-Tuned Frontier（[§6.1](#61-搜参目标与搜索流程)），否则小模型超参没调好会带偏拟合出的幂律斜率；做第 1 类实验则直接按给定的超参公式运行。

### 2.2 评估指标与设置

设计评估指标时需注意以下几点：

- 区分代理指标与最终指标：小模型上 training loss 的下降不一定能转化为大模型下游 benchmark 的提升（[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) §1）。
- 小模型上的可区分度：小模型在数学、代码等较难任务上 accuracy 常接近随机水平，而简单任务在大模型上容易饱和。对小模型 accuracy 拉不开差距的任务，可以改用正确答案的 bits-per-byte（BPB）等连续指标（[OLMo 3, 2025](https://arxiv.org/abs/2512.13961)）。
- 跨规模排序一致性：用小模型代理指标筛选方案时，需要验证小模型上的排名与大模型最终表现的排序一致（[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3, Appendix A.4）。
- 控制高波动任务：评测集需覆盖核心目标能力；对波动较大的任务可以多次采样取平均或单独统计（[Phi-4, 2024](https://arxiv.org/abs/2412.08905) §5；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3.3–3.3.4）。
- 固定评测设置：统一 prompt 模板、解码参数与打分脚本版本，并排查训练集对测试集的数据污染。

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

如果最终模型需要经过 SFT 或 RL，只看预训练指标可能漏掉后训练阶段才会暴露的问题。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 报告了两个例子：去掉位置编码（NoPE）和引入稀疏读取残差分支在预训练时对 loss 影响很小，但后训练后分别出现无法停止的重复生成和效果退化。[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.5.1 的做法是在候选方案完成退火后，接一个轻量级的 instruction tuning，同时检查下游任务得分、生成能否正常停止以及输出长度。

### 2.3 验收标准

外推误差的合格标准需要根据方案间的最小有效差距和随机种子波动提前定好，不能等跑完 holdout 再看结果定标准。[Choshen et al., 2025](https://arxiv.org/abs/2410.11840) 统计发现，文献中足以支持架构或算法改动的最小相对提升约为 4%，而换随机种子重跑本身的波动就可达到约 3.5%；Delphi 在特定设置下做到的 0.2%–0.5% 预测误差不能直接当成通用标准。

不同模型规模下不宜使用固定的相对 loss 误差。在幂律 $L(C)=E+A\,C^{-\gamma}$ 下，loss 差 $\Delta L$ 对应的等效算力倍数满足：

$$\ln\frac{C_2}{C_1}\approx\frac{\Delta L}{\gamma\,(L-E)}$$

模型越大、$L$ 越接近不可约损失 $E$，同样的 $\Delta L$ 对应的算力倍数就越大。因此建议用等效算力倍数（比如节省 15% 算力）配合绝对 $\Delta L$ 来衡量收益和误差。

## 3. 参数量、算力与 Loss 计算

### 3.1 参数量计算

拟合 $L(N,D)$ 和计算算力 $C$ 时，参数量 $N$ 的算法不同，需要分开记录。

拟合 $L(N,D)$ 时，$N$ 取排除 input embedding 和 output head 的主干参数量 $N_{\text{body}}$。因为 embedding 和 head 的参数量是 $O(Vd)$，主干是 $O(d^2 n)$，在小模型上词表层占比很大，算进去会带偏跨尺寸的幂律指数。[Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972v3) Appendix G 的消融实验表明，包含 embedding 的参数量外推到 25.1B 时误差明显大于 $N_{\text{body}}$；[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 同样排除了 embedding。

但计算训练算力 $C$ 时，必须把 output head 加回来。Output head 对每个 token 都要做一次 $d\times V$ 的矩阵乘；即使与 input embedding 共享权重（tied weights），也只是省了显存，矩阵乘计算省不掉。Input embedding 本身是按 token ID 查表，不产生矩阵乘，两边都不计入。

对比不同文献的系数或 TPP 时要注意各自的参数量算法：[Porian et al., 2024](https://arxiv.org/abs/2406.19146) Table 2 的 $N$ 计入了 output head、不含 input embedding；Chinchilla 的 20 TPP 则是按包含 embedding 的总参数量算的（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) Appendix F）。

对于 MoE 模型，每个实验点需要同时记录：

- 总参数量 $N_{total}$ 与单 token 激活参数量 $N_{active}$（均不含 embedding 与 head）；
- 路由专家总数 $E_{total}$、每 token 激活专家数 $E_{active}$ 以及稀疏比 $S=E_{total}/E_{active}$；
- 共享专家的数量与大小，以及单个路由专家的大小（专家粒度）；
- 路由器与负载均衡设置（[§4.4](#44-moe-结构缩放规则)）。

### 3.2 计算量

常用的 $C\approx6ND$ 忽略了注意力和 output head 的计算。以标准 MHA 和 $4d$ FFN 为例，按完整 $L\times L$ 注意力矩阵计算，前反向总 FLOPs 为：

$$C \approx \left(6 + \frac{L}{d}\right) N_{\text{body}}D + 6VdD$$

其中 $L$ 为序列长度，$d$ 为隐藏维，$V$ 为词表大小，$N_{\text{body}}\approx12d^2n$（$n$ 为层数）。$L/d$ 项来自注意力，末项来自 output head。如果底层 attention kernel 跳过了 causal mask 的上三角部分，注意力项减半；使用 SwiGLU、GQA 或 MoE 时系数也会变，实际应按模型结构和 kernel 统计 $C$。

MoE 模型的矩阵乘 FLOPs 按 $N_{active}$ 加路由器开销计算；专家并行（EP）的 all-to-all 通信不产生 FLOPs，但会影响实际训练速度（[§10.4](#104-实际训练速度与推理成本)）。

### 3.3 Loss 与 token 统计

Ladder 各实验需统一以下统计方式：

| 项目 | 要求 |
|---|---|
| Loss 项 | 只统计主任务 cross-entropy；MoE aux loss、z-loss、MTP 等附加项单独记录 |
| 验证集 | 所有方案共用同一套 eval 数据与领域权重，核心领域单独看指标 |
| Loss 平均方式 | 统一按有效预测 token 还是按文档求平均 |
| 序列拼接 | 统一 BOS/EOS、文档边界、packing、attention mask 与截断方式 |
| Token 计数 | 区分总处理 tokens、实际参与 loss 计算的 tokens、unique tokens 与重复轮数 |
| 随机种子 | 分别记录初始化、数据采样与 shuffle 的随机种子 |
| 跨词表对比 | 在相同原始文本上换算为 BPB（bits-per-byte），并对齐相同文本长度下的训练 FLOPs |

如果实验调整了训练数据配比，各自训练集上的 training loss 会受数据本身难度影响（比如代码占比高时 loss 天然偏低），不能直接拿来比好坏，必须看固定 eval 集上的 loss。

此外，eval 间隔应按总训练步数 $T$ 的固定比例（如每 2% 训练进度）设置，不要固定步数（如每 1,000 步）。否则长训练任务的 eval 点远多于短任务，拟合整条 loss 曲线时权重会偏大（[§8.3](#83-数值拟合细节)）。

## 4. 基础设置与缩放规则

### 4.1 变量分类

Ladder 中的变量分为三类：

| 类别 | 含义 | 示例 |
|---|---|---|
| 固定项 | 所有实验点保持一致 | 模型架构、数据版本、优化器类型、seq_len |
| 缩放规则 | 随 $N$ 或 $D$ 按固定公式变化 | LR（幂律或 $\mu$P）、BSZ（如 $D^{0.4}$）、warmup 比例 |
| 自变量 | 待搜索或待拟合的量 | $N$、$D$、待搜索的最优 LR 与 BSZ |

Warmup 步数应按总训练量的固定比例设置（如 Delphi 固定取总训练量的 10%，见[附录 A.3](#a3-delphi-ladder)）。如果所有规模都固定相同的 warmup 步数，小预算实验中 warmup 占比过高，会带偏拟合出的 compute-optimal 指数（[Porian et al., 2024](https://arxiv.org/abs/2406.19146)）。

当模型架构、优化器、数据版本、训练精度或 tokenizer 改变时，原有的缩放系数可能失效，需要先在小模型上验证能否继续沿用，再决定是微调系数还是重新拟合（[§7.1](#71-数据质量与数据源评估)）。

### 4.2 架构与训练配置一致性

除当前正在研究的变量外，Ladder 各尺寸模型应保持架构和训练配置一致：

| 配置项 | 要求 |
|---|---|
| 模型结构 | 统一主干结构（如 decoder-only Transformer）、归一化（如 Pre-RMSNorm）、位置编码（如 RoPE）与激活函数（如 SwiGLU） |
| Embedding | 全系保持 tied 或全系 untied |
| 注意力 | 统一注意力类型（如 GQA），小模型的分组比例与 `head_dim` 贴近目标大模型 |
| 宽深比 | 按固定比例同步放大宽度与层数（[§4.3](#43-宽深配置)） |
| MoE 路由 | 统一路由方式、负载均衡策略与共享专家比例（[§4.4](#44-moe-结构缩放规则)） |
| 训练与评测 | 统一 seq_len（如 4096）、数据版本、tokenizer、初始化方式、优化器（如 AdamW）及验证集设置（[§3.3](#33-loss-与-token-统计)） |

### 4.3 宽深配置

相同主干参数量 $N_{\text{body}}$ 下，改变宽度 $d$ 与层数 $n$ 的比值会改变注意力项 $L/d$ 的计算量（[§3.2](#32-计算量)）。[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 报告宽深比在较大范围内对预训练 loss 影响很小，但后续实验表明宽深比会影响下游 benchmark 表现（[Gemstones, 2025](https://arxiv.org/abs/2502.06857) §4.4）和推理能力（[GLM-4.5, 2025](https://arxiv.org/abs/2508.06471)）。因此 Ladder 内应保持统一的宽深缩放规则；如果要对比不同深度或 head 维度，建议单独增加对照实验。

### 4.4 MoE 结构缩放规则

除 [§4.2](#42-架构与训练配置一致性) 的通用设置外，MoE Ladder 还会涉及以下结构参数：

- 专家粒度（Granularity）：[Krajewski et al., 2024](https://arxiv.org/abs/2402.07871) 把单个专家大小作为独立变量研究，发现直接沿用 dense FFN 大小的常见做法通常不是算力最优的；把专家切得更细，MoE 相对 dense 的优势会随规模继续扩大。
- 共享专家：固定共享专家的数量及其与路由专家的大小比例。
- 负载均衡：统一负载均衡机制。[Kimi K3, 2026](https://arxiv.org/abs/2607.24653)（Appendix C–D）采用 Quantile Balancing（QB），直接按目标负载对应的 router logits 分位数设置各专家的路由偏置（通过直方图跨 microbatch 聚合），省去了传统无 aux loss 方法中需要手动调节的偏置更新步长。负载均衡方式会同时影响模型 loss 与专家并行吞吐，Ladder 内建议保持一致。
- Capacity factor 与 token dropping：训练与推理的 capacity factor 保持一致，并监控 token 丢弃比例；DeepSeek-V3 在训练和推理中均不丢弃 token（§2.1.2）。
- 稀疏度（Sparsity）：[Abnar et al., 2025](https://arxiv.org/abs/2501.12370) 在不计显存与通信开销时发现，固定训练 FLOPs 下提高稀疏度（即增大 $N_{total}$）可以持续降低预训练 loss；固定 $N_{total}$ 时，loss 随稀疏度呈 U 型变化，且最优稀疏度随总算力增加而提高。多数下游任务表现主要取决于预训练 loss，对稀疏度不敏感，但在 CoQA、SQuAD 等阅读理解任务上，激活比例更高的模型效果更好。[Ludziejewski et al., 2025](https://arxiv.org/abs/2502.05172) 进一步在显存约束下对专家总数、$N_{active}$ 与 $D$ 做了联合拟合（最大到 2.7B 激活参数 / 5B 总参数）。

### 4.5 词表与训练精度

- 词表大小：[Tao et al., 2024](https://arxiv.org/abs/2407.13623) 在 33M–3B 模型上的实验表明，最优词表大小随算力预算增加而变大，目前多数开源模型的词表偏小。由于改词表会同时改变 $O(Vd)$ 的计算量和单个 token 对应的文本长度，跨词表对比建议统一换算为字节级的 BPB（[§3.3](#33-loss-与-token-统计)）。
- 训练精度：[Kumar et al., 2024](https://arxiv.org/abs/2411.04330)（实验在 1.7B 参数、26B tokens 以内）将训练和量化精度纳入了 scaling law：低精度训练相当于降低了模型的有效参数量；同时训练 TPP 越高，后训练量化带来的 loss 退化越明显。因此 Ladder 实验建议直接采用目标大模型的训练精度方案（[§10.3](#103-训练框架与硬件对齐)）。

## 5. 实验设计与预算

### 5.1 实验矩阵结构

常见的 $(N, D)$ 实验网格如下（模型尺寸与倍数为示例）：

```text
训练量 D →    0.5×    1×     2×     4×
模  130M       ●      ●      ●      ●
型  520M       ●      ●      ●      ●
尺  2.3B       ●      ●      ●      ●
寸  8B         ○      ○      ○      ○  ← holdout（不参与拟合）
N   ↓
```

- 拟合点（`●`）：在小到中等尺寸上跑多档训练量，用于拟合 Scaling Law。
- Holdout 点（`○`）：在最大尺寸或更高算力档位留出实验点，不参与拟合，只用来验证外推精度。

实验点布局可以根据研究目标和预算来定：除了上图的完整网格，也可以用固定算力的 IsoFLOP 扫描（[§5.3](#53-训练量档位)）或更省算力的 L-shape 布局（[§8.1](#81-函数形式)）。需要注意，如果所有实验点只沿着同一个固定 TPP（例如全部取 $D=20N$）采样，$N$ 和 $D$ 完全共线，拟合二元函数 $L(N,D)$ 时就无法分开估计两者各自的指数。

### 5.2 尺寸数量、跨度与外推倍数

| 维度 | 建议 | 来源 |
|---|---|---|
| 模型尺寸档数 | 至少 3–4 个拟合尺寸并另设 holdout；增加尺寸档数有助于稳定斜率估计 | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) |
| 尺寸跨度 | 覆盖目标外推方向；部分模型族在 34× 跨度内仍符合幂律 | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840) |
| 相邻尺寸比 | 等比递增（常见 $2\times$–$4\times$） | 工程惯例 |

外推倍数指目标训练算力与最大拟合点算力之比。例如 [Delphi](https://openathena.ai/blog/delphi/) 在 $3\times 10^{18}$–$3\times 10^{20}$ FLOPs 区间拟合，在 $3\times$–$333\times$ 的 holdout 上验证外推；[Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.2.1 在 $6\times 10^{18}$–$10^{22}$ FLOPs（40M–16B）上拟合，外推到 $3.8\times 10^{25}$ FLOPs（约 $3,800\times$）。外推倍数越大，公式本身的偏差和训练后期不稳定的风险也越大；如果目标规模远大于最大 holdout，建议在正式启动前加一次中等规模试运行（[§10.1](#101-holdout-与组合验证)）。

为了节省预算，单次训练过程中的中间 checkpoint 也可以纳入拟合（比如在 WSD 调度下）。由于训练初期 loss 下降还不稳定，[Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) 发现去掉前 10% 训练进度或前 10B tokens 的早期点能明显降低外推误差；截断位置建议在拟合前定好，或按 [Lourie et al., 2026](https://arxiv.org/abs/2608.11859) 的做法把模型尺寸分成拟合、验证、测试三段，在验证段上确定。此外，有限预算下是再加一个大尺寸实验点，还是在小模型上多跑几个随机种子，取决于随机种子波动与外推跨度的相对大小（[Choshen et al., 2025](https://arxiv.org/abs/2410.11840)）。

### 5.3 训练量档位

训练量档位根据目标模型的预期 TPP 来定。Chinchilla（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)）的 IsoFLOP 方法固定几个算力预算 $C$，在每个 $C$ 下训练一组不同尺寸 $N$ 的模型，拟合出 loss 最低点，得到该算力下的最优 $(N_{opt}, D_{opt})$。

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

基于 400 多个模型（70M–16B，5B–500B tokens），Chinchilla 得到：

$$N_{opt}\propto C^a,\qquad D_{opt}\propto C^b,\qquad a\approx b\approx0.5$$

即在算力最优（compute-optimal）条件下，模型参数量与训练数据量近似等比例放大（约 $D/N\approx 20$，按含 embedding 的总参数量计）。Llama 3 用同样的 IsoFLOP 方法算出 $3.8\times 10^{25}$ FLOPs 下的最优尺寸约为 402B，最终选定 405B（[Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1）。

实际设计训练量档位时，可以根据目标场景调整范围：

- 接近算力最优：常取 0.5×–4× Chinchilla 配比，包住 1× 最优配比两侧；[Fantastic Optimizers](https://arxiv.org/abs/2509.02046v2) 则用 1×、2×、4×、8× 四档。
- 大幅过训练：[Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2) 在 104 个模型上验证了最高 32× Chinchilla 配比下的幂律外推。[Sardana et al., 2024](https://arxiv.org/abs/2401.00448) 在 47 个模型上测试到 10,000 TPP，发现 loss 虽然还在降，但如果只用低 TPP 数据拟合 Chinchilla 公式，会高估高 TPP 下新增 token 的收益。因此如果目标 TPP 很高，Ladder 需要包含高 TPP 的实验点。
- 考虑推理成本：把部署后的推理开销算进总成本后，最优解会偏向更小的模型和远高于 Chinchilla 的训练量（[Sardana et al., 2024](https://arxiv.org/abs/2401.00448)）。比如 Llama 3 405B 接近训练算力最优，而 8B 和 70B 则训练了超过 15T tokens，用来降低推理延迟和显存占用（[Llama 3](https://arxiv.org/abs/2407.21783) §1；见 [§10.4](#104-实际训练速度与推理成本)）。
- 数据不够用：当目标训练量 $D$ 超过去重后的可用数据量 $U$ 时，需要同时记录数据重复轮数，并在拟合公式里加上重复衰减项（[§7.3](#73-数据受限训练与重复)）。

### 5.4 MoE 实验设计

MoE 模型的变量比 dense 多，一般拆成三组实验分别设计并独立拟合：

1. 规模 Ladder：固定稀疏比 $S$ 与专家大小，同步放大 $N_{active}$ 与 $D$，拟合时直接用 $N_{active}$ 代替公式中的 $N$；
2. 稀疏度 Ladder：固定 $N_{active}$、$D$ 与激活专家数 $E_{active}$，只改变总专家数 $E_{total}$；
3. 粒度 Ladder：固定 $N_{active}$ 与 $N_{total}$，同步调整单个专家大小与激活专家数（[Krajewski et al., 2024](https://arxiv.org/abs/2402.07871)；[Ludziejewski et al., 2025](https://arxiv.org/abs/2502.05172) 将专家数、$N_{active}$ 与 $D$ 做了联合建模）。

例如，[Kimi K2, 2025](https://arxiv.org/abs/2507.20534) 在固定 $N_{active}$ 和训练 FLOPs 的条件下改变 $E_{total}$ 拟合稀疏度 scaling law，发现稀疏比从 8 提高到 48 时，达到相同 loss 所需的 FLOPs 持续下降，但跨节点通信和推理显存开销会相应增加。规模 Ladder 中还可以加上相同 $N_{active}$ 或相同 FLOPs 的 dense 对照实验，用来观察 MoE 相对 dense 的等效算力优势如何随规模变化。

## 6. 超参数搜索与训练配置

### 6.1 搜参目标与搜索流程

[Lourie et al., 2026](https://arxiv.org/abs/2608.11859) 与 [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) 发现，小模型对超参非常敏感，偏离最优值会明显抬高 loss 并改变拟合出的幂律斜率，Scaling Law 只有在各实验点充分调优的 Fully-Tuned Frontier 上才成立。而大模型最优点附近则存在很宽的平坦区间（Wide Plateau）：[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320v1) 在 156B-A7B 模型上把学习率乘除 $\sqrt{2}$、batch size 上调 25%，最终 training loss 的变化不超过 $7\times10^{-4}$。

各超参对 loss 的影响程度按 $\text{LR} > \text{BSZ} > \text{Schedule} > \text{WD}$ 递减，$\beta_2$、$\epsilon$ 等数值稳定性参数在基线确认稳定后即可固定。实际搜参可以把算力集中在小模型上，按坐标下降分步推进（尺寸仅为示例）：

1. 小模型（如 130M）网格搜索：铺开二维网格扫描 $(\eta, B)$，确保最优点落在网格内部而非停在边界；
2. 中等模型（如 500M）验证斜率：用缩窄的网格检验幂律趋势；
3. 大模型局部确认：以外推值为中心，LR 在 $[\eta_{pred}/\sqrt{2}, \sqrt{2}\eta_{pred}]$ 内取点，BSZ 在 $[B_{opt}/2, 2B_{opt}]$ 内取点（[§6.3](#63-lr-与-bsz-的-scaling-law)），只有边界点更优时才向外扩。

需要注意，在多次带随机波动的试验里直接挑 loss 最低的配置，会让选中配置的收益估计偏高（[Cawley & Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)）。选出最优配置后，建议换一个新随机种子重跑一次，用复验结果去拟合 Scaling Law；同时不要根据训练前期的 loss 排名提前停掉候选实验，因为退火阶段经常出现排名翻转（[§6.6](#66-方案对比注意事项)）。

### 6.2 参数化与跨优化器迁移

- $\mu$P 与宽度/深度迁移：$\mu$-Transfer（[Yang et al., 2022](https://arxiv.org/abs/2203.03466v2)）通过调整各层初始化和学习率缩放因子，使最优 LR 在宽度方向上可以零样本迁移（带非零 weight decay 的修正见原论文 Appendix G.1.2 与 [Power Lines](https://arxiv.org/abs/2505.13738v2)）。深度方向上，[Bordelon et al., 2023](https://arxiv.org/abs/2309.16620) 将残差分支按 $1/\sqrt{\text{depth}}$ 缩放并结合 $\mu$P，在 ResNet 和 ViT 上实现了跨宽深度迁移；但 [Tensor Programs VI (Yang et al., 2023)](https://arxiv.org/abs/2310.02244) 证明 Depth-$\mu$P 只在每个残差块包含单层变换时成立，对残差块内含多层非线性变换的 Transformer 存在理论局限。此外，$\mu$P 本身不解决跨 batch size 和训练量 $D$ 的迁移。因此 $\mu$P 虽然理论上很优美，但业界实践通常还是倾向于在 Ladder 上直接拟合超参 scaling law，而非默认直接按 $\mu$P 迁移。
- 跨优化器迁移：[Liu et al., 2025](https://arxiv.org/abs/2502.16982) 将 Muon 扩展到大模型时，加入 weight decay 并把正交化更新矩阵的 RMS 幅度对齐到 AdamW 常见的 0.2–0.4 区间（取 0.2），从而直接复用 AdamW 调好的 LR 与 WD。但如果架构与优化器同时改动，[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 观察到切换到 Muon 后最优 LR 和 BSZ 依然会发生偏移，需要在小模型上重新搜索。

### 6.3 LR 与 BSZ 的 Scaling Law

如果 Ladder 只跑单一的固定 $D/N$ 配比，可以直接把最优学习率和 batch size 拟合成总算力 $C$ 的幂律：

$$\eta_{opt} = a \cdot C^{-\alpha}, \quad B_{opt} = b \cdot C^{\beta}$$

[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) 提供了该形式下的拟合系数。当 Ladder 覆盖多个 $D/N$ 档位时，$C$ 把 $N$ 和 $D$ 混在了一起，需要将两者拆开拟合：

| 方案 | $\eta_{opt}$ 公式 | $B_{opt}$ 公式 | 适用条件与结论 |
|---|---|---|---|
| Step Law ([Li et al., 2025](https://arxiv.org/abs/2503.04715v3)) | $c\cdot N^{-\alpha}D^{\beta}$ | $0.58\,D^{0.571}$ | 经 3,700+ 模型验证，支持跨 $D/N$ 外推；$B_{opt}$ 主要由训练量 $D$ 决定，几乎不随 $N$ 变化（Appendix A.5） |
| Power Lines ([Bergsma et al., 2025](https://arxiv.org/abs/2505.13738v2)) | 由 timescale $\tau$ 决定（[§6.4](#64-weight-decay)） | $\propto D^{0.4}$（与 $N$ 基本无关） | 通过 EMA 时间尺度 $\tau$ 统一 $(\eta, B, \lambda)$ 的关系 |
| Token Horizons ([Bjorck et al., 2024](https://arxiv.org/abs/2409.19913)) | $c \cdot N^{-\alpha} D^{-\beta}$ | 固定 BSZ | 固定 $N$ 与 BSZ 时，训练量 $D$ 越长，最优 peak LR 越低 |

Token Horizons 与 Step Law 中，$\eta_{opt}$ 关于 $D$ 的指数符号相反（前者为 $D^{-\beta}$，后者为 $D^{+\beta}$）。两者的实验设置不同：Token Horizons 固定 batch size，而 Step Law 对每个 $D$ 联合搜索并按 $D^{0.571}$ 放大 $B_{opt}$。该设置差异能否完全解释符号相反尚未验证，实际使用时可根据自身调度和 BSZ 策略在 holdout 上验证。

对于 MoE 模型，由于稀疏结构很难搭建极小尺寸的 Ladder，直接在 MoE 上做全网格搜参算力开销较大。实践中，改变 attention 结构或从 dense 切换到 MoE 时，最优 LR 的幂律指数通常变化很小，主要只差一个常数倍。因此可以先搭建低成本的 dense Ladder 拟合出 $\eta_{\text{dense}}^{\ast}(N, D)$，再用几个小尺寸 MoE 实验点拟合一个迁移系数 $c_{\text{MoE}}$，直接按 $\eta_{\text{MoE}}^{\ast} = c_{\text{MoE}}\,\eta_{\text{dense}}^{\ast}$ 得到 MoE 的 LR scaling law。

{% include figure.liquid
  path='assets/img/pretrain-scaling/step-law-hyperparameter-validation.png'
  id='fig-step-law-hyperparameter-validation'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 4. Step Law 在 N = 1B、D = 100B 的测试条件下，对照超参数公式的预测配置与实验确定的最优配置。等高线依据 120 次不同 LR 与 BSZ 组合的训练实验得到；该条件超出论文的拟合范围。图示比较限于论文采用的训练配方与学习率调度。<a href="https://arxiv.org/pdf/2503.04715v3#page=1">来源：Predictable Scale: Part I, 2025, v3, Fig. 1（PDF 第 1 页）</a>。'
  alt='Step Law 原论文 Figure 1：LR 与 BSZ 对应的 loss 等高线，以及 Step Law、其他超参数公式和实验确定的最优配置。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

在最优 batch size 上，Step Law 和 Power Lines 都发现 $B_{opt}$ 主要随训练数据量 $D$ 按幂律增长（指数约 0.4–0.57），与模型尺寸 $N$ 基本无关。因此不同尺寸、相同 token 量的实验可以先共用同一个 BSZ，减少搜索维度。

Power Lines 还测量了临界 batch size $B_{crit}\propto D_{min}^{0.5}$（$D_{min}$ 为无限小 batch 下达到目标 loss 所需的最少 tokens）。如图 5 所示，达到同一目标 loss 所需的训练 tokens $D$ 与更新步数 $S$ 呈双曲线关系：$B=B_{crit}$ 时耗费 $2D_{min}$ tokens；超过 $B_{crit}$ 后继续增大 batch size，步数减少很少，但算力开销明显增加。

{% include figure.liquid
  path='assets/img/pretrain-scaling/cerebras-bcrit-hyperbola.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 5. 达到相同目标 loss 所需训练 tokens 与 steps 的双曲线关系，左、右分别为 610M 和 1.7B 模型，颜色表示 loss。B_crit 表示 token 效率与步数之间的权衡转折。<a href="https://arxiv.org/pdf/2505.13738v2#page=6">来源：Bergsma et al., 2025, v2, Fig. 4</a>。'
  alt='610M 和 1.7B 模型的两幅 tokens 对 steps 曲线图：不同目标 loss 对应不同双曲线，颜色表示 loss，曲线标出 B_crit。'
  avoid_scaling=true
  zoomable=true
%}

[Schaipp, 2026](https://arxiv.org/abs/2607.01487v1) 测算发现，算力损耗不超过 5% 的次优 batch 区间宽度约为 $4\times$，对应 $[B_{opt}/2, 2B_{opt}]$，中大尺寸模型在此范围内做局部确认或对齐硬件并行倍数即可。此外，大规模训练有时会在前期采用 batch size warmup（例如 [Llama 3](https://arxiv.org/abs/2407.21783) §3.4.1 将 BSZ 从 4M 分段增至 16M tokens，[Nemotron-4, 2024](https://arxiv.org/abs/2402.16819) 做法类似），以兼顾训练初期的收敛速度与稳定性。

### 6.4 Weight Decay

在绝大多数工业界实践中，weight decay 都是直接固定为常数，最常见的取值是 $\lambda=0.1$，不随 $N$ 和 $D$ 调整（例如 [Kimi K2, 2025](https://arxiv.org/abs/2507.20534v2) 全程固定 $\lambda=0.1$；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961v1) 在 AdamW 中固定 $\lambda$ 并排除 embedding；[Qwen3, 2025](https://arxiv.org/abs/2505.09388) 也只对 LR 和 BSZ 做外推而不调 WD）。

学界也有一些工作探讨 WD 的缩放规律：比如 [Han et al., 2026](https://arxiv.org/abs/2602.11137v2) 认为使预训练 loss 最低的 $\lambda_{opt}$ 会随 TPP 增大而下降（同时指出稍大的 WD 有助于保持权重有效秩和后训练可塑性）；[Power Lines](https://arxiv.org/abs/2505.13738) 则用权重 EMA 的时间尺度 $\tau = B/(\eta\lambda D)$ 把三个超参联系起来，提出在 $\mu$P 下固定 $\eta$ 时按 $\lambda \propto B/D$ 调整 WD 以维持恒定的 $\tau_{opt}$。不过这些结论在更大规模上的必要性还有待商榷，实际搭建 Ladder 时直接固定 $\lambda=0.1$ 通常就够了。

### 6.5 LR Schedule 与训练收尾

预训练常用的两类学习率调度：

- Cosine 调度：warmup 后按余弦曲线平滑衰减到终点学习率 $\eta_{\text{end}}$，适合固定总训练步数的运行。
- WSD（Warmup-Stable-Decay）：大部分步数保持恒定学习率，最后阶段快速退火（[River Valley (Wen et al., 2024)](https://arxiv.org/abs/2410.05192v3) 从损失曲面几何解释了 stable 与 decay 的作用；[Tissue et al., 2024](https://arxiv.org/abs/2408.11029v2) 与 [Wang et al., 2025](https://arxiv.org/abs/2512.13705) 研究了退火曲线与退火比例随训练规模的变化）。WSD 的便利之处在于可以在同一条 stable 轨迹的不同位置分叉出多条退火分支，省去重复跑前期训练的开销。

关于终点学习率 $\eta_{\text{end}}$ 与调度选择，工业界实践中有几条重要经验：

- $\eta_{\text{end}}$ 通常不衰减到 0：预训练结束后往往还要接继续训练（CPT / CT）和后训练，把 $\eta_{\text{end}}$ 留在一个较小的非零值（例如 $3\times 10^{-5}$），能为后续阶段留出优化空间。
- 固定 $\eta_{\text{end}}$ 优于固定 $\eta_{\text{end}}/\eta_{\text{peak}}$ 比例：早期工作（如 [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361)、[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954)）常固定 $\eta_{\text{end}}/\eta_{\text{peak}}=0.1$，但这会把峰值学习率和终点学习率绑死——当尝试更大的 $\eta_{\text{peak}}$ 时，$\eta_{\text{end}}$ 也会跟着变大，导致训练末期无法充分消除梯度噪声，从而限制了 $\eta_{\text{peak}}$ 取到更优的较大值。直接固定绝对值 $\eta_{\text{end}}$（如 [Step Law](https://arxiv.org/abs/2503.04715v3) 的做法）能搜到更大的最优 $\eta_{\text{peak}}$ 和更低的最终 loss。
- 充分调优下 Cosine 优于 WSD：同尺寸、同训练量下 Cosine 与 WSD 的最优 $(\eta, B)$ 差异明显；在固定 $\eta_{\text{end}}$ 并分别为两者独立调优超参后，Cosine 的最优峰值学习率高于 WSD（$\eta_{\text{cosine}}^{\ast} > \eta_{\text{WSD}}^{\ast}$），且最终预训练 loss 稳定低于 WSD（[Kimi K3, 2026](https://arxiv.org/abs/2607.24653) §3.2）。

如果正式大模型训练包含以下收尾步骤，Ladder 实验也建议保持一致：

- 退火期数据切换（Continued Training / Cooldown）：在训练末期切换到高质量数据并配合学习率退火（[Nemotron-4, 2024](https://arxiv.org/abs/2402.16819)；[OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3)；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961)；[Nemotron 3 Super, 2026](https://arxiv.org/abs/2604.12374)），详见 [§11](#11-多阶段训练-ladder)。
- 权重平均（Weight Averaging）：包括把同一起点独立微调的多个分支合并（[Model Soups](https://arxiv.org/abs/2203.05482v3)）或对单条轨迹做滑动窗口平均（[LAWA](https://arxiv.org/abs/2209.14981)；[Sanyal et al., 2024](https://arxiv.org/abs/2306.03241)）。[Model Merging (ByteDance Seed, 2025)](https://arxiv.org/abs/2505.12082v3) 在 1.3B 与 13B 上展示了用 WSD stable 阶段的 checkpoint 权重平均（PMA）来低成本近似学习率退火的收益；不过经验上权重平均一般只能作为近似手段，拿不到真实 LR decay 的全部收益。

### 6.6 方案对比注意事项

用 Ladder 对比不同优化器、架构或 LR schedule 时，需要注意三点：

- 对齐各方案的调参程度：[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) 指出，直接沿用未针对当前规模调优的 AdamW 默认超参会夸大新优化器的收益；双方都充分搜参后，差距会明显缩小。对比不同 LR schedule 时同样需要分别为各自拟合最优超参（如 [§6.5](#65-lr-schedule-与训练收尾) 中 Cosine 与 WSD 的对比）。
- 在退火终点对比：Fantastic Optimizers 观察到不同优化器或超参在 stable 阶段的 loss 排名经常在 LR decay 阶段翻转，不能只看中途的排名下结论。
- 观察收益是否随规模衰减：在 Fantastic Optimizers 的 8× Chinchilla 实验中，矩阵类优化器相对 AdamW 的 token 效率优势从 0.1B 的 $1.4\times$ 缩小到了 1.2B 的 $1.1\times$（且 token 效率提升不等于实际训练时间缩短）。小模型上的改进需要在多个尺寸上观察是否随规模增大而衰减。

## 7. 数据 Ladder

我本人并非数据方向出身，对数据管线的了解有限，本章只结合公开文献，简要整理与 Scaling Ladder 直接相关的三个问题：评估新数据源（[§7.1](#71-数据质量与数据源评估)）、搜索多领域配比（[§7.2](#72-数据配比-ladder)），以及数据受限下的多 epoch 重复（[§7.3](#73-数据受限训练与重复)）。

### 7.1 数据质量与数据源评估

[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) 对比不同预训练语料时发现：

- 数据质量越高，最优算力分配越偏向增大模型参数量 $N$；
- 不同数据集拟合出的 scaling law 系数差异明显，不能直接跨数据集套用。

因此更换数据版本后需要重新验证 scaling law 参数。例如 [Kimi K3, 2026](https://arxiv.org/abs/2607.24653) §3.2 在同时更新架构、数据与训练方式后，重新拟合了 BSZ、LR、最优 TPP 与宽深比。

日常评估单个新数据源时不需要重跑完整 $(N, D)$ 矩阵，常用三种低成本做法：

- Checkpoint 微退火（Micro-annealing）：[OLMo 2, 2025](https://arxiv.org/abs/2501.00656) 与 [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) 从主干训练中后期的 checkpoint 出发，把待测数据与基础语料混合做短程退火，直接看下游指标变化。
- 小模型单域消融：[Kimi K3](https://arxiv.org/abs/2607.24653) §3.1 在固定小模型上通过消融实验确定各领域的采样权重。
- 领域合成数据消融：[Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) 将合成的法律推理数据混入 [Nemotron 3 Nano, 2025](https://arxiv.org/abs/2512.20848) 预训练，使 LegalBench 平均分从 64.6 提升到 74.7。

### 7.2 数据配比 Ladder

数据配比 Ladder 固定模型尺寸 $N$、训练量 $D$ 与优化超参，只改变 $m$ 个数据桶的采样权重 $\mathbf{w}=(w_1,\ldots,w_m)$。结合 [Olmix, 2026](https://arxiv.org/abs/2602.12237)、[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.4.4 与 [Marin 数据流程, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/)，常见流程如下：

1. 划分数据桶并统计 token 量：按数据源、领域或实例属性（如教育价值、代码、语言、安全；[Qwen3, 2025](https://arxiv.org/abs/2505.09388)）切分数据桶，并在每次调整过滤阈值后重新统计各桶去重后的可用 token 数 $N_j$（[Marin 数据流程, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/)）。如果训练分阶段切换数据分布（如 [Nemotron 3 Super, 2026](https://arxiv.org/abs/2604.12374) 在 25T tokens 中前 80% 偏重多样性、后 20% 偏重高质量数据），各阶段需分别搜索配比（[§11](#11-多阶段训练-ladder)）。
2. 设定基线：以自然 token 比例采样和限制最大重复轮数的 [UniMax](https://arxiv.org/abs/2304.09151) 作为对比基线；[Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) 指出，如果回归模型或约束没设好，搜出来的配比可能还不如这两个简单基线。
3. 训练小模型代理：
   - 代理尺寸：在 5× Chinchilla 训练量下，15M 及以上小模型与 1B 目标模型的配比排序 Spearman 相关系数高于 0.89，而降到 1M 时只有 0.73（[Olmix](https://arxiv.org/abs/2602.12237)）；[OLMo 3](https://arxiv.org/abs/2512.13961) 选用 30M 模型训练 3B tokens。
   - 按比例缩小数据池：小模型训练 token 少，如果直接在全量数据池里采样，不会出现大模型训练时的多轮重复过拟合。因此 [Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) 按小模型的激活参数量同比例缩小各桶数据池，使小模型在相同的 791 tokens/active param 下经历与目标大模型一致的重复轮数。个人认为在 Ladder 各尺寸上保持数据重复轮数一致非常重要，尤其是在评估 n-gram 这类与数据规模、重复次数和记忆过拟合高度相关的结构组件时，如果不按比例缩池，小模型在全量数据上只跑不到 1 个 epoch，表现会与多轮重复下的大模型完全不同。
   - 采样组数与模型合并替代：拟合 log-linear 回归一般至少需要 $3(m+1)$ 组实验，候选配比从以自然分布为中心的 Dirichlet 分布采样。[DeMix, 2026](https://arxiv.org/abs/2602.00747) 则提出给每个数据桶单独训一个组件模型，直接通过权重加权合并来评估任意配比，排序一致性比重新训练小模型更高。
4. 拟合与求解：对每个下游任务单独拟合 log-linear 模型再汇总，在留出配比上的相关性（0.983）明显高于直接拟合单一总分（0.866，[Olmix](https://arxiv.org/abs/2602.12237)）。另外，单桶权重过高引发过度重复时 loss 会回升（[Marin](https://openathena.ai/blog/marin-data-pipeline-overview/)），因此求解时需加入单桶最大重复轮数约束 $w_j \le k N_j / R$（$R$ 为目标总训练量，$k$ 为最大 epoch 数；[§7.3](#73-数据受限训练与重复)）以及指向自然分布的 KL 正则（如 $\lambda=0.05$），最后在中等尺寸模型上与自然采样和 UniMax 做对比确认。
5. 数据更新后的增量复用：如果只有少数数据桶增删，[Olmix](https://arxiv.org/abs/2602.12237) 的 mixture reuse 固定未变动桶之间的相对比例，只重跑涉及变动桶的实验。在 5 轮更新、扩展到 64 个领域的 1B/100B 实验中，该方法省去了 74% 的代理实验，保留了全量重算 95% 的收益（相对自然分布提升 11.6%），被用于 [OLMo 3](https://arxiv.org/abs/2512.13961) 的三轮数据迭代。

### 7.3 数据受限训练与重复

当目标训练量 $D$ 超过可用 unique tokens 总量 $U$ 时，需要做多 epoch 重复训练。统计重复轮数前要先做跨数据源的全局去重——[Marin 数据流程, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/) 在全局去重时发现，不同开源数据集之间有大量重叠，其中最大的重叠源来自 [Nemotron-CC](https://arxiv.org/abs/2412.02595) 及其合成改写版本。

[Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264v5) 假设每次额外重复的边际信息量呈指数衰减，定义了等效数据量 $D^\prime$：

$$D^{\prime} = U_D + U_D \cdot R_D^{\ast} \left(1 - e^{-R_D / R_D^{\ast}}\right)$$

其中 $U_D$ 为 unique tokens 数量，$R_D$ 为额外重复轮数（单 epoch 时 $R_D=0$），$R_D^\ast$ 为衰减半衰期参数（实验拟合约为 4）。当 $R_D\to\infty$ 时，等效数据量最多收敛到 $U_D(1+R_D^\ast)$。

{% include figure.liquid
  path='assets/img/pretrain-scaling/muennighoff-epoch-returns.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 6. 重复数据的边际收益（4.2B 模型，12B unique tokens）。前几轮重复接近新数据的收益，随后边际收益下降，约 40 epoch 后趋于饱和。<a href="https://arxiv.org/pdf/2305.16264v5#page=1">来源：Muennighoff et al., 2023, v5, Fig. 1 左</a>。'
  alt='4.2B 模型在 12B unique tokens 上的重复训练曲线：横轴为累计训练 tokens 和 epochs，纵轴为 final test loss；约 40 epochs 后收益趋于饱和。'
  avoid_scaling=true
  zoomable=true
%}

关于多轮重复的边界和缓解方式，主要有以下结论：

- 过拟合程度取决于参数量 $N$ 和数据池大小 $U$：[Yan et al., 2025](https://arxiv.org/abs/2511.13421v2) 在线性回归下推导出最优重复轮数随数据集大小对数增长；[Lovelace et al., 2026](https://arxiv.org/abs/2605.01640) 与 [Xue et al., 2023](https://arxiv.org/abs/2305.13230v2) 的实验表明，参数量 $N$ 越大、unique tokens $U$ 越少、重复轮数越多，过拟合越严重。因此在固定 $U$ 时，一味增大 $N$ 会加剧过拟合，算力最优解会向更小的 $N$ 偏移。
- 数据质量与 dropout：[Xue et al., 2023](https://arxiv.org/abs/2305.13230v2) 发现，相同 unique token 数量下提高数据质量并不能消除多轮重复带来的过拟合，而在训练中加入适当的 dropout（v2 Table 4）能明显缓解重复过拟合。
- 稀缺数据与通用数据混合训练：把小体量稀缺数据混入大规模通用语料一起训练时，可承受的重复轮数远高于单域训练。[Sedova et al., 2026](https://arxiv.org/abs/2605.12715) 通过 2,000 多次训练发现，混合训练下稀缺目标语料重复 15–20 轮仍有正收益，且带重复项的混合 scaling law 可以从小模型外推到大模型。
- 按质量限制重复上限：[Kimi K2.5, 2026](https://arxiv.org/abs/2602.02276) 从 [Kimi K2](https://arxiv.org/abs/2507.20534) 末期 checkpoint 继续联合预训练时，严格限制各数据源的最大 epoch 数；[OLMo 3](https://arxiv.org/abs/2512.13961)（§3.4.4, Appendix A.2.5）只对高质量分桶做上采样并允许最多重复 7 轮，效果优于直接按高阈值过滤。
- 用合成改写替代直接重复：[Kimi K2, 2025](https://arxiv.org/abs/2507.20534) §2.2 在 SimpleQA 上对比了三种方式：原文直接重复 10 轮得分为 23.76，改写 1 次训练 10 轮升至 27.39，生成 10 个不同改写版本各训 1 轮则达到 28.94；正式训练中对知识语料限制最多改写 2 次，并在 [Kimi K3](https://arxiv.org/abs/2607.24653) §3.1 中沿用。统计时建议把改写生成的 token 与原始 unique token 分开记录。

## 8. Loss Scaling Law 拟合

拟合 $L(N,D)$ 前需确认三件事（[Porian et al., 2024](https://arxiv.org/abs/2406.19146)）：算力 $C$ 已计入 output head 并与 $N_{\text{body}}$ 分开记录（[§3.1](#31-参数量计算)）、warmup 步数按训练量固定比例缩放（[§4.1](#41-变量分类)），以及求最优分配或对比方案时各点已调参到 Fully-Tuned Frontier（[§6.1](#61-搜参目标与搜索流程)）。

### 8.1 函数形式

如果实验只固定单一的 $D/N$ 配比，loss 直接拟合成总算力 $C$ 的一维幂律 $L(C) = A/C^\gamma + E$ 即可。但如果要跨不同 $D/N$ 预测或求解最优 $(N, D)$ 分配，就需要对 $N$ 和 $D$ 二维建模。最经典的是 Chinchilla 的加性形式：

$$L(N,D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + E$$

这个公式默认 $N$ 和 $D$ 的作用完全独立（混合偏导 $\partial^2L/\partial N\partial D\equiv0$）。但在大跨度 $(N, D)$ 网格上实测会发现，Chinchilla 的预测残差在网格四角呈明显的鞍形（图 7 左），且实测混合偏导持续为负，说明参数量和数据量之间存在耦合。

为了刻画 $(N, D)$ 耦合，近年主要有两种改进形式：

- [Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972)：让数据项的系数和指数都随 $N$ 变化，共有 9 个参数；
- [Skaling (Videau et al., 2026)](https://arxiv.org/abs/2608.07222v1)：直接在加性项外面加一个耦合指数 $k$，共 6 个参数（$k=1$ 时退化为 Chinchilla，实测 $k \approx 0.31$–$0.45$）：

$$L(N,D)=\left(\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}\right)^{k}+E$$

相比之下，Skaling 只比 Chinchilla 多 1 个参数，求最优 $(N_{opt}, D_{opt})$ 的闭式解结构不变；在 Farseer 和 SK-Grid 两套公开数据集上，6 参数的 Skaling 在插值和外推上的误差比 Chinchilla 和 9 参数的 Farseer 都更低（[Skaling](https://arxiv.org/abs/2608.07222) Table 1，图 7 中、右），推荐作为默认拟合公式。

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-vs-chinchilla-error.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 7. 两种形式在同一 (N, D) 网格上的预测残差，每点为一次训练。左、中：带符号百分比误差，共用色标；Chinchilla 的残差呈鞍形并向四角增大，Skaling 全网格接近零。右：两者误差之比，Skaling 在 76% 的配置上更准，中位数 2.2 倍。<a href="https://arxiv.org/abs/2608.07222">来源：Videau et al., 2026, Fig. 1</a>。'
  alt='三张 (N, D) 网格散点图：左为 Chinchilla 的带符号百分比误差，中为 Skaling 的同类误差，右为两者误差之比。'
  avoid_scaling=true
  zoomable=true
%}

采用 Skaling 的另一个实际好处是可以省去全网格采样。如图 8 所示，只跑“小 $D$ 覆盖多档 $N$ + 小 $N$ 覆盖多档 $D$”的 L-shape 布局，算力只需完整网格的 $1/5$–$1/10$，配合 Skaling 外推到大 $(N,D)$ 区域的精度就接近全网格；而 Chinchilla 加性形式在 L-shape 下外推误差会大幅退化。

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-v1-sampling-evaluation.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 8. Skaling 的采样策略与评价范围。左：Random 随机留出验证点；L-shape 在小 D 下覆盖不同 N，并在小 N 下覆盖不同 D。右：插值、沿 N 或 D 外推及联合外推的区域。横轴为 D，纵轴为 N。<a href="https://arxiv.org/pdf/2608.07222v1#page=5">来源：Videau et al., 2026, v1, Fig. 4（PDF 第 5 页）</a>。'
  alt='Skaling v1 Figure 4 原图：左侧为 Random 与 L-shape 采样，右侧为插值及沿模型尺寸、训练量和二者联合外推的评价区域；横轴 D，纵轴 N。'
  avoid_scaling=true
  zoomable=true
%}

### 8.2 Loss 曲线与退火 Scaling Law

[Tissue et al., 2024](https://arxiv.org/abs/2408.11029) 将训练步数 $s$ 上的 loss 表示为学习率调度的函数：

$$\hat L(s) = L_0 + A \cdot S_1(s)^{-\alpha} - C \cdot S_2(s)$$

其中 $S_1(s)=\sum_{i\leq s}\eta_i$ 为累计学习率面积，$S_2(s)$ 为带遗忘核的累计退火量。这个公式的好处是可以利用单条训练曲线上的所有中间 eval 点来拟合整条 loss 曲线，而不仅限于终点 loss，并且能预测 cosine、WSD 以及 continued training 中二次 warmup 后的曲线（原文 §4.7）。不过该公式以固定的 peak LR 为前提，跨规模的 peak LR 外推仍要靠 [§6.3](#63-lr-与-bsz-的-scaling-law) 的超参 scaling law。

个人经验：公式中 $S_2(s)$ 的遗忘核改用幂律核 $K(\Delta) = (1+\Delta)^{-p}$，拟合效果通常比原论文的单指数核更好。

### 8.3 数值拟合细节

直接对幂律公式做普通最小二乘回归很容易陷入局部极小值——因为指数 $\alpha, \beta$ 与系数 $A, B$ 之间强共线，优化器初值或异常点权重的微小变化都可能让外推出来的最优 $(N_{opt}, D_{opt})$ 差出几倍（[Besiroglu et al., 2024](https://arxiv.org/abs/2404.10102)；[(Mis)Fitting (Li et al., 2025)](https://arxiv.org/abs/2502.18969)）。实际拟合通常沿用 Chinchilla 原文的做法：把系数改写为对数参数（$a=\log A, b=\log B, e=\log E$），通过 `log-sum-exp` 计算 $\log \hat{L}(N,D)$，以 $\log L$ 上的 Huber loss（如 $\delta=10^{-3}$）作为优化目标来降低异常点影响，并在参数网格上取多组初值运行 L-BFGS，最后选 loss 最低的一组收敛解。

## 9. 下游任务预测

除了预测预训练 loss，大家自然也希望用预训练 Ladder 直接预测旗舰模型（hero run）在下游 benchmark 上的分数。不过我个人并非专门做这一方向的，了解有限；而且从目前的实践来看，下游任务的 loss 和离散 accuracy 远不如预训练 loss 那样干净好拟合（[Schaeffer et al., 2024](https://arxiv.org/abs/2406.04391) 也分析过多选题中错误干扰项概率分布和 argmax 截断带来的预测噪声）。如何用小模型准确预测大模型的下游表现，学界和工业界都还在不断探索新方法，目前并没有收敛出一套公认的标准做法。

下面列举几类有代表性的探索思路供参考：

| 代表思路 | 基本做法 | 相关工作 |
|---|---|---|
| 通用 Loss $\to$ 下游分数 | 先由 $(N,D)$ 外推预训练 loss，再用单调函数（幂律或 sigmoid）映射为下游得分 | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2)；[Delphi](https://openathena.ai/blog/delphi/) |
| $(N,D) \to$ Task NLL $\to$ Acc | 分两步：先对目标任务正确选项的负对数似然（NLL）做幂律外推，再拟合 NLL 到 accuracy 的 sigmoid 映射 | [Bhagia et al., 2024](https://arxiv.org/abs/2412.04403)（OLMo Task Ladder）；[Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1 |
| 按题目难度聚类分桶 | 根据小模型表现将题库按难度聚类，筛出有清晰上升信号的子集外推算力曲线，再映射回全集总分 | [COD (Xu et al., 2026, v4)](https://arxiv.org/pdf/2502.17262v4)；[GPT-4 Technical Report](https://arxiv.org/abs/2303.08774v6) |
| 给小模型加提示（Hint）辅助外推 | 对小模型完全做不出的难题，在评测时提供不同程度的提示（hint）使其脱离零分，联合拟合得分随提示强度与算力的变化，再外推大模型无提示表现 | [Hinted Scaling Laws, 2026](https://arxiv.org/abs/2611.02270) |

其中，传统两阶段方法以 [Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1 为代表：第一阶段用 $\le 10^{22}$ FLOPs 的 Ladder 小模型拟合正确选项归一化 NLL 对 $\log\text{FLOPs}$ 的线性关系；第二阶段结合 Ladder 模型与 Llama 2 系列模型拟合 NLL 到 accuracy 的 sigmoid 曲线，在 ARC-Challenge 上对 405B 给出了比较接近的预测。

而在 MATH、AIME、GSM8K 这类推理任务上，最大的痛点是小模型几乎全得 0 分，没有随算力上升的趋势可供外推。近期的两类工作分别从不同角度绕开这一零分瓶颈：

一是**从题库侧按难度分桶**（[COD, Xu et al., 2026, v4](https://arxiv.org/pdf/2502.17262v4)）：用一组小模型对每道题多次采样的平均通过率做难度聚类，剔除小模型全 0 分的极难题和早早满分的简单题，只在有单调上升信号的中等难度簇上拟合双指数曲线外推目标算力，最后通过校准曲线映射回完整测试集总分。在 8 个主流 benchmark 上，COD 对 70B 模型的平均绝对预测误差为 1.55 个百分点（图 9；不过跨架构从 dense 预测 MoE 时误差会放大到约 3.11 个百分点）。

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

二是**从评测侧给提示降低门槛**（[Hinted Scaling Laws, van Adrichem et al., 2026](https://arxiv.org/abs/2611.02270)）：不筛题，而是通过揭示前 $H$ 比例的标准解答步骤（Truncation）、随机保留 $H$ 比例的解答词（Masking）或按权重 $H$ 混入强教师模型的 logits（Logit mixing），把原题库变成随提示强度 $H$ 连续变容易的版本，让原本得 0 分的小模型也能拿到平滑上升的非零准确率（图 10 左）。由于提示强度 $H$ 与对数算力 $C$ 对准确率的提升近似满足线性置换关系，只在最小几档模型上拟合一个关于 $(C, H)$ 的联合带底 sigmoid 公式，再令 $H=0$ 即可预测大模型在无提示原题上的得分。例如在 Delphi 预训练 Ladder 上，仅用 $\le 10^{20}$ FLOPs 的小模型预测 $10^{23}$ FLOPs 模型在 GSM8K 上的准确率，误差在 3 个百分点以内，而只用无提示分数直接外推的基线偏差达 13 个百分点（图 10 右）。

{% include figure.liquid
  path='assets/img/pretrain-scaling/hinted-scaling-laws-delphi.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 10. Hinted Scaling Laws 示意图及其在 Delphi 预训练 Ladder（GSM8K）上的预测效果。上：三种给提示方式（Truncation、Masking、Logit mixing）；下：准确率随提示强度 H 平滑上升（左），在小模型上拟合关于训练算力与 H 的联合曲线（中），令 H=0 外推大模型无提示准确率，跨越 5600× 算力准确命中目标点（右）。<a href="https://arxiv.org/pdf/2611.02270v1#page=2">来源：van Adrichem et al., 2026, v1, Fig. 1</a>。'
  alt='Hinted Scaling Laws 原论文 Figure 1：上方为 Truncation、Masking 与 Logit mixing 三种提示方法示意，下方三个子图为 Delphi 预训练 Ladder 在 GSM8K 上随提示强度的平滑曲线、联合拟合曲面及 5600 倍算力外推预测对比。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

## 10. 外推验证

### 10.1 Holdout 与组合验证

Holdout 实验点应沿目标外推方向（更大的 $N$、更长的 $D$ 或更高的算力 $C$）设置，预算允许时拉开多档外推倍数（见 [§1.1](#11-什么是-scaling-ladder)、[§5.2](#52-尺寸数量跨度与外推倍数)）。如果根据某组 holdout 的结果修改了模型方案或拟合公式，这组点就变成了验证集，评估最终外推精度需要新的 holdout 点。目标算力远大于最大 holdout 时，正式训练前可以用最终配置先跑一次中等规模试运行，确认 loss 曲线落在预测区间内。

架构、优化器、超参与数据配比通常分开做消融，但单项改动的收益不能直接相加（[Marin 后续](https://openathena.ai/blog/pretraining-speedup/)；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) Appendix A.2.5）。不同改动之间可能互相影响，比如换优化器会改变多 epoch 数据上的过拟合速度，调数据配比也会改变最优 LR。定下最终方案前，建议把选中的改动合在一起，在目标 TPP 和重复轮数下做一次联合验证。

### 10.2 稳定性压力测试

小模型层数浅、步数短，很难暴露大模型训练中后期的数值溢出、loss spike 和路由失衡。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 的做法是在中等规模模型上做高 LR 压力测试：把 LR 调到预测最优值的 $2\times$ 和 $4\times$，在更高优化压力下对比新旧架构或优化器的训练稳定性。

不过短程高 LR 测试替代不了长程训练后期的监控。例如 [Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) §2.7 在训练后期出现过两次发散：第一次是输出层低精度梯度溢出，将该层梯度改回 FP32 后恢复正常；第二次原因未查明，靠提前进入退火阶段解决。正式训练时建议全程监控各层梯度范数、激活最大值与 loss spike，MoE 模型还要关注专家负载和 token 丢弃率。

### 10.3 训练框架与硬件对齐

跑小模型 Ladder 和跑大模型预训练经常不是同一套环境：小模型常用 FSDP 加标准 kernel，大模型则多用 Megatron 等框架，开启 TP/PP/EP 混合并行、融合 kernel 和低精度训练（如 [DeepSeek-V3, 2024](https://arxiv.org/abs/2412.19437) §3.3 的 FP8 方案），还会频繁经历断点续训（[Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.3.4）。

从个人经验来看，哪怕架构、优化器和数据完全相同，不同训练框架跑出来的 loss 和训练内部观测量（如各层 grad norm、激活值等）往往会有明显差异；显卡型号不同（比如 H 卡和 B 卡），跑出来的 loss 也可能差出 0.01 左右。

因此，搭 research 用的 Ladder 最好直接和大模型用同一套训练框架与同代显卡。如果条件不允许、没法统一，就应该在相同配置下把 pre-training loss、validation/test loss、评测集上的 loss 与评测分数，以及训练过程中的各项内部观测量都尽量对齐。

### 10.4 实际训练速度与推理成本

理论 FLOPs 的节省不一定能等比例缩短实际训练时间（wall-clock time）。[Marin 后续](https://openathena.ai/blog/pretraining-speedup/)区分了理论效率（theoretical efficiency）与实际效率（realized efficiency），[Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) 也指出不同精度与算子方案要以目标集群的实测吞吐为准——尤其是 MoE 模型，跨节点 EP 的 all-to-all 通信不计入 FLOPs，但会直接影响训练吞吐。

如果模型最终要上线部署，定 $(N, D)$ 时还要把推理开销算进去（见 [§5.3](#53-训练量档位)）。[Sardana et al., 2024](https://arxiv.org/abs/2401.00448) 将训练成本与 prefill、decode 推理成本联合建模；实际选型时可以结合线上请求的输入输出长度、延迟要求（TTFT / TPOT）、显存限制和量化精度，按集群实测成本算出更合适的尺寸与训练量。

## 11. 多阶段训练 Ladder

现在主流 LLM 普遍采用多阶段预训练，比如先做通用预训练（PT），再接高质量数据的继续训练（CT / Mid-training）和长上下文扩展。不同阶段的数据分布与 LR schedule 不同，PT 阶段拟合出的 Scaling Law 系数不能直接套用到后续阶段（[Qwen3, 2025](https://arxiv.org/abs/2505.09388)；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961)）。

如果 CT / Mid-training 阶段也需要定量预测超参、数据配比或新增 token 预算，可以在 PT Ladder 之上接着搭一套 CT Ladder：各尺寸模型直接从对应的 PT checkpoint 出发（PT 终点保留非零 $\eta_{\text{end}}$，见 [§6.5](#65-lr-schedule-与训练收尾)），单独拟合这一阶段的规律，并同时观察目标能力提升与原有基础能力的变化。

需要注意的是，CT 阶段的表现依赖起点 checkpoint 的状态：即使两个 PT checkpoint 的 validation loss 相近，只要结束时的 LR、已见数据的重复轮数或优化器动量不同，接上同一套 CT 配置后的效果就可能不一样。因此，小模型起点 checkpoint 的训练方式，以及阶段切换时是否二次 warmup、是否重置优化器状态，都建议与目标大模型保持一致。

至于最后的长上下文扩展阶段，本身训练 token 占比很小，实际中未必会专门搭一套独立的 Scaling Ladder；加上我个人不做长上下文方向的训练与评测，也不是本文重点，这里就不展开了（长序列下的 FLOPs 计算修正见 [§3.2](#32-计算量)，相关评测与注意力配置可参考 [RULER, 2024](https://arxiv.org/abs/2404.06654) 与 [Chen et al., 2025](https://arxiv.org/abs/2503.09579)）。

## 12. 搭建 Checklist

下面这份清单只是经验性的参考，不必面面俱到。实际上哪怕在各家 Frontier Lab，搭 Ladder 的过程中也总会遇到各种细节问题，最终拼的往往是谁发现和修掉的 bug 更多、更快。这里只整理前文最核心、最容易踩坑的几项检查点：

- [ ] **参数量与算力分开统计**：拟合 $L(N,D)$ 用不含 embedding 与 head 的主干参数 $N_{\text{body}}$；算 FLOPs $C$ 时把 output head 和长序列注意力项加回来（[§3.1](#31-参数量计算)、[§3.2](#32-计算量)）。
- [ ] **按训练进度比例设 warmup 与 eval 间隔**：不要跨规模固定死步数；统一主任务 cross-entropy、固定验证集，跨词表对比换算为 BPB（[§3.3](#33-loss-与-token-统计)、[§4.1](#41-变量分类)、[§4.5](#45-词表与训练精度)）。
- [ ] **小模型搜参搜透，完整退火后再比**：把算力花在小模型网格搜索上以落在 Fully-Tuned Frontier；对比不同架构、优化器或 schedule 时各自独立调优，不在退火完成前提前下结论（[§6.1](#61-搜参目标与搜索流程)、[§6.6](#66-方案对比注意事项)）。
- [ ] **LR 调度与 MoE 迁移**：PT 终点 $\eta_{\text{end}}$ 留非零值给 CT，且固定绝对 $\eta_{\text{end}}$ 优于固定 $\eta_{\text{end}}/\eta_{\text{peak}}$ 比例（[§6.5](#65-lr-schedule-与训练收尾)）；MoE 可先用低成本 dense Ladder 拟合 LR 幂律，再用几个小 MoE 点拟合迁移常数 $c_{\text{MoE}}$（[§6.3](#63-lr-与-bsz-的-scaling-law)）。
- [ ] **对齐各尺寸的数据重复轮数**：涉及多 epoch 重复或数据配比实验时，按模型规模同比例缩小数据池，并在拟合时计入重复衰减（[§7.2](#72-数据配比-ladder)、[§7.3](#73-数据受限训练与重复)）。
- [ ] **选带耦合项的公式并截掉早期点**：二维 $(N, D)$ 拟合优先考虑 Skaling（算力紧时可配 L-shape 采样）；纳入中间 checkpoint 时截掉前 10% 不稳定段，用对数空间 Huber loss 加多初值优化（[§5.2](#52-尺寸数量跨度与外推倍数)、[§8.1](#81-函数形式)、[§8.3](#83-数值拟合细节)）。
- [ ] **多档 holdout、联合验证与高 LR 压测**：沿外推方向留独立 holdout 且不反向调参；单项改动合并做一次联合验证，新架构或优化器做 $2\times / 4\times$ 高 LR 压力测试（[§10.1](#101-holdout-与组合验证)、[§10.2](#102-稳定性压力测试)）。
- [ ] **对齐训练框架与显卡**：尽量与目标大模型共用同一套训练框架和同代显卡；没法统一时，把各类 loss、评测分数和训练内部观测量尽量对齐（[§10.3](#103-训练框架与硬件对齐)）。

## 附录 A. 公开 Ladder 配置

### A.1 公开规模配置

部分公开 Scaling Law 研究的实验规模与训练量（参数量统计方式依各原始文献）：

| 来源 | 模型参数量范围 | 训练数据量 / 算力范围 | 参考文献 |
|---|---|---|---|
| OpenAI | 多尺寸，最大 1.5B（不含 embedding） | 22M–23B tokens | [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) |
| DeepMind Chinchilla | 70M–16B（400+ 模型） | 5B–500B tokens | [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) |
| StepFun | 多尺寸（3,700+ 模型） | 全实验累计约 100T tokens | [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) |
| Gadre et al. | 11M–6.9B（104 个模型） | 最高 32× Chinchilla 配比 | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540) |
| Fantastic Optimizers | 0.1B–1.2B（4 个尺寸） | 1×–8× Chinchilla 配比 | [Wen et al., 2025](https://arxiv.org/abs/2509.02046) |
| Delphi | 按算力扫描尺寸与训练量，最大 holdout 25B | 拟合 $3\times 10^{18}$–$3\times 10^{20}$ FLOPs，holdout 至 $10^{23}$ FLOPs | [Marin, 2026](https://openathena.ai/blog/delphi/) |
| Llama 3 | 40M–16B | 拟合 $6\times 10^{18}$–$10^{22}$ FLOPs，外推目标 $3.8\times 10^{25}$ FLOPs | [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.2.1 |

部分开源基座模型的训练量（供参考目标 TPP）：

| 来源 | 发布模型尺寸 | 累计训练量 | 参考文献 |
|---|---|---|---|
| NVIDIA Nemotron | Nemotron-4：15B, 340B；Nemotron 3（Hybrid MoE）：Nano 30B-A3B, Super 120B-A12B, Ultra 550B-A55B | Nemotron-4：8T, 9T；Nemotron 3：Nano / Super 各 25T，Ultra 20T | [Nemotron-4 15B](https://arxiv.org/abs/2402.16819)；[340B](https://arxiv.org/abs/2406.11704)；[Nemotron 3 Nano](https://arxiv.org/abs/2512.20848)；[Super](https://arxiv.org/abs/2604.12374)；[Ultra](https://arxiv.org/abs/2606.15007) |
| OLMo | OLMo 1：1B, 7B；OLMo 2：7B, 13B, 32B；OLMo 3：7B, 32B | OLMo 1：2T–2.46T；OLMo 2：最高 5T+；OLMo 3：约 6T（5.9T PT + 100B mid-train + 50B 长文本） | [OLMo, 2024](https://arxiv.org/abs/2402.00838v4)；[OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3)；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) |
| Llama 3 | 8B, 70B, 405B | 405B：15.6T tokens（接近算力最优）；8B/70B：约 15T tokens（大幅过训练） | [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §1、§3.4 |

### A.2 Fantastic Optimizers Ladder

[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046v2) Table 2–3 给出了一套用于对比优化器与搜索超参的 dense Ladder：采用 Llama 2 架构，四个尺寸均固定 32 层、MHA、`seq_len` 4096，只通过调整 `hidden_dim` 改变参数量。

| 尺寸 | hidden_dim | inter_dim | heads | 训练量档位 |
|---|---|---|---|---|
| 130M | 512 | 2,048 | 8 | 1×–8× Chinchilla |
| 300M | 768 | 3,072 | 12 | 同上 |
| 520M | 1,024 | 4,096 | 16 | 同上 |
| 1.2B | 1,536 | 6,144 | 24 | 同上 |

数据由 DCLM-baseline、StarCoder V2 Data 与 ProofPile 2 混合而成。每个 $(N, D)$ 网格点单独搜索超参（如 Table 3 中 130M/1× 档 AdamW 为 Peak LR 8e-3、WD 0.1、warmup 2000 steps、BSZ 128 条序列约 0.5M tokens；520M/1× 档为 WD 0.2、BSZ 256）。

### A.3 Delphi Ladder

[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) 给出了用于端到端 loss 预测与外推验证的 dense Ladder：基于 Qwen 3 decoder-only 架构（MLP ratio 4，`seq_len` 4096），后续扩展到 MoE（[535B-A23B](https://openathena.ai/blog/pretraining-speedup/)）。

全系固定采用 AdamH 优化器、WSD 调度（10% warmup，最后 20% 线性衰减到 0）、FP32 主权重配 BF16 计算与 FSDP，数据由 Nemotron-CC、StarCoderData 与 ProofPile 2 组成。实验在 $3\times 10^{18}$–$3\times 10^{20}$ FLOPs 范围内做 IsoFLOP 扫描，取 7 个最优点拟合幂律，超参直接由预设公式算出而不逐点搜索，最后在 $10^{21}$–$10^{23}$ FLOPs（最大 25B，对应 $3\times$–$333\times$ 外推）的多档 holdout 上验证外推精度。

### A.4 选择建议

- 侧重搜参与算法对比：可参考 Fantastic Optimizers 的 $(N \times \text{TPP})$ 网格，把算力花在将每个小尺寸网格点搜到 Fully-Tuned Frontier。
- 侧重算力分配与 loss 外推：可参考 Delphi 的 IsoFLOP 扫描与预设超参公式，用多档 holdout 检验外推精度与稳定性。
- 对齐目标大模型架构：两套公开配置都是 dense 起点（其中 Fantastic Optimizers 固定了 32 层 MHA）；实际搭建时建议让小模型的宽深比、GQA 分组等与目标模型保持一致（[§4.2](#42-架构与训练配置一致性)），目标为 MoE 时按 [§5.4](#54-moe-实验设计) 与 [§6.3](#63-lr-与-bsz-的-scaling-law) 设计实验与超参迁移。

## 参考文献

按主题分类汇总，每条文献后附正文中引用的对应章节。

### Scaling Law 基础、函数形式与拟合

1. Scaling Laws for Neural Language Models — Kaplan et al., OpenAI, 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)  
   跨越 7 个数量级的经验幂律与排除 embedding 的参数统计（[§3.1](#31-参数量计算)、[§4.3](#43-宽深配置)、[§6.5](#65-lr-schedule-与训练收尾)、[附录 A.1](#a1-公开规模配置)）
2. Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., DeepMind, 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)  
   IsoFLOP 扫描方法、$N$ 与 $D$ 等比例缩放及加性幂律公式（[§3.1](#31-参数量计算)、[§5.3](#53-训练量档位)、[§8.1](#81-函数形式)、[附录 A.1](#a1-公开规模配置)）
3. Language models scale reliably with over-training and on downstream tasks — Gadre et al., 2024. [arXiv:2403.08540](https://arxiv.org/abs/2403.08540)  
   最高 32× Chinchilla 过训练区间的幂律外推与下游错误率映射（[§5.3](#53-训练量档位)、[§9](#9-下游任务预测)、[附录 A.1](#a1-公开规模配置)）
4. Skaling: Chinchilla's Exponents Meet Kaplan's Coupling — Videau et al., FAIR at Meta, 2026. [arXiv:2608.07222](https://arxiv.org/abs/2608.07222)  
   引入外耦合指数 $k$ 消除四角鞍形残差，并验证 L-shape 采样（[§8.1](#81-函数形式)）
5. Predictable Scale: Part II, Farseer: A Refined Scaling Law in Large Language Models — Li et al., StepFun & Fudan, NeurIPS 2025. [arXiv:2506.10972](https://arxiv.org/abs/2506.10972)  
   9 参数耦合形式与排除 embedding 的外推消融实验（[§3.1](#31-参数量计算)、[§8.1](#81-函数形式)）
6. Scaling Law with Learning Rate Annealing — Tissue et al., 2024. [arXiv:2408.11029](https://arxiv.org/abs/2408.11029)  
   基于累计学习率面积与退火核的整条 loss 曲线建模（[§6.5](#65-lr-schedule-与训练收尾)、[§8.2](#82-loss-曲线与退火-scaling-law)）
7. A Hitchhiker's Guide to Scaling Law Estimation — Choshen et al., MIT/IBM, ICML 2025. [arXiv:2410.11840](https://arxiv.org/abs/2410.11840)  
   早期 checkpoint 截断、尺寸跨度与随机种子波动分析（[§2.3](#23-验收标准)、[§5.2](#52-尺寸数量跨度与外推倍数)）
8. Resolving Discrepancies in Compute-Optimal Scaling of Language Models — Porian et al., NeurIPS 2024. [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)  
   分析 output head 算力、固定 warmup 与搜参不足对幂律指数的影响（[§3.1](#31-参数量计算)、[§4.1](#41-变量分类)、[§8](#8-loss-scaling-law-拟合)）
9. Chinchilla Scaling: A Replication Attempt — Besiroglu et al., Epoch AI, 2024. [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)  
   复现 Chinchilla 参数拟合，分析优化初值与损失函数对拟合结果的影响（[§8.3](#83-数值拟合细节)）
10. (Mis)Fitting: A Survey of Scaling Laws — Li et al., 2025. [arXiv:2502.18969](https://arxiv.org/abs/2502.18969)  
    综述拟合目标与数值细节缺失对结论可复现性的影响（[§8.3](#83-数值拟合细节)）
11. Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws — Sardana et al., 2024. [arXiv:2401.00448](https://arxiv.org/abs/2401.00448)  
    计入推理成本的最优配比偏移及最高 10,000 TPP 的过训练实验（[§5.3](#53-训练量档位)、[§10.4](#104-实际训练速度与推理成本)）
12. Gemstones: A Model Suite for Multi-Faceted Scaling Laws — McLeish et al., 2025. [arXiv:2502.06857](https://arxiv.org/abs/2502.06857)  
    分析模型宽深比对下游任务表现的影响（[§4.3](#43-宽深配置)）
13. Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies — Tao et al., NeurIPS 2024. [arXiv:2407.13623](https://arxiv.org/abs/2407.13623)  
    最优词表大小随训练算力幂律增长（[§4.5](#45-词表与训练精度)）
14. Scaling Laws for Precision — Kumar et al., 2024. [arXiv:2411.04330](https://arxiv.org/abs/2411.04330)  
    训练数值精度与后训练量化的联合 scaling law（[§4.5](#45-词表与训练精度)）

### 超参 Scaling Law 与优化器

{:start="15"}
15. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism — DeepSeek, 2024. [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)  
    按算力 $C$ 拟合的超参幂律及数据质量对最优 $N/D$ 分配的影响（[§6.3](#63-lr-与-bsz-的-scaling-law)、[§6.5](#65-lr-schedule-与训练收尾)、[§7.1](#71-数据质量与数据源评估)）
16. Predictable Scale: Part I, Step Law — Optimal Hyperparameter Scaling Law in Large Language Model Pre-training — Li et al., StepFun, 2025. [arXiv:2503.04715v3](https://arxiv.org/abs/2503.04715v3)  
    基于 3,700+ 模型拟合的 $\eta_{opt}(N,D)$ 与 $B_{opt}(D)$ 二元超参公式（[§6.1](#61-搜参目标与搜索流程)、[§6.3](#63-lr-与-bsz-的-scaling-law)、[§6.5](#65-lr-schedule-与训练收尾)、[附录 A.1](#a1-公开规模配置)）
17. Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training — Bergsma et al., Cerebras, 2025. [arXiv:2505.13738](https://arxiv.org/abs/2505.13738)  
    $B_{opt}\propto D^{0.4}$、$B_{crit}\propto D_{min}^{0.5}$ 双曲线及基于 EMA 时间尺度 $\tau$ 的超参关系（[§6.2](#62-参数化与跨优化器迁移)、[§6.3](#63-lr-与-bsz-的-scaling-law)、[§6.4](#64-weight-decay)）
18. Scaling Optimal LR Across Token Horizons — Bjorck et al., Microsoft, 2024. [arXiv:2409.19913](https://arxiv.org/abs/2409.19913)  
    固定模型尺寸与 BSZ 时峰值学习率随训练长度的幂律衰减（[§6.3](#63-lr-与-bsz-的-scaling-law)）
19. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer ($\mu$-Transfer) — Yang et al., Microsoft, 2022. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)  
    $\mu$P 参数化下的跨宽度零样本学习率迁移（[§6.2](#62-参数化与跨优化器迁移)）
20. Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks — Yang et al., 2023. [arXiv:2310.02244](https://arxiv.org/abs/2310.02244)  
    单层残差块的 Depth-$\mu$P 与多层残差块无限深度参数化的理论局限（[§6.2](#62-参数化与跨优化器迁移)）
21. Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit — Bordelon et al., 2023. [arXiv:2309.16620](https://arxiv.org/abs/2309.16620)  
    残差分支 $1/\sqrt{\text{depth}}$ 缩放在 ResNet 与 ViT 上的跨宽深度超参迁移（[§6.2](#62-参数化与跨优化器迁移)）
22. Muon is Scalable for LLM Training — Liu et al., Moonshot AI, 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)  
    对齐更新 RMS 并加入 weight decay，使 Muon 直接复用 AdamW 超参（[§6.2](#62-参数化与跨优化器迁移)）
23. Fantastic Pretraining Optimizers and Where to Find Them — Wen et al., Stanford, 2025. [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)  
    优化器公平搜参基准、收益随尺寸衰减及退火期排名翻转（[§5.3](#53-训练量档位)、[§6.6](#66-方案对比注意事项)、[附录 A.1](#a1-公开规模配置)、[附录 A.2](#a2-fantastic-optimizers-ladder)）
24. Weight Decay Improves Language Model Plasticity — Han et al., 2026. [arXiv:2602.11137](https://arxiv.org/abs/2602.11137)  
    最优 WD 随 TPP 增大而下降及较大 WD 对后训练可塑性的影响（[§6.4](#64-weight-decay)）
25. Small-Scale Experiments: Are We There Yet? — Lourie et al., NYU & Meta, 2026. [arXiv:2608.11859](https://arxiv.org/abs/2608.11859)  
    小模型超参高敏感性、Fully-Tuned Frontier 及拟合/验证/测试三段尺寸划分（[§5.2](#52-尺寸数量跨度与外推倍数)、[§6.1](#61-搜参目标与搜索流程)）
26. How to Allocate Your Tokens? Scaling Laws with Training Steps and Batch Size — Schaipp, Inria, 2026. [arXiv:2607.01487](https://arxiv.org/abs/2607.01487)  
    5% 算力损耗内约 $4\times$ 宽度的次优 batch size 区间（[§6.3](#63-lr-与-bsz-的-scaling-law)）
27. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation — Cawley & Talbot, JMLR 2010. [JMLR 11](https://www.jmlr.org/papers/v11/cawley10a.html)  
    有限样本多次试验择优带来的选择偏差分析（[§6.1](#61-搜参目标与搜索流程)）

### MoE

{:start="28"}
28. Scaling Laws for Fine-Grained Mixture of Experts — Krajewski et al., 2024. [arXiv:2402.07871](https://arxiv.org/abs/2402.07871)  
    将专家粒度作为独立变量的 MoE scaling law（[§4.4](#44-moe-结构缩放规则)、[§5.4](#54-moe-实验设计)）
29. Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models — Abnar et al., Apple, 2025. [arXiv:2501.12370](https://arxiv.org/abs/2501.12370)  
    固定算力与固定总参数下的最优稀疏度规律及下游任务表现（[§4.4](#44-moe-结构缩放规则)）
30. Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient — Ludziejewski et al., 2025. [arXiv:2502.05172](https://arxiv.org/abs/2502.05172)  
    显存约束下联合建模专家数、激活参数与训练量（[§4.4](#44-moe-结构缩放规则)、[§5.4](#54-moe-实验设计)）

### 数据构建、配比与重复

{:start="31"}
31. Scaling Data-Constrained Language Models — Muennighoff et al., 2023. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)  
    多 epoch 重复训练的指数衰减等效数据量公式（[§7.3](#73-数据受限训练与重复)）
32. To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis — Xue et al., 2023. [arXiv:2305.13230](https://arxiv.org/abs/2305.13230)  
    重复过拟合随参数量 $N$ 的变化及 dropout 的缓解作用（[§7.3](#73-数据受限训练与重复)）
33. Prescriptive Scaling Laws for Data Constrained Training — Lovelace et al., 2026. [arXiv:2605.01640](https://arxiv.org/abs/2605.01640)  
    参数量、unique tokens 与重复轮数共同决定的过拟合与算力分配（[§7.3](#73-数据受限训练与重复)）
34. Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression — Yan et al., 2025. [arXiv:2511.13421](https://arxiv.org/abs/2511.13421)  
    最优重复轮数随数据集规模对数增长的理论分析与 LLM 实验（[§7.3](#73-数据受限训练与重复)）
35. UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining — Chung et al., ICLR 2023. [arXiv:2304.09151](https://arxiv.org/abs/2304.09151)  
    限制单桶最大重复轮数的配比基线算法（[§7.2](#72-数据配比-ladder)）
36. Olmix: A Framework for Data Mixing Throughout LM Development — Chen et al., Allen Institute, ICLR 2026. [arXiv:2602.12237](https://arxiv.org/abs/2602.12237)  
    配比代理尺寸、分任务 log-linear 回归、重复上限约束及 mixture reuse（[§7.2](#72-数据配比-ladder)）
37. Scaling Laws for Mixture Pretraining Under Data Constraints — Sedova et al., Apple, 2026. [arXiv:2605.12715](https://arxiv.org/abs/2605.12715)  
    稀缺领域与通用语料混合训练下的重复耐受度与配比外推（[§7.3](#73-数据受限训练与重复)）
38. Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training (DeMix) — Li et al., 2026. [arXiv:2602.00747](https://arxiv.org/abs/2602.00747)  
    用单域组件模型加权合并替代重训小模型的数据配比评估方法（[§7.2](#72-数据配比-ladder)）
39. Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset — Su et al., NVIDIA, 2024. [arXiv:2412.02595](https://arxiv.org/abs/2412.02595)  
    大规模清洗与合成改写语料，跨源全局去重中的主要重叠来源分析（[§7.3](#73-数据受限训练与重复)）

### LR Schedule 与训练收尾

{:start="40"}
40. Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al., 2024. [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)  
    从损失曲面几何解释 WSD 调度中 stable 与 decay 阶段的作用（[§6.5](#65-lr-schedule-与训练收尾)）
41. Scaling and Transferability of Annealing Strategies in Large Language Model Training — Wang et al., 2025. [arXiv:2512.13705](https://arxiv.org/abs/2512.13705)  
    退火比例与退火曲线的跨规模迁移规律（[§6.5](#65-lr-schedule-与训练收尾)）
42. Model Merging in Pre-training of Large Language Models — ByteDance Seed, 2025. [arXiv:2505.12082](https://arxiv.org/abs/2505.12082)  
    用 WSD stable 阶段 checkpoint 权重平均（PMA）近似退火收益（[§6.5](#65-lr-schedule-与训练收尾)）
43. Model Soups: Averaging Weights of Multiple Fine-tuned Models — Wortsman et al., 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)  
    同起点多分支独立微调后的权重平均（[§6.5](#65-lr-schedule-与训练收尾)）
44. Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging (LAWA) — Kaddour, 2022. [arXiv:2209.14981](https://arxiv.org/abs/2209.14981)  
    单条训练轨迹末端的滑动窗口权重平均（[§6.5](#65-lr-schedule-与训练收尾)）
45. Early Weight Averaging meets High Learning Rates for LLM Pre-training — Sanyal et al., COLM 2024. [arXiv:2306.03241](https://arxiv.org/abs/2306.03241)  
    高学习率预训练过程中的早期滑动权重平均（[§6.5](#65-lr-schedule-与训练收尾)）

### 下游任务预测与评估

{:start="46"}
46. Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective (COD) — Xu et al., ICLR 2026, v4（2026-03-09）. [arXiv:2502.17262v4](https://arxiv.org/pdf/2502.17262v4)  
    按题目难度聚类分桶外推下游 benchmark 准确率（[§9](#9-下游任务预测)）
47. Hinted Scaling Laws — van Adrichem et al., Stanford, 2026. [arXiv:2611.02270](https://arxiv.org/abs/2611.02270)  
    评测时给小模型提供连续强度的提示（hint）使其脱离零分，联合拟合提示强度与算力外推大模型无提示准确率（[§9](#9-下游任务预测)）
48. GPT-4 Technical Report — OpenAI, 2023. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)  
    按小模型通过率对 HumanEval 题目分桶并外推目标表现（[§9](#9-下游任务预测)）
49. Establishing Task Scaling Laws via Compute-Efficient Model Ladders (OLMo Task Ladder) — Bhagia et al., Allen Institute, 2024. [arXiv:2412.04403](https://arxiv.org/abs/2412.04403)  
    $(N,D) \to \text{Task NLL} \to \text{Accuracy}$ 的两阶段下游预测方法（[§9](#9-下游任务预测)）
50. Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive? — Schaeffer et al., 2024. [arXiv:2406.04391](https://arxiv.org/abs/2406.04391)  
    分析多选题错误干扰项分布与 argmax 截断带来的下游准确率预测噪声（[§9](#9-下游任务预测)）
51. RULER: What's the Real Context Size of Your Long-Context Language Models? — Hsieh et al., NVIDIA, 2024. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)  
    涵盖多跳追踪、信息聚合与问答的有效上下文长度评测集（[§11](#11-多阶段训练-ladder)）

### 模型技术报告与 Ladder 实例

{:start="52"}
52. Delphi: An Open Scaling Suite from 3e18 to 1e23 FLOPs — Marin Team, 2026. [openathena.ai/blog/delphi](https://openathena.ai/blog/delphi/)  
    IsoFLOP 扫描、预设公式超参与多档 holdout 外推验证（[§1.1](#11-什么是-scaling-ladder)、[§2.1](#21-决策目标)、[§5.2](#52-尺寸数量跨度与外推倍数)、[§9](#9-下游任务预测)、[§10.1](#101-holdout-与组合验证)、[附录 A.1](#a1-公开规模配置)、[附录 A.3](#a3-delphi-ladder)）
53. The Llama 3 Herd of Models — Meta, 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)  
    405B IsoFLOP 选型、小模型过训练、两阶段下游预测、BSZ warmup 与断点续训恢复（[§5.2](#52-尺寸数量跨度与外推倍数)、[§5.3](#53-训练量档位)、[§6.3](#63-lr-与-bsz-的-scaling-law)、[§9](#9-下游任务预测)、[§10.3](#103-训练框架与硬件对齐)、[附录 A.1](#a1-公开规模配置)）
54. DeepSeek-V3 Technical Report — DeepSeek, 2024. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)  
    无 token 丢弃路由与 FP8 混合精度训练方案（[§4.4](#44-moe-结构缩放规则)、[§10.3](#103-训练框架与硬件对齐)）
55. Nemotron-4 15B Technical Report — NVIDIA, 2024. [arXiv:2402.16819](https://arxiv.org/abs/2402.16819)  
    Batch size warmup 与训练末期高质量数据退火（[§6.3](#63-lr-与-bsz-的-scaling-law)、[§6.5](#65-lr-schedule-与训练收尾)、[附录 A.1](#a1-公开规模配置)）
56. Nemotron-4 340B Technical Report — NVIDIA, 2024. [arXiv:2406.11704](https://arxiv.org/abs/2406.11704)  
    340B 模型训练配置与 9T token 训练量参考（[附录 A.1](#a1-公开规模配置)）
57. OLMo: Accelerating the Science of Language Models — Groeneveld et al., Allen Institute, 2024. [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)  
    早期开源预训练模型、数据与中间 checkpoint 套件（[附录 A.1](#a1-公开规模配置)）
58. 2 OLMo 2 Furious — OLMo Team, Allen Institute, 2025. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)  
    多阶段训练、micro-annealing 数据源评估与 model souping（[§6.5](#65-lr-schedule-与训练收尾)、[§7.1](#71-数据质量与数据源评估)、[附录 A.1](#a1-公开规模配置)）
59. Olmo 3 — Team Olmo, Allen Institute, 2025. [arXiv:2512.13961](https://arxiv.org/abs/2512.13961)  
    BPB 代理指标、Olmix 配比迭代、高质量分桶上采样、多阶段训练与轻量后训练验证（[§2.2](#22-评估指标与设置)、[§6.4](#64-weight-decay)、[§6.5](#65-lr-schedule-与训练收尾)、[§7.1](#71-数据质量与数据源评估)、[§7.2](#72-数据配比-ladder)、[§7.3](#73-数据受限训练与重复)、[§10.1](#101-holdout-与组合验证)、[§11](#11-多阶段训练-ladder)、[附录 A.1](#a1-公开规模配置)）
60. Qwen3 Technical Report — Qwen Team, 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)  
    多阶段超参预测与实例级属性数据配比（[§6.4](#64-weight-decay)、[§7.2](#72-数据配比-ladder)、[§11](#11-多阶段训练-ladder)）
61. On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability — Qwen Team, 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)  
    大模型超参平坦区间、切换 Muon 的超参偏移、后训练退化验证与高 LR 压力测试（[§2.2](#22-评估指标与设置)、[§6.1](#61-搜参目标与搜索流程)、[§6.2](#62-参数化与跨优化器迁移)、[§10.2](#102-稳定性压力测试)）
62. Kimi K2: Open Agentic Intelligence — Moonshot AI, 2025. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)  
    固定激活规模的 MoE 稀疏度 scaling law 与知识语料改写消融（[§5.4](#54-moe-实验设计)、[§6.4](#64-weight-decay)、[§7.3](#73-数据受限训练与重复)）
63. Kimi K2.5: Visual Agentic Intelligence — Moonshot AI, 2026. [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)  
    继续联合预训练中的单源最大 epoch 限制（[§7.3](#73-数据受限训练与重复)）
64. Kimi K3: Open Frontier Intelligence — Moonshot AI, 2026. [arXiv:2607.24653](https://arxiv.org/abs/2607.24653)  
    Quantile Balancing（QB）负载均衡、更新架构与数据后重做 scaling law、Cosine 与 WSD 独立调参对比及知识改写复用（[§4.4](#44-moe-结构缩放规则)、[§6.5](#65-lr-schedule-与训练收尾)、[§7.1](#71-数据质量与数据源评估)、[§7.3](#73-数据受限训练与重复)）
65. Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2025. [arXiv:2512.20848](https://arxiv.org/abs/2512.20848)  
    30B-A3B Hybrid MoE 架构与 25T tokens 预训练配置（[§7.1](#71-数据质量与数据源评估)、[附录 A.1](#a1-公开规模配置)）
66. Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2026. [arXiv:2604.12374](https://arxiv.org/abs/2604.12374)  
    120B-A12B 模型及 25T tokens 前 80% 偏重多样性、后 20% 偏重高质量的两阶段数据配比（[§6.5](#65-lr-schedule-与训练收尾)、[§7.2](#72-数据配比-ladder)、[附录 A.1](#a1-公开规模配置)）
67. Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2026. [arXiv:2606.15007](https://arxiv.org/abs/2606.15007)  
    550B-A55B 模型、训练后期发散的 FP32 输出层梯度修复、提前退火与领域合成数据消融（[§7.1](#71-数据质量与数据源评估)、[§10.2](#102-稳定性压力测试)、[§10.4](#104-实际训练速度与推理成本)、[附录 A.1](#a1-公开规模配置)）
68. Marin：MoE 与训练效率后续 — Marin Team, 2026. [openathena.ai/blog/pretraining-speedup](https://openathena.ai/blog/pretraining-speedup/)  
    理论效率与实际训练时间效率的区分，以及多项改动的组合验证（[§10.1](#101-holdout-与组合验证)、[§10.4](#104-实际训练速度与推理成本)、[附录 A.3](#a3-delphi-ladder)）
69. Marin 数据流程 — Marin Team, 2026. [openathena.ai/blog/marin-data-pipeline-overview](https://openathena.ai/blog/marin-data-pipeline-overview/)  
    跨源全局去重、代理数据池同比例缩池与带重复上限约束的配比回归（[§7.2](#72-数据配比-ladder)、[§7.3](#73-数据受限训练与重复)）
70. Phi-4 Technical Report — Microsoft, 2024. [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)  
    高波动下游评测任务的多次采样处理（[§2.2](#22-评估指标与设置)）
71. GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models — Zhipu AI, 2025. [arXiv:2508.06471](https://arxiv.org/abs/2508.06471)  
    模型深度与注意力头配置对推理表现的影响（[§4.3](#43-宽深配置)）

### 延伸阅读

以下文献与大 batch 训练和注意力结构相关，供扩展参考：

{:start="72"}
72. Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training — Merrill et al., 2025. [arXiv:2505.23971](https://arxiv.org/abs/2505.23971)
73. An Empirical Model of Large-Batch Training — McCandlish et al., 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
74. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al., Google, 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
75. Cost-Optimal Grouped-Query Attention for Long-Context Modeling — Chen et al., 2025. [arXiv:2503.09579](https://arxiv.org/abs/2503.09579)（[§11](#11-多阶段训练-ladder)）
