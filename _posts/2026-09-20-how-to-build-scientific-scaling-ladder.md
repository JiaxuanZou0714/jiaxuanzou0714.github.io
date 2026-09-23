---
layout: post
title: "如何搭建一个科学的 Scaling Ladder"
date: 2026-09-20 10:00:00
description: "按执行顺序整理 Scaling Ladder 的设计与搭建：决策目标与验收、测量口径、dense 与 MoE 的缩放规则、实验矩阵与预算、超参搜索、数据配比与数据受限训练、Loss Scaling Law 拟合、下游预测、外推验证与交付流程，综合 Chinchilla、DeepSeek、StepFun、Cerebras、Llama 3、Delphi 等公开文献与工程实践。"
tags: [scaling-laws, pretraining, hyperparameter, optimization, llm, empirical-methodology]
categories: [research]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: zh-CN
---

本文整理自**<u>公开文献与技术报告，不含涉密内容</u>**。内容覆盖如何搭建 Scaling Ladder：通过小规模实验拟合 Scaling Law，外推目标模型的训练配置与性能，并据此判断是否启动目标训练。


---

## 1. 适用范围与术语

### 1.1 Scaling Ladder 的定义

Scaling Ladder 是一组覆盖不同参数量 $N$、训练量 $D$ 及其他待研究变量的训练实验，用于拟合 Scaling Law，外推目标规模下的训练配置、loss 与下游表现，为模型尺寸、训练量、超参数和候选方案的选择提供定量依据，降低大规模训练的配置风险。

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

图 1 中的配方在拟合范围内没有异常，外推到目标规模后 loss 高于预测并发散。Ladder 必须在交付前验证外推精度（[§10](#10-外推验证与启动决策)）。

### 1.2 典型场景

本文覆盖三类场景。第 2–10 章为共同流程，各章中针对 MoE 与数据受限条件的内容单独列出。

| 场景 | 主要约束 | 专门章节 |
|---|---|---|
| Dense | 结构缩放规则与宽深配置；用于 dense 交付模型，也作为 MoE 的对照 | [§4.3](#43-宽深配置) |
| MoE | 总参数量与激活参数量分离；稀疏率、专家粒度、路由与负载均衡；专家并行的实际吞吐 | [§3.1](#31-参数量口径)、[§4.4](#44-moe-结构缩放规则)、[§5.6](#56-moe-实验轴)、[§6.6](#66-moe-训练超参)、[§8.5](#85-moe-拟合)、[§10.3](#103-稳定性压力测试) |
| 数据受限 | 可用 unique tokens 少于目标训练量；重复次数、数据质量与配比共同决定可达 loss | [§5.3](#53-训练量档位)、[§7.3](#73-数据受限训练与重复) |

MoE 与数据受限可同时成立。目标 TPP 可以低于或高于 compute-optimal 配比，训练量轴按目标 TPP 设计（[§5.3](#53-训练量档位)）。

以下内容不在本文范围内：后训练（SFT/RL）自身的 scaling law、原生多模态、蒸馏与合成数据的专项协议。后训练只作为预训练候选的验收条件出现（[§2.2](#22-评价协议)）。

### 1.3 Ladder 的产出

| 产出 | 预测目标 | 章节 |
|---|---|---|
| 超参 scaling law（LR、BSZ、WD） | 目标规模的训练超参 | [§6](#6-超参数搜索与训练配置) |
| Loss scaling law | 最终 training/eval loss | [§8.1](#81-函数形式) |
| Loss 曲线 scaling law | 完整训练曲线与退火效果 | [§8.2](#82-loss-曲线与退火-scaling-law) |
| 退火比例 | LR decay 占总训练量的比例 | [§6.5](#65-lr-schedule)；本文只给出初始参考值，未给出拟合方法 |
| 数据配比与数据重复 scaling law | 最优配比；多 epoch 训练的等效数据量 | [§7](#7-数据-ladder) |
| MoE 稀疏度 scaling law | 固定激活规模下的专家配置 | [§5.6](#56-moe-实验轴)、[§8.5](#85-moe-拟合) |
| 下游任务 scaling law | Benchmark 指标 | [§9](#9-下游任务预测) |

### 1.4 术语与符号

| 术语 | 定义 |
|---|---|
| $N_{\text{body}}$ | Transformer 主干参数量，排除 input embedding 与 output head（[§3.1](#31-参数量口径)） |
| $N_{total}$、$N_{active}$ | MoE 总参数量（含全部专家）；单 token 参与计算的参数量 |
| $D$ | 累计训练 tokens，按参与 loss 计算的 token 计（[§3.3](#33-loss-与-token-口径)） |
| $U$ | Unique tokens |
| $C$ | 训练 FLOPs，按实际架构统计（[§3.2](#32-计算量)） |
| TPP | Tokens per parameter，即 $D/N$；引用外部结果时注明其 $N$ 口径 |
| LR（$\eta$）、BSZ（$B$）、WD（$\lambda$） | 学习率、batch size、weight decay。BSZ 以 tokens 计，引用以序列计的结果时注明 |
| Holdout | 不参与拟合与方案选择、只用于验收的实验点 |
| Fully-Tuned Frontier | 各实验点在充分调优超参时达到的 loss；有限预算下的操作定义见 [§6.1](#61-搜参目标与近优区间) |
| 等效算力倍数 | 达到相同 loss 所需的算力之比，用于把 loss 差换算为算力差（[§2.3](#23-验收阈值与判定规则)） |

## 2. 决策目标与验收标准

### 2.1 决策目标

Scaling Ladder 可支持四类决策：

1. 既定配方预测：预测固定配方在目标规模的表现。有效的 scaling law 不要求每个实验点穷尽调优。
2. 最优资源分配：在预算约束下选择 $N$ 与 $D$ 的分配。
3. 候选方案比较：比较不同架构、优化器或数据方案。
4. 超参数预测：预测跨规模或跨训练量的超参数。

搭建前必须确定目标类型，并明确目标模型、训练阶段、预算约束、推理部署条件，以及何种预测误差会改变最终决策。[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) 分别考察性能竞争力（competitive）与可预测性（predictable），二者需分别验证。目标 2–4 要求各拟合点达到 Fully-Tuned Frontier（[§6.1](#61-搜参目标与近优区间)）；目标 1 按配方缩放规则运行。

### 2.2 评价协议

- 代理指标与最终指标：代理指标（training loss、answer BPB）与最终验收指标（下游 benchmark accuracy）应分别明确。小模型上 loss 的改善不保证目标规模下游指标的改善（[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) §1）。
- 信号可分辨性：各指标在拟合规模上必须有可分辨信号。小模型在数学、代码等任务上可能接近随机水平，部分指标在大规模上饱和，两种情况均无法支持决策。对 accuracy 无法区分的任务，使用 answer BPB 等连续代理指标（[OLMo 3, 2025](https://arxiv.org/abs/2512.13961)）。
- 跨规模排序相关性：代理指标用于方案选择时，需与目标规模的能力指标建立排序相关性（[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3, Appendix A.4）。迁移证据不足时，应标明代理结论的适用范围与待验证内容。
- 目标能力代表性：说明评估任务覆盖哪些交付能力、聚合方式及关键单项退化的容忍条件。高噪声任务可增加采样或单独报告（[Phi-4, 2024](https://arxiv.org/abs/2412.08905) §5；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3.3–3.3.4）。
- 评价版本控制：记录提示模板、生成设置、评分方法及版本，并检查训练与评估数据的重叠。

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

若交付后训练模型（SFT/RL），代表性候选应经过条件可比的后训练验证。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 报告两项改动各自在预训练中影响很小：NoPE 在后训练后出现无法终止的生成，稀疏读取残差分支在后训练后质量退化。[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.5.1 对候选配比完整退火后做快速 instruction tuning 验收。后训练验收除能力分数外，还应检查终止行为、输出长度等交付指标。

### 2.3 验收阈值与判定规则

验收对象分四类，各自需要不同的实验和阈值（本文建议）：

1. 固定配方在目标条件下的最终指标预测误差；
2. 候选方案在目标规模的差值及排序；
3. 资源分配方案相对于可行替代方案的预期损失；
4. 稳定性与关键能力约束。

验收阈值由目标方案间的最小可区分差异与随机种子方差决定，必须在观察 holdout 结果之前确定。[Choshen et al., 2025](https://arxiv.org/abs/2410.11840) 报告文献中推动建模改动的最小相对差异约 4%，随机重启波动可达约 3.5%；Delphi 的 0.2%–0.5% 为特定实验结果，不能作为通用合格线。

Loss 的相对误差不宜在不同规模间使用同一阈值。设拟合得 $L(C)=E+A\,C^{-\gamma}$，loss 差 $\Delta L$ 对应的等效算力倍数近似为

$$\ln\frac{C_2}{C_1}\approx\frac{\Delta L}{\gamma\,(L-E)}$$

$L$ 越接近 $E$，同样的 $\Delta L$ 对应的算力倍数越大。本文建议以等效算力倍数表达阈值，并同时报告绝对 loss 差。

比较候选方案时，必须估计二者在目标条件下差值的不确定性。两套方案共享的误差来源可部分抵消，不能把两个独立误差条直接相加。例如两套方案的预测 loss 相差 0.2%，已有 holdout 的绝对相对误差为 1%，这两个数值不足以判断排序是否可靠。

拟合的不确定性应传播到目标规模的方案差值、最优参数量和训练量上，必要时报告一组近优可行配置及其在质量、成本和稳定性上的差异。部分系数估计不精确时，目标决策仍可能稳定；平均 loss 误差较小时，最优资源分配也可能不稳定。[Gemstones, 2025](https://arxiv.org/abs/2502.06857) §4.2 表明实验点的选择可改变资源分配建议。

判定结果分为三类：选择、追加实验、证据不足暂缓判断。“证据不足”是正式结果。决策表应包含：候选配置、目标条件、预期收益、差值区间、关键约束、待验证条件，以及预先约定的处理规则。

## 3. 测量规范

### 3.1 参数量口径

拟合 $L(N,D)$ 与计算 $C$ 所需的参数量口径不同，必须分开记录。

本文拟合 $L(N,D)$ 时采用 Transformer 主干参数量 $N_{\text{body}}$，排除 input embedding 与 output head。Embedding/head 为 $O(Vd)$，主干为 $O(d^2 n)$，两者的比例随模型尺寸变化。[Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972v3) Appendix G 的消融表明，含 embedding 的口径外推到 25.1B 时误差更高；[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 也排除 embedding。

计算 $C$ 时需计入 output head（每 token 一次 $d\times V$ 矩阵乘）。Input embedding 按 token ID 查表，单独记录其参数与访存开销。权重绑定减少存储，output head 的计算仍需计入。

外部研究的口径不同，引用时必须注明：Porian 的 $N$ 含 output head、排除 input embedding（[Porian et al., 2024](https://arxiv.org/abs/2406.19146) Table 2）；Chinchilla 的 20 TPP 按含 embedding 的总参数量计（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) Appendix F）。

MoE 的每个实验点必须同时记录：

- $N_{total}$ 与 $N_{active}$，两者均按主干口径计，embedding 与 head 单独记录；
- 路由专家总数 $E_{total}$、每 token 激活的路由专家数 $E_{active}$、稀疏率 $S=E_{total}/E_{active}$；
- 共享专家的数量与尺寸、单个专家的尺寸（专家粒度）；
- 路由器与负载均衡配置（[§4.4](#44-moe-结构缩放规则)）。

### 3.2 计算量

常用的 $C\approx6ND$ 未计入注意力矩阵运算。以标准 MHA、$4d$ FFN 为例，按完整 $L\times L$ 注意力矩阵计数：

$$C \approx \left(6 + \frac{L}{d}\right) N_{\text{body}}D + 6VdD$$

$L$ 为序列长度，$d$ 为隐藏维度，$V$ 为词表大小。$L/d$ 项来自注意力，末项为 output head。上述架构下 $N_{\text{body}}\approx12d^2n$（$n$ 为层数）。Causal attention 的 kernel 若跳过被 mask 的部分，注意力项约减半；SwiGLU、GQA、MoE 等也会改变系数。计算量应按实际架构与实际 kernel 统计。

MoE 的矩阵乘计算量按 $N_{active}$ 统计，另计路由器计算。专家并行的 all-to-all 通信不计入 FLOPs，但计入实际训练时间（[§10.5](#105-实际效率与部署约束)）。

### 3.3 Loss 与 token 口径

Ladder 内必须固定以下口径，并写成可检查的定义（本文建议）：

| 项目 | 需说明的内容 |
|---|---|
| 主要 loss | 主任务交叉熵；MoE auxiliary loss、z-loss、MTP 等附加目标分别记录 |
| 评价分布 | 各方案共用固定的评估语料与领域权重，另报重要领域的指标 |
| 聚合 | 按有效预测 token、文档或任务聚合；说明分布式归约方式与分母 |
| 序列处理 | BOS/EOS、文档边界、packing、attention mask、position ID、截断 |
| 数据量 | 已处理 tokens、参与 loss 的 tokens、unique tokens、重复暴露量 |
| 随机性 | 参数初始化、数据抽样与数据顺序的种子分别记录；配对对照的条件 |
| Tokenizer 对照 | 在共同原始文本上报告按字节归一化的指标（如 BPB）与实际计算成本 |

候选方案改变训练数据配比时，各自在训练分布上计算的 training loss 同时反映分布难度的变化，不能单独支持质量排序，必须使用固定的评估分布。

评估频率应按总步数 $T$ 的固定比例设置。固定 step 间隔会使不同训练长度的有效 eval 点数不同，较长轨迹在拟合中权重更大；拟合时需明确 eval 点的采样与加权方式（[§8.3](#83-拟合协议)）。

交付物：评价函数与 token 计数函数的定义，以及从单个 batch 到全局汇总的一份可检查样例。

## 4. 基线配方与缩放规则

### 4.1 变量分类

Ladder 中的变量分三类：

| 变量类别 | 含义 | 示例 |
|---|---|---|
| 固定量 | 所有 Ladder 点取同一值 | 架构、数据版本、优化器类型、seq_len |
| 缩放规则 | 随 $N$ 或 $D$ 按预定规则变化 | LR（幂律或 $\mu$P）、BSZ（$D^{0.4}$ 为待验证先验）、warmup |
| 实验自变量 | 待拟合或待搜索的量 | $N$、$D$、最终的最优 LR 和 BSZ |

Warmup 必须有明确的缩放规则，并记录各实验点 warmup 占总训练量的比例。固定 warmup 步数会使其在小预算实验中占比过高，影响 compute-optimal 指数的估计（[Porian et al., 2024](https://arxiv.org/abs/2406.19146)）。Delphi 使用总训练量的 10% 作为 warmup（[附录 A.3](#a3-delphi-ladder)）。

架构、优化器、数据、精度或 tokenizer 变更后，应先在小规模上验证原有结论的迁移性，再决定局部校准或完整重拟合（[§12.3](#123-ladder-维护)）。

### 4.2 架构与训练配置一致性

同一模型族内，除明确研究的架构变量外，各实验点的以下属性或缩放规则必须一致：

| 属性 | 要求 |
|---|---|
| 架构类型 | 一致，如均为 decoder-only Transformer |
| 归一化 | 一致，如均为 RMSNorm |
| 位置编码 | 一致，如均为 RoPE |
| 激活函数 | 一致，如均为 SwiGLU |
| Embedding 共享 | tied / untied 在 Ladder 内保持一致 |
| 注意力类型 | 一致，如均为 GQA |
| 宽深配置 | 按预定规则缩放（[§4.3](#43-宽深配置)） |
| MoE 路由 | 路由方式、负载均衡方法、共享专家比例一致或按预定规则缩放（[§4.4](#44-moe-结构缩放规则)） |

同一配方内，除明确研究的变量外，以下训练配置保持一致：

| 配置项 | 示例 |
|---|---|
| 序列长度 | 4,096 |
| 训练数据版本 | 同一版本 |
| 优化器 | AdamW |
| 评估集 | 同一 eval set（[§3.3](#33-loss-与-token-口径)） |
| 评估频率 | 按训练进度等比例设置 |
| Tokenizer、初始化、参数化、loss 统计口径 | 同一版本 |

模型族与结构缩放规则应对目标模型具有代理有效性，GQA 分组、head_dim、宽深配置等应与目标模型的设计方向一致。

### 4.3 宽深配置

给定参数量，宽深配置影响实际算力（[§3.2](#32-计算量)），Ladder 应记录结构缩放规则并使用实际 $C$。[Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) 报告给定参数量时宽深比在较大范围内对 loss 影响很小；后续实验表明，不同配置在 benchmark（[Gemstones, 2025](https://arxiv.org/abs/2502.06857) §4.4）和推理能力（[GLM-4.5, 2025](https://arxiv.org/abs/2508.06471)）上可产生差异。Ladder 内应使用一致的宽深缩放规则；涉及结构选择时增加代表性对照。

### 4.4 MoE 结构缩放规则

MoE Ladder 在 [§4.2](#42-架构与训练配置一致性) 的基础上，还需为以下配置规定缩放规则：

- 专家粒度：[Krajewski et al., 2024](https://arxiv.org/abs/2402.07871) 将专家粒度作为独立的缩放变量。在所测范围内，专家尺寸等于 dense FFN 尺寸的常见设置在几乎所有算力预算下均非最优，MoE 相对 dense 的优势随规模增大。
- 共享专家：数量与尺寸在 Ladder 内按固定比例设置。
- 负载均衡：辅助损失系数，或无辅助损失方法中的偏置更新速率，在 Ladder 内按固定规则设置。[DeepSeek-V3, 2024](https://arxiv.org/abs/2412.19437) §2.1.2 按专家负载调整路由偏置，偏置更新速率在前 14.3T tokens 取 0.001、最后 500B tokens 取 0。负载均衡强度同时影响 loss 与专家并行吞吐，改变时视为配方变更。
- Capacity factor 与 token dropping：训练与推理的设置应一致；被丢弃 token 的比例记录为诊断量。DeepSeek-V3 在训练与推理中均不丢弃 token（§2.1.2）。
- 稀疏度：[Abnar et al., 2025](https://arxiv.org/abs/2501.12370) 在不计显存与通信开销的条件下发现，固定训练算力时提高稀疏度并相应增加总参数量，可降低预训练 loss；固定总参数量时，loss 随稀疏度呈抛物线变化，最优稀疏度随模型尺寸与训练算力增大。多数下游任务上，预训练 loss 相近的模型下游表现相近，与稀疏度无关；阅读理解类任务（如 CoQA、SQuAD）上 dense 程度较高的模型表现更好。[Ludziejewski et al., 2025](https://arxiv.org/abs/2502.05172) 将专家数与激活参数量、训练量联合建模，并纳入显存约束，实验规模最大为 2.7B 激活参数、5B 总参数。

### 4.5 词表与数值精度

- 词表大小：[Tao et al., 2024](https://arxiv.org/abs/2407.13623) 在 33M–3B 模型上发现最优词表大小随算力增大，多数公开模型的词表偏小。词表改变 embedding 与 head 的参数量和计算量，也改变每个 token 覆盖的文本长度；比较不同词表时，loss 必须换算为按字节归一化的指标（[§3.3](#33-loss-与-token-口径)）。
- 训练精度：[Kumar et al., 2024](https://arxiv.org/abs/2411.04330) 将精度纳入 scaling law：低精度训练降低有效参数量；训练数据量越大，后训练量化造成的 loss 退化越大。该文的验证范围为 1.7B 参数、26B tokens 以内。Ladder 应使用目标训练精度方案；精度方案变更视为配方变更，需要重新验证（[§10.4](#104-实现一致性验证)）。

## 5. 实验设计与预算

### 5.1 实验矩阵结构

示例布局：

```text
训练量 D →    0.5×    1×     2×     4×
模  130M       ●      ●      ●      ●
型  520M       ●      ●      ●      ●
尺  2.3B       ●      ●      ●      ●
寸  8B         ○      ○      ○      ○  ← holdout（不参与拟合）
N   ↓
```

- 拟合点：较小尺寸用于拟合 Scaling Law。
- Holdout 点：最大尺寸不参与拟合，只用于验证外推精度。

上述尺寸、训练量和完整网格布局均为示例。实验点布局与拟合形式必须联合设计：

- 明确 $N$、$D$、数据重复程度的拟合范围与目标范围，区分沿 $N$、沿 $D$ 和联合外推；
- 根据 [§2.1](#21-决策目标) 的目标选择全网格、IsoFLOP 或 L-shape 等稀疏布局（[§8.1](#81-函数形式)）；
- 预先留出验证点，并约定何种结果触发补充实验（[§5.5](#55-预算分配与追加实验)）。

布局还需检查参数可辨识性。只沿固定 TPP 采样，无法区分 $N$ 与 $D$ 的独立效应。可通过拟合敏感性分析、整体留出一个尺寸、定向补点，检查结论是否依赖采样位置。

### 5.2 尺寸数量、跨度与外推倍数

| 维度 | 建议 | 来源 |
|---|---|---|
| 模型尺寸数 | 覆盖目标外推方向，另设 holdout；增加尺寸可检验拟合稳定性 | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) |
| 尺寸跨度 | 覆盖目标外推区间；部分模型族上 34× 跨度仍可用 | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840) |
| 相邻尺寸比 | 可等比递增，间距按预算选择 | 本文建议 |

外推倍数定义为目标算力与最大拟合点算力之比。公开实例中，[Delphi](https://openathena.ai/blog/delphi/) 在 3e18–3e20 FLOPs 上拟合，holdout 覆盖 3×–333× 外推；[Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.2.1 在 6e18–1e22 FLOPs 上拟合（40M–16B），外推至 3.8e25 FLOPs，约 3,800×。外推倍数越大，函数形式误差与配方稳定性问题在总误差中的占比越高。目标规模远超最大 holdout 时，应安排中等规模试运行（[§10.6](#106-中等规模试运行)）。

### 5.3 训练量档位

训练量档位按目标 TPP 设计，compute-optimal 配比为常用参照。Chinchilla（[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)）的 IsoFLOP 方法：固定每个算力预算 $C$，在不同尺寸的模型上训练，取 loss 曲线的最低点，得到该算力下的最优 $(N, D)$ 组合。

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

即该范围内模型与数据近似等比例增长，不同拟合方法的指数略有差异。Chinchilla 的 $D/N\approx20$ 按含 embedding 的总参数量计；换算到 $N_{\text{body}}$ 时比值更大，最优配比需在自身配方上验证。Llama 3 用同类 IsoFLOP 实验得到 3.8e25 FLOPs 下的最优尺寸约 402B，最终选择 405B（[Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1）。

训练量档位必须覆盖目标 TPP：

- 示例档位 0.5×–4× Chinchilla 配比覆盖 under-trained 到轻度 over-trained 区间，适用于目标接近 compute-optimal 的项目；[Fantastic Optimizers](https://arxiv.org/abs/2509.02046v2) 使用 1×、2×、4×、8× Chinchilla 配比四档。
- 过训练：[Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2) 在 104 个模型中发现不同 $D/N$ 下 loss 对 $C$ 的幂律指数接近，为最高 32× Chinchilla 配比的外推提供了经验证据。[Sardana et al., 2024](https://arxiv.org/abs/2401.00448) 训练的 47 个模型中，TPP 增至 10,000 时质量仍在提升；只用常规 TPP 的数据拟合 Chinchilla 系数，会高估极端 TPP 下新增 tokens 的作用。目标 TPP 超出已验证范围时，必须在接近目标 TPP 的 holdout 上确认外推。
- 推理成本：计入推理需求后，最优尺寸小于 compute-optimal 尺寸，训练量相应增大（[Sardana et al., 2024](https://arxiv.org/abs/2401.00448)）。Llama 3 的 405B 为近似 compute-optimal 尺寸，较小模型的训练时长远超 compute-optimal，以换取同等推理预算下的更好表现（[Llama 3](https://arxiv.org/abs/2407.21783) §1）。部署约束的处理见 [§10.5](#105-实际效率与部署约束)。
- 数据受限：目标 $D$ 超过可用 $U$ 时，训练量轴必须同时标注重复次数，拟合形式需加入重复项（[§7.3](#73-数据受限训练与重复)）。

### 5.4 中间 checkpoint、随机种子与共享轨迹

中间 checkpoint 可纳入拟合，但需检查训练初期点的影响。[Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) 发现排除初期 checkpoint 可降低预测误差；其前 10% 或前 10B tokens 的截断设置不宜直接用于所有训练预算。截断规则必须预先确定，或将模型尺寸划分为拟合、验证和测试三段，在验证段上选择（[Lourie et al., 2026](https://arxiv.org/abs/2608.11859)）。截断规则与 warmup 规则分别确定。

同一轨迹的多个 checkpoint 存在序列相关性，不能替代目标外推方向上的独立实验（[§8.3](#83-拟合协议)）。随机种子方差必须纳入实验设计。

共享 stable 轨迹的 WSD 分支应分别记录：实际累计成本、各分支的有效训练预算、共享起点。成本不重复计算，共享前缀的分支也不能按独立运行计数。

### 5.5 预算分配与追加实验

搜索、种子、评估、验证、追加实验与失败重跑均计入总预算。本文建议的分配流程：

1. 先做少量试验，估计种子噪声、吞吐和候选方案间的差异；
2. 按试验结果分配搜索、拟合、独立复验、最终 holdout、评估和预备预算；
3. 标明每组实验要消除哪项决策不确定性。

追加实验的决策示例：

- 种子波动是误差的主要来源时，增加重复运行；
- 候选函数形式在目标区域的预测分歧较大时，补充对应方向的实验点；
- 候选方案在较长训练预算下排序发生变化时，延长代表性运行。

增加较大模型与增加小模型随机种子，哪一项更有价值取决于具体条件（[Choshen et al., 2025](https://arxiv.org/abs/2410.11840)）。公开文献中没有可跨项目通用的预算比例。

### 5.6 MoE 实验轴

MoE Ladder 按决策需要选择实验轴，各轴分别设计：

1. 规模 Ladder：固定稀疏率与专家粒度，改变 $N_{active}$ 与 $D$；
2. 稀疏度 Ladder：固定 $N_{active}$、$D$ 和 $E_{active}$，只改变 $E_{total}$；
3. 粒度 Ladder：固定 $N_{active}$ 与 $N_{total}$，改变单个专家尺寸与专家数（[Krajewski et al., 2024](https://arxiv.org/abs/2402.07871)）。

[Kimi K2, 2025](https://arxiv.org/abs/2507.20534) 在固定 $N_{active}$ 和 FLOPs 下改变 $E_{total}$ 拟合 sparsity scaling law：稀疏率从 8 增至 48 时，达到相同目标 loss 所需的 FLOPs 持续下降，通信和推理复杂度随之增加。

各轴的拟合变量不能混用。不需要全交叉搜索。规模 Ladder 中应包含同 $N_{active}$ 或同 $C$ 的 dense 对照点，用于判断 MoE 相对 dense 的收益随规模的变化（本文建议）。

## 6. 超参数搜索与训练配置

本章标定充分调优的 $(\eta,B,\lambda)$，并写成可随 $N$、$D$（或 $C$）外推的函数。既定配方预测（[§2.1](#21-决策目标) 目标 1）按配方缩放规则运行。§6.1 说明搜参目标；§6.2–§6.4 给出各超参随规模迁移的先验；§6.5–§6.6 为 LR 调度与 MoE 超参的设置；§6.7 为搜索流程；§6.8 为配方之间的比较方法；§6.9 为收尾流程。

### 6.1 搜参目标与近优区间

[Lourie et al., 2026](https://arxiv.org/abs/2608.11859) 与 [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) 表明：小模型在次优超参下性能下降明显，Scaling Law 只在 Fully-Tuned Frontier 上显现；搜参不充分会改变幂律曲线的形状，导致目标规模的预测出现偏差。

大模型实验中观察到的近优区间较宽。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320v1) 在 156B-A7B 上测试 LR 乘除 $\sqrt{2}$、BSZ 增加 25%，最终 training loss 的差不超过 $7\times10^{-4}$。

资源分配原则（目标 2–4）：算力集中用于小模型的穷举搜索，确保拟合点位于 Fully-Tuned Frontier 上；大模型先在外推值附近局部确认，必要时扩大搜索（[§6.7](#67-搜索流程与停止规则)）。

有限预算下，“位于 Fully-Tuned Frontier”采用以下操作定义（本文建议）：最优点不位于搜索边界；在最优点邻域的局部联合扰动下，loss 变化不超过预定容差，该容差不小于种子方差。结论表述为“在已搜索范围与预算内达到指定近优容差”。

### 6.2 参数化与优化器迁移

- 宽度方向：$\mu$-Transfer（[Yang et al., 2022](https://arxiv.org/abs/2203.03466v2)）支持宽度方向的 LR 迁移，含非零 WD 的情形见原论文 Appendix G.1.2 与 [Power Lines](https://arxiv.org/abs/2505.13738v2)。BSZ、训练长度方向只有有限实验。可与超参幂律结合使用。
- 深度方向：[Bordelon et al., 2023](https://arxiv.org/abs/2309.16620) 将残差分支按 $1/\sqrt{\text{depth}}$ 缩放并结合 $\mu$P，在 CIFAR-10 与 ImageNet 上的 ResNet 和 ViT 中观察到跨宽度与深度的超参迁移。[Tensor Programs VI (Yang et al., 2023)](https://arxiv.org/abs/2310.02244) 对每个残差块只含一层的网络给出 Depth-$\mu$P；残差块含多层时（如 Transformer），该文指出所有无限深度参数化均存在局限。语言模型上的深度方向迁移缺少直接证据，宽深规则改变深度时，必须在代表性尺度重新确认超参。
- 优化器：[Liu et al., 2025](https://arxiv.org/abs/2502.16982) 为 Muon 加入 WD，并将更新的 RMS 缩放到 AdamW 常见的 0.2–0.4 范围（取 0.2），使 AdamW 调好的 LR 与 WD 可以直接复用。[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 切换架构与 Muon 后最优 LR 和 BSZ 均发生偏移。更换优化器后，必须重新标定超参 scaling law，或先在小规模上验证原系数的迁移性。

### 6.3 LR 与 BSZ 的 Scaling Law

单一 $D/N$ 配比时，可用 $C$ 的幂律：

$$\eta_{opt} = a \cdot C^{-\alpha}, \quad B_{opt} = b \cdot C^{\beta}$$

[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) 提供的拟合系数可作参考。覆盖多档 $D/N$ 时，必须将 $N$、$D$ 分开建模：

| 方案 | $\eta_{opt}$ | $B_{opt}$ | 适用条件与局限 |
|---|---|---|---|
| Step Law ([Li et al., 2025](https://arxiv.org/abs/2503.04715v3)) | $c\cdot N^{-\alpha}D^{\beta}$ | $0.58\,D^{0.571}$ | 可跨 $D/N$ 建模；3,700+ 模型验证；$B_{opt}$ 与 $N$ 无关的假设经回归检验（附录 A.5），目标范围仍需验证 |
| Power Lines ([Bergsma et al., 2025](https://arxiv.org/abs/2505.13738v2)) | 由 timescale $\tau$ 联合约束（[§6.4](#64-weight-decay)） | $\propto D^{0.4}$，所测范围内对 $N$ 的依赖较弱 | 所测配方下的近似拟合；仍需检查 LR 稳定性与 BSZ 效率范围 |
| Token Horizons ([Bjorck et al., 2024](https://arxiv.org/abs/2409.19913)) | $c \cdot N^{-\alpha} D^{-\beta}$ | — | 固定模型尺寸与 BSZ 时，peak LR 随训练长度衰减 |

Token Horizons 与 Step Law 中 $\eta_{opt}$ 对 $D$ 的指数符号相反（$D^{-\beta}$ 与 $D^{+\beta}$）。两者的实验设置不同：Token Horizons 固定 BSZ，Step Law 对每个 $D$ 联合搜索 BSZ。该设置差异能否完全解释符号差异尚未验证。应结合自身配置选择候选形式，并在 holdout 上验证。

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

两项 $B_{opt}$ 拟合的指数相近，可据此初始化 BSZ 迁移规则，在自身数据上验证。

Power Lines 同时测得 $B_{crit}\propto D_{min}^{0.5}$，$D_{min}$ 为达到目标 loss 所需的最少 tokens。$B_{crit}$ 为 token 效率与步数之间的权衡转折点：在该文的双曲线模型中，$B=B_{crit}$ 时达到同一 loss 约需 $2D_{min}$ tokens；继续增大 $B$，步数下降，token 与算力开销增加。实际训练时间还取决于硬件利用率，需系统实测。

{% include figure.liquid
  path='assets/img/pretrain-scaling/cerebras-bcrit-hyperbola.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 5. 达到相同目标 loss 所需训练 tokens 与 steps 的双曲线关系，左、右分别为 610M 和 1.7B 模型，颜色表示 loss。B_crit 表示 token 效率与步数之间的权衡转折。<a href="https://arxiv.org/pdf/2505.13738v2#page=6">来源：Bergsma et al., 2025, v2, Fig. 4</a>。'
  alt='610M 和 1.7B 模型的两幅 tokens 对 steps 曲线图：不同目标 loss 对应不同双曲线，颜色表示 loss，曲线标出 B_crit。'
  avoid_scaling=true
  zoomable=true
%}

[Schaipp, 2026](https://arxiv.org/abs/2607.01487v1) 将近优 batch 区间定义为算力损失约 5% 以内的范围，所测宽度约 4 倍；在该文对数对称的拟合模型下对应 $[B_{opt}/2,2B_{opt}]$，可作为局部搜索的起点。

BSZ 的单位与训练过程中的变化也需统一：

- 改变序列长度时，按 tokens 保持 BSZ，同时检查注意力计算量与数据打包方式的变化。
- 部分目标训练使用 batch size ramp，例如 Llama 3 405B 在训练中分阶段将 BSZ 从 4M tokens 增至 16M tokens（[Llama 3](https://arxiv.org/abs/2407.21783) §3.4.1）；Nemotron-4 15B 也使用 BSZ 递增（[Nemotron-4, 2024](https://arxiv.org/abs/2402.16819)）。Ladder 未包含 ramp 时，必须验证 ramp 对 loss 曲线和超参外推的影响。

### 6.4 Weight Decay

若干公开配方固定 WD：[Kimi K2, 2025](https://arxiv.org/abs/2507.20534v2) 使用 $\lambda=0.1$；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961v1) 使用 AdamW，embedding 不做 decay；[Qwen3, 2025](https://arxiv.org/abs/2505.09388) 未报告 WD scaling。

最优 WD 取决于训练量和评价目标。[Han et al., 2026](https://arxiv.org/abs/2602.11137v2) 发现预训练 loss 偏好的 WD 随 TPP 增大而降低，较强的 WD 可能有利于后训练可塑性。复现配方时沿用其 WD；研究充分调优或跨 TPP 关系时，应检查 WD 的影响。

在 AdamW 设置下，WD 可与 LR、BSZ 通过 timescale $\tau = B/(\eta\lambda D)$ 联合校准：[Power Lines](https://arxiv.org/abs/2505.13738) 改变 $B$ 时固定 μP 下的 $\eta$，调整 $\lambda$ 以维持最优 $\tau$；LR 还受最大稳定学习率的限制（[Power Lines §2.4](https://arxiv.org/abs/2505.13738)）。

### 6.5 LR Schedule

两类常用调度：

- Cosine：短 warmup 后平滑衰减至 0 或较小的残余值。多阶段训练中各阶段的最小 LR 可能不同。
- WSD（Warmup-Stable-Decay）：便于复用 stable 轨迹，并在选定预算处退火。[River Valley (Wen et al., 2024)](https://arxiv.org/abs/2410.05192v3) 在特定损失几何假设下解释了 stable 和 decay 阶段的作用。

退火比例约 10%–20% 可作为初始参考（[Tissue et al., 2024](https://arxiv.org/abs/2408.11029v2)），目标预算上仍需验证。[Wang et al., 2025](https://arxiv.org/abs/2512.13705) 研究了退火策略的跨规模迁移。

### 6.6 MoE 训练超参

MoE 的 LR 与 BSZ 规则应在 MoE 配方上标定，dense 配方的系数需验证后才能复用（本文建议）。路由相关超参，包括负载均衡系数或偏置更新速率、路由器的数值精度、capacity factor，按 [§4.4](#44-moe-结构缩放规则) 的固定规则设置，并在代表性尺度检查专家负载分布与 token 丢弃比例。

### 6.7 搜索流程与停止规则

应优先搜索 LR 和 BSZ，再检查 schedule 与 WD；$\epsilon$、$\beta_2$ 等稳定性超参可在基线和代表性尺度验证后固定。可采用坐标下降逐步缩小范围（尺寸为示例）：

1. 小模型（约 130M）网格搜索，确定核心区间；
2. 中模型（约 500M）检验幂律趋势；
3. 大模型局部确认：LR 先在外推值乘除 $\sqrt{2}$ 范围内确认，BSZ 先在 $[B_{opt}/2,2B_{opt}]$ 内确认，检查边界后决定是否扩展（[§6.1](#61-搜参目标与近优区间)、[§6.3](#63-lr-与-bsz-的-scaling-law)）。

BSZ 的搜索维度可借助 [§6.3](#63-lr-与-bsz-的-scaling-law) 的迁移规则削减。在已选定 LR 的条件下，可通过调整 WD 搜索最优 timescale $\tau$（[§6.4](#64-weight-decay)），减少联合搜索的成本；等 $\tau$ 不保证不同的 $(\eta, \lambda, B)$ 组合性能等价，仍需检查 LR 稳定性与 BSZ 效率范围。

从多次带随机波动的试验中选出最低 loss，会使选中配置的收益估计偏高（[Cawley & Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)）。处理规则（本文建议）：

- 记录每个拟合点的搜索范围、边界结果、局部联合扰动结果、重复运行方差与预算消耗；
- 用于挑选超参的评价与用于报告收益的评价分开；最终候选使用新的随机种子复验；
- 最终规模的 holdout 保留验收用途，不参与搜参；
- 各候选方案的调参次数与算力分别列出。

提前停止 trial 时，不能默认用早期排名淘汰后期可能有效的配置，退火期间可能出现排名翻转（[§6.8](#68-配方比较)）。主动停止、算法发散、基础设施故障与实现错误的运行必须分别标记状态，不能统一记为某个终点 loss，也不能不加记录地删除（[§12.1](#121-实验记录)）。

### 6.8 配方比较

比较优化器、LR 调度等配方时，需满足以下条件：

- 各方案分别调优，并报告各自的搜索预算。[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) 发现 AdamW 基线调优不足会夸大新优化器的收益。[Kimi K3, 2026](https://arxiv.org/abs/2607.24653) §3.2 观察到，同一模型尺寸与训练量下，cosine 与 WSD 的最优峰值 LR 和 BSZ 差异显著；分别做 scaling law 搜索后，cosine 的最终 loss 始终低于 WSD，因此采用 cosine 作为默认调度。
- 在可比预算下比较退火终点。Fantastic Optimizers 观察到 loss 排名在退火期间翻转，stable 阶段中途的排名不能直接替代最终排名；经过验证的代理筛选除外。
- 在多个尺寸上比较。Fantastic Optimizers 的 8× Chinchilla 设置中，token 效率的改善随模型增大而缩小（0.1B 的 $1.4\times$ 降至 1.2B 的 $1.1\times$），且不等同于训练时间的加速比。

### 6.9 收尾流程

若交付流程包含以下步骤，Ladder 的对应阶段也应执行，或使用经过验证的代理，以保持拟合目标与交付目标一致：

- Continued training：主训练后切换数据配比并衰减 LR（[Nemotron-4, 2024](https://arxiv.org/abs/2402.16819)；[OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3)）。预测最终性能时需纳入该阶段（[§11.1](#111-分阶段-ladder)）。
- Weight averaging：对共享初始化的 checkpoint 做权重平均，包括独立微调后合并（[Model Soups](https://arxiv.org/abs/2203.05482v3)）和同一轨迹的滑动窗口平均（[LAWA](https://arxiv.org/abs/2209.14981)；[Sanyal et al., 2024](https://arxiv.org/abs/2306.03241)）。[Model Merging (ByteDance Seed, 2025)](https://arxiv.org/abs/2505.12082v3) 在 1.3B 和 13B 对照中发现，WSD stable 阶段的 PMA 可接近退火终点的下游表现；合并窗口和起点仍需验证。

## 7. 数据 Ladder

数据 Ladder 固定模型结构与训练配置，改变数据变量，用小规模实验支持目标规模的数据决策。本章讨论三类决策：数据源取舍（[§7.1](#71-数据质量与数据源评估)）、配比（[§7.2](#72-数据配比-ladder)）、数据受限条件下的重复（[§7.3](#73-数据受限训练与重复)）。数据清洗与过滤流程不在本文范围内。

### 7.1 数据质量与数据源评估

[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) 在所比较的训练语料中观察到：

- 质量较高的语料对应更偏向参数量的最优算力分配；
- 不同数据集的 scaling law 参数差异显著，不能直接复用；
- 控制配方和评估分布时，最优 $N/D$ 的差异可辅助判断数据质量。

因此数据版本必须作为 Ladder 的固定项记录，版本变化后必须重新验证 scaling law 参数。Kimi K3 的架构、数据与训练配方同时变化后，重新做了 scaling law 研究，重调 BSZ、LR、TPP 与模型形状（[Kimi K3, 2026](https://arxiv.org/abs/2607.24653) §3.2）。

评估新数据源时不需要重跑完整矩阵：

- [OLMo 2, 2025](https://arxiv.org/abs/2501.00656) 的 micro-annealing 从指定 checkpoint 出发，将候选数据与通用数据混合做短期退火，判断增益。该方法的结论限于所用 checkpoint 的起点和阶段，不能推断全程的数据排序。
- [Kimi K3](https://arxiv.org/abs/2607.24653) 的各领域采样率由较小模型上的消融确定（§3.1）。
- [Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) 发布的法律领域合成数据，在加入 Nemotron 3 Nano 预训练的消融中，使 LegalBench 代理评测的平均准确率由 64.6 提高到 74.7。

### 7.2 数据配比 Ladder

数据配比 Ladder 固定模型结构、参数量、训练量和优化器，只改变数据混合向量 $\mathbf{w}=(w_1,\ldots,w_k)$。配比实验选择训练分布，规模 Ladder 在该分布上拟合 $N$、$D$ 与 loss；配比随训练量调整时，两类实验需要迭代。从头预训练与继续训练的数据选择分开处理，各自标明迁移范围。

执行流程如下。代理实验的设计数值来自 [Olmix, 2026](https://arxiv.org/abs/2602.12237)，以 1B 目标模型为参照；该方法用于 [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.4.4。

1. 定义配比变量。按来源、领域等属性划分数据桶，记录各桶去重后的可用 tokens；调整过滤器或阈值后，重新检查容量与重复条件（[Marin 数据流程, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/)）。配比维度可细化到实例属性，如教育价值、领域、语言、安全（[Qwen3, 2025](https://arxiv.org/abs/2505.09388)）。训练分阶段切换配比时，各阶段分别设计配比实验（[§11.1](#111-分阶段-ladder)）：[Nemotron 3 Super, 2026](https://arxiv.org/abs/2604.12374) 在 25T tokens 训练中，前 80% 偏重多样性，后 20% 偏重高质量数据。
2. 设定基线。基线为按比例采样与 [UniMax](https://arxiv.org/abs/2304.09151)；[Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) 指出，执行不当的学习配比可能差于这两类基线。
3. 设计代理实验。
   - 尺寸：5× Chinchilla 训练量下，15M 及以上代理与 1B 目标的 Spearman 相关高于 0.89，1M 代理为 0.73（[Olmix](https://arxiv.org/abs/2602.12237)）；[OLMo 3](https://arxiv.org/abs/2512.13961) 采用 30M 代理、3B tokens。
   - 训练量与数据池：按目标训练的重复条件同比例缩放。[Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) 按激活参数量设置代理预算，并按相同比例缩小各桶的数据池，使代理的重复次数与目标 ladder 一致（每个激活参数 791 tokens）。
   - 数量与采样：所需代理数量随领域数 $m$ 线性增长，使用 log-linear 回归时不少于 $3(m+1)$ 个；配比从以自然分布为中心的 Dirichlet 分布采样，主题级领域用稀疏采样，来源级领域用稠密采样。
   - 替代方法：[DeMix, 2026](https://arxiv.org/abs/2602.00747) 为每个候选数据集训练一个组件模型，以组件模型的加权合并代替按配比训练的代理，其排序一致性高于小规模训练的代理。
4. 拟合回归。[Olmix](https://arxiv.org/abs/2602.12237) 中 log-linear 模型的下游结果最好；不同回归模型族在不同代理数量下各有优势，已有文献的结论因此不一致。每个任务单独拟合，留出配比上的拟合相关为 0.983，聚合指标拟合为 0.866。回归形式需能表示非单调响应：[Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) 观察到单个桶权重增加时，loss 先下降，随后饱和，重复过多时回升。单个桶权重与指标的相关系数不代表该桶的独立效应，因为权重之和为 1；[Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) 中该相关还随评测集、随预训练与 cooldown 阶段变化。
5. 求解配比。约束 $w_j \le k N_j / R$（$N_j$ 为数据桶 $j$ 的可用 tokens，$R$ 为目标训练 tokens，$k$ 为重复上限）会显著改变求得的配比（[§7.3](#73-数据受限训练与重复)）。精确求解加向自然分布的 KL 正则（$\lambda=0.05$）效果最好。
6. 跨规模确认。目标训练前，在一组更大的模型上比较入选配比与基线（[Marin](https://openathena.ai/blog/marin-data-pipeline-overview/)）。该流程迁移到更大模型的效果仍待验证。
7. 数据更新后重估。[Olmix](https://arxiv.org/abs/2602.12237) 的 mixture reuse 保留未受影响数据桶的相对比例，只重算受影响的部分。在 5 次更新、最终 64 个领域、1B 模型训练 100B tokens 的设置中，该方法相对自然分布提升 11.6%，达到完全重算增益的 95%，代理数量减少 74%。[OLMo 3](https://arxiv.org/abs/2512.13961) 的配比经过 3 轮该流程。

交付物：数据清单、候选配比表、可行域约束、回归验收结果、跨规模确认结果与最终采样配置之间的可追溯关系。

### 7.3 数据受限训练与重复

数据受限时，unique tokens 总量 $U$ 必须在设计阶段作为约束写入实验矩阵，训练量轴同时标注累计训练 tokens、unique tokens 与重复次数。去重范围需与重复次数的统计口径一致：[Marin 数据流程, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/) 在全部来源上做全局去重，以便将 epoch 数作为受控变量，其中最大的跨来源重叠来自 [Nemotron-CC](https://arxiv.org/abs/2412.02595) 与其合成改写版本。

[Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264v5) 假设重复数据的边际收益指数衰减，给出等效数据量：

$$D^{\prime} = U_D + U_D \cdot R_D^{\ast} \left(1 - e^{-R_D / R_D^{\ast}}\right)$$

$U_D$ 为 unique token 数，$R_D$ 为额外重复次数（$R_D=0$ 即单 epoch），$R_D^\ast$ 为拟合的衰减尺度。$R_D\to\infty$ 时 $D^\prime\to U_D(1+R_D^\ast)$。该形式描述收益饱和；若 loss 随重复回升，还需建模过拟合项。

{% include figure.liquid
  path='assets/img/pretrain-scaling/muennighoff-epoch-returns.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 6. 重复数据的边际收益（4.2B 模型，12B unique tokens）。前几轮重复接近新数据的收益，随后边际收益下降，约 40 epoch 后趋于饱和。<a href="https://arxiv.org/pdf/2305.16264v5#page=1">来源：Muennighoff et al., 2023, v5, Fig. 1 左</a>。'
  alt='4.2B 模型在 12B unique tokens 上的重复训练曲线：横轴为累计训练 tokens 和 epochs，纵轴为 final test loss；约 40 epochs 后收益趋于饱和。'
  avoid_scaling=true
  zoomable=true
%}

- 重复次数的上限取决于数据与配方。[Yan et al., 2025](https://arxiv.org/abs/2511.13421v2) 在线性回归假设下得到最优重复次数随样本数对数增长；[Lovelace et al., 2026](https://arxiv.org/abs/2605.01640) 发现较大的参数量、较少的 unique tokens 和较多的重复次数共同加剧过拟合。
- [Xue et al., 2023](https://arxiv.org/abs/2305.13230v2) 的受控实验中，重复导致的过拟合主要随参数量变化；增加数据量可缓解，同等数据量下提高质量未带来同样的改善。Dropout 对重复过拟合的改善较明显（v2 Table 4），较大模型需重新调节 dropout rate。
- 数据受限时，模型尺寸的选择需同时考虑重复带来的收益下降：固定 $U$ 时，增大 $N$ 会加剧过拟合，compute-optimal 配比需在含重复项的拟合形式下重新求解。
- 稀缺数据与大量通用数据混合训练时，可承受的重复次数高于单一来源训练。[Sedova et al., 2026](https://arxiv.org/abs/2605.12715) 在 2,000 余次训练中发现，稀缺目标语料可重复 15–20 次，最优重复次数取决于目标数据量、算力和模型尺寸；其含重复项的配比 scaling law 在小规模拟合后可外推到更大规模。
- 各数据源的最大重复次数应作为配比约束（[§7.2](#72-数据配比-ladder)）。[Kimi K2.5, 2026](https://arxiv.org/abs/2602.02276) 从 [Kimi K2](https://arxiv.org/abs/2507.20534) 接近结束的 checkpoint 继续做联合预训练时，控制每个数据源的最大 epoch 数；[OLMo 3](https://arxiv.org/abs/2512.13961) 的质量感知上采样只对高质量数据重复，最大重复 7 次，在模拟的数据受限对照中优于按阈值过滤（§3.4.4、附录 A.2.5）。
- 改写可替代部分重复。[Kimi K2, 2025](https://arxiv.org/abs/2507.20534) 在早期 checkpoint 上以 SimpleQA 比较三种设置：原文训练 10 epoch 为 23.76，改写 1 次训练 10 epoch 为 27.39，改写 10 次各训练 1 次为 28.94；推广到其他知识语料时每个语料最多改写 2 次（§2.2）。[Kimi K3](https://arxiv.org/abs/2607.24653) 沿用该改写方法（§3.1）。改写数据与原文需分开统计 unique tokens。

## 8. Loss Scaling Law 拟合

拟合前必须确认三项设置（[Porian et al., 2024](https://arxiv.org/abs/2406.19146)）：

1. 算力统计计入 output head，与拟合用的参数量口径分开记录（[§3.1](#31-参数量口径)）；
2. warmup 按缩放规则设置（[§4.1](#41-变量分类)）；
3. 目标 2–4 各尺寸搜索至 Fully-Tuned Frontier（[§6.1](#61-搜参目标与近优区间)）；目标 1 按配方运行。

### 8.1 函数形式

Chinchilla 的参数化形式为 $L(N,D)=E+A/N^\alpha+B/D^\beta$，加性形式隐含 $\partial^2L/\partial N\partial D\equiv0$。[Skaling (Videau et al., 2026)](https://arxiv.org/abs/2608.07222v1) 在所分析数据上发现负的混合偏导数，增加外指数 $k$：

$$L(N,D)=\left(\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}\right)^{k}+E$$

$k=1$ 时退化为 Chinchilla 形式；该文拟合得 $k\approx0.31$–$0.45$。Skaling 只多一个参数，compute-optimal 闭式解的代数形式不变。

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-vs-chinchilla-error.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 7. 两种形式在同一 (N, D) 网格上的预测残差，每点为一次训练。左、中：带符号百分比误差，共用色标；Chinchilla 的残差呈鞍形并向四角增大，Skaling 全网格接近零。右：两者误差之比，Skaling 在 76% 的配置上更准，中位数 2.2 倍。<a href="https://arxiv.org/abs/2608.07222">来源：Videau et al., 2026, Fig. 1</a>。'
  alt='三张 (N, D) 网格散点图：左为 Chinchilla 的带符号百分比误差，中为 Skaling 的同类误差，右为两者误差之比。'
  avoid_scaling=true
  zoomable=true
%}

[Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972) 令数据侧的系数与指数均依赖 $N$，共九个参数。Skaling 六个参数，在 Farseer 与 SK-Grid 两套数据上的插值和单轴外推误差最低（[Skaling](https://arxiv.org/abs/2608.07222) Table 1）。可将 Skaling 作为默认候选，在自身数据上与 Chinchilla 形式比较；重新拟合可能改变 compute-optimal 配比，需由 holdout 验证。

固定 $D/N$ 时，可用一维近似 $L=G(M)/C^\gamma+E$（$M=D/N$）；跨配比仍需对 $N$、$D$ 分别建模。

布局必须与函数形式共同验证。在 Skaling 论文的两套实验中，L-shape 布局的算力约为完整网格的 1/5–1/10；Chinchilla 加性形式在这些稀疏布局上误差明显增加。算力受限时可评估 L-shape 布局。

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

[Tissue et al., 2024](https://arxiv.org/abs/2408.11029) 将 loss 表示为 step 的函数：

$$\hat L(s) = L_0 + A \cdot S_1(s)^{-\alpha} - C \cdot S_2(s)$$

$S_1(s)=\sum_{i\leq s}\eta_i$ 为累计学习率面积，$S_2(s)$ 为带遗忘核的累计退火量。该式以 schedule 为输入，可利用同一轨迹的多个 eval 点拟合整条 loss 曲线，也可预测 continued training 中不同 re-warmup LR 下的曲线（原文 §4.7）。跨规模或跨数据分布复用系数仍需验证。外推跨度过大时会出现系统性偏差，需要补充实验点。

### 8.3 拟合协议

以下选择必须在拟合前固定并记录（本文建议）：

- 拟合目标：拟合 $L$ 还是 $\log L$；拟合 $E$ 还是固定 $E$。
- 目标函数：平方误差或稳健损失（如 Huber）；各尺寸、轨迹与 checkpoint 的权重。
- 数值设置：参数约束、变量变换、初始化方式、多起点优化与收敛检查。
- 异常值：预先规定的处理规则。
- 函数形式选择：在开发验证集上比较候选形式，holdout 不参与选择。
- 不确定性：bootstrap 的重采样单位需符合数据的依赖结构，共享前缀的多个分支不能视为独立运行。参数置信区间、单次未来运行的预测区间、函数形式不确定性分别报告。重采样无法消除函数形式错误或超出验证范围的误差。

参数拟合与决策计算应一起复验：对每次有效拟合重新计算目标配置与候选差值，观察结论是否稳定（[§2.3](#23-验收阈值与判定规则)）。只报告参数的标准误差，无法支持方案选择。

[Besiroglu et al., 2024](https://arxiv.org/abs/2404.10102) 的复现表明，参数化拟合与置信区间的设置会影响结论；[(Mis)Fitting (Li et al., 2025)](https://arxiv.org/abs/2502.18969) 讨论了拟合细节缺失对复现的影响。

交付物：读取实验表的拟合脚本，输出残差、敏感性分析、目标预测与决策表，并附一个使用公开数据或明确标注的合成数据的完整示例。

### 8.4 拟合诊断与失效处理

拟合完成后需检查：

- 残差结构：残差是否随 $N$、$D$ 或训练阶段呈系统性变化；结论是否依赖少数点或特定函数形式。
- Checkpoint 相关性：同一轨迹的多个 checkpoint 存在序列相关性，不能按点数判断独立信息量（[Delphi](https://openathena.ai/blog/delphi/) 按 IsoFLOP 最优点做 bootstrap）。
- 不确定性分离：分别报告种子方差、评估方差和拟合不确定性。
- 截断与验证划分：截断规则预先确定或用开发验证集选择；holdout 保持验收用途（[§5.4](#54-中间-checkpoint随机种子与共享轨迹)）。
- 失效处理：误差过大时，列出待追加的实验和暂时不能作出的决策。没有证据的解释记为“原因未查明”。
- 产物：配方版本、实验记录、拟合方法、预测区间和未解决问题（[§12.1](#121-实验记录)）。

若决策涉及下游能力，还需验证任务指标的预测（[§9](#9-下游任务预测)）；不同样本的 scaling 规律可能不同。

### 8.5 MoE 拟合

- 固定稀疏率与粒度的规模 Ladder，可用 $N_{active}$ 替代 $N$ 作为拟合自变量，同时记录 $N_{total}$。
- 稀疏率或粒度变化时，$S$（或 $E_{total}$）与粒度必须作为独立变量进入拟合形式（[Krajewski et al., 2024](https://arxiv.org/abs/2402.07871)；[Ludziejewski et al., 2025](https://arxiv.org/abs/2502.05172)）。
- 规模 Ladder 与稀疏度 Ladder 的数据不能混合拟合同一组系数（[§5.6](#56-moe-实验轴)）。
- dense 对照点单独拟合，用于比较 MoE 与 dense 的等效算力倍数随规模的变化。

## 9. 下游任务预测

### 9.1 方法路线

| 方法路线 | 核心思路 | 代表工作 | 局限 |
|---|---|---|---|
| Loss → Performance | 先预测 loss 或 perplexity，再映射到下游指标 | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2)（错误率对 perplexity 的幂律，等价于对交叉熵 loss 的指数关系）；[Delphi](https://openathena.ai/blog/delphi/)（sigmoid 映射） | 映射依赖任务与评估协议 |
| 算力或 $(N,D)$ → Task NLL → Acc | 两阶段：先预测任务上正确答案的 NLL，再拟合 NLL 到 accuracy 的映射 | [Bhagia et al., 2024](https://arxiv.org/abs/2412.04403)（OLMo Task Ladder）；[Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1（sigmoid 映射，外推至 405B） | 任务间噪声与预测误差差异大 |
| End-to-End | 直接对任务指标随算力的变化建模；可按难度分组 | [COD (Xu et al., 2026, v4)](https://arxiv.org/pdf/2502.17262v4)（难度特征聚类）；[GPT-4 Technical Report](https://arxiv.org/abs/2303.08774v6)（HumanEval 难度分桶） | 需要可分辨的评估信号；适用范围取决于方法与评分协议 |

Llama 3 的第一阶段只用 1e22 FLOPs 以内的 scaling law 模型，拟合正确答案的归一化 NLL 与训练 FLOPs 的线性关系；第二阶段同时使用 scaling law 模型与 Llama 2 模型，拟合 NLL 与 accuracy 的 sigmoid 关系。在 ARC-Challenge 上，对 405B 的预测略低于实测值（[Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1）。[Delphi](https://openathena.ai/blog/delphi/) 的 IsoFLOP 加 sigmoid 映射可作为初始方案。不同路线的优劣需在相同任务和协议下比较。

### 9.2 COD 框架

[COD (Xu et al., 2026, v4)](https://arxiv.org/pdf/2502.17262v4) 的四个阶段：

1. 聚类：多个小模型对每道题多次采样，以平均正确率为难度特征，按难度聚类；
2. 拟合：每个聚类拟合 $E[\mathrm{Acc}(C)] = g + (1-g) \cdot e^{-aC^{-b}-c}$；
3. 外推：筛选可靠聚类，代入目标算力，按样本数加权平均；
4. 映射：校准可外推子集到全评估集的映射曲线。

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

适用条件：

- 样本过少时聚类指标不稳定；新架构或新数据分布下需验证难度特征与映射的稳定性。
- 主预训练实验采用 warmup 后恒定 LR。Continued training 实验另含数据变化与退火，小模型需匹配两阶段的分布与 TPP（v4 Appendix D–E）。
- v4 用 dense 聚类预测激活参数 32B 的 MoE 目标，平均与最大绝对误差分别为 3.11 和 8.11 个百分点，提供了有限的跨架构证据（§5.3.1、Table 2）。
- CoT 已有经验预测结果，理论尚未充分覆盖非唯一答案和推理路径（Appendix H）。

### 9.3 可预测性的限制

[Schaeffer et al., 2024](https://arxiv.org/abs/2406.04391) 分析了下游指标难以预测的原因：多选题 accuracy 由正确选项与特定错误选项上的概率质量共同决定，从 loss 到 accuracy 的逐步变换会削弱与算力的统计关系，仅预测正确答案的概率会丢失这部分信息。

因此，下游预测必须满足 [§2.2](#22-评价协议) 的信号可分辨性要求，并在接近目标规模的 holdout 上单独验证。下游预测未通过验证时，决策只能依据 loss 预测与代理指标，并标明该限制。

## 10. 外推验证与启动决策

### 10.1 Holdout 验证

必须在目标外推方向上设置 holdout，并按预算安排多个外推倍数（[§5.2](#52-尺寸数量跨度与外推倍数)）。[Delphi](https://openathena.ai/blog/delphi/) 在 3e18–3e20 FLOPs 上拟合，holdout 至 1e23 FLOPs；首次配方在约 33× 外推时 loss 比预测高 2.5%，在 333× 外推时发散（图 1）。

误差阈值按 [§2.3](#23-验收阈值与判定规则) 预先确定，同时报告外推的置信区间。Holdout 结果一旦用于调整模型或拟合设置，即转为开发数据，新的验收需要独立证据。

### 10.2 组合验证

单项改动通过后，必须验证最终组合，各项收益不能直接相加（[Marin 后续](https://openathena.ai/blog/pretraining-speedup/)；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961) Appendix A.2.5）：

- 数据配比与训练量、重复程度联合检查；
- 代理实验的结论标明迁移范围；
- 组合验证只在小规模完成时，不能表述为目标规模已验证。

### 10.3 稳定性压力测试

小规模训练无法充分暴露大规模训练中的 loss spike 和梯度异常。架构或优化器变更时应增加压力测试：[Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) 用中等规模模型和 2×/4× 预测最优 LR 提高优化压力，在相同压力下比较新旧配方。

短期高 LR 下的稳定与目标训练量附近的稳定是两个不同条件。对关键候选应记录梯度范数、激活范围与 loss spike；MoE 还应记录各专家负载、token 丢弃比例与路由分布的变化。[Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) §2.7 报告训练后期两次发散：第一次与输出层梯度精度有关，恢复 FP32 后稳定；第二次原因未查明，通过提前退火缓解。

### 10.4 实现一致性验证

大规模训练与 Ladder 实验常使用不同的并行方式、梯度累积、融合算子、通信精度和分布式优化器。配置名称相同不能证明训练的数学运算一致。验证步骤（本文建议）：

1. 用同一 checkpoint 与受控 batch，比较 forward、loss、梯度与单次参数更新；
2. 比较短轨迹的偏差是否在预定容限内；
3. 改变并行配置与梯度累积后，检查有效 global batch、归约方式、梯度裁剪顺序与精度差异；
4. 检查恢复训练时 optimizer state、scheduler、随机状态与数据加载位置是否正确恢复。

浮点归约会带来数值差异，应定义误差容限或统计容限，不要求逐位一致。恢复后重复或跳过数据、重置动量、schedule 位移，都需记入实验记录。新 kernel 或目标精度路径未经上述检查时，必须限制 Ladder 结论的适用范围。[DeepSeek-V3, 2024](https://arxiv.org/abs/2412.19437) §3.3 对计算、累加和优化器状态分别设置精度；[Llama 3](https://arxiv.org/abs/2407.21783) §3.3.4 描述了生产训练系统中的故障恢复需求。

### 10.5 实际效率与部署约束

理论 FLOPs 的改善未必带来相同幅度的训练时间改善，必须同时报告实际训练时间（[Marin 后续](https://openathena.ai/blog/pretraining-speedup/)区分 theoretical efficiency 与 realized efficiency）。精度方案需要验证，不能假设等价（[Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007)）。

用于选择生产配置时，应把以下量纳入约束（本文建议）：

- 训练侧：目标硬件上的吞吐、并行可行性、显存、有效训练时间，以及评估、checkpoint、故障恢复与重跑的开销；MoE 另计专家并行的通信开销。以交付期限为约束时，单独估计资源可用性。
- 部署侧：按目标负载给出输入与输出长度分布、并发、延迟与显存约束，使用实际精度与推理实现测量。[Sardana et al., 2024](https://arxiv.org/abs/2401.00448) 将训练、输入处理与输出生成的成本分开建模，其成本最优尺寸需用自身测量值重新计算。

不涉及部署的研究型 Ladder 可省略部署侧。交付物为目标硬件与负载下的质量、成本和约束对照表，以及由此选择模型尺寸与训练量的计算过程。

### 10.6 中等规模试运行

目标规模远超最大 holdout 时，应在启动目标训练前做一次中等规模试运行（本文建议）：

- 使用目标配方、目标基础设施、目标并行方式与目标精度方案；
- 在运行前冻结该规模的预测 loss 曲线及其区间；
- 通过条件为 loss 曲线落在预测区间内，且稳定性诊断量（[§10.3](#103-稳定性压力测试)）无持续异常；
- 未通过时按 [§12.2](#122-运行中偏离处理) 处理，查明原因前不启动目标训练。

试运行的规模由预算与外推倍数决定，公开文献中没有通用比例。

## 11. 专项 Ladder

### 11.1 分阶段 Ladder

不同训练阶段的数据分布、序列长度和 schedule 不同，scaling law 系数可能随阶段改变（[Qwen3, 2025](https://arxiv.org/abs/2505.09388)；[OLMo 3, 2025](https://arxiv.org/abs/2512.13961)）。需要预测或选择配置的阶段，可分别建立 Ladder：

1. 预训练 Ladder：从随机初始化开始，拟合 $N$、$D$、LR 与 BSZ；
2. Mid-training Ladder：从对应的预训练 checkpoint 开始，拟合新增 token 量与 LR schedule；
3. 长上下文 Ladder：从对应的 mid-training checkpoint 开始，搜索序列长度、RoPE 配置与 LR（[§11.2](#112-长上下文-ladder)）。

阶段间不直接复用 scaling law 系数，并同时记录基础能力的变化与目标能力的增量。

总 loss 接近的 checkpoint，其领域能力、已见数据、optimizer state 与最近的 LR 轨迹可能不同。固定一个 checkpoint 比较后续配方，结论以该起点为条件；外推到其他起点时，应保留代表性起点的交叉对照。还应记录（本文建议）：

- 阶段边界是否重置 optimizer、是否 re-warmup、哪些数据继续重复；
- 预算在预训练、mid-training 与长上下文阶段之间的分配；
- 各阶段局部最优的选择组合后，在端到端交付指标上的确认结果（[§10.2](#102-组合验证)）。

### 11.2 长上下文 Ladder

长上下文 Ladder 需说明训练与评价的长度分布、关键信息在序列中的位置、跨文档打包方式，并同时评估短上下文能力是否退化及计算成本的变化。[RULER (Hsieh et al., 2024)](https://arxiv.org/abs/2404.06654v3) 表明，通过简单检索测试不代表在多跳追踪、聚合等任务上具有同等能力，评估集应覆盖多类任务。

## 12. 执行流程与交付物

### 12.1 实验记录

每次运行必须有唯一标识，并关联：代码版本、配置、数据清单、tokenizer、评价版本、初始化与数据随机种子、父 checkpoint 与共享前缀、硬件与精度方案。

运行状态分为：完成、主动停止、算法异常、基础设施失败、实现错误。配置改动后的运行建立新记录，并说明旧结果是否仍可用于当前拟合。

### 12.2 运行中偏离处理

Ladder 交付时应同时交付各实验点的训练轨迹与阶段配置，目标训练偏离预测时据此调查。运行前应明确：用于比较的评价数据、按训练进度对齐的预测区间、判定持续偏离的窗口、需要保存的诊断量。

单次越界可能来自评估噪声，不能直接归因于 scaling law 失效。处理顺序（本文建议）：先复核测量与配置，再检查数据流、实现与硬件，最后判断是否需要修改配方并重新校准。原因未查明时保留“原因未查明”状态。

独立验收应能从原始实验表重新生成目标预测，并核对预测的冻结时间与实际结果。

### 12.3 Ladder 维护

Ladder 需要随配方与基础设施的变化持续维护（本文建议）：

- 回归 Ladder：保留一组固定的参考配置，在代码、kernel、并行方式或集群变更后重跑，检查 loss 是否在种子方差范围内；
- 系数版本：拟合系数与对应的配方版本、数据版本、拟合脚本版本一起存档；
- 重拟合条件：架构、优化器、数据版本、精度方案、tokenizer 变更时，先在小规模验证原系数的迁移性，偏差超出阈值时局部校准或完整重拟合。

### 12.4 Checklist

设计阶段：

- [ ] 明确适用场景（[§1.2](#12-典型场景)）、决策目标（[§2.1](#21-决策目标)）、评价协议（[§2.2](#22-评价协议)）
- [ ] 各指标在 Ladder 规模上有可分辨信号；交付后训练模型时安排 SFT/RL 验收（[§2.2](#22-评价协议)）
- [ ] 在观察 holdout 前确定验收阈值与判定规则，以等效算力倍数表达（[§2.3](#23-验收阈值与判定规则)）
- [ ] 区分固定量、缩放规则与实验自变量，规定 warmup 缩放规则（[§4.1](#41-变量分类)）
- [ ] 确定尺寸数量与外推倍数，设多档 holdout（[§5.2](#52-尺寸数量跨度与外推倍数)）
- [ ] 训练量档位覆盖目标 TPP；数据受限时写入 $U$ 与重复次数（[§5.3](#53-训练量档位)、[§7.3](#73-数据受限训练与重复)）
- [ ] 制定包括试验、搜索、拟合、验收与预备预算的实验计划（[§5.5](#55-预算分配与追加实验)）
- [ ] MoE：记录总参、激活参、路由配置，分别设计规模、稀疏度与粒度实验，设 dense 对照（[§3.1](#31-参数量口径)、[§5.6](#56-moe-实验轴)）
- [ ] 固定数据版本（[§7.1](#71-数据质量与数据源评估)）；去重范围与重复次数的统计口径一致（[§7.3](#73-数据受限训练与重复)）
- [ ] 数据配比未确定时安排配比实验，并与按比例采样和 UniMax 基线比较（[§7.2](#72-数据配比-ladder)）；多阶段训练分别设计实验（[§11.1](#111-分阶段-ladder)）

配置阶段：

- [ ] $N_{\text{body}}$ 与 $C$ 分开记录，按实际架构与 kernel 统计 FLOPs（[§3.1](#31-参数量口径)、[§3.2](#32-计算量)）
- [ ] 固定 loss、token 与评价口径，写出计数与聚合样例（[§3.3](#33-loss-与-token-口径)）
- [ ] 模型族内架构属性与缩放规则一致，MoE 路由规则固定（[§4.2](#42-架构与训练配置一致性)、[§4.4](#44-moe-结构缩放规则)）
- [ ] 使用目标训练精度方案；比较不同词表时使用按字节归一化的指标（[§4.5](#45-词表与数值精度)）

搜参阶段：

- [ ] 目标 2–4：小模型搜索至 Fully-Tuned Frontier 的操作定义；目标 1：按配方运行（[§6.1](#61-搜参目标与近优区间)）
- [ ] 更换优化器或参数化后重新标定超参规则（[§6.2](#62-参数化与优化器迁移)）
- [ ] BSZ 以 tokens 计，用 $B_{opt}\propto D^{0.4}$ 作先验并验证；检查 batch size ramp 的影响（[§6.3](#63-lr-与-bsz-的-scaling-law)）
- [ ] 复现配方沿用其 WD；跨 TPP 研究时检查 WD 的影响（[§6.4](#64-weight-decay)）
- [ ] 记录搜索覆盖情况，用新种子复验最终候选，分类记录失败 trial（[§6.7](#67-搜索流程与停止规则)）

拟合阶段：

- [ ] 核对 FLOPs 与参数量口径和 warmup 规则（[§8](#8-loss-scaling-law-拟合)）
- [ ] 比较候选函数形式（[§8.1](#81-函数形式)）；多 epoch 训练明确 unique tokens 与重复次数（[§7.3](#73-数据受限训练与重复)）
- [ ] 截断规则预先确定（[§5.4](#54-中间-checkpoint随机种子与共享轨迹)）
- [ ] 按拟合协议固定设置，按依赖结构做 bootstrap，复验决策结论（[§8.3](#83-拟合协议)）
- [ ] 检查残差结构、相关性与不确定性来源；误差过大时记录待补充实验（[§8.4](#84-拟合诊断与失效处理)）

验证阶段：

- [ ] Holdout 按预定阈值验收（[§10.1](#101-holdout-验证)）
- [ ] 下游能力决策需验证任务指标预测（[§9](#9-下游任务预测)）
- [ ] 单项改动通过后验证最终组合（[§10.2](#102-组合验证)）
- [ ] 架构或优化器变更后做稳定性压力测试（[§10.3](#103-稳定性压力测试)）
- [ ] 研究实现与目标实现一致性验证（[§10.4](#104-实现一致性验证)）
- [ ] 同时报告理论 FLOPs、实际训练时间与部署约束（[§10.5](#105-实际效率与部署约束)）
- [ ] 外推倍数大时做中等规模试运行（[§10.6](#106-中等规模试运行)）
- [ ] 保存实验记录、拟合系数版本、预测区间与未解决问题（[§12](#12-执行流程与交付物)）

## 附录 A. 公开 Ladder 配置

### A.1 公开规模配置

Ladder 扫描的规模与训练量（参数量按各来源口径）：

| 来源 | 模型参数量 | 训练量 | 文献 |
|---|---|---|---|
| OpenAI | 多尺寸，最大 1.5B（非 embedding 参数） | 22M–23B tokens | [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) |
| DeepMind Chinchilla | 70M–16B（400+ 模型） | 5B–500B tokens | [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) |
| StepFun | 多尺寸（3,700+ 模型） | 全部实验累计 100T tokens | [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) |
| Gadre et al. | 11M–6.9B（104 个模型） | 最高 32× Chinchilla 配比 | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540) |
| Fantastic Optimizers | 0.1B–1.2B（4 个尺寸） | 1×–8× Chinchilla 配比 | [Wen et al., 2025](https://arxiv.org/abs/2509.02046) |
| Delphi | 按算力预算扫描尺寸与训练量，最大 holdout 为 25B | 拟合 3e18–3e20 FLOPs，holdout 至 1e23 FLOPs | [Marin, 2026](https://openathena.ai/blog/delphi/) |
| Llama 3 | 40M–16B | 拟合 6e18–1e22 FLOPs，目标 3.8e25 FLOPs | [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.2.1 |

单次发布模型的训练量，用于参考目标 TPP，不属于 Ladder 扫描：

| 来源 | 模型参数量 | 训练量 | 文献 |
|---|---|---|---|
| NVIDIA Nemotron | 15B, 340B | 8T, 9T tokens | [15B 报告](https://arxiv.org/abs/2402.16819)；[340B 报告](https://arxiv.org/abs/2406.11704) |
| OLMo | 1B, 7B, 13B, 32B | OLMo 1：2T–2.46T；OLMo 2：按模型设置多阶段预算 | [OLMo, 2024](https://arxiv.org/abs/2402.00838v4)；[OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3) |
| Llama 3 | 8B, 70B, 405B | 405B：15.6T tokens；8B、70B 使用类似配方，训练时长远超 compute-optimal | [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §1、§3.4 |

### A.2 Fantastic Optimizers Ladder

[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046v2) Table 2–3。Dense，Llama 2 架构，四个尺寸均固定 32 层、MHA、seq_len 4096。目标为公平对比优化器，侧重超参搜索。

| 尺寸 | hidden_dim | inter_dim | heads | 数据比 |
|---|---|---|---|---|
| 130M | 512 | 2,048 | 8 | 1×–8× Chinchilla |
| 300M | 768 | 3,072 | 12 | 同上 |
| 520M | 1,024 | 4,096 | 16 | 同上 |
| 1.2B | 1,536 | 6,144 | 24 | 同上 |

Table 3 给出 AdamW 的搜参示例：Peak LR 8e-3、WD 0.1、warmup 2000 steps、BSZ 128 条序列（seq_len 4096，约 0.5M tokens）。该结果对应特定尺寸与配比，不同尺寸的配置不同，例如 520M/1× 使用 WD 0.2、BSZ 256。数据由 DCLM-baseline、StarCoder V2 Data 与 ProofPile 2 混合。

### A.3 Delphi Ladder

[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/)。Dense decoder-only，Qwen 3 架构，MLP ratio 4，seq_len 4096。目标为拟合 IsoFLOP scaling law 并外推到 1e23 FLOPs（25B），侧重端到端 loss 预测。后续扩展到 MoE（[535B-A23B](https://openathena.ai/blog/pretraining-speedup/)）。

共同设置：AdamH、WSD（10% warmup、20% decay to 0）、f32 参数与 bf16 计算、FSDP。数据为 Nemotron-CC、StarCoderData 与 ProofPile 2。

结构：在 3e18–3e20 FLOPs 上做 IsoFLOP 扫描，取 7 个最优点拟合；holdout 为 1e21–1e23 FLOPs（3×–333× 外推）。超参按配方规则设置，不逐点手动搜索。

### A.4 选择建议

- 需要搜索最优超参并拟合超参 scaling law 时，参考 Fantastic Optimizers 的网格设计。
- 需要端到端 loss 预测并外推到大规模时，参考 Delphi 的 IsoFLOP 布局与配方公式。
- 两套配置均为起点，均为 dense；MoE Ladder 需按 [§5.6](#56-moe-实验轴) 另行设计。模型族与结构缩放规则应对目标模型具有代理有效性（[§4.2](#42-架构与训练配置一致性)）。

## 参考文献

按主题分组，条目后标注正文中的对应章节。

### Scaling Law 基础、函数形式与拟合

1. Scaling Laws for Neural Language Models — Kaplan et al., OpenAI, 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)  
   在所测范围内观察到跨越 7 个数量级的经验幂律；给定参数量时宽深比对 loss 影响较小。见 [§3.1](#31-参数量口径)、[§4.3](#43-宽深配置)
2. Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., DeepMind, 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)  
   IsoFLOP 方法与模型、数据近似等比例扩展的结果；加性形式 $L=E+A/N^\alpha+B/D^\beta$；20 TPP 的参数量口径。见 [§3.1](#31-参数量口径)、[§5.3](#53-训练量档位)、[§8.1](#81-函数形式)
3. Language models scale reliably with over-training and on downstream tasks — Gadre et al., 2024. [arXiv:2403.08540](https://arxiv.org/abs/2403.08540)  
   104 个模型验证过训练区域的 scaling law；不同 $D/N$ 下幂律指数接近。见 [§5.3](#53-训练量档位)、[§9.1](#91-方法路线)
4. Skaling: Chinchilla's Exponents Meet Kaplan's Coupling — Videau et al., FAIR at Meta, 2026. [arXiv:2608.07222](https://arxiv.org/abs/2608.07222)  
   在所分析数据上观察到负混合偏导，提出外指数 $k$ 和 L-shape 采样，并验证其外推表现。见 [§8.1](#81-函数形式)
5. Predictable Scale: Part II, Farseer: A Refined Scaling Law in Large Language Models — Li et al., StepFun & Fudan, NeurIPS 2025. [arXiv:2506.10972](https://arxiv.org/abs/2506.10972)  
   数据侧系数与指数显式依赖 $N$ 的九参数形式；参数量口径消融。见 [§3.1](#31-参数量口径)、[§8.1](#81-函数形式)
6. Scaling Law with Learning Rate Annealing — Tissue et al., 2024. [arXiv:2408.11029](https://arxiv.org/abs/2408.11029)  
   将 loss 表示为累计学习率面积与退火量的函数，可拟合整条曲线。见 [§6.5](#65-lr-schedule)、[§8.2](#82-loss-曲线与退火-scaling-law)
7. A Hitchhiker's Guide to Scaling Law Estimation — Choshen et al., MIT/IBM, ICML 2025. [arXiv:2410.11840](https://arxiv.org/abs/2410.11840)  
   中间 checkpoint 的使用与初期截断；尺寸数量、外推跨度与种子的价值；文献中的最小有效差异。见 [§2.3](#23-验收阈值与判定规则)、[§5.2](#52-尺寸数量跨度与外推倍数)、[§5.4](#54-中间-checkpoint随机种子与共享轨迹)、[§5.5](#55-预算分配与追加实验)
8. Resolving Discrepancies in Compute-Optimal Scaling of Language Models — Porian et al., NeurIPS 2024. [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)  
   分析 output head、warmup 和超参调优对 compute-optimal 指数差异的影响。见 [§3.1](#31-参数量口径)、[§4.1](#41-变量分类)、[§8](#8-loss-scaling-law-拟合)
9. Chinchilla Scaling: A Replication Attempt — Besiroglu et al., Epoch AI, 2024. [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)  
   复现 Chinchilla 的参数化拟合，指出拟合与置信区间设置的问题。见 [§8.3](#83-拟合协议)
10. (Mis)Fitting: A Survey of Scaling Laws — Li et al., 2025. [arXiv:2502.18969](https://arxiv.org/abs/2502.18969)  
    讨论拟合细节缺失对复现和结论的影响。见 [§8.3](#83-拟合协议)
11. Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws — Sardana et al., 2024. [arXiv:2401.00448](https://arxiv.org/abs/2401.00448)  
    将推理成本纳入资源分配，分开建模训练、输入处理与输出生成的成本；TPP 至 10,000 的过训练实验。见 [§5.3](#53-训练量档位)、[§10.5](#105-实际效率与部署约束)
12. Gemstones: A Model Suite for Multi-Faceted Scaling Laws — McLeish et al., 2025. [arXiv:2502.06857](https://arxiv.org/abs/2502.06857)  
    宽深配置对 benchmark 的影响；实验点选择对资源分配建议的影响。见 [§2.3](#23-验收阈值与判定规则)、[§4.3](#43-宽深配置)
13. Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies — Tao et al., NeurIPS 2024. [arXiv:2407.13623](https://arxiv.org/abs/2407.13623)  
    最优词表大小随算力增大。见 [§4.5](#45-词表与数值精度)
14. Scaling Laws for Precision — Kumar et al., 2024. [arXiv:2411.04330](https://arxiv.org/abs/2411.04330)  
    训练精度与后训练量化对 loss 的影响。见 [§4.5](#45-词表与数值精度)

### 超参 Scaling Law 与优化器

{:start="15"}
15. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism — DeepSeek, 2024. [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)  
    按算力 $C$ 拟合超参幂律；所测语料的最优资源分配存在差异。见 [§6.3](#63-lr-与-bsz-的-scaling-law)、[§7.1](#71-数据质量与数据源评估)
16. Predictable Scale: Part I, Step Law — Optimal Hyperparameter Scaling Law in Large Language Model Pre-training — Li et al., StepFun, 2025. [arXiv:2503.04715v3](https://arxiv.org/abs/2503.04715v3)  
    3,700+ 模型验证的 $\eta_{opt}=c\,N^{-\alpha}D^{\beta}$、$B_{opt}=d\,D^{\gamma}$。见 [§6.1](#61-搜参目标与近优区间)、[§6.3](#63-lr-与-bsz-的-scaling-law)
17. Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training — Bergsma et al., Cerebras, 2025. [arXiv:2505.13738](https://arxiv.org/abs/2505.13738)  
    在所测配方下得到 $B_{opt}\propto D^{0.4}$、$B_{crit}\propto D_{min}^{0.5}$，对 $N$ 的依赖较弱；timescale $\tau=B/(\eta\lambda D)$ 用于联合搜索。见 [§6.2](#62-参数化与优化器迁移)、[§6.3](#63-lr-与-bsz-的-scaling-law)、[§6.4](#64-weight-decay)
18. Scaling Optimal LR Across Token Horizons — Bjorck et al., Microsoft, 2024. [arXiv:2409.19913](https://arxiv.org/abs/2409.19913)  
    固定模型尺寸与 BSZ 时 peak LR 随训练长度衰减。见 [§6.3](#63-lr-与-bsz-的-scaling-law)
19. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer ($\mu$-Transfer) — Yang et al., Microsoft, 2022. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)  
    $\mu$P 下的宽度方向超参迁移。见 [§6.2](#62-参数化与优化器迁移)
20. Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks — Yang et al., 2023. [arXiv:2310.02244](https://arxiv.org/abs/2310.02244)  
    每个残差块含一层时的 Depth-$\mu$P；残差块含多层时无限深度参数化的局限。见 [§6.2](#62-参数化与优化器迁移)
21. Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit — Bordelon et al., 2023. [arXiv:2309.16620](https://arxiv.org/abs/2309.16620)  
    按 $1/\sqrt{\text{depth}}$ 缩放残差分支，在 ResNet 与 ViT 上观察到跨深度的超参迁移。见 [§6.2](#62-参数化与优化器迁移)
22. Muon is Scalable for LLM Training — Liu et al., Moonshot AI, 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)  
    为 Muon 加入 weight decay 并匹配更新 RMS，复用 AdamW 的超参设置。见 [§6.2](#62-参数化与优化器迁移)
23. Fantastic Pretraining Optimizers and Where to Find Them — Wen et al., Stanford, 2025. [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)  
    四尺寸、1×–8× Chinchilla 范围内的公平调参基准；token 效率收益随尺寸变化；退火期间的排名翻转。见 [§5.3](#53-训练量档位)、[§6.8](#68-配方比较)、[附录 A.2](#a2-fantastic-optimizers-ladder)
24. Weight Decay Improves Language Model Plasticity — Han et al., 2026. [arXiv:2602.11137](https://arxiv.org/abs/2602.11137)  
    预训练 loss 偏好的 WD 随 TPP 增大而降低；后训练可塑性可能受益于更强的 WD。见 [§6.4](#64-weight-decay)
25. Small-Scale Experiments: Are We There Yet? — Lourie et al., NYU & Meta, 2026. [arXiv:2608.11859](https://arxiv.org/abs/2608.11859)  
    小模型对超参的敏感度与 Fully-Tuned Frontier；尺寸划分为拟合、验证、测试三段。见 [§5.4](#54-中间-checkpoint随机种子与共享轨迹)、[§6.1](#61-搜参目标与近优区间)
26. How to Allocate Your Tokens? Scaling Laws with Training Steps and Batch Size — Schaipp, Inria, 2026. [arXiv:2607.01487](https://arxiv.org/abs/2607.01487)  
    按算力损失定义近优 batch 区间，所测宽度约 4 倍。见 [§6.3](#63-lr-与-bsz-的-scaling-law)
27. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation — Cawley & Talbot, JMLR 2010. [JMLR 11](https://www.jmlr.org/papers/v11/cawley10a.html)  
    有限样本上模型选择带来的选择偏差。见 [§6.7](#67-搜索流程与停止规则)

### MoE

{:start="28"}
28. Scaling Laws for Fine-Grained Mixture of Experts — Krajewski et al., 2024. [arXiv:2402.07871](https://arxiv.org/abs/2402.07871)  
    将专家粒度纳入 scaling law。见 [§4.4](#44-moe-结构缩放规则)、[§5.6](#56-moe-实验轴)、[§8.5](#85-moe-拟合)
29. Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models — Abnar et al., Apple, 2025. [arXiv:2501.12370](https://arxiv.org/abs/2501.12370)  
    固定算力下稀疏度与预训练 loss 的关系；稀疏度对下游迁移的影响。见 [§4.4](#44-moe-结构缩放规则)
30. Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient — Ludziejewski et al., 2025. [arXiv:2502.05172](https://arxiv.org/abs/2502.05172)  
    联合建模专家数、激活参数量与训练量，纳入显存约束。见 [§4.4](#44-moe-结构缩放规则)、[§8.5](#85-moe-拟合)

### 数据构建、配比与重复

{:start="31"}
31. Scaling Data-Constrained Language Models — Muennighoff et al., 2023. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)  
    重复数据的等效 token 量公式与边际收益衰减曲线。见 [§7.3](#73-数据受限训练与重复)
32. To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis — Xue et al., 2023. [arXiv:2305.13230](https://arxiv.org/abs/2305.13230)  
    参数量、数据量和质量对重复过拟合的影响；dropout 的作用。见 [§7.3](#73-数据受限训练与重复)
33. Prescriptive Scaling Laws for Data Constrained Training — Lovelace et al., 2026. [arXiv:2605.01640](https://arxiv.org/abs/2605.01640)  
    参数量、unique tokens 和重复次数共同影响的过拟合。见 [§7.3](#73-数据受限训练与重复)
34. Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression — Yan et al., 2025. [arXiv:2511.13421](https://arxiv.org/abs/2511.13421)  
    线性回归假设下重复次数与样本数的关系，并提供 LLM 实验。见 [§7.3](#73-数据受限训练与重复)
35. UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining — Chung et al., ICLR 2023. [arXiv:2304.09151](https://arxiv.org/abs/2304.09151)  
    限制每个语料最大重复次数的采样方法。见 [§7.2](#72-数据配比-ladder)
36. Olmix: A Framework for Data Mixing Throughout LM Development — Chen et al., Allen Institute, ICLR 2026. [arXiv:2602.12237](https://arxiv.org/abs/2602.12237)  
    配比代理实验的设计选择（代理尺寸、代理数量、采样分布、回归模型与粒度、重复约束、求解方式）；领域更新后的 mixture reuse。见 [§7.2](#72-数据配比-ladder)
37. Scaling Laws for Mixture Pretraining Under Data Constraints — Sedova et al., Apple, 2026. [arXiv:2605.12715](https://arxiv.org/abs/2605.12715)  
    稀缺数据混合训练中的重复次数；含重复项的配比 scaling law。见 [§7.3](#73-数据受限训练与重复)
38. Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training (DeMix) — Li et al., 2026. [arXiv:2602.00747](https://arxiv.org/abs/2602.00747)  
    以组件模型加权合并代替按配比训练的代理。见 [§7.2](#72-数据配比-ladder)
39. Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset — Su et al., NVIDIA, 2024. [arXiv:2412.02595](https://arxiv.org/abs/2412.02595)  
    Common Crawl 数据集及其合成改写版本；Marin 全局去重中最大的跨来源重叠。见 [§7.3](#73-数据受限训练与重复)

### LR Schedule 与训练收尾

{:start="40"}
40. Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al., 2024. [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)  
    在特定损失几何与优化动力学假设下解释 WSD 的 stable 和 decay 阶段。见 [§6.5](#65-lr-schedule)
41. Scaling and Transferability of Annealing Strategies in Large Language Model Training — Wang et al., 2025. [arXiv:2512.13705](https://arxiv.org/abs/2512.13705)  
    退火策略的 scaling 与跨规模迁移。见 [§6.5](#65-lr-schedule)
42. Model Merging in Pre-training of Large Language Models — ByteDance Seed, 2025. [arXiv:2505.12082](https://arxiv.org/abs/2505.12082)  
    预训练中的 checkpoint 合并；stable 阶段 PMA 与退火终点表现的比较。见 [§6.9](#69-收尾流程)
43. Model Soups: Averaging Weights of Multiple Fine-tuned Models — Wortsman et al., 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)  
    对共享预训练起点、独立微调所得模型做权重平均。见 [§6.9](#69-收尾流程)
44. Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging (LAWA) — Kaddour, 2022. [arXiv:2209.14981](https://arxiv.org/abs/2209.14981)  
    单条轨迹的滑动窗口权重平均。见 [§6.9](#69-收尾流程)
45. Early Weight Averaging meets High Learning Rates for LLM Pre-training — Sanyal et al., COLM 2024. [arXiv:2306.03241](https://arxiv.org/abs/2306.03241)  
    高学习率下的早期权重平均。见 [§6.9](#69-收尾流程)

### 下游任务预测与评估

{:start="46"}
46. Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective (COD) — Xu et al., ICLR 2026, v4（2026-03-09）. [arXiv:2502.17262v4](https://arxiv.org/pdf/2502.17262v4)  
    按难度特征聚类、筛选可预测簇并映射到全集；包含 dense 与 MoE 目标预测及 continued training 实验。见 [§9.1](#91-方法路线)、[§9.2](#92-cod-框架)
47. GPT-4 Technical Report — OpenAI, 2023. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)  
    按小模型表现对 HumanEval 难度分桶，在子集上拟合外推。见 [§9.1](#91-方法路线)
48. Establishing Task Scaling Laws via Compute-Efficient Model Ladders (OLMo Task Ladder) — Bhagia et al., Allen Institute, 2024. [arXiv:2412.04403](https://arxiv.org/abs/2412.04403)  
    两阶段下游预测：由 $N,D$ 拟合 task-specific loss，再拟合 loss 到 accuracy；任务间噪声差异显著。见 [§9.1](#91-方法路线)
49. Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive? — Schaeffer et al., 2024. [arXiv:2406.04391](https://arxiv.org/abs/2406.04391)  
    多选题 accuracy 依赖错误选项上的概率质量，削弱与算力的统计关系。见 [§9.3](#93-可预测性的限制)
50. RULER: What's the Real Context Size of Your Long-Context Language Models? — Hsieh et al., NVIDIA, 2024. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)  
    简单检索测试与多类长上下文任务表现的差异。见 [§11.2](#112-长上下文-ladder)

### 模型技术报告与 Ladder 实例

{:start="51"}
51. Delphi: An Open Scaling Suite from 3e18 to 1e23 FLOPs — Marin Team, 2026. [openathena.ai/blog/delphi](https://openathena.ai/blog/delphi/)  
    IsoFLOP 拟合与配方公式驱动的超参；多档外推 holdout；性能竞争力与可预测性分别验证；IsoFLOP 最优点的 bootstrap；sigmoid 下游映射。见 [§1.1](#11-scaling-ladder-的定义)、[§2.1](#21-决策目标)、[§5.2](#52-尺寸数量跨度与外推倍数)、[§8.4](#84-拟合诊断与失效处理)、[§9.1](#91-方法路线)、[§10.1](#101-holdout-验证)、[附录 A.3](#a3-delphi-ladder)
52. The Llama 3 Herd of Models — Meta, 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)  
    IsoFLOP 确定 405B 尺寸；较小模型过训练；两阶段下游预测；batch size ramp；训练系统的故障恢复。见 [§5.2](#52-尺寸数量跨度与外推倍数)、[§5.3](#53-训练量档位)、[§6.3](#63-lr-与-bsz-的-scaling-law)、[§9.1](#91-方法路线)、[§10.4](#104-实现一致性验证)
53. DeepSeek-V3 Technical Report — DeepSeek, 2024. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)  
    无辅助损失的负载均衡与不丢弃 token（§2.1.2）；FP8 训练中计算、累加与存储的精度设置（§3.3）。见 [§4.4](#44-moe-结构缩放规则)、[§10.4](#104-实现一致性验证)
54. Nemotron-4 15B Technical Report — NVIDIA, 2024. [arXiv:2402.16819](https://arxiv.org/abs/2402.16819)  
    batch size 递增；训练末期的 continued training。见 [§6.3](#63-lr-与-bsz-的-scaling-law)、[§6.9](#69-收尾流程)
55. Nemotron-4 340B Technical Report — NVIDIA, 2024. [arXiv:2406.11704](https://arxiv.org/abs/2406.11704)  
    发布模型的训练量。见 [附录 A.1](#a1-公开规模配置)
56. OLMo: Accelerating the Science of Language Models — Groeneveld et al., Allen Institute, 2024. [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)  
    完全开放的训练数据、代码与中间 checkpoint。见 [附录 A.1](#a1-公开规模配置)
57. 2 OLMo 2 Furious — OLMo Team, Allen Institute, 2025. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)  
    两阶段训练；micro-annealing 验证数据源；model souping。见 [§6.9](#69-收尾流程)、[§7.1](#71-数据质量与数据源评估)
58. OLMo 3 — Allen Institute, 2025. [arXiv:2512.13961](https://arxiv.org/abs/2512.13961)  
    评价指标的有效规模范围；Olmix 配比流程与质量感知上采样；embedding 不做 decay；分阶段训练与组合验证。见 [§2.2](#22-评价协议)、[§6.4](#64-weight-decay)、[§7.2](#72-数据配比-ladder)、[§7.3](#73-数据受限训练与重复)、[§10.2](#102-组合验证)、[§11.1](#111-分阶段-ladder)
59. Qwen3 Technical Report — Qwen Team, 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)  
    分阶段预测 LR 与 batch size；未报告 WD scaling；实例属性维度的数据配比。见 [§6.4](#64-weight-decay)、[§7.2](#72-数据配比-ladder)、[§11.1](#111-分阶段-ladder)
60. On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability — Qwen Team, 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)  
    大模型的近优区间；架构与优化器变更后的超参偏移；后训练验收；稳定性压力测试。见 [§2.2](#22-评价协议)、[§6.1](#61-搜参目标与近优区间)、[§6.2](#62-参数化与优化器迁移)、[§10.3](#103-稳定性压力测试)
61. Kimi K2: Open Agentic Intelligence — Moonshot AI, 2025. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)  
    固定激活规模下的稀疏率 scaling law；固定 WD 配方；改写与重复的对照。见 [§5.6](#56-moe-实验轴)、[§6.4](#64-weight-decay)、[§7.3](#73-数据受限训练与重复)
62. Kimi K2.5: Visual Agentic Intelligence — Moonshot AI, 2026. [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)  
    联合预训练中控制各数据源的最大 epoch 数。见 [§7.3](#73-数据受限训练与重复)
63. Kimi K3: Open Frontier Intelligence — Moonshot AI, 2026. [arXiv:2607.24653](https://arxiv.org/abs/2607.24653)  
    配方变化后重做 scaling law 研究；cosine 与 WSD 分别搜参后的比较；小模型消融确定领域采样率；沿用 K2 的改写方法。见 [§6.8](#68-配方比较)、[§7.1](#71-数据质量与数据源评估)、[§7.3](#73-数据受限训练与重复)
64. Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2026. [arXiv:2604.12374](https://arxiv.org/abs/2604.12374)  
    两阶段配比：前 80% 偏重多样性，后 20% 偏重高质量数据。见 [§7.2](#72-数据配比-ladder)
65. Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2026. [arXiv:2606.15007](https://arxiv.org/abs/2606.15007)  
    训练后期发散与高精度分支对照；领域合成数据的消融。见 [§7.1](#71-数据质量与数据源评估)、[§10.3](#103-稳定性压力测试)、[§10.5](#105-实际效率与部署约束)
66. Marin：MoE 与训练效率后续 — Marin Team, 2026. [openathena.ai/blog/pretraining-speedup](https://openathena.ai/blog/pretraining-speedup/)  
    区分 theoretical 与 realized efficiency；多尺度方案比较与组合实验设计。见 [§10.5](#105-实际效率与部署约束)、[§10.2](#102-组合验证)
67. Marin 数据流程 — Marin Team, 2026. [openathena.ai/blog/marin-data-pipeline-overview](https://openathena.ai/blog/marin-data-pipeline-overview/)  
    全局去重与重复次数；配比基线；代理实验中按比例缩放训练量与数据池；跨规模确认。见 [§7.2](#72-数据配比-ladder)、[§7.3](#73-数据受限训练与重复)
68. Phi-4 Technical Report — Microsoft, 2024. [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)  
    高噪声评估任务的处理。见 [§2.2](#22-评价协议)
69. GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models — Zhipu AI, 2025. [arXiv:2508.06471](https://arxiv.org/abs/2508.06471)  
    宽深配置对推理能力的影响。见 [§4.3](#43-宽深配置)

### 延伸阅读

以下文献与 Ladder 设计相关，正文未展开：

{:start="70"}
70. Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training — Merrill et al., 2025. [arXiv:2505.23971](https://arxiv.org/abs/2505.23971)
71. An Empirical Model of Large-Batch Training — McCandlish et al., 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
72. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al., Google, 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
73. Cost-Optimal Grouped-Query Attention for Long-Context Modeling — Chen et al., 2025. [arXiv:2503.09579](https://arxiv.org/abs/2503.09579)
