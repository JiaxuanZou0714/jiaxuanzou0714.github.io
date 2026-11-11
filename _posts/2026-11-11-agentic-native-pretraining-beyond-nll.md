---
layout: post
title: "Agentic 时代的预训练不能只看 NLL：从 MAI、Beam、ML4 与 Nemotron 说起"
date: 2026-11-11 12:00:00
description: "结合近期 MAI、Reflection Beam、Mistral Large 4 与 Nemotron 3.5 的技术报告，讨论旧 GDM 一脉的训练习惯，以及为什么 PT 与 CPT 不能只做退火，而需要提前注入 Agentic 交互 pattern、提高 RL 起点。"
tags: [llm, pretraining, mid-training, agent, scaling-law, rl]
categories: [deep-learning]
featured: true
giscus_comments: true
toc:
  sidebar: left
lang: zh-CN
published: true
---

最近一个月，微软 MAI（MAI-Base-2）、Reflection AI（Beam）、Mistral（Mistral Large 4，简称 ML4）和 NVIDIA（Nemotron 3.5）接连放出了技术报告。这几家团队里都有不少 Google DeepMind（GDM）出来的核心成员。

读下来有两个很明显的现象：

1. MAI-Base-2 训到了 2.8T 参数、50T tokens，整篇报告却还在只看静态 NLL / BPB，没有考虑面向 Agent 的预训练指标；几个月前的 MAI-Thinking-1 连在线 OPD（On-Policy Distillation）都没跑通，CPT（Mid-train）只用纯自然数据做 LR 退火与 long context 训练。
2. Reflection 的 Beam 进入 RL 时起点极低，靠上万张卡跑了约一个月才慢慢把分数训上去，RL 的 sample complexity 极高，忽视了多轮先验（Prior）训练和快速迭代。

这两个现象背后是同一个问题：**没有从 Agentic-Native（Agentic 本位）的角度去设计预训练与 CPT / Mid-train，还把 Mid-train 单纯当成换到高质量静态数据上做退火，没有在进入 RL 之前让模型先具备多轮交互的 pattern 和足够高的起点分数。**

---

## 1. 四份报告的做法对比

### 1.1 MAI 系列

MAI-Base-2（Mustafa Suleyman、Karen Simonyan 等领衔）在传统预训练实验上做得很工整，包括对齐数据重复轮数、用 Muon 控制训练稳定性、拟合超参 scaling law [[1]](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/)。但全篇没有任何生成式或工具调用评测，声称预训练算力效率是外部最强开源基座 DeepSeek-V4-Pro-Base 的 4–12 倍，比的也只是内部静态验证集上的 BPB；Figure 1 甚至把只有 post-trained 权重的 Kimi K3、Qwen3.8-2.4T-A95B 和 MiMo-V2.6-Pro-RL 拿来测静态文本 BPB，而经过长思维链 RL 的模型输出分布本来就会偏离自然语料。

6 月发布的 MAI-Thinking-1 报告也延续了同一套思路 [[2]](https://microsoft.ai/pdf/mai-thinking-1.pdf)：

- 报告专门解释预训练为什么只看 NLL 而不测生成或 tool use：NLL 每步都用 ground-truth 前缀做 teacher forcing，能限制单步小错误向后累积。但长程 Agent 的主要难点恰恰是多步交互下的状态漂移和误差累积，teacher forcing 直接把这个问题掩盖掉了。
- 出于“能力应靠学习而非继承”的旧 GDM 观念以及商业售卖权重的版权顾虑，它的 Mid-train 完全取自原预训练自然语料，只按 NLL 重调配比、做 LR 退火和 long context 训练，零合成数据、没见过任何推理轨迹就直接进 RL 从头学；多领域专家合并也没跑通在线 MOPD，只能退回离线轨迹 SFT 再补轻量 RL。

### 1.2 Beam 与 MiMo-V2.6

Reflection 的 Beam（Ioannis Antonoglou、Misha Laskin 等大批前 GDM 骨干领衔）在 Mid-train 里加入了不少开源模型改写的论文推导、竞赛教程与静态代码任务，后训练也跑通了全词表 MOPD，单轮数学竞赛分数不错 [[3]](https://reflection.ai/documents/beam-model-tech-report)。

但在长程 Agent 上，它的 Mid-train（不到四天）和 SFT（整本报告只写了三句话）做得很薄，单次 RL 消耗的卡时反而超过整个预训练的两倍 [[3]](https://reflection.ai/documents/beam-model-tech-report), [[4]](https://reflection.ai/blog/introducing-beam)。Mid-train 结束时模型连 chat template 和官方 tool-call 语法都没有，代码与终端数据停留在单轮静态改写，缺少多轮试错轨迹。结果进入 RL 时 DeepSWE 起点只有约 5%，动用上万张 GB300 连续跑了约四周、采了超过 1 亿条 rollouts，才把分数训到 43% 左右。

对比同期小米发布的 MiMo-V2.6：它在 Mid-train 阶段混入了代码、通用、视觉等领域的真实多轮 Agent 轨迹，并提前把优化器从 AdamW 切到 Muon 变体以适配后续大 batch 训练。经过简短 SFT 后，MiMo-V2.6-Pro 进入 RL 时 DeepSWE 起点就已经接近 60%（高于 Beam 跑完一个月的终点），后面只做了 30 步超大 batch RL（总计 70 多万条 rollouts，不到 Beam 的百分之一），分数就提高到了 72% 以上 [[5]](https://arxiv.org/abs/2610.11959)。

### 1.3 Nemotron 3 Ultra 与 3.5 Super

年中发布的 Nemotron 3 Ultra 用重度 Mamba-2 换高吞吐，没有专门的 Agentic Mid-train，底座只看静态榜单；即便后训练蒸馏了多款强开源模型并迭代两轮 MOPD，长程代码修复和深度搜索表现仍然偏弱 [[6]](https://arxiv.org/abs/2606.15007)。到了 10 月底的 Nemotron 3.5 Super，团队专门补上了带多框架真实 Agent 轨迹的 SWE Mid-train，同时放弃用 NLL 验收，改看轻量 SFT 后在多套 Agent Harness 上的解决率，最终以小得多的激活参数在 SWE-bench Pro 上反超了前代 Ultra [[7]](https://arxiv.org/abs/2610.38095)。

---

## 2. 旧 GDM 的训练习惯与对 Gemini 3.1 Pro 的猜测

GDM 背景越浓的团队（MAI、Beam，以及早期调整前的 ML4），在训练上越容易延续老 GDM 的两套习惯：

- **Chinchilla / Gemma 式的预训练习惯**：在小模型 Ladder 上只看静态验证集 NLL，只要静态 loss 不掉点，就大胆采用 5:1 SWA-512 这类大幅压缩 KV cache 的结构，默认不会影响多轮交互中的远距离信息回溯。
- **AlphaGo Zero 式的“白板探索”习惯**：习惯让 PT/CPT 只管压缩干净的静态知识，把思考和行动留给在线 RL 在沙盒里从头学。但围棋动作空间极小、模拟极快；开放域软件工程与工具调用面对十多万词表和超长上下文，单次容器执行动辄十几分钟，奖励还极稀疏。无 Critic 的 GRPO 主要只是放大底座在 pass@k 里本来就能采出来的成功路径；如果前期的先验不足，后面靠上亿条 rollout 去搜，效率就会很低。

由此我捕风捉影地猜测：**为什么 GDM 本部的 Gemini 3.1 Pro 刚出来时那么强，后来却落后了 OpenAI 和 Anthropic 一段时间？**

3.1 Pro 占上风的那段时间，主要比拼的还是预训练知识加上**单轮超大 CoT**。纯文本长 CoT 和预训练里的教材、论文推导分布很接近；哪怕 CPT 只做退火和静态知识提纯，底座里也已经有了推演基础，靠 RL 就能把单轮 CoT 激发得很强。但很快 GPT 和 Claude 就进入了高频做 tool use 的阶段，而当时我用 Gemini 的明显感受是：**它几乎不会用 tool use，完全停留在纯超大 CoT 的时代。**

做过预训练数据的人都知道：**你想要模型输出怎样的 pattern，数据里就得先有怎样的 pattern。** 适合多轮、long-horizon agentic 任务的输出 pattern（比如先简短分析两句就调工具试探环境 $\to$ 读报错或搜索结果 $\to$ 发现不对立刻回退改方案 $\to$ 跨几十轮维护状态）在自然网页和静态代码文件里几乎不存在。如果 CPT / Mid-train 还停留在“换到高质量静态数据上做退火和 long context 训练”，没有把这类多轮交互 pattern 注入进去，纯靠 RL 几乎不可能自己激发出来。

---

## 3. 从 Agentic 本位思考预训练与 CPT

### 3.1 预训练指标

ML4 最初把 KV cache 压小 16 倍时，预训练 perplexity 和 MMLU 几乎无损；中途做 Mid-train + SFT 评测才发现，**相同的 validation perplexity 下 SWE-bench 直接低了 20 多个百分点**，只能拆掉局部滑窗和跨层共享重训后半程 [[8]](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/resolve/main/assets/technical_report.pdf)。相近的语言模型 loss 并不代表相近的长程 Agent 表现。目前比较务实的做法是在定架构或数据配比时接一个固定小预算的轻量 Mid-train + Agentic SFT，看多套 Agent Harness 下的解决率与 pass@k；更具体的评测设计我个人做得不多，了解有限，这里不展开。

### 3.2 PT 与 CPT：注入交互 Pattern，提高 RL 起点

PT 与 CPT（尤其是连接底座和后训练的 Mid-train）不能只是换到高质量数据上做退火，更重要的作用是充当连接底座与后训练的桥梁，往模型里注入目标输出 pattern，**让进入 RL 之前的 checkpoint 就有尽量好的交互 pattern 和尽量高的起点分数，后面的 RL 才能更好发挥作用。**

PT 和 CPT 是全序列并行训练，MFU 高，不用等漫长的容器执行，也不会因为整组 rollout 全错（零方差）浪费算力，因此很适合用来注入复杂先验：

- **PT 阶段**：除了常规网页、教材与静态代码，尽量按软件和任务的真实推进过程组织数据，比如把 `Issue -> PR 讨论 -> 跨文件 Commit Diff -> Review 与 CI 报错修复` 串成因果序列，让模型在预训练阶段就接触到“问题—动作—反馈—修正”的 pattern。
- **CPT / Mid-train 阶段**：集中注入多轮交互 pattern。把真实或合成的多轮 tool use 轨迹、读报错定位回滚的过程、跨不同 Agent 框架（multi-harness）的交互格式放进 CPT（对环境返回观测做 loss mask，只对思考和动作算梯度），也可以像 MiMo-V2.6 那样提前切换优化器状态来适配后续大 batch 训练。让 RL 前的 checkpoint 先具备完整的交互模式和较高的起点分数，下游 RL 只需要几十万条 rollouts、几十步大 batch 更新就能把能力充分激发出来。
- **RL + MOPD 与数据回流 CPT 的快速迭代**：CPT 与 SFT 提供了较好的初始策略之后，在线 RL 用较小样本量在能力边界上做探索，各领域专家模型再通过同源 MOPD 合并，筛出来的高质量长程轨迹清洗后回流到下一轮 CPT / CT（Mid-train）中，把按月计算的大规模 RL 训练缩短成几天一轮的迭代。工业界做 1.1、1.2 这类小版本更新时，除了补充特定领域数据（如 code-to-video、3D 建模等），也经常把线上用户回流的超长 trace（包含 tool use、agentic markdown、上下文 summary 等）整理清洗成从问题、动作到最终答案的干净轨迹，和特定任务的 RL 回流数据一起放回 CPT 阶段做定向增强。

---

## 4. 两个尚未解决的问题

1. **CPT / SFT 先验强度与探索多样性的平衡。**
   如果在 CPT 和 SFT 里加入过多单一来源的同质化轨迹，虽然 pass@1 起点高了，但输出分布容易变窄、熵降得太快，可能压低 pass@k 上限，限制后续 RL 的探索空间（这可能也是旧 GDM 路线抗拒合成数据与蒸馏的原因之一）。如何在注入多轮 tool use pattern、提高起点的同时保留足够的路径多样性，配比和训练方式还需要继续摸索。
2. **小规模 Scaling Ladder 上如何选取观测量来反映和预测旗舰模型的 Agentic 能力。**
   传统 NLL Scaling Law 能用很小的算力把大模型的最终 loss 预测得很准；但在多轮 Agent 任务上，小模型能力尚未涌现、环境交互方差大，哪怕接轻量 SFT 探针，跨两三个数量级外推时的排序稳定性和定量精度也还远不如静态 BPB 成熟。

---

## 参考文献

[1] Microsoft AI (2026). [Frontier Pre-training at Microsoft AI](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/). Microsoft AI Blog.

[2] The Microsoft AI Team (2026). [MAI-Thinking-1: Building a Hill-Climbing Machine](https://microsoft.ai/pdf/mai-thinking-1.pdf). Microsoft AI Technical Report.

[3] Reflection Team (2026). [Beam Model Technical Report](https://reflection.ai/documents/beam-model-tech-report). Reflection AI Technical Report.

[4] Reflection AI (2026). [Introducing Beam: Reflection's 501B Open-Weight Model](https://reflection.ai/blog/introducing-beam). Reflection AI Blog.

[5] LLM-Core Xiaomi (2026). [MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement](https://arxiv.org/abs/2610.11959). arXiv preprint arXiv:2610.11959.

[6] NVIDIA (2026). [Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](https://arxiv.org/abs/2606.15007). arXiv preprint arXiv:2606.15007.

[7] NVIDIA (2026). [Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI](https://arxiv.org/abs/2610.38095). arXiv preprint arXiv:2610.38095.

[8] Mistral AI (2026). [Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/resolve/main/assets/technical_report.pdf). Mistral AI Technical Report.

## 引用

引用本文可使用以下 BibTeX：

```bibtex
@article{zou2026agenticpretrain,
  title={Agentic 时代的预训练不能只看 NLL：从 MAI、Beam、ML4 与 Nemotron 说起},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/agentic-native-pretraining-beyond-nll/}
}
```
