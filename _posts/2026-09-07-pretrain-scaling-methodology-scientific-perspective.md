---
layout: post
title: "pretrain 和 scaling 作为一种方法论和科学观"
date: 2026-09-07 12:00:00
description: "从 GEN-1.5 的长期预训练出发，讨论通用学习框架、training-time 与 test-time scaling，以及效率、训练稳定性和可预测性如何影响研究选择。"
tags: [llm, pretraining, scaling-law, optimization, embodied-ai, reasoning]
categories: [deep-learning]
featured: true
giscus_comments: true
toc:
  sidebar: left
lang: zh-CN
published: true
---

我做 LLM pretraining 与 scaling 研究。看到 GEN-1.5 八个多月的 pretraining loss curve 时，我的第一反应是：

> This is a universal principle for turning energy into intelligence.

{% include figure.liquid
  path='assets/img/pretrain-scaling/gen15-pretraining.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block '
  width='100%'
  caption='图 1. GEN-1.5 八个多月的预训练：验证集动作预测误差与三个训练阶段。<a href="https://generalistai.com/blog/gen-1.5#scaling-pretraining">来源：Generalist，Figure 3</a>。'
  alt='GEN-1.5 在 2025 年 12 月至 2026 年 8 月三个训练阶段中的验证集动作预测误差曲线。'
  avoid_scaling=true
  zoomable=true
  loading='eager'
%}

**我把 pretraining 与 scaling 理解为一种通用方法：通过计算，从经验中形成可复用的能力。** 能源支持计算，数据提供经验，训练将其中的结构编码到权重中。语言模型、世界模型和具身智能，可以采用相通的研究方法。

## 1. 通用框架与可扩展性

杨植麟与张小珺的对谈中，我记住了两个条件：[[2]](https://www.xiaoyuzhoufm.com/episode/65e16b5b6144a933b1d968b5)

- 足够通用的框架，能够利用广泛经验。
- 可扩展的学习过程，能够持续利用更多数据和计算。

我理解的框架包括架构、数据表示、训练目标与优化方法。确定这些条件之后，我最关注新增计算的收益、效率、稳定性和可预测性。

Dyna-2 将人类视频预训练扩展到一百万小时，在未见过的机器人数据上取得预测改善；经过相同设置的后训练，实机任务表现也随数据规模改善。数据组成和视频预测目标同样影响这一结果。[[3]](https://www.dyna.co/dyna-2)

{% include figure.liquid
  path='assets/img/pretrain-scaling/dyna2-data-scaling.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 2. 相同后训练设置下，Dyna-2 的实机表现随人类预训练数据增加而改善。纵轴为 14 项任务的平均归一化得分。<a href="https://www.dyna.co/dyna-2#fig-8">来源：Dyna-2，Figure 8</a>。'
  alt='Dyna-2 的人类预训练数据从一千小时增加至一百万小时，实机任务平均归一化得分为 20%、28%、45% 和 53%。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

这里同时存在研究方法的跨领域适用性和学到的能力的迁移。**预训练的改善能够迁移到目标任务，规模扩展才具有实际价值。** 各领域可以采用不同架构、指标与 scaling 曲线。

## 2. 设计学习条件

初学 deep learning 时，我曾根据长期记忆、短期记忆和临时记忆的划分，设想新的 attention 机制，希望超过 standard attention。

现在，我认为认知类比可以启发假设，却不足以说明算法有效。新增机制需要通过学习效果、效率或可扩展性说明其必要性。“如无必要，勿增实体”也适用于模型设计。

Sutton 的 The Bitter Lesson 支持优先研究能持续利用更多计算的通用方法。Feature engineering 向 feature learning 的变化，以及大规模预训练，都体现了这种取向。[[4]](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)

Karpathy 的 Software 2.0 则说明实现方式的变化：研究者提供架构、数据和目标，通过优化确定具体行为。**人类能够判断结果，却未必能写出完整的实现程序。** 学习方法扩大了这类问题的可处理范围。[[5]](https://karpathy.medium.com/software-2-0-a64152b37c35)

由此，我形成了两个判断：

- 构建能力，可以先于完整解释其内部机制。
- 简单的训练目标，可以对应复杂的表示与能力。

## 3. Pretraining 与 scaling

**在我看来，pretraining 形成可复用能力，scaling 研究如何持续有效地增加资源。**

预训练的价值包括改善后续学习效率。GEN-1.5 展示了单次示范的 ICL，以及少量梯度更新的任务适应。大规模预训练之后，新任务所需的数据和计算可以减少。[[1]](https://generalistai.com/blog/gen-1.5)

<figure>
  <video controls playsinline preload="metadata" width="100%" class="img-fluid rounded z-depth-1" aria-label="GEN-1.5 根据上下文中的示范执行两个任务的官方演示">
    <source src="https://generalistai.com/assets/pages/blog/gen-1.5/assets/videos/two_task_in_context.mp4" type="video/mp4">
    <a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">观看 GEN-1.5 官方演示</a>
  </video>
  <figcaption class="caption">视频 1. 将示范放入上下文后，GEN-1.5 分别执行打开笔袋拉链、从笔袋取钱两个任务，无须梯度更新。<a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">来源：Generalist 官方演示</a>。</figcaption>
</figure>

SFT 通过示范更新参数，ICL 利用上下文中的示例。两者都说明，压缩到权重中的结构需要通过泛化和新任务适应来评价。

Scaling 的范围包括预训练、RL，以及推理阶段的 CoT、搜索和环境交互。增加输出长度或运行时间能否改善结果，取决于模型利用额外计算的能力。

因此，我评价模型时，也关注后续改善速度和成本。初始得分接近的两个模型，在获得更多示范、反馈或计算之后，可能表现出不同的学习效率。

## 4. 训练与推理的两个里程碑

我认为，这条研究路线有两个重要的里程碑。

2020 年，OpenAI 系统研究语言模型的 scaling laws，使参数量、数据量、计算量与 loss 之间的关系能够用于预测和资源分配。这是 training-time scaling 的重要进展；更早已有相关经验研究。[[6]](https://openai.com/index/scaling-laws-for-neural-language-models/)[[7]](https://arxiv.org/abs/1712.00409)

{% include figure.liquid
  path='assets/img/pretrain-scaling/openai-2020-scaling-laws.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 3. 语言模型的 loss 与训练计算量、参数量、数据量的关系。在其余因素不构成限制时，各自呈幂律关系。左图计算量为小 batch size 条件下的估计。<a href="https://arxiv.org/html/2001.08361v1#S1.F1">来源：Kaplan 等，2020，Figure 1</a>。'
  alt='2020 年语言模型 scaling laws 原图：loss 分别随计算量、模型参数量和数据集大小增加而下降。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

2024 年，o1 展示了另一个重要进展：经过强化学习训练的模型，能够有效利用更多推理计算。**训练可以改善模型使用计算的方式，推理时再将额外计算用于当前问题。** [[8]](https://openai.com/index/learning-to-reason-with-llms/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/o1-compute.webp'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 4. o1 的 AIME pass@1 准确率。左：强化学习训练计算；右：推理计算。横轴均为对数尺度。<a href="https://openai.com/index/learning-to-reason-with-llms/">来源：OpenAI</a>。'
  alt='o1 的 AIME 准确率随强化学习训练计算和推理计算增加而提高。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

CoT prompting 在 2022 年已由 Wei 等人系统提出，o1 发布前也已有 test-time compute 分配研究。o1 的意义在于通过强化学习扩大了这种能力的实际用途。[[9]](https://arxiv.org/abs/2201.11903)[[10]](https://arxiv.org/abs/2408.03314)

我也把 Seed Edge 团队的 EdgeBench 理解为 test-time scaling 的重要突破之一。它将额外计算用于长期环境交互：agent 持续获取观测和反馈，修改方案、验证结果，部分扩展实验超过 72 小时。团队观察到，跨任务平均得分随交互时间呈 log-sigmoid 关系。[[11]](https://seed.bytedance.com/en/blog/edgebench-measuring-real-world-environment-learning-and-discovering-a-new-scaling-law)

{% include figure.liquid
  path='assets/img/pretrain-scaling/edgebench-environment-scaling.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 5. EdgeBench 在 134 项任务上的汇总表现。左：12 小时环境交互中的得分变化；右：对数时间坐标下的 log-sigmoid 拟合。这是跨任务平均结果，单个任务的轨迹存在差异。<a href="https://edge-bench.org/">来源：EdgeBench 官方项目页，原图</a>。'
  alt='EdgeBench 官方曲线：不同模型的汇总得分随环境交互时间增加而改善，log-sigmoid 拟合描述平均学习轨迹。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

这里的经验可以保留在上下文、外部记录和修改后的方案中，无须更新权重。我看重的联系是：预训练改善利用新经验的能力，推理和环境交互再将这种能力用于具体任务。

在我看来，这两个里程碑明确了 scaling 的两个方向：**training-time scaling 通过增加训练计算改善模型能力；test-time scaling 将更多计算用于推理、搜索和环境交互，改善当前任务的结果。**

## 5. 效率与训练稳定性

**效率决定相同预算能够达到的能力水平，稳定性决定这些收益能否在完整训练中实现。**

我把效率改善理解为 scaling 曲线向左移动：达到相同 loss，所需资源更少。横轴是 token 数时，对应 token efficiency；横轴是 FLOPs 时，对应 compute efficiency。两者与实际能耗还受单 token 计算成本和硬件效率影响。

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-compute-efficiency.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='图 6. Marin 改进训练方案后的计算效率。蓝线为基线，棕点为新方案；水平线比较达到相同 loss 所需的 FLOPs。在图示预算范围内，新方案所需计算量约为基线的一半。<a href="https://openathena.ai/blog/pretraining-speedup/#combining-all-4">来源：Marin 团队官方博客 [18]，Figure 12</a>。'
  alt='Marin 官方计算效率图：新方案相对基线在相同 loss 下的计算效率提升约为 2.09 至 2.26 倍。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

我关注的训练稳定性，具体指激活值 outlier、gradient norm、weight norm、logits 和 loss 的异常变化。面对 3T、10T 参数这样的规模，只有控制好这些数值与优化问题，我才有信心继续 scale up。

Kimi K2 团队在扩展时遇到 attention logits 异常增大，通过 MuonClip 中的 QK-Clip 调整 Q、K 投影权重，完成了 15.5T tokens 的预训练，报告期间没有出现 loss spike。[[12]](https://www.kimi.ai/blog/kimi-k2)

{% include figure.liquid
  path='assets/img/pretrain-scaling/kimi-muonclip.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-dark'
  width='100%'
  caption='图 7. Kimi K2 使用 MuonClip 完成 15.5T tokens 预训练的 loss 曲线。<a href="https://www.kimi.ai/blog/kimi-k2">来源：Kimi K2</a>。'
  alt='Kimi K2 官方展示的预训练损失随训练 token 数变化的曲线。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

稳定训练允许正常的数值增长与波动。干预依据应来自完整训练动态，以及这些变化对学习效果的影响。

## 6. 可预测性与精细研究

**既然依赖 scaling 获得收益，我就需要在扩大投入之前建立可预测性。**

Scaling laws 使资源分配有据可依。预测需要明确变量、指标和适用范围：loss 的幂律、对数数据轴上的任务得分、长期交互的 log-sigmoid 曲线，具有不同含义。[[13]](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)

我关注 Marin，正是因为他们公开了这个过程。535B-A23B 训练记录中的做法是：先用 scaling ladder 预测大模型的完整轨迹，再检查实际训练是否符合预期。小规模训练还能提供梯度范数等动态的参考，帮助区分正常变化与需要调查的偏离。[[14]](https://github.com/marin-community/marin/issues/8435)

Delphi 的首次实验更能说明我为什么关注这个问题。小规模实验的 scaling law 拟合良好，扩展到 $10^{22}$ FLOPs 时，实际 loss 却比预测高 2.5%；$10^{23}$ FLOPs 的训练则出现了发散。**小规模拟合良好，仍不足以保证大规模训练符合预测。** [[17]](https://openathena.ai/blog/delphi/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 8. Delphi 首次实验，采用 Cautious AdamC recipe。右图显示大规模训练偏离预测，并标出发散的运行。<a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">来源：Marin 团队官方博客，原图</a>。'
  alt='Delphi 首次 scaling 实验：10 的 22 次方 FLOPs 运行的 loss 比预测高 2.5%，10 的 23 次方 FLOPs 运行发散。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

随后，团队采用 AdamH，加入随训练 token 数变化的学习率修正，并重新调参。改进后的 $10^{23}$ FLOPs 训练完成，最终 loss 比预测高约 0.2%。我看重的是这个过程：通过失败结果修正训练方案，再验证 scaling 的可预测性。

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 9. 改进后的 Delphi AdamH recipe。左：不同计算预算下的 IsoFLOP 实验；右：小规模拟合与更大规模验证。<a href="https://openathena.ai/blog/delphi/#delphi-attempt-2-forecasts-1e23-to-within-02">来源：Marin 团队官方博客，原图</a>。'
  alt='Marin 官方 Delphi 图：IsoFLOP 实验及 scaling law 对更大计算预算的预测与验证。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Marin 曾通过 scaling ladder 发现梯度增长问题，并采用 logit z-loss；公开 recipe 也记录了 Hyperball 相关方法 MuonH。具体方法的效果需要逐项验证。[[15]](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/README.md)

Greg Yang 等人的 Tensor Programs V 则将参数化理论用于规模扩展：μP 支持许多最优超参数随模型宽度保持稳定，μTransfer 据此将小模型调参结果迁移到大模型。[[16]](https://arxiv.org/abs/2203.03466)

**这也是我对 scaling 研究的理解：通用框架可以简洁，规模扩展仍需要精细研究。** 参数化、优化器、数据与训练动态，共同决定预测在更大规模下是否成立。

> Make scaling more predictable and stable.

## 7. 总结

从 GEN-1.5 的预训练曲线，到 o1 的推理计算，再到 EdgeBench 的长期环境交互，我关注的是同一个问题：如何通过通用的学习框架，持续利用经验和计算改善能力。

Pretraining 将广泛经验编码为可复用的能力，也改善后续学习效率。Scaling 则贯穿训练与推理：training-time scaling 改善模型能力，test-time scaling 将更多计算用于具体任务。它们在不同领域的实现和曲线可以不同，研究方法仍有共通之处。

**对我而言，方法论是设计能够持续学习和扩展的条件；科学观是要求资源投入与能力改善之间的关系能够预测、检验和修正。** 效率决定相同投入的收益，稳定性保证学习过程能够持续，可预测性为进一步扩大投入提供依据。这也是我选择 pretraining 与 scaling 研究问题的标准。

## 参考文献

[1] Generalist Team (2026). [GEN-1.5: Embodied Foundation Models are One-Shot Learners](https://generalistai.com/blog/gen-1.5). Generalist Blog.

[2] 张小珺、杨植麟（2024）。[和杨植麟聊大模型创业这一年：人类理想的增量、有概率的非共识和 Sora](https://www.xiaoyuzhoufm.com/episode/65e16b5b6144a933b1d968b5)。《张小珺 Jùn｜商业访谈录》，第 59 期。

[3] Dyna Robotics (2026). [Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models](https://www.dyna.co/dyna-2). Dyna Research.

[4] Sutton, R. S. (2019). [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf). 个人文章，PDF 副本。

[5] Karpathy, A. (2017). [Software 2.0](https://karpathy.medium.com/software-2-0-a64152b37c35). Medium.

[6] Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). arXiv preprint arXiv:2001.08361.

[7] Hestness, J., Narang, S., Ardalani, N., et al. (2017). [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409). arXiv preprint arXiv:1712.00409.

[8] OpenAI (2024). [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/). OpenAI Blog.

[9] Wei, J., Wang, X., Schuurmans, D., et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903). arXiv preprint arXiv:2201.11903.

[10] Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314). arXiv preprint arXiv:2408.03314.

[11] ByteDance Seed (2026). [EdgeBench: Measuring Real-World Environment Learning and Discovering a New Scaling Law](https://seed.bytedance.com/en/blog/edgebench-measuring-real-world-environment-learning-and-discovering-a-new-scaling-law). Seed Blog.

[12] Moonshot AI (2025). [Kimi K2: Open Agentic Intelligence](https://www.kimi.ai/blog/kimi-k2). Kimi Blog.

[13] Weng, L. (2026). [Scaling Laws, Carefully](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/). Lil’Log.

[14] Marin Community (2026). [[Hero Run] 535B-A23B on 18T Tokens](https://github.com/marin-community/marin/issues/8435). GitHub Issue #8435.

[15] Marin Community. [MoE Recipe](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/README.md). GitHub 项目文档。访问日期：2026-09-07。

[16] Yang, G., Hu, E. J., Babuschkin, I., et al. (2022). [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466). arXiv preprint arXiv:2203.03466.

[17] Held, W. (2026). [Scaling Laws That Extrapolate 300× Past the Fit](https://openathena.ai/blog/delphi/). Open Athena Blog.

[18] Dial, L. (2026). [Improving our LLM Pretraining Efficiency](https://openathena.ai/blog/pretraining-speedup/). Open Athena Blog.

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026pretrainscaling,
  title={pretrain 和 scaling 作为一种方法论和科学观},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/pretrain-scaling-methodology-scientific-perspective/}
}
```
