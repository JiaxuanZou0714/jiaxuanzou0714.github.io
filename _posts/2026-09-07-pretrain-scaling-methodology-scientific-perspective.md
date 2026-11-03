---
layout: post
title: "预训练与 scaling：一种通用方法论"
date: 2026-09-07 12:00:00
last_modified_at: 2026-11-03
description: "把预训练与 scaling 看作一种跨领域的方法论：从数据中学习世界知识，在不同规模上检验改进，并用效率、稳定性和预测指导训练。"
tags: [llm, pretraining, scaling-law, optimization, embodied-ai, reasoning]
categories: [deep-learning]
featured: true
giscus_comments: true
toc:
  sidebar: left
lang: zh-CN
published: true
---

本文最初发表于 2026 年 9 月 7 日，次日修订。2026 年 11 月 3 日更新了措辞和排版，并在第 2 节加入注明日期的补充。

先看 GEN-1.5 报告里的训练曲线。八个多月、三个训练阶段，模型对下一步动作的预测误差一直在下降。

{% include figure.liquid
  path='assets/img/pretrain-scaling/gen15-pretraining.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block '
  width='100%'
  caption='图 1. GEN-1.5 预训练过程：三个训练阶段中的验证集动作预测误差。<a href="https://generalistai.com/blog/gen-1.5#scaling-pretraining">来源：Generalist，Figure 3</a>。'
  alt='GEN-1.5 在 2025 年 12 月至 2026 年 8 月三个训练阶段中的验证集动作预测误差曲线。'
  avoid_scaling=true
  zoomable=true
  loading='eager'
%}

我是这样理解预训练与 scaling 的。数据里包含世界知识（world knowledge），比如事物如何运作、事件之间有什么联系、人会如何行动。预训练把其中一部分知识压缩进模型权重。随着参数和数据增加，训练得当的模型可以学到、存下更多知识。对新任务的泛化、微调和上下文学习，都在一定程度上利用了权重里已有的知识。

所以，我把预训练与 scaling 看作一种通用方法论。语言模型、世界模型和具身智能只是几个例子。这个思路也可以用到其他智能领域，包括我们还没有探索的领域。

## 1. 通用框架与可扩展性

在 2024 年与张小珺的对谈中，月之暗面创始人兼 CEO 杨植麟谈到，选择学习框架时要关注两点。我概括为：[[2]](https://www.xiaoyuzhoufm.com/episode/65e16b5b6144a933b1d968b5)

- 框架要足够通用，能用同一套方法处理许多不同的问题。
- 框架要 scalable。

这里的框架，包括模型、数据的表示方式、训练目标和训练方法。接下来，我关心几个实际问题：多投入一些计算，效果能好多少？能不能用更少的计算达到同样的效果？规模变大以后，训练还能不能保持稳定？小实验能不能告诉我们，大规模训练会得到什么结果？

Dyna-2 就是一个具体例子。团队把用于预训练的第一视角人类视频扩展到一百万小时。在相同的后训练设置下，视频数据越多，模型在 14 项真实机器人任务上的平均表现越好。[[3]](https://www.dyna.co/dyna-2)

{% include figure.liquid
  path='assets/img/pretrain-scaling/dyna2-data-scaling.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 2. 相同后训练设置下，Dyna-2 的实机表现随人类预训练数据增加的变化。纵轴为 14 项任务的平均归一化得分。<a href="https://www.dyna.co/dyna-2#fig-8">来源：Dyna-2，Figure 8</a>。'
  alt='Dyna-2 的人类预训练数据从 1,000 小时增加至 1,000,000 小时，实机任务平均归一化得分依次为 20%、28%、45% 和 53%。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Dyna-2 表明，模型在预训练中学到的东西，可以帮助它完成新任务。我也想在其他领域看到同样的结果：给模型更多经验，它能不能把我们希望它做的事情做得更好？

## 2. 奥卡姆剃刀原则

*如无必要，勿增实体。*

任何改进，无论是 attention、MoE、优化器，还是整个模型架构，都应该经过 Scaling Ladder 的检验。在相同训练 FLOPs 下比较性能，或者比较达到相同性能所需的 FLOPs，再把这组比较逐步扩展到更大规模。相对于充分调优的 baseline，收益应该随着规模增大保持下来，而不是越来越小。

这也是为什么我对 nanoGPT、nanochat 这类 speedrun 的夸张宣传比较谨慎。反复调优可能只是在过拟合一个 benchmark、一套固定的训练设置。小规模上的好成绩，还不足以证明这个方法适合大规模预训练。看到更广泛的宣传结论时，我会先看它有没有经过 scaling 验证。

*2026 年 11 月 3 日补充。* 围绕 Dust 的 speedrun 热潮就是一个例子。在固定设置上反复调优取得更好的分数，还不足以让我相信这些收益能在更大规模下保持。[[18]](https://qlabs.sh/research/dust)

我在 Sutton 的 The Bitter Lesson 中也看到了类似的思想。回看传统机器学习到深度学习的变化，很多手工特征和复杂流程逐渐被端到端学习取代。Sutton 的观点是，能够持续从更多计算中获益的通用方法，长期来看往往会胜出。对我来说，这也是预训练与 scaling 背后的一条指导原则。[[4]](https://heartyhaven.github.io/files/bitter_lesson.pdf)

Karpathy 的 Software 2.0 描述了同样的转变：我们选择模型、数据和目标，再让训练把程序学到权重里。[[5]](https://karpathy.medium.com/software-2-0-a64152b37c35)

简单的训练目标，也能产生能力远超训练任务本身的复杂系统。比如，预测下一个 token，可以让模型学到后来用于回答问题、编写代码的知识。

## 3. 预训练与 scaling

有了预训练，模型遇到新任务时就能利用权重中已有的世界知识，无须每次从零开始。微调通过更新权重来适应任务。上下文学习则用输入中的示例引导模型，权重保持不变。

GEN-1.5 展示了这两种方式。在上下文里给出一次示范，模型就可能学会一个新任务，无须更新权重。它也可以用很少的梯度更新来适应新任务。[[1]](https://generalistai.com/blog/gen-1.5)

<figure>
  <video controls playsinline preload="metadata" width="100%" class="img-fluid rounded z-depth-1" aria-label="GEN-1.5 根据上下文中的示范执行两个任务的官方演示">
    <source src="https://generalistai.com/assets/pages/blog/gen-1.5/assets/videos/two_task_in_context.mp4" type="video/mp4">
    <a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">观看 GEN-1.5 官方演示</a>
  </video>
  <figcaption class="caption">视频 1. GEN-1.5 根据上下文中的示范打开笔袋拉链、从笔袋取钱，全程无须更新权重。 <a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">来源：Generalist 官方演示</a>。</figcaption>
</figure>

预训练的质量，也体现在模型学新东西有多省力。下一个任务需要多少示例、多少次权重更新、多少额外计算？这些问题和 benchmark 分数一样值得关注。预训练的成本付出一次，学到的知识就能在许多任务上反复使用。

Scaling 关心的是，增加参数、数据或计算后，模型能好多少。增加预训练或强化学习的计算，可以改善那个供许多任务复用的模型。在推理时，增加思考、搜索或交互的计算，则可以帮助模型更好地完成眼前的任务。训练和推理使用计算的方式不同，我会分别衡量它们的收益。

## 4. 两个里程碑：训练时与测试时的 scaling

两个里程碑可以帮助我们看清这个区别：2020 年的语言模型 scaling laws 对应 train-time scaling，2024 年的 o1 对应 test-time scaling。

2020 年，OpenAI 把 train-time scaling 变成了可量化的规律。随着模型参数量、数据量和训练计算增加，语言模型的 loss 沿着可预测的曲线下降。研究者可以用这些曲线估计更大规模训练的结果。更早之前，Hestness 等人也研究过类似的扩展规律。[[6]](https://arxiv.org/abs/2001.08361)[[7]](https://arxiv.org/abs/1712.00409)

{% include figure.liquid
  path='assets/img/pretrain-scaling/openai-2020-scaling-laws.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 3. 语言模型的 loss 随训练计算量、模型参数量和数据量增加而下降。左图计算量为小 batch size 条件下的估计。<a href="https://arxiv.org/html/2001.08361v1#S1.F1">来源：Kaplan 等，2020，Figure 1</a>。'
  alt='2020 年语言模型 scaling laws 原图：loss 分别随计算量、模型参数量和数据集大小增加而下降。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

2024 年，o1 让 test-time scaling 成为一个重要方向。强化学习教会模型更有效地推理。在回答前给训练好的模型更多计算，让它多想一会儿，又能提高它解决难题的表现。下面两张图分别展示了增加强化学习训练计算和测试时计算带来的收益。[[8]](https://openai.com/index/learning-to-reason-with-llms/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/o1-compute.webp'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 4. o1 的 AIME 准确率随强化学习训练计算（左）和测试时计算（右）增加而提高。准确率使用 pass@1，两个计算量坐标轴均为对数尺度。<a href="https://openai.com/index/learning-to-reason-with-llms/">来源：OpenAI</a>。'
  alt='o1 的 AIME 准确率随强化学习训练计算和测试时计算增加而提高。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

思维链提示和测试时计算的研究都早于 o1。o1 的里程碑意义在于，通过大规模强化学习，教会模型更好地利用额外的推理时间。[[9]](https://arxiv.org/abs/2201.11903)[[10]](https://arxiv.org/abs/2408.03314)

Seed Edge 团队的 EdgeBench 把 test-time scaling 扩展到了真实环境中的智能体。智能体可以先尝试，检查结果，再调整下一步。有些运行持续超过 72 小时。交互时间越长，任务平均得分越高；在对数时间坐标下，这个趋势可以用 log-sigmoid 曲线拟合。[[11]](https://seed.bytedance.com/en/blog/edgebench-measuring-real-world-environment-learning-and-discovering-a-new-scaling-law)

{% include figure.liquid
  path='assets/img/pretrain-scaling/edgebench-environment-scaling.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 5. EdgeBench 在 134 项任务上的平均得分。左：12 小时交互中的得分变化。右：对数时间坐标下的 log-sigmoid 拟合。<a href="https://edge-bench.org/">来源：EdgeBench 官方项目页</a>。'
  alt='EdgeBench 官方曲线：不同模型的平均得分随环境交互时间增加而提高，log-sigmoid 拟合描述平均学习轨迹。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

在这些运行中，模型权重保持不变。新经验积累在上下文、日志和智能体修改过的文件里。模型利用训练时学到的知识理解反馈，再决定下一步尝试什么。

Train-time scaling 训练出能力更强的模型。Test-time scaling 则给这个模型更多计算，让它更好地解决眼前的任务。

## 5. Scaling 的效率与稳定性

既然预训练与 scaling 是核心方法论，我最关注的就是两件事：效率和稳定性。每投入一份计算，能换来多少进步？规模扩大以后，训练还能不能保持稳定？

计算效率提高，意味着用更少的 FLOPs 达到相同的 loss。在 loss 对计算量的图上，这表现为曲线向左移动。用更少的 token 达到同样的效果，是另一种效率，但每个 token 的计算开销可能不同。硬件利用率也会影响实际花费的时间和能耗。

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-compute-efficiency.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='图 6. 在图示范围内，Marin 的新配方用约一半的 baseline FLOPs 达到相同 loss。蓝色为 baseline，棕色为新配方。水平线比较达到相同 loss 所需的计算。<a href="https://openathena.ai/blog/pretraining-speedup/#combining-all-4">来源：Marin 团队官方博客 [17]，Figure 12</a>。'
  alt='Marin 官方计算效率图：新配方相对 baseline 在相同 loss 下的计算效率提升约为 2.09 至 2.26 倍。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

保持稳定，就是让模型在长时间训练中持续学到东西。我们会观察 loss spike，以及激活值、梯度、权重和 attention logits 的异常增长。这些信号可以帮助我们在训练出问题之前发现异常。

Kimi K2 团队在训练时发现 attention logits 增长过大，于是在 MuonClip 中加入 QK-Clip 来控制这种增长。最终，模型完成了 15.5T tokens 的预训练，没有出现 loss spike。[[12]](https://www.kimi.ai/blog/kimi-k2)

{% include figure.liquid
  path='assets/img/pretrain-scaling/kimi-muonclip.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-dark'
  width='100%'
  caption='图 7. Kimi K2 使用 MuonClip 完成 15.5T tokens 预训练的 loss 曲线。<a href="https://www.kimi.ai/blog/kimi-k2">来源：Kimi K2</a>。'
  alt='Kimi K2 官方展示的预训练 loss 随训练 token 数变化的曲线。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

看到梯度范数增大，先检查它是否伴随着 loss spike 或学习变慢，再决定要不要干预。如果提出了改进，就在相同计算预算下与原配方比较，同时看训练是否更稳定、最终 loss 是否更好。

## 6. 把 scaling 做成一门可预测的科学

即使还没有完整的学习理论，我们仍然可以把 scaling 做成一门可预测的科学。我想到的是开普勒时代的天文学：认真观测，建立数学规律，再用新的观测检验预测。

Scaling laws 可以帮助我们估计，更大的计算预算能把 loss 降到什么程度，选择什么训练配方，以及训练是否符合预期。它的价值，在于预测超出已有拟合实验范围的结果。[[13]](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)

Marin 的实践就是这样。团队先用小模型的 Scaling Ladder 预测大规模训练的 loss 和训练动态，再把实际结果与预测对照。[[14]](https://github.com/marin-community/marin/issues/8435)[[15]](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/README.md)

Delphi 的早期实验说明了为什么要做这种检验。在小规模上拟合得很好的曲线，给出了过于乐观的预测：到了 10²² FLOPs，实际 loss 比预测高 2.5%；到了 10²³ FLOPs，训练直接发散。[[16]](https://openathena.ai/blog/delphi/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 8. Delphi 的第一套配方在小规模上拟合良好，到了更大规模却偏离预测。10²³ FLOPs 的训练出现发散。<a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">来源：Marin 团队官方博客</a>。'
  alt='Delphi 首次 scaling 实验：10²² FLOPs 训练的 loss 比预测高 2.5%，10²³ FLOPs 的训练发散。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

团队修改了优化器和学习率调度。下一次 10²³ FLOPs 的训练顺利完成，最终 loss 与预测的偏差约在 0.2% 以内。

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='图 9. 修改后的 Delphi 配方对 10²³ FLOPs 训练结果的预测误差约在 0.2% 以内。左：固定计算预算下的实验。右：预测与更大规模训练的结果。<a href="https://openathena.ai/blog/delphi/#delphi-attempt-2-forecasts-1e23-to-within-02">来源：Marin 团队官方博客</a>。'
  alt='Marin 官方 Delphi 图：IsoFLOP 实验，以及 scaling law 对更大计算预算的预测与验证。'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

好的 scaling 研究会先做预测，再到更大规模上检验。结果不一致时，就查找原因、修正方法。这样，小实验才能指导昂贵的大规模训练。

## 7. 一种跨领域的方法论

预训练把数据中的世界知识压缩进权重，让许多任务都能使用。Scaling 研究的是，增加资源以后，模型还能多学到多少、多做成多少事情。我把它看作一种可以带到其他智能领域的方法论，包括我们还没有探索的领域。

我们要做的，是让学习框架保持简单，在不同规模上检验效率和稳定性，并在投入更多计算前做出预测。如果大规模训练与预测不符，就从差异中找原因，再修改配方。

## 参考文献

[1] Generalist Team (2026). [GEN-1.5: Embodied Foundation Models are One-Shot Learners](https://generalistai.com/blog/gen-1.5). Generalist Blog.

[2] 张小珺、杨植麟（2024）。[和杨植麟聊大模型创业这一年：人类理想的增量、有概率的非共识和 Sora](https://www.xiaoyuzhoufm.com/episode/65e16b5b6144a933b1d968b5)。《张小珺 Jùn｜商业访谈录》，第 59 期。

[3] Dyna Robotics (2026). [Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models](https://www.dyna.co/dyna-2). Dyna Research.

[4] Sutton, R. S. (2019). [The Bitter Lesson](https://heartyhaven.github.io/files/bitter_lesson.pdf). 个人文章，PDF 存档。

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

[15] Marin Community. [MoE Recipe](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/README.md). GitHub 项目文档。访问日期：2026-11-03。

[16] Held, W. (2026). [Scaling Laws That Extrapolate 300× Past the Fit](https://openathena.ai/blog/delphi/). Open Athena Blog.

[17] Dial, L. (2026). [Improving our LLM Pretraining Efficiency](https://openathena.ai/blog/pretraining-speedup/). Open Athena Blog.

[18] Dahal, S., Mandal, B., Gülbahar, S., & Vegesna, A. (2026). [Dust: Pretraining Transformers Without Backpropagation](https://qlabs.sh/research/dust). Q Labs Research. 2026 年 10 月。

## 引用

引用本文可使用以下 BibTeX：

```bibtex
@article{zou2026pretrainscaling,
  title={预训练与 scaling：一种通用方法论},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/pretrain-scaling-methodology-scientific-perspective/}
}
```
