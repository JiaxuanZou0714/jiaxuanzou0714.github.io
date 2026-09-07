---
layout: post
title: "Pretraining and scaling as a methodology and scientific perspective"
date: 2026-09-07 12:00:00
description: "Starting from GEN-1.5's long-term pretraining, I discuss general learning frameworks, training-time and test-time scaling, and how efficiency, training stability, and predictability influence research choices."
tags: [llm, pretraining, scaling-law, optimization, embodied-ai, reasoning]
categories: [deep-learning]
featured: true
giscus_comments: true
toc:
  sidebar: left
lang: en
published: true
permalink: /en/blog/2026/pretrain-scaling-methodology-scientific-perspective/
ref: pretrain-scaling-methodology-scientific-perspective
related_posts: false
source_sha: 26541c62ceeee52d
---

I research LLM pretraining and scaling. When I saw the GEN-1.5 loss curve spanning more than eight months of pretraining, my first reaction was:

> This is a universal principle for turning energy into intelligence.

{% include figure.liquid
  path='assets/img/pretrain-scaling/gen15-pretraining.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block '
  width='100%'
  caption='Figure 1. More than eight months of GEN-1.5 pretraining: validation action prediction error across three training stages. <a href="https://generalistai.com/blog/gen-1.5#scaling-pretraining">Source: Generalist, Figure 3</a>.'
  alt='Validation action prediction error curves for GEN-1.5 across three training stages from December 2025 to August 2026.'
  avoid_scaling=true
  zoomable=true
  loading='eager'
%}

**I see pretraining and scaling as a general method for developing reusable capabilities from experience through computation.** Energy powers computation, data provides experience, and training encodes structure from that experience into weights. Language models, world models, and embodied AI can share this research approach.

## 1. General framework and scalability

What conditions does this approach require? Two points from the conversation between Yang Zhilin and Zhang Xiaojun stayed with me: [[2]](https://www.xiaoyuzhoufm.com/episode/65e16b5b6144a933b1d968b5)

- A sufficiently general framework that can learn from a broad range of experience.
- A scalable learning process that can continue to make use of more data and compute.

By framework, I mean the architecture, data representation, training objective, and optimization method. Once these are in place, I focus on the gains from additional compute, efficiency, stability, and predictability.

Dyna-2 scales human video pretraining to one million hours and achieves improved prediction on unseen robot data; after post-training with the same setup, real-world task performance also improves with data scale. Data composition and the video prediction objective also influence this result. [[3]](https://www.dyna.co/dyna-2)

{% include figure.liquid
  path='assets/img/pretrain-scaling/dyna2-data-scaling.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 2. Under the same post-training setup, robot task performance of Dyna-2 improves with more human pretraining data. The y-axis is the average normalized score across 14 tasks. <a href="https://www.dyna.co/dyna-2#fig-8">Source: Dyna-2, Figure 8</a>.'
  alt='As the human pretraining data for Dyna-2 increases from one thousand hours to one million hours, the average normalized scores on robot tasks are 20%, 28%, 45%, and 53%.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

The Dyna-2 results draw my attention to how pretraining improvements transfer to practical tasks. Different domains can use different architectures, metrics, and scaling curves, but each requires research into the learning conditions that support this transfer and continued improvement.

## 2. Designing learning conditions

When I first studied deep learning, I imagined new attention mechanisms based on long-term, short-term, and temporary memory, hoping to outperform standard attention. I would look for an explanation in human cognition, then use it to design a model.

Studying pretraining and scaling shifted my attention to another question: **What architectures, data, and training objectives allow capabilities to develop through learning and improve with scale?** Cognitive analogies can inspire hypotheses, but new mechanisms still need to justify their inclusion through learning outcomes, efficiency, and scalability. "Entities should not be multiplied without necessity" also applies to model design.

This is how I read Sutton's The Bitter Lesson: prioritize general methods that can continue to benefit from more compute. The shift from feature engineering to feature learning leaves the specific form of features to training. Pretraining applies this approach to a broader range of data and tasks. [[4]](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)

Karpathy's Software 2.0 further explains the researcher's role: provide the architecture, data, and objective, then obtain a concrete implementation through optimization. This approach is especially valuable for tasks whose results we can evaluate but for which a complete program is difficult to write by hand. [[5]](https://karpathy.medium.com/software-2-0-a64152b37c35)

I therefore focus my research on the conditions for learning. Simple training objectives can produce complex representations and capabilities. Our understanding of the learning process may be sufficient to support capability improvements while our explanation of the model's internal mechanisms remains incomplete. **Building a capability can precede fully explaining it.**

## 3. Pretraining and scaling

When studying these learning conditions, I focus on two questions: how to develop reusable capabilities, and how to keep improving them with more resources. Pretraining and scaling address these two questions, respectively. Pretraining learns from large datasets, encoding transferable structure into weights so that one model can adapt to many tasks. Scaling studies how increasing data and compute can continue to improve capabilities.

**One aspect of pretraining I value is its ability to reduce the cost of subsequent learning.** A pretrained model can adapt to new tasks through SFT or use demonstrations in context through ICL. GEN-1.5 makes this value tangible by demonstrating ICL from a single demonstration and task adaptation with a few gradient updates. [[1]](https://generalistai.com/blog/gen-1.5)

<figure>
  <video controls playsinline preload="metadata" width="100%" class="img-fluid rounded z-depth-1" aria-label="Official demonstration of GEN-1.5 performing two tasks using examples in context">
    <source src="https://generalistai.com/assets/pages/blog/gen-1.5/assets/videos/two_task_in_context.mp4" type="video/mp4">
    <a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">Watch the official GEN-1.5 demo</a>
  </video>
  <figcaption class="caption">Video 1. After placing demonstrations in the context, GEN-1.5 performs two tasks without gradient updates: unzipping a pencil case and taking money from it. <a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">Source: Generalist official demo</a>.</figcaption>
</figure>

When evaluating a pretrained model, I therefore also consider how efficiently it uses new experience: how many demonstrations, parameter updates, and how much compute it needs to learn a new task. An investment in pretraining can reduce learning costs across many subsequent tasks.

Scaling applies more broadly. Pretraining and RL can improve models with more training compute. At inference time, CoT, search, and environment interaction can also make use of additional compute. The gains from additional compute therefore need to be examined separately during training and inference.

## 4. Two milestones in training and inference

I see OpenAI's 2020 scaling laws research and o1 in 2024 as important advances in training and inference, respectively.

In 2020, OpenAI systematically studied scaling laws for language models, enabling the relationship between parameter count, data size, compute, and loss to be used for prediction and resource allocation. This was a significant advance in training-time scaling; earlier empirical research already existed. [[6]](https://openai.com/index/scaling-laws-for-neural-language-models/)[[7]](https://arxiv.org/abs/1712.00409)

{% include figure.liquid
  path='assets/img/pretrain-scaling/openai-2020-scaling-laws.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 3. The relationship between language model loss and training compute, parameter count, and data size. When other factors are not limiting, each follows a power law. The left plot shows compute estimates under small batch size conditions. <a href="https://arxiv.org/html/2001.08361v1#S1.F1">Source: Kaplan et al., 2020, Figure 1</a>.'
  alt='Original 2020 language model scaling laws figure: loss decreases with compute, model parameter count, and dataset size, respectively.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

In 2024, o1 demonstrated another important advance: models trained with reinforcement learning can effectively use more inference-time compute. **Training can improve the way models use compute, and at inference time, additional compute is then applied to the current problem.** [[8]](https://openai.com/index/learning-to-reason-with-llms/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/o1-compute.webp'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 4. AIME pass@1 accuracy of o1. Left: reinforcement learning training compute; right: inference-time compute. Both x-axes are on a logarithmic scale. <a href="https://openai.com/index/learning-to-reason-with-llms/">Source: OpenAI</a>.'
  alt='The AIME accuracy of o1 improves with increasing reinforcement learning training compute and inference-time compute.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

CoT prompting was systematically proposed by Wei et al. in 2022, and research on test-time compute allocation existed before o1's release. The significance of o1 lies in expanding the practical use of this capability through reinforcement learning. [[9]](https://arxiv.org/abs/2201.11903)[[10]](https://arxiv.org/abs/2408.03314)

I also see the Seed Edge team's EdgeBench as an important advance in test-time scaling. It allocates additional compute to long-horizon environment interaction: the agent continuously obtains observations and feedback, modifies plans, and verifies results, with some extended experiments exceeding 72 hours. The team observed a log-sigmoid relationship between the average score across tasks and interaction time. [[11]](https://seed.bytedance.com/en/blog/edgebench-measuring-real-world-environment-learning-and-discovering-a-new-scaling-law)

{% include figure.liquid
  path='assets/img/pretrain-scaling/edgebench-environment-scaling.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 5. Aggregate performance of EdgeBench across 134 tasks. Left: score changes over 12 hours of environment interaction; right: log-sigmoid fit on a logarithmic time axis. This is the average result across tasks; individual task trajectories vary. <a href="https://edge-bench.org/">Source: EdgeBench official project page, original figure</a>.'
  alt='Official EdgeBench curves: aggregate scores of different models improve with increasing environment interaction time, and the log-sigmoid fit describes the average learning trajectory.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

The experience here can be retained in context, external records, and modified plans, without updating weights. The connection that matters to me is that pretraining improves the ability to use new experience, and inference and environment interaction apply that ability to specific tasks.

In my view, these two milestones clarify two directions of scaling: **training-time scaling improves model capability by increasing training compute; test-time scaling allocates more compute to inference, search, and environment interaction, improving results on the current task.**

## 5. Efficiency and training stability

Once we establish where additional compute can improve results, we also need to study the cost of those improvements. In my own field of pretraining, **efficiency determines the capability level achievable with a given budget, and training stability determines whether those gains can be realized over a complete training run.**

I understand efficiency improvement as a leftward shift of the scaling curve: achieving the same loss requires fewer resources. When the horizontal axis is token count, it corresponds to token efficiency; when the horizontal axis is FLOPs, it corresponds to compute efficiency. Relating either measure to actual energy consumption also requires accounting for per-token compute cost and hardware efficiency.

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-compute-efficiency.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Figure 6. Compute efficiency of the improved Marin training recipe. The blue line is the baseline, and the brown points are the new recipe; horizontal lines compare the FLOPs required to reach the same loss. Within the illustrated budget range, the new recipe requires about half the compute of the baseline. <a href="https://openathena.ai/blog/pretraining-speedup/#combining-all-4">Source: Marin team official blog [18], Figure 12</a>.'
  alt='Official Marin compute efficiency figure: the new recipe improves compute efficiency by approximately 2.09 to 2.26 times relative to the baseline at the same loss.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

By training stability, I mean controlling activation outliers and anomalous changes in gradient norms, weight norms, logits, and loss. At scales such as 3T or 10T parameters, I need to control these numerical and optimization issues to have confidence in scaling up further.

When scaling up, the Kimi K2 team encountered anomalous growth in attention logits. By using QK-Clip in MuonClip to adjust the Q and K projection weights, they completed pretraining on 15.5T tokens, reporting no loss spikes during the period. [[12]](https://www.kimi.ai/blog/kimi-k2)

{% include figure.liquid
  path='assets/img/pretrain-scaling/kimi-muonclip.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-dark'
  width='100%'
  caption='Figure 7. Loss curve of Kimi K2 pretraining with MuonClip on 15.5T tokens. <a href="https://www.kimi.ai/blog/kimi-k2">Source: Kimi K2</a>.'
  alt='Official Kimi K2 curve showing pretraining loss as a function of training token count.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Stable training allows normal numerical growth and fluctuation. Decisions about intervention should be based on dynamics over the full training run and how these changes affect learning.

## 6. Predictability and detailed research

Efficiency and stability also need to be assessed before scaling up. **I need to know whether the gains and training dynamics observed in small-scale experiments will hold at larger scales.** This makes predictability another requirement for scaling research.

Scaling laws provide a basis for resource allocation. Prediction requires clarifying variables, metrics, and applicability: the power law of loss, task scores on logarithmic data axes, and log-sigmoid curves for long-horizon interaction have different meanings. [[13]](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)

I follow Marin because the team makes this process public. For the 535B-A23B run, they first use a scaling ladder to predict the full training trajectory, then check actual training against that prediction. Small-scale training can also provide references for dynamics such as gradient norm, helping to distinguish normal variation from deviations that require investigation. [[14]](https://github.com/marin-community/marin/issues/8435)

Delphi's first experiment better illustrates why I care about this issue. The scaling law fit for small-scale experiments was good, but when scaled to $10^{22}$ FLOPs, the actual loss was 2.5% higher than predicted; training at $10^{23}$ FLOPs diverged. **A good small-scale fit is still insufficient to guarantee that large-scale training matches predictions.** [[17]](https://openathena.ai/blog/delphi/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 8. Delphi first attempt, using the Cautious AdamC recipe. The right panel shows large-scale training deviating from predictions and flags the diverging run. <a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">Source: Marin team official blog, original figure</a>.'
  alt='Delphi first scaling experiment: the 1e22 FLOPs run has a loss 2.5% higher than predicted, and the 1e23 FLOPs run diverges.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Subsequently, the team adopted AdamH, added a learning rate correction that varies with training token count, and re-tuned hyperparameters. The improved $10^{23}$ FLOPs training completed, with a final loss about 0.2% higher than predicted. What I value is this process: correcting the training recipe based on failed results, then validating the predictability of scaling.

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 9. Improved Delphi AdamH recipe. Left: IsoFLOP experiments at different compute budgets; right: small-scale fit and validation at larger scale. <a href="https://openathena.ai/blog/delphi/#delphi-attempt-2-forecasts-1e23-to-within-02">Source: Marin team official blog, original figure</a>.'
  alt='Official Marin Delphi figure: IsoFLOP experiments and scaling law predictions and validation for larger compute budgets.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Marin previously discovered gradient growth issues through the scaling ladder and adopted logit z-loss; the public recipe also documents Hyperball-related methods like MuonH. Each method needs to be evaluated separately. [[15]](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/README.md)

Greg Yang et al.'s Tensor Programs V applies parameterization theory to scale-up: μP allows many optimal hyperparameters to remain stable as model width changes, and μTransfer transfers small-model hyperparameter tuning results to large models. [[16]](https://arxiv.org/abs/2203.03466)

**This is also my understanding of scaling research: the general framework can be simple, but scale-up still requires detailed research.** Parameterization, optimizer, data, and training dynamics jointly determine whether predictions hold at larger scales.

> Make scaling more predictable and stable.

## 7. Summary

GEN-1.5, o1, and EdgeBench show me how general learning frameworks can use more experience and compute across domains and stages. The Marin experiments show that sustaining these gains also requires testing predictions, analyzing deviations, and revising training recipes.

Pretraining encodes broad experience into reusable capabilities and also improves subsequent learning efficiency. Scaling runs through both training and inference: training-time scaling improves model capabilities, while test-time scaling allocates more compute to specific tasks. Their implementations and curves may differ across domains, but the research methodology shares commonalities.

**For me, the methodology is about designing conditions that enable continuous learning and scaling; the scientific perspective requires that the relationship between resource investment and capability improvement be predictable, testable, and correctable.** Efficiency determines the returns on the same investment, stability ensures the learning process can continue, and predictability provides the basis for further scaling up investment. This is also my criterion for choosing research problems in pretraining and scaling.

## References

[1] Generalist Team (2026). [GEN-1.5: Embodied Foundation Models are One-Shot Learners](https://generalistai.com/blog/gen-1.5). Generalist Blog.

[2] Zhang Xiaojun, Yang Zhilin (2024). [A Conversation with Yang Zhilin on a Year of Building an LLM Startup: Expanding Human Aspirations, Probabilistic Non-Consensus, and Sora](https://www.xiaoyuzhoufm.com/episode/65e16b5b6144a933b1d968b5). Zhang Xiaojun Jùn | Business Interviews, Episode 59.

[3] Dyna Robotics (2026). [Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models](https://www.dyna.co/dyna-2). Dyna Research.

[4] Sutton, R. S. (2019). [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf). Personal blog, PDF copy.

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

[15] Marin Community. [MoE Recipe](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/README.md). GitHub project documentation. Accessed: 2026-09-07.

[16] Yang, G., Hu, E. J., Babuschkin, I., et al. (2022). [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466). arXiv preprint arXiv:2203.03466.

[17] Held, W. (2026). [Scaling Laws That Extrapolate 300× Past the Fit](https://openathena.ai/blog/delphi/). Open Athena Blog.

[18] Dial, L. (2026). [Improving our LLM Pretraining Efficiency](https://openathena.ai/blog/pretraining-speedup/). Open Athena Blog.

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026pretrainscaling,
  title={pretrain 和 scaling 作为一种方法论和科学观},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/pretrain-scaling-methodology-scientific-perspective/}
}
```
