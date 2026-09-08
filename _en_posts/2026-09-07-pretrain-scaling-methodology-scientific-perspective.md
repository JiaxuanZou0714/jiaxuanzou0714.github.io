---
source_sha: 16d93abdebe95fba
layout: post
title: "Pretraining and scaling as a methodology and scientific perspective"
date: 2026-09-07 12:00:00
description: "Starting from GEN-1.5's long-term pretraining, this article discusses general learning frameworks, training-time and test-time scaling, and how efficiency, training stability, and predictability guide research decisions."
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
---

The technical report for GEN-1.5 presents a pretraining loss curve spanning over eight months: across three consecutive training phases, the validation next-action prediction error shows a continuous downward trend.

{% include figure.liquid
  path='assets/img/pretrain-scaling/gen15-pretraining.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block '
  width='100%'
  caption='Figure 1. GEN-1.5 pretraining process: validation action prediction error across three training stages. <a href="https://generalistai.com/blog/gen-1.5#scaling-pretraining">Source: Generalist, Figure 3</a>.'
  alt='GEN-1.5 validation action prediction error curve across three training phases from December 2025 to August 2026.'
  avoid_scaling=true
  zoomable=true
  loading='eager'
%}

Pretraining and scaling constitute a systematic methodology for extracting general computational capabilities from data experience. Data supplies experience, compute powers numerical optimization, and training encodes transferable structures from that experience into model parameters. This methodology shares underlying principles across language models, world models, and embodied intelligence.

## 1. General framework and scalability

The effective operation of general learning methods rests on two basic conditions: [[2]](https://www.xiaoyuzhoufm.com/episode/65e16b5b6144a933b1d968b5)

- A sufficiently general framework capable of accommodating and utilizing heterogeneous, multi-source experiential data.
- A scalable learning process capable of continuously translating additional data and compute budgets into performance gains.

The learning framework encompasses model architecture, data representation, training objectives, and optimization algorithms. Once the framework is established, the core engineering questions center on the returns from additional compute, compute efficiency, numerical stability, and extrapolation predictability.

Dyna-2 scaled first-person human video pretraining data to one million hours, achieving prediction accuracy gains on unannotated robot data. Holding post-training configurations constant, average normalized scores across 14 physical evaluation tasks improved with increasing pretraining data volume, while data composition and video prediction objectives also exerted clear effects on transfer performance. [[3]](https://www.dyna.co/dyna-2)

{% include figure.liquid
  path='assets/img/pretrain-scaling/dyna2-data-scaling.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 2. Under identical post-training configurations, physical evaluation performance of Dyna-2 as a function of human pretraining data scale. The vertical axis shows average normalized scores across 14 tasks. <a href="https://www.dyna.co/dyna-2#fig-8">Source: Dyna-2, Figure 8</a>.'
  alt='As human pretraining data for Dyna-2 scales from 1,000 to 1,000,000 hours, average normalized scores on physical robot tasks reach 20%, 28%, 45%, and 53%.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Dyna-2 demonstrates the feasibility of transferring pretrained representations to concrete downstream tasks. While different domains may adopt distinct model architectures, evaluation metrics, and scaling curves, the central research objective remains identifying which learning conditions reliably support cross-task transfer and sustained performance gains.

## 2. Designing learning conditions

An early heuristic in deep learning exploration was designing specialized attention mechanisms based on intuitive divisions of human cognition, such as mapping long-term, short-term, and working memory to separate computational modules. However, cognitive analogies only offer hypothesis inspiration; the efficacy of added mechanisms in large-scale training must be established through empirical results.

The central problem of model design is determining which architectures, data, and training objectives allow capabilities to emerge naturally through optimization and improve systematically with scale. Newly introduced manual mechanisms must demonstrate necessity through final learning outcomes, parameter efficiency, and scalability. Occam's razor applies equally to network architecture design.

In The Bitter Lesson, Sutton observed that general methods leveraging computation and continuously benefiting from scaled compute outperform handcrafted priors over the long run. The transition from feature engineering to feature learning essentially delegate representation construction to numerical optimization; pretraining extends this principle across multiple tasks and massive data volumes. [[4]](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)

In Software 2.0, Karpathy described the corresponding engineering paradigm: researchers define network topology, assemble datasets, and specify loss functions, while optimization algorithms search parameter space for the concrete program. For tasks where output quality can be evaluated reliably but explicit logic rules are intractable to specify by hand, this paradigm offers distinct advantages. [[5]](https://karpathy.medium.com/software-2-0-a64152b37c35)

Simple objective functions can drive the formation of complex representations and high-level capabilities. In engineering systems, building effective capabilities can precede complete theoretical explanations of internal representation mechanisms.

## 3. Pretraining and scaling

Pretraining and scaling address two distinct problem dimensions. Pretraining addresses the formation of reusable capabilities: by training on massive unlabeled or weakly labeled data, general statistical structures are encoded into weights, providing a single base model with foundations for diverse tasks. Scaling addresses the continuous enhancement of capabilities, investigating how performance evolves as parameter count, data volume, and compute budgets expand.

The core value of pretraining lies in lowering downstream adaptation costs. Pretrained models can adapt to new tasks either by updating parameters via supervised fine-tuning (SFT) or by leveraging input demonstrations in prompt context via in-context learning (ICL). GEN-1.5 demonstrates single-shot in-context learning without gradient updates, as well as task adaptation requiring only minimal gradient steps. [[1]](https://generalistai.com/blog/gen-1.5)

<figure>
  <video controls playsinline preload="metadata" width="100%" class="img-fluid rounded z-depth-1" aria-label="Official demonstration of GEN-1.5 performing two tasks using examples in context">
    <source src="https://generalistai.com/assets/pages/blog/gen-1.5/assets/videos/two_task_in_context.mp4" type="video/mp4">
    <a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">Watch the official GEN-1.5 demo</a>
  </video>
  <figcaption class="caption">Video 1. With demonstrations placed in context, GEN-1.5 executes two tasks without gradient updates: unzipping a pencil case and retrieving money from inside. <a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">Source: Generalist official demo</a>.</figcaption>
</figure>

Evaluating pretraining quality involves not only absolute performance on static benchmarks, but also the efficiency with which a model absorbs new experience: the demonstration count, optimization steps, and compute required to learn a new task. Substantial upfront investment in pretraining amortizes adaptation costs across multiple downstream tasks.

The scope of scaling spans the full model lifecycle. During training, expanding pretraining and reinforcement learning compute enhances foundational representations; during inference, allocating compute through chain-of-thought rollout, search algorithms, or environment interactions similarly elevates output quality. Compute allocation benefits must therefore be modeled and assessed separately at training time and inference time.

## 4. Two milestones in training and inference

The empirical validation of language model scaling laws in 2020 and the release of the o1 model in 2024 mark major scaling milestones across the training and inference phases, respectively.

In 2020, OpenAI systematically characterized scaling laws for language models, formalizing the relationships between model parameters, training tokens, compute budgets, and pretraining cross-entropy loss into measurable empirical regularities, establishing quantitative prediction and resource planning benchmarks for large-scale training. Prior to this work, Hestness et al. had conducted early empirical scaling investigations in deep learning. [[6]](https://openai.com/index/scaling-laws-for-neural-language-models/)[[7]](https://arxiv.org/abs/1712.00409)

{% include figure.liquid
  path='assets/img/pretrain-scaling/openai-2020-scaling-laws.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 3. Relationship between language model cross-entropy loss and training compute, parameter count, and data volume. The left plot estimates compute under small batch size conditions. <a href="https://arxiv.org/html/2001.08361v1#S1.F1">Source: Kaplan et al., 2020, Figure 1</a>.'
  alt='Original 2020 language model scaling laws figure: loss decreases with compute, model parameter count, and dataset size, respectively.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

In 2024, o1 marked a significant development in test-time compute scaling: by optimizing reasoning trajectories through reinforcement learning, the model effectively leverages larger inference-time compute budgets. Training optimizes the efficiency with which the model generates intermediate reasoning steps and conducts strategy search, while inference directly converts additional compute into problem-solving accuracy on complex tasks. [[8]](https://openai.com/index/learning-to-reason-with-llms/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/o1-compute.webp'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 4. AIME pass@1 accuracy of o1. Left: reinforcement learning training compute; right: inference-time compute. Both horizontal axes are on logarithmic scales. <a href="https://openai.com/index/learning-to-reason-with-llms/">Source: OpenAI</a>.'
  alt='The AIME accuracy of o1 improves with increasing reinforcement learning training compute and inference-time compute.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Chain-of-thought (CoT) prompting was formally introduced by Wei et al. in 2022, accompanied by prior research on test-time compute allocation. The contribution of o1 was operationalizing inference compute scaling systematically through large-scale reinforcement learning. [[9]](https://arxiv.org/abs/2201.11903)[[10]](https://arxiv.org/abs/2408.03314)

EdgeBench from the Seed Edge team pushed test-time scaling into long-horizon interaction with real environments. Agents iteratively adjust strategies and verify execution outcomes while receiving ongoing environment observations and feedback, with extended task runs surpassing 72 hours. Aggregate statistics across tasks show average scores following a log-sigmoid trajectory relative to logarithmic interaction time. [[11]](https://seed.bytedance.com/en/blog/edgebench-measuring-real-world-environment-learning-and-discovering-a-new-scaling-law)

{% include figure.liquid
  path='assets/img/pretrain-scaling/edgebench-environment-scaling.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 5. Aggregate performance of EdgeBench across 134 tasks. Left: score progression over 12 hours of environment interaction; right: log-sigmoid fit on a logarithmic time axis. <a href="https://edge-bench.org/">Source: EdgeBench official project page</a>.'
  alt='Official EdgeBench curves: aggregate scores of different models improve with increasing environment interaction time, and the log-sigmoid fit describes the average learning trajectory.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

During such long-horizon interactions, newly acquired experience is maintained within context sequences, external logs, and modified project files, requiring no online weight updates. Pretraining supplies the baseline capacity to interpret feedback and adjust policies, while inference and interactive execution convert that capacity into practical returns on complex tasks.

These two developments delineate the two primary axes of compute scaling. Training-time scaling expands offline training compute to construct foundational representations and model capacity; test-time scaling, holding model weights fixed, leverages inference rollout, heuristic search, or multi-turn interaction to elevate current task execution quality.

## 5. Efficiency and training stability

The viability of compute scaling depends on conversion costs and engineering robustness. In pretraining, compute efficiency determines the performance ceiling achievable under fixed budgets, while training stability governs whether theoretical scaling returns are fully realized over complete distributed runs.

Algorithmic efficiency gains translate to leftward shifts of scaling curves, meaning fewer compute resources are required to achieve target loss values. Evaluated against token volume, this reflects token efficiency; evaluated against total floating-point operations, it reflects compute efficiency. Actual energy consumption is further governed by model architecture, model FLOPs utilization (MFU), and per-token compute overhead.

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-compute-efficiency.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Figure 6. Compute efficiency of the improved Marin training recipe. The blue line represents the baseline, and brown points indicate the revised recipe; horizontal lines compare FLOPs required to achieve identical loss. Across the indicated budget window, the revised recipe requires roughly half the baseline compute. <a href="https://openathena.ai/blog/pretraining-speedup/#combining-all-4">Source: Marin team official blog [18], Figure 12</a>.'
  alt='Official Marin compute efficiency figure: the new recipe improves compute efficiency by approximately 2.09 to 2.26 times relative to the baseline at the same loss.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Training stability centers on numerical and optimization anomalies encountered over long training durations, including activation outliers, gradient norms, weight norms, anomalous logit growth, and loss spikes. At scales spanning trillions of parameters, uncontrolled numerical drift frequently leads to gradient explosion or training divergence.

During scale-up, the Kimi K2 team observed anomalous growth in attention logits, subsequently implementing the QK-Clip mechanism within MuonClip to constrain Q and K projection layer weights, successfully completing 15.5T tokens of pretraining without encountering loss spikes. [[12]](https://www.kimi.ai/blog/kimi-k2)

{% include figure.liquid
  path='assets/img/pretrain-scaling/kimi-muonclip.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-dark'
  width='100%'
  caption='Figure 7. Loss curve of Kimi K2 pretraining with MuonClip across 15.5T tokens. <a href="https://www.kimi.ai/blog/kimi-k2">Source: Kimi K2</a>.'
  alt='Official Kimi K2 curve showing pretraining loss as a function of training token count.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Healthy training runs accommodate expected numerical progression and normal variance. Adjustments to training recipes should be guided by comprehensive training dynamics tracking and the empirical impact of numerical shifts on final loss convergence.

## 6. Predictability and detailed research

Before committing large compute investments, teams must verify whether efficiency advantages and numerical dynamics observed in small-scale experiments extrapolate reliably to target scales. Extrapolation predictability serves as a foundational criterion for scaling research.

Scaling laws offer quantitative baselines for engineering resource allocation. Reliable extrapolation requires explicit constraints on variables, observation metrics, and the valid domains of fitted models: power-law loss fits, benchmark accuracy curves on logarithmic data axes, and log-sigmoid curves for long-horizon interaction represent distinct mathematical properties and interpretive scopes. [[13]](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)

In their 535B-A23B sparse model training logs, the Marin team documented comprehensive prediction and monitoring procedures: constructing scaling ladders from small-model sequences to forecast global loss trajectories and gradient dynamics for large runs, then comparing observed training metrics against predictions in real time. Baselines established at smaller scales facilitate rapid differentiation between normal parameter evolution and engineering discrepancies that demand diagnostic investigation. [[14]](https://github.com/marin-community/marin/issues/8435)

Initial experiments in Project Delphi illustrate typical failure modes in scaling extrapolation. While scaling laws achieved tight fits on small configurations, actual loss exceeded predictions by 2.5% when scaled to a budget of $10^{22}$ FLOPs, and training diverged outright at $10^{23}$ FLOPs. This finding underscores that accurate fits on local small-scale runs do not guarantee that large-scale executions adhere to extrapolation forecasts. [[17]](https://openathena.ai/blog/delphi/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 8. Delphi initial trial using the Cautious AdamC recipe. The right panel shows large-scale runs deviating from forecasts alongside a diverging execution. <a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">Source: Marin team official blog</a>.'
  alt='Delphi first scaling experiment: the 1e22 FLOPs run has a loss 2.5% higher than predicted, and the 1e23 FLOPs run diverges.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

The team subsequently switched to the AdamH optimizer, incorporated a token-count-dependent learning rate schedule correction, and retuned hyperparameters. The revised recipe completed training smoothly at $10^{23}$ FLOPs, with final loss deviating by only ~0.2% from predictions. Diagnosing engineering root causes from empirical failures and adjusting training recipes provides the primary path to building dependable scaling forecasts.

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 9. Revised Delphi AdamH recipe. Left: IsoFLOP experiments across compute budgets; right: small-scale fits and larger-scale validation. <a href="https://openathena.ai/blog/delphi/#delphi-attempt-2-forecasts-1e23-to-within-02">Source: Marin team official blog</a>.'
  alt='Official Marin Delphi figure: IsoFLOP experiments and scaling law predictions and validation for larger compute budgets.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Regarding engineering diagnostics, Marin leveraged scaling ladders to detect anomalous gradient growth early, introducing logit z-loss to stabilize output distributions; their public recipe also details the MuonH configuration based on hyperspherical projection optimization. The benefits of individual adjustments require dedicated validation within specific engineering pipelines. [[15]](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/README.md)

Tensor Programs V by Greg Yang et al. provides theoretical foundations for scale extrapolation: maximal update parameterization (μP) ensures that as neural networks widen toward infinite width limits, key hyperparameters such as optimal learning rates remain stable, allowing μTransfer to transfer hyperparameter configurations tuned on small models zero-shot directly to massive architectures. [[16]](https://arxiv.org/abs/2203.03466)

While general learning frameworks can remain conceptually concise, realizing reliable scale-up demands meticulous engineering execution. Network parameterization, optimizer selection, data mixture ratios, and training dynamics collectively govern whether extrapolation forecasts hold at target scale.

## 7. Summary

GEN-1.5, o1, and EdgeBench demonstrate that general learning frameworks can consistently convert experiential data and compute budgets across diverse modalities and execution stages; the engineering practices of Marin and Delphi illustrate that realizing scaling dividends requires rigorous quantitative forecasting, discrepancy diagnosis, and recipe revision mechanisms.

Pretraining condenses multi-source experience into reusable model parameters, reducing downstream adaptation costs. Scaling spans both ends of the lifecycle: training-time scaling expands foundational representations and model capacity, while test-time scaling converts additional compute into reasoning and decision-making gains on targeted complex tasks. Although implementation details and empirical curves vary across domains, the scientific approach of extracting capability through compute and data remains consistent.

The core of the methodology lies in constructing effective conditions that support continuous learning and scale expansion; the core of the scientific perspective requires that the relationship between resource investment and capability gain remains predictable, testable, and correctable. Compute efficiency determines the return ceiling under fixed budgets, training stability ensures reliable convergence throughout large-scale optimization, and extrapolation predictability provides the objective baseline for further expanding engineering commitments.

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
