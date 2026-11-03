---
source_sha: a6376129ffcd9cdf
layout: post
title: "Pretraining and scaling: a universal methodology"
date: 2026-09-07 12:00:00
last_modified_at: 2026-11-03
description: "Pretraining and scaling as a methodology across domains: learn world knowledge from data, test improvements across scales, and use efficiency, stability, and prediction to guide training."
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

First published on September 7, 2026, and revised on September 8, 2026. Updated on November 3, 2026, with clearer wording and formatting, plus a dated addition in Section 2.

Take the training curve in the GEN-1.5 report. Over more than eight months and three training phases, the model's next-action prediction error keeps falling.

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

Here's how I think about pretraining and scaling. Data contains world knowledge: how things work, how events relate, and how people act. Pretraining compresses some of that knowledge into a model's weights. With more parameters and more data, a well-trained model can learn and store more of it. Generalizing to new tasks, fine-tuning, and in-context learning all draw, in part, on this knowledge already in the weights.

That's why I see pretraining and scaling as a general methodology. Language models, world models, and embodied intelligence are a few examples. The same idea can carry into other areas of intelligence, including ones we haven't developed yet.

## 1. General framework and scalability

In his 2024 conversation with Zhang Xiaojun, Moonshot AI founder and CEO Yang Zhilin described two things to look for in a learning framework. To paraphrase: [[2]](https://www.xiaoyuzhoufm.com/episode/65e16b5b6144a933b1d968b5)

- It should be general enough to handle many different problems within the same framework.
- It should keep getting better as we put more compute into it.

By 'framework,' I mean the model, how we represent the data, what we train it to do, and how we train it. From there, I want to know a few practical things. If we put in more compute, how much better does it get? Can we reach the same result with less compute? Will training stay stable as we scale up, and can small experiments tell us what to expect from a large run?

Dyna-2 gives us a concrete example. The team scaled pretraining on first-person human video to one million hours. With the same post-training setup, more video data led to better average performance across 14 real-world robot tasks. [[3]](https://www.dyna.co/dyna-2)

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

Dyna-2 shows that what a model learns during pretraining can help it on new tasks. That's the pattern I want to look for in other fields too: as we give the model more experience, does it get better at the things we want it to do?

## 2. Occam's razor

*Don't add complexity unless you need it.*

Any proposed improvement, whether it changes attention, MoE, the optimizer, or the broader architecture, should pass a scaling ladder. Compare performance at the same training FLOPs, or compare the FLOPs needed to reach the same performance. Repeat that comparison at increasing scales. The gain over a well-tuned baseline should hold up as we scale, rather than shrink.

That's why I'm cautious about big claims from nanoGPT or nanochat speedruns. Repeated tuning can overfit one benchmark and one training setup. A strong result at that scale does not establish a better method for large-scale pretraining. Ask to see the scaling evidence before accepting the broader claim.

*Added November 3, 2026.* The speedrun hype around Dust is one example: better scores from tuning a fixed setup don't convince me that the gains will survive scaling. [[18]](https://qlabs.sh/research/dust)

I see a similar philosophy in Sutton's The Bitter Lesson. In the shift from traditional machine learning to deep learning, many hand-built features and elaborate pipelines gave way to end-to-end learning. Sutton's point is that general methods that benefit from more compute tend to win over time. For me, that is a guiding idea behind pretraining and scaling. [[4]](https://heartyhaven.github.io/files/bitter_lesson.pdf)

Karpathy's Software 2.0 describes the same shift: we choose the model, data, and objective, then let training learn the program in the weights. [[5]](https://karpathy.medium.com/software-2-0-a64152b37c35)

A simple training objective can produce a complex system whose capabilities go far beyond the training task. Next-token prediction, for example, can teach a model knowledge it later uses to answer questions or write code.

## 3. Pretraining and scaling

Pretraining means a model doesn't have to start from scratch on every new task. It can draw on world knowledge already compressed into its weights. Fine-tuning adapts those weights to a task. In-context learning uses examples to guide the model while leaving its weights unchanged.

As GEN-1.5 shows, one demonstration in context can be enough to learn a new task without changing the model's weights. It can also adapt to tasks with only a few gradient steps. [[1]](https://generalistai.com/blog/gen-1.5)

<figure>
  <video controls playsinline preload="metadata" width="100%" class="img-fluid rounded z-depth-1" aria-label="Official demonstration of GEN-1.5 performing two tasks using examples in context">
    <source src="https://generalistai.com/assets/pages/blog/gen-1.5/assets/videos/two_task_in_context.mp4" type="video/mp4">
    <a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">Watch the official GEN-1.5 demo</a>
  </video>
  <figcaption class="caption">Video 1. GEN-1.5 uses demonstrations in context to unzip a pencil case and retrieve money from inside, without updating its weights. <a href="https://generalistai.com/blog/gen-1.5#one-shot-in-context">Source: Generalist official demo</a>.</figcaption>
</figure>

Pretraining quality also shows up in how easily a model learns something new. How many examples or weight updates does it need for the next task? How much extra compute? Those questions matter alongside benchmark scores. We pay for pretraining once and reuse what the model learned across many tasks.

Scaling asks how much better a model gets as we add parameters, data, or compute. More pretraining or reinforcement learning compute can improve the model we reuse across tasks. At inference time, more reasoning, search, or interaction can help it do the current task better. Training and inference use compute in different ways, so I would measure their gains separately.

## 4. Two milestones: train-time and test-time scaling

Two milestones make this distinction concrete: language-model scaling laws in 2020 for train-time scaling, and o1 in 2024 for test-time scaling.

In 2020, OpenAI put numbers on train-time scaling. Language-model loss followed predictable curves as model size, data, and training compute increased. Researchers could use those curves to estimate what a larger run might achieve. Hestness et al. had studied similar scaling patterns earlier. [[6]](https://arxiv.org/abs/2001.08361)[[7]](https://arxiv.org/abs/1712.00409)

{% include figure.liquid
  path='assets/img/pretrain-scaling/openai-2020-scaling-laws.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 3. Language-model loss falls as training compute, model size, and data increase. The left plot estimates compute for small batch sizes. <a href="https://arxiv.org/html/2001.08361v1#S1.F1">Source: Kaplan et al., 2020, Figure 1</a>.'
  alt='Original 2020 language model scaling laws figure: loss decreases with compute, model parameter count, and dataset size, respectively.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

In 2024, o1 made test-time scaling a major part of the picture. Reinforcement learning taught the model to reason more effectively. Giving that trained model more compute to reason before answering then improved its performance on difficult problems. The two plots show gains from both more RL training and more computation at test time. [[8]](https://openai.com/index/learning-to-reason-with-llms/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/o1-compute.webp'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 4. o1 accuracy on AIME improves with more reinforcement learning compute on the left and more test-time compute on the right. Accuracy is measured by pass@1. Both compute axes use logarithmic scales. <a href="https://openai.com/index/learning-to-reason-with-llms/">Source: OpenAI</a>.'
  alt='The AIME accuracy of o1 improves with increasing reinforcement learning training compute and inference-time compute.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Chain-of-thought prompting and research on test-time compute came before o1. The milestone was using reinforcement learning at scale to teach a model how to make better use of extra reasoning time. [[9]](https://arxiv.org/abs/2201.11903)[[10]](https://arxiv.org/abs/2408.03314)

EdgeBench from the Seed Edge team extends test-time scaling to agents working in real environments. An agent can try something, check the result, and adjust its next step. Some runs last more than 72 hours. Average task scores improve with more interaction time, following a log-sigmoid curve on a logarithmic time axis. [[11]](https://seed.bytedance.com/en/blog/edgebench-measuring-real-world-environment-learning-and-discovering-a-new-scaling-law)

{% include figure.liquid
  path='assets/img/pretrain-scaling/edgebench-environment-scaling.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 5. Average EdgeBench scores across 134 tasks. Left: scores over 12 hours of interaction. Right: a log-sigmoid fit with interaction time on a logarithmic axis. <a href="https://edge-bench.org/">Source: EdgeBench official project page</a>.'
  alt='Official EdgeBench curves: aggregate scores of different models improve with increasing environment interaction time, and the log-sigmoid fit describes the average learning trajectory.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

During these runs, the weights stay fixed. New experience accumulates in the context, logs, and files the agent changes. The model uses what it learned during training to make sense of feedback and decide what to try next.

Train-time scaling builds a more capable model. Test-time scaling gives that model more computation to solve the task in front of it.

## 5. Scaling efficiency and stability

So if pretraining and scaling are our core methodology, I would focus on two things: efficiency and stability. How much progress do we get for the compute we spend? And can training stay stable as we scale up?

An efficiency gain means reaching the same loss with fewer FLOPs. On a loss-versus-compute plot, that shifts the curve to the left. Using fewer tokens is another kind of efficiency, but each token can cost a different amount of compute. Hardware utilization also affects the time and energy we spend.

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-compute-efficiency.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Figure 6. The revised Marin recipe reaches the same loss with roughly half the baseline FLOPs over the range shown. Blue: baseline. Brown: revised recipe. Horizontal lines compare compute at the same loss. <a href="https://openathena.ai/blog/pretraining-speedup/#combining-all-4">Source: Marin team official blog [17], Figure 12</a>.'
  alt='Official Marin compute efficiency figure: the new recipe improves compute efficiency by approximately 2.09 to 2.26 times relative to the baseline at the same loss.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Maintaining stability means the model keeps learning through a long training run. We watch for loss spikes and unusual growth in activations, gradients, weights, or attention logits. These signals help us catch problems before training breaks down.

During Kimi K2 training, the team saw attention logits grow too large. They introduced QK-Clip as part of MuonClip to control that growth. The resulting run completed 15.5T tokens of pretraining without loss spikes. [[12]](https://www.kimi.ai/blog/kimi-k2)

{% include figure.liquid
  path='assets/img/pretrain-scaling/kimi-muonclip.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-dark'
  width='100%'
  caption='Figure 7. Kimi K2 training loss over 15.5T tokens with MuonClip. <a href="https://www.kimi.ai/blog/kimi-k2">Source: Kimi K2</a>.'
  alt='Official Kimi K2 curve showing pretraining loss as a function of training token count.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Not every increase in gradient norm needs a fix. First check whether it comes with loss spikes or slower learning. Then compare any fix with the original recipe at the same compute budget, checking both stability and final loss.

## 6. Scaling as a predictive science

Before we have a complete theory of learning, we can still build a predictive science of scaling. I think of astronomy in Kepler's time: careful observations, mathematical laws, and predictions we can test.

Scaling laws help us estimate what loss a larger compute budget should buy, choose a training recipe, and check whether a run is on track. Their value lies in predicting results beyond the experiments used to fit them. [[13]](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)

In practice, Marin uses small-model scaling ladders to forecast loss and training dynamics in larger runs, then checks actual results against those forecasts. [[14]](https://github.com/marin-community/marin/issues/8435)[[15]](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/README.md)

Initial Delphi experiments show why those checks matter. A fit that looked good at small scales gave a loss prediction that was too optimistic: the actual loss was 2.5% higher at 10²² FLOPs, and training diverged at 10²³ FLOPs. [[16]](https://openathena.ai/blog/delphi/)

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 8. The first Delphi recipe fits the small runs but misses at larger scales. The 10²³-FLOP run diverges. <a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">Source: Marin team official blog</a>.'
  alt='Delphi first scaling experiment: the 1e22 FLOPs run has a loss 2.5% higher than predicted, and the 1e23 FLOPs run diverges.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

The team revised the optimizer and learning-rate schedule. The next 10²³-FLOP run finished with loss within about 0.2% of the prediction.

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 9. The revised Delphi recipe predicts the 10²³-FLOP result to within about 0.2%. Left: experiments at fixed compute budgets. Right: forecasts and larger runs. <a href="https://openathena.ai/blog/delphi/#delphi-attempt-2-forecasts-1e23-to-within-02">Source: Marin team official blog</a>.'
  alt='Official Marin Delphi figure: IsoFLOP experiments and scaling law predictions and validation for larger compute budgets.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Good scaling research makes a prediction and tests it at a larger scale. When the result differs, we investigate and revise. That is how small experiments can guide expensive training decisions.

## 7. A methodology across domains

Pretraining compresses world knowledge from data into weights that many tasks can use. Scaling asks how much more a model can learn and do with more resources. I see this as a methodology we can carry into other areas of intelligence, including ones we haven't explored yet.

The work is to keep the learning setup simple, test efficiency and stability across scales, and make predictions before committing more compute. When larger runs disagree with those predictions, we learn from the gap and revise the recipe.

## References

[1] Generalist Team (2026). [GEN-1.5: Embodied Foundation Models are One-Shot Learners](https://generalistai.com/blog/gen-1.5). Generalist Blog.

[2] Zhang Xiaojun, Yang Zhilin (2024). [A Conversation with Yang Zhilin on a Year of Building an LLM Startup: Expanding Human Aspirations, Probabilistic Non-Consensus, and Sora](https://www.xiaoyuzhoufm.com/episode/65e16b5b6144a933b1d968b5). Zhang Xiaojun Jùn | Business Interviews, Episode 59.

[3] Dyna Robotics (2026). [Dyna-2: A 1-Million-Hour Scaling Law for World-Action Models](https://www.dyna.co/dyna-2). Dyna Research.

[4] Sutton, R. S. (2019). [The Bitter Lesson](https://heartyhaven.github.io/files/bitter_lesson.pdf). Personal blog; archived PDF copy.

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

[15] Marin Community. [MoE Recipe](https://github.com/marin-community/marin/blob/main/experiments/grug/moe/README.md). GitHub project documentation. Accessed: 2026-11-03.

[16] Held, W. (2026). [Scaling Laws That Extrapolate 300× Past the Fit](https://openathena.ai/blog/delphi/). Open Athena Blog.

[17] Dial, L. (2026). [Improving our LLM Pretraining Efficiency](https://openathena.ai/blog/pretraining-speedup/). Open Athena Blog.

[18] Dahal, S., Mandal, B., Gülbahar, S., & Vegesna, A. (2026). [Dust: Pretraining Transformers Without Backpropagation](https://qlabs.sh/research/dust). Q Labs Research. October 2026.

## Citation

To cite this article:

```bibtex
@article{zou2026pretrainscaling,
  title={Pretraining and scaling: a universal methodology},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/en/blog/2026/pretrain-scaling-methodology-scientific-perspective/}
}
```
