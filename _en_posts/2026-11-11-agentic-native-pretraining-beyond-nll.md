---
source_sha: ba66a76682bd9feb
layout: post
title: "Pretraining in the Agentic Era Cannot Rely on NLL Alone: Notes on MAI, Beam, ML4, and Nemotron"
date: 2026-11-11 12:00:00
description: "Reading the recent technical reports on MAI, Reflection Beam, Mistral Large 4, and Nemotron 3.5: legacy GDM training habits, and why PT and CPT should go beyond LR annealing to inject agentic interaction patterns and raise the RL starting point."
tags: [llm, pretraining, mid-training, agent, scaling-law, rl]
categories: [deep-learning]
featured: true
giscus_comments: true
toc:
  sidebar: left
lang: en
published: true
permalink: /en/blog/2026/agentic-native-pretraining-beyond-nll/
ref: agentic-native-pretraining-beyond-nll
related_posts: false
---

Over the past month, Microsoft AI (MAI-Base-2), Reflection AI (Beam), Mistral (Mistral Large 4, or ML4), and NVIDIA (Nemotron 3.5) all released new technical reports. Each of these teams includes a strong group of former Google DeepMind (GDM) researchers.

Two observations stand out:

1. MAI-Base-2 scales up to 2.8T parameters and 50T tokens, yet the entire report still relies exclusively on static NLL / BPB without considering agent-oriented pretraining metrics; a few months earlier, MAI-Thinking-1 did not get online OPD (On-Policy Distillation) working, and its CPT (Mid-train) stage used only natural data for LR annealing and long-context training.
2. Reflection's Beam enters RL from a very low starting point and takes roughly a month on over ten thousand GPUs to slowly train its scores up, resulting in high RL sample complexity and little emphasis on multi-turn prior training and fast iteration.

Both observations point to the same issue: **pretraining and CPT / Mid-train are not designed from an agentic-native perspective; Mid-train is still treated simply as switching to clean static data for annealing, rather than giving the pre-RL checkpoint multi-turn interaction patterns and a high starting score.**

---

## 1. Comparing the Four Reports

### 1.1 The MAI Series

MAI-Base-2 (led by Mustafa Suleyman, Karen Simonyan, and colleagues) is methodical on classical pretraining experiments, aligning data repetition across scales, stabilizing training with Muon, and fitting hyperparameter scaling laws [[1]](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/). Yet the entire post includes zero generative or tool-use evaluations. Its claim of 4 to 12x higher pretraining compute efficiency than the strongest external open-weight base model, DeepSeek-V4-Pro-Base, rests solely on BPB over internal static validation sets; Figure 1 even measures static text BPB on Kimi K3, Qwen3.8-2.4T-A95B, and MiMo-V2.6-Pro-RL, which only release post-trained weights whose output distributions naturally shift away from natural text after long-CoT RL.

The June MAI-Thinking-1 report follows the same approach [[2]](https://microsoft.ai/pdf/mai-thinking-1.pdf):

- The report explicitly defends using NLL alone instead of generative or tool-use evaluations during pretraining: because NLL conditions every step on a ground-truth prefix via teacher forcing, it limits how much a minor single-step error can compound. Yet state drift and compounding errors across multi-step interactions are the main difficulties in long-horizon agents, and teacher forcing masks that exact problem.
- Driven by the old GDM view that capabilities should be learned rather than inherited, alongside copyright concerns around commercial weight sales, its Mid-train corpus is drawn entirely from the natural pretraining set, reweighted by NLL for LR annealing and long-context training with zero synthetic data and zero exposure to reasoning traces before RL; multi-domain specialist merging also lacked online MOPD, falling back on offline trajectory SFT followed by lightweight RL.

### 1.2 Beam vs. MiMo-V2.6

Reflection's Beam (led by Ioannis Antonoglou, Misha Laskin, and many former GDM researchers) incorporates open-model rewrites of paper derivations, competitive programming tutorials, and static code tasks during Mid-train, and implements full-vocabulary MOPD in post-training, giving it solid single-turn math competition scores [[3]](https://reflection.ai/documents/beam-model-tech-report).

On long-horizon agent tasks, however, both its Mid-train stage (under four days) and its SFT stage (three sentences in the entire report) are very thin, while a single RL run consumes more than twice the GPU-days of the entire pretraining run [[3]](https://reflection.ai/documents/beam-model-tech-report), [[4]](https://reflection.ai/blog/introducing-beam). At the end of Mid-train, the model has neither a chat template nor official tool-call syntax, and its code and terminal data remain single-turn static rewrites without multi-turn trial-and-error trajectories. As a result, Beam starts RL at roughly 5% on DeepSWE and needs about four weeks on over ten thousand GB300s, generating more than 100M rollouts, to train its score up to roughly 43%.

Compare this with Xiaomi's MiMo-V2.6 from the same period: during Mid-train, Xiaomi mixes realistic multi-turn agent trajectories across coding, general, and visual domains, and switches the optimizer from AdamW to a Muon variant early on to prepare for large-batch training. After a brief SFT stage, MiMo-V2.6-Pro enters RL already near 60% on DeepSWE (higher than Beam's final score after a month of RL), and just 30 ultra-large-batch RL steps (around 700K rollouts in total, less than 1% of Beam's budget) raise its score above 72% [[5]](https://arxiv.org/abs/2610.11959).

### 1.3 Nemotron 3 Ultra and 3.5 Super

Released mid-year, Nemotron 3 Ultra relies heavily on Mamba-2 for high throughput, skips dedicated agentic Mid-train, and only checks static benchmarks on the base model; even after distilling multiple strong open-weight models and running two rounds of MOPD in post-training, its performance on long-horizon bug fixing and deep search remains weak [[6]](https://arxiv.org/abs/2606.15007). With Nemotron 3.5 Super in late October, the team added an SWE Mid-train stage containing real multi-harness agent trajectories, dropped NLL as the acceptance metric in favor of solve rates across multiple agent harnesses after lightweight SFT, and beat the much larger Ultra on SWE-bench Pro with far fewer active parameters [[7]](https://arxiv.org/abs/2610.38095).

---

## 2. Legacy GDM Habits and a Guess About Gemini 3.1 Pro

The deeper a team's GDM roots (MAI, Beam, and early ML4 before its mid-run adjustment), the more clearly two legacy GDM training habits appear:

- **The Chinchilla / Gemma pretraining habit**: evaluating small-scale ladders strictly on static validation NLL, and adopting aggressive KV-cache compression (such as 5:1 local SWA-512 with global NoPE) whenever static loss stays flat, assuming long-range information recall across multi-turn interactions will not be affected.
- **The AlphaGo Zero tabula-rasa habit**: keeping PT and CPT focused on compressing clean static knowledge while leaving reasoning and action learning to online RL in sandboxes from scratch. Go has a tiny action space and fast simulation; open-domain software engineering and tool use face vocabularies over 100K tokens and ultra-long contexts, where a single container run often takes over ten minutes and rewards are sparse. Critic-free GRPO mainly amplifies successful paths that the base model can already sample under pass@k; if upstream priors are weak, searching over 100M rollouts becomes very inefficient.

This leads me to a speculative guess: **why did GDM's own Gemini 3.1 Pro look so strong at launch, yet fall behind OpenAI and Anthropic for a period afterward?**

When 3.1 Pro held the lead, models mainly competed on pretraining knowledge plus **single-turn ultra-long CoT**. Pure text CoT sits close to the distribution of textbooks and paper derivations in pretraining corpora; even if CPT only anneals on clean static text, the base model already supports step-by-step derivations that RL can amplify into strong single-turn CoT. Soon after, however, GPT and Claude moved into frequent multi-turn tool use, whereas my clear impression using Gemini at the time was that **it barely knew how to use tools and remained stuck in the era of pure ultra-long CoT.**

Anyone who has worked on pretraining data knows a simple truth: **if you want a model to emit a certain pattern, that pattern needs to exist in your data first.** Output patterns suited for multi-turn, long-horizon agentic tasks (briefly analyzing the state $\to$ calling a tool to probe the environment $\to$ reading error logs or search results $\to$ backtracking and revising the plan $\to$ maintaining state across dozens of turns) barely exist in natural web pages or static code files. If CPT / Mid-train stays limited to annealing and long-context training on clean static data without injecting these multi-turn interaction patterns, RL alone can almost never elicit them from scratch.

---

## 3. Rethinking Pretraining and CPT from an Agentic-Native Perspective

### 3.1 Pretraining Metrics

When Mistral initially compressed the KV cache by 16x in ML4, pretraining perplexity and MMLU were virtually unchanged; only after running Mid-train + SFT evaluations mid-run did they find that **at the same validation perplexity, SWE-bench dropped by more than 20 percentage points**, forcing them to remove local sliding-window attention and cross-layer sharing and retrain most of the second half [[8]](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/resolve/main/assets/technical_report.pdf). Comparable language-modeling loss does not mean comparable long-horizon agent performance. Right now, a practical approach when choosing architectures or data mixtures is to attach a fixed small-budget Mid-train + agentic SFT probe and check solve rates and pass@k across multiple agent harnesses; I do not work extensively on evaluation design myself, so I will not expand further here.

### 3.2 PT and CPT: Injecting Interaction Patterns to Raise the RL Starting Point

PT and CPT (especially Mid-train, which connects the base model and post-training) should not just mean switching to high-quality static data for annealing. Their more important role is to bridge the base model and post-training by injecting target output patterns, **giving the pre-RL checkpoint strong interaction patterns and as high a starting score as possible so that downstream RL can work well.**

Because PT and CPT train in parallel across full sequences at high MFU without waiting on long container executions or wasting compute on zero-variance all-fail rollout groups, they are well suited for injecting complex priors:

- **PT stage**: beyond standard web text, textbooks, and static code, organize data around real task progression whenever possible, such as chaining `Issue -> PR discussion -> cross-file commit diff -> code review and CI failure fixes` into causal sequences so the model encounters "problem, action, feedback, revision" patterns during pretraining.
- **CPT / Mid-train stage**: focus on injecting multi-turn interaction patterns. Mix real or synthetic multi-turn tool-use trajectories, error localization and rollback traces, and interaction formats across diverse agent frameworks (multi-harness) into CPT (masking loss on environment observations and computing gradients only on reasoning and action tokens), and even switch optimizer states early like MiMo-V2.6 to match downstream large-batch training. Once the pre-RL checkpoint has complete interaction patterns and a high starting score, downstream RL only needs hundreds of thousands of rollouts and a few dozen large-batch updates to bring out strong performance.
- **Fast iteration via RL + MOPD and feeding data back into CPT**: once CPT and SFT provide a solid initial policy, online RL can explore capability boundaries with modest sample budgets, merge domain specialists via shared-base MOPD, and feed filtered high-quality long-horizon trajectories back into the next round of CPT / CT (Mid-train), shortening month-long RL runs into multi-day iteration cycles. In industrial 1.1 or 1.2 minor-version updates, alongside adding targeted domain data (such as code-to-video or 3D modeling), teams routinely clean ultra-long user-returned traces (preserving tool use, agentic markdown, context summaries, and so on) into clean problem-to-action-to-answer trajectories, and fold them back into CPT together with task-specific RL trajectories for targeted capability improvements.

---

## 4. Two Unresolved Questions

1. **Balancing CPT / SFT prior strength against exploration diversity.**
   Adding too many homogeneous trajectories from a single source into CPT and SFT raises pass@1 quickly, but can narrow the output distribution and drop entropy too fast, potentially lowering the pass@k ceiling and shrinking the exploration space for RL (likely one reason legacy GDM teams resisted synthetic data and distillation). How to inject multi-turn tool-use patterns and raise the starting score while preserving path diversity still needs further exploration.
2. **Selecting observables on small-scale scaling ladders to reflect and predict flagship agentic capability.**
   Classical NLL scaling laws predict flagship loss accurately from small compute budgets; on multi-turn agent tasks, however, small models have not yet crossed emergence thresholds and environment variance is high. Even with lightweight SFT probes, rank stability and quantitative accuracy when extrapolating across two or three orders of magnitude remain far less mature than static BPB.

---

## References

[1] Microsoft AI (2026). [Frontier Pre-training at Microsoft AI](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/). Microsoft AI Blog.

[2] The Microsoft AI Team (2026). [MAI-Thinking-1: Building a Hill-Climbing Machine](https://microsoft.ai/pdf/mai-thinking-1.pdf). Microsoft AI Technical Report.

[3] Reflection Team (2026). [Beam Model Technical Report](https://reflection.ai/documents/beam-model-tech-report). Reflection AI Technical Report.

[4] Reflection AI (2026). [Introducing Beam: Reflection's 501B Open-Weight Model](https://reflection.ai/blog/introducing-beam). Reflection AI Blog.

[5] LLM-Core Xiaomi (2026). [MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement](https://arxiv.org/abs/2610.11959). arXiv preprint arXiv:2610.11959.

[6] NVIDIA (2026). [Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](https://arxiv.org/abs/2606.15007). arXiv preprint arXiv:2606.15007.

[7] NVIDIA (2026). [Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI](https://arxiv.org/abs/2610.38095). arXiv preprint arXiv:2610.38095.

[8] Mistral AI (2026). [Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/resolve/main/assets/technical_report.pdf). Mistral AI Technical Report.

## Citation

To cite this article:

```bibtex
@article{zou2026agenticpretrain,
  title={Pretraining in the Agentic Era Cannot Rely on NLL Alone: Notes on MAI, Beam, ML4, and Nemotron},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/en/blog/2026/agentic-native-pretraining-beyond-nll/}
}
```
