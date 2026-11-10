---
source_sha: ff4220a2eccc8387
layout: post
title: "How to Build a Scientific Scaling Ladder"
date: 2026-09-20 10:00:00
description: "A practical guide to designing and building LLM pretraining scaling ladders: parameter and compute accounting, dense and MoE experimental design, hyperparameter search and schedules, data mixtures and multi-epoch repetition, loss scaling law fitting, downstream prediction, and extrapolation validation."
tags: [scaling-laws, pretraining, hyperparameter, optimization, llm, empirical-methodology]
categories: [research]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/how-to-build-scientific-scaling-ladder/
ref: how-to-build-scientific-scaling-ladder
related_posts: false
---

This post draws strictly on **<u>public literature and personal hands-on experience</u>** and **<u>contains no confidential information</u>**. In my previous post, [Pretraining and scaling: a universal methodology](/en/blog/2026/pretrain-scaling-methodology-scientific-perspective/), I discussed why pretraining and scaling form a general methodology across domains; here I walk through how to build a reliable LLM pretraining scaling ladder from scratch.

---

## 1. Basics and Scope

### 1.1 What Is a Scaling Ladder

A scaling ladder is a matrix of small-scale pretraining runs spanning multiple model sizes $N$ and training token budgets $D$. In industry pretraining teams and frontier labs, it is much more than a tool for picking hyperparameters; it is the core experimental framework across the entire research cycle, serving three main roles:

1. **Studying scaling rules for hyperparameters and model configurations**: mapping out how LR, BSZ, aspect ratio, MoE sparsity ratio, data mixtures, and epoch counts transfer as $(N, D)$ grows.
2. **Acting as a gold-standard baseline (optimal baseline) for new techniques**: it serves as a common ruler. Any new optimizer, architectural change, or data batch must show consistent gains on the ladder before merging into the mainline recipe for the next base model. Many academic papers overclaim gains by testing at a single small scale or against an undertuned baseline; evaluating across scales on a well-tuned ladder filters out ideas that do not hold up and shows whether a real gain stays flat, grows, or fades with scale.
3. **Forecasting the flagship run (hero run)**: once the architecture, hyperparameter rules, and data recipe are locked in, the fitted curves predict the loss trajectory, final loss, and even downstream benchmark scores before committing full-scale compute.

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 1. Initial Delphi scaling run (Cautious AdamC). Large-scale runs on the right deviate from the forecast and diverge. <a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">Source: Marin, 2026</a>.'
  alt='Delphi initial scaling experiment: the 10^22 FLOPs run ends 2.5% above predicted loss, and the 10^23 FLOPs run diverges.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Making that ruler reliable takes more than running a few small models and fitting a curve. If small baseline runs are undertuned, parameter counts and FLOPs are tallied inconsistently, data repetition epochs are mismatched across scales, or the functional form is misspecified, both method comparisons and large-scale forecasts will be off. In the first [Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) attempt (Figure 1), the fit looked clean across $3\times 10^{18}$–$3\times 10^{20}$ FLOPs, yet actual loss came in 2.5% higher than predicted at $10^{22}$ FLOPs and training diverged outright at $10^{23}$ FLOPs. Building a scientific scaling ladder therefore requires both careful small-scale experiments and curve fitting, and multi-tier holdout runs plus stress tests to check that extrapolation actually holds ([§10](#10-extrapolation-validation)).

### 1.2 Typical Regimes

This post covers three pretraining regimes. Sections 2–10 lay out the shared workflow, while MoE-specific and data-constrained questions are addressed in dedicated subsections along the way.

| Regime | Key Questions | Sections |
|---|---|---|
| Dense | Choosing aspect ratio and allocating compute; used both for dense models and as a baseline for MoE | [§4.3](#43-aspect-ratio) |
| MoE | Tracking total vs. active parameters; setting sparsity ratio, expert size, routing, and load balancing; measuring real expert-parallel throughput | [§3.1](#31-parameter-counting), [§4.4](#44-moe-structural-scaling), [§5.4](#54-moe-experimental-design), [§6.3](#63-scaling-laws-for-lr-and-bsz), [§10.2](#102-stability-stress-tests), [§10.4](#104-wall-clock-speed-and-inference-cost) |
| Data-constrained | Choosing repetition epochs and domain weights when unique tokens fall short of the target training budget | [§5.3](#53-token-horizons), [§7.3](#73-data-constrained-training-and-repetition) |

Real-world runs often use MoE while either running short of unique data (for instance, having less than $1\times$ Chinchilla tokens available) or overtraining deliberately to cut inference cost. Because target TPP (tokens per parameter) can sit well below or well above the Chinchilla optimum, the ladder's token horizons should be chosen around that target TPP ([§5.3](#53-token-horizons)).

This post focuses on LLM pretraining and does not cover scaling laws for SFT/RL, native multimodal models, distillation, or synthetic data generation.

### 1.3 What a Ladder Produces

| Deliverable | Prediction Target | Section |
|---|---|---|
| Hyperparameter scaling laws (LR, BSZ, WD) | Optimal training hyperparameters | [§6](#6-hyperparameter-search-and-training-setup) |
| Loss scaling law | Final training / eval loss | [§8.1](#81-functional-forms) |
| Loss trajectory scaling law | Full loss curve and annealing gain | [§8.2](#82-loss-trajectory-and-annealing-scaling-laws) |
| LR schedule and cooldown | End LR, schedule choice, and weight averaging | [§6.5](#65-lr-schedule-and-cooldown) |
| Data mixture and repetition scaling laws | Optimal domain weights; effective tokens under multi-epoch training | [§7](#7-data-ladders) |
| MoE sparsity scaling law | Expert count and expert size at fixed active parameters | [§5.4](#54-moe-experimental-design) |
| Downstream task scaling law | Benchmark scores | [§9](#9-downstream-task-prediction) |

### 1.4 Notation and Terms

| Term | Definition |
|---|---|
| $N_{\text{body}}$ | Transformer backbone parameters, excluding input embedding and output head ([§3.1](#31-parameter-counting)) |
| $N_{total}$, $N_{active}$ | Total MoE backbone parameters and active parameters per token |
| $D$ | Cumulative training tokens (counted over tokens that contribute to the loss; see [§3.3](#33-loss-and-token-accounting)) |
| $U$ | Deduplicated available unique tokens |
| $C$ | Total training FLOPs ([§3.2](#32-compute)) |
| TPP | Tokens per parameter ($D/N$); check whether $N$ includes embeddings when comparing papers |
| LR ($\eta$), BSZ ($B$), WD ($\lambda$) | Learning rate, batch size (in tokens), and weight decay |
| Holdout | Runs excluded from curve fitting and reserved strictly to test extrapolation accuracy |
| Fully-Tuned Frontier | Lowest loss achieved at each scale once hyperparameters are thoroughly tuned ([§6.1](#61-search-goals-and-procedure)) |
| Compute multiplier | Ratio of compute required to reach the same loss ([§2.3](#23-acceptance-criteria)) |

## 2. Objectives and Acceptance Criteria

### 2.1 Decision Objectives

Before running anything, clarify what decision the ladder is meant to inform. Most ladders target one of four goals:

1. Forecasting a fixed recipe: given a preset hyperparameter scaling rule, predict its large-scale performance without tuning each grid point individually.
2. Optimal compute allocation: find the $(N, D)$ pair that minimizes loss under a compute budget $C$.
3. Recipe comparison: compare architectures, optimizers, or data mixtures to see which wins at scale.
4. Hyperparameter prediction: fit how optimal $(\eta, B, \lambda)$ scale with $N$ and $D$.

[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) separates whether a recipe is *competitive* (achieves lower loss) from whether it is *predictable* (extrapolates cleanly from small to large runs); both properties need to be checked separately. Goals 2–4 require tuning every fitting point onto the Fully-Tuned Frontier ([§6.1](#61-search-goals-and-procedure)), because poorly tuned small models tilt the fitted power-law slope. Goal 1 simply runs the prescribed hyperparameter formula directly.

### 2.2 Evaluation Metrics and Setup

A few practical points matter when setting up evaluation:

- Separate proxy metrics from end goals: lower training loss on small models does not always translate into better downstream benchmark scores at scale ([Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) §1).
- Signal at small scales: small models often score near chance on hard math and code benchmarks, whereas easy tasks saturate at large scales. When discrete accuracy shows no signal on small models, continuous metrics such as bits-per-byte (BPB) on gold answers provide much cleaner separation ([OLMo 3, 2025](https://arxiv.org/abs/2512.13961)).
- Rank consistency across scales: when using small-model proxy metrics to screen candidates, verify that small-scale rankings actually match large-scale ordering ([OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3, Appendix A.4).
- Handling high-variance benchmarks: cover core target capabilities, and average over multiple samples or track noisy benchmarks separately ([Phi-4, 2024](https://arxiv.org/abs/2412.08905) §5; [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3.3–3.3.4).
- Fixed evaluation harness: keep prompt templates, decoding parameters, and scoring script versions identical across runs, and check for train-test contamination.

{% include figure.liquid
  path='assets/img/pretrain-scaling/olmo3-evaluation.png'
  id='fig-olmo3-evaluation'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 2. Math evaluation in OLMo 3: left and center show bits-per-byte (BPB) on the Easy suite and pass@1 on the Main suite vs. compute; small models already separate cleanly on BPB while pass@1 remains near zero. Right shows the relationship between the two metrics across evaluated models; cross-scale rank consistency still needs checking on target tasks. <a href="https://arxiv.org/pdf/2512.13961v1#page=12">Source: Olmo 3, 2025, v1, Fig. 6 (PDF p. 12)</a>.'
  alt='Three subplots from OLMo 3 Figure 6: math Easy suite BPB vs. compute, Main suite pass@1 vs. compute, and BPB vs. pass@1.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

If the final model will undergo SFT or RL, looking only at pretraining metrics can miss failure modes that surface during post-training. [Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) gives two examples: removing positional encodings (NoPE) and adding sparse-read residual branches barely affected pretraining loss, yet caused non-terminating repetition and post-training degradation, respectively. [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.5.1 addresses this by running lightweight instruction tuning after cooldown on candidate checkpoints and checking task scores, EOS termination, and output length together.

### 2.3 Acceptance Criteria

Set acceptable extrapolation error bounds ahead of time based on the smallest meaningful gap between candidate recipes and run-to-run seed noise, rather than after looking at holdout numbers. [Choshen et al., 2025](https://arxiv.org/abs/2410.11840) report that the smallest relative improvement typically used to justify an architectural or algorithmic change in the literature is around 4%, while random-seed variance alone can reach roughly 3.5%; Delphi's 0.2%–0.5% prediction error in its specific setup should not be treated as a universal bar.

A fixed relative loss error also means different things at different scales. Under a power law $L(C)=E+A\,C^{-\gamma}$, a loss gap $\Delta L$ translates into a compute multiplier of:

$$\ln\frac{C_2}{C_1}\approx\frac{\Delta L}{\gamma\,(L-E)}$$

As models grow and $L$ approaches the irreducible loss $E$, the same $\Delta L$ corresponds to a much larger compute ratio. Measuring gains and errors via the compute multiplier (e.g., 15% compute saved) alongside absolute $\Delta L$ is more informative than relative loss error alone.

## 3. Parameter, Compute, and Loss Accounting

### 3.1 Parameter Counting

Fitting $L(N,D)$ and computing FLOPs $C$ use different definitions of $N$, so record both separately.

When fitting $L(N,D)$, set $N$ to the backbone parameter count $N_{\text{body}}$, excluding both input embeddings and the output head. Embedding and head layers scale as $O(Vd)$ whereas the backbone scales as $O(d^2 n)$; at small model sizes, vocabulary layers make up a large fraction of total parameters and skew the fitted scaling exponent across sizes. Ablations in [Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972v3) Appendix G show noticeably larger extrapolation error at 25.1B when embeddings are included than when using $N_{\text{body}}$; [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) likewise excluded embeddings.

When computing training FLOPs $C$, however, you must add the output head back in. The output head performs a $d\times V$ matrix multiply on every token; tying weights with the input embedding saves memory, not that matrix multiply. Input embedding itself is just a table lookup by token ID and incurs no matrix-multiply FLOPs, so it is excluded from both.

Watch parameter definitions when comparing coefficients or TPP across papers: [Porian et al., 2024](https://arxiv.org/abs/2406.19146) Table 2 includes the output head in $N$ while excluding input embeddings, whereas Chinchilla's 20 TPP rule counts all parameters including embeddings ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) Appendix F).

For MoE runs, log the following at every grid point:

- Total backbone parameters $N_{total}$ and active parameters per token $N_{active}$ (both excluding embeddings and head);
- Total routed experts $E_{total}$, active experts per token $E_{active}$, and sparsity ratio $S=E_{total}/E_{active}$;
- Number and size of shared experts, plus individual routed expert size (expert granularity);
- Router and load-balancing settings ([§4.4](#44-moe-structural-scaling)).

### 3.2 Compute

The common shorthand $C\approx6ND$ drops attention and output-head FLOPs. For standard MHA with a $4d$ FFN and full $L\times L$ attention matrices, forward-plus-backward FLOPs are:

$$C \approx \left(6 + \frac{L}{d}\right) N_{\text{body}}D + 6VdD$$

where $L$ is sequence length, $d$ is hidden dimension, $V$ is vocabulary size, and $N_{\text{body}}\approx12d^2n$ for $n$ layers. The $L/d$ term comes from attention and the last term from the output head. If the attention kernel skips the masked upper triangle, the attention term halves; SwiGLU, GQA, and MoE also change the exact coefficients, so compute $C$ from the actual architecture and kernels in use.

For MoE models, matrix-multiply FLOPs follow $N_{active}$ plus router overhead; expert-parallel (EP) all-to-all communication adds zero FLOPs, though it directly affects wall-clock training speed ([§10.4](#104-wall-clock-speed-and-inference-cost)).

### 3.3 Loss and Token Accounting

Keep loss and token accounting consistent across every run on the ladder:

| Item | Rule |
|---|---|
| Loss term | Report main next-token cross-entropy only; log MoE aux loss, z-loss, MTP, and other auxiliary terms separately |
| Validation set | Share the exact same eval set and domain weights across all candidates, while tracking core domains individually |
| Loss reduction | Average consistently per valid target token (or per document) across all runs |
| Sequence packing | Standardize BOS/EOS placement, document boundaries, packing, attention masks, and truncation |
| Token accounting | Distinguish total processed tokens, loss-contributing tokens, unique tokens, and repetition epochs |
| Random seeds | Log initialization, data sampling, and shuffle seeds separately |
| Cross-tokenizer comparison | Convert to BPB (bits-per-byte) on identical raw text and compare at matched FLOPs over the same text span |

When experiments change the training data mixture, training loss reflects the intrinsic entropy of each mixture (for example, higher code share naturally lowers training loss) and cannot be compared across runs; evaluate on a fixed validation set instead.

Also space evaluations at fixed fractions of total training steps $T$ (e.g., every 2% of training) rather than fixed step intervals (e.g., every 1,000 steps). Fixed step intervals give long runs far more evaluation points than short runs and overweight them when fitting full loss curves ([§8.3](#83-numerical-fitting-details)).

## 4. Baseline Setup and Scaling Rules

### 4.1 Grouping Variables

Variables on a ladder fall into three groups:

| Group | Meaning | Examples |
|---|---|---|
| Held fixed | Kept identical across all runs | Architecture family, dataset version, optimizer type, `seq_len` |
| Scaled by rule | Varied deterministically with $N$ or $D$ | LR (power law or $\mu$P), BSZ (e.g., $D^{0.4}$), warmup fraction |
| Swept / fitted | Independent variables to search or fit | $N$, $D$, and the optimal LR and BSZ being swept |

Set warmup steps as a fixed fraction of total training tokens (for example, Delphi uses a 10% warmup across all scales; see [Appendix A.3](#a3-delphi-ladder)). Keeping a constant warmup step count across scales makes warmup dominate short, small-budget runs and biases the fitted compute-optimal exponents ([Porian et al., 2024](https://arxiv.org/abs/2406.19146)).

Whenever the architecture, optimizer, dataset version, numerical precision, or tokenizer changes, existing scaling coefficients may stop holding; check on small runs first before deciding whether to reuse, adjust, or refit them ([§7.1](#71-data-quality-and-source-evaluation)).

### 4.2 Architectural and Training Consistency

Aside from the specific variable under study, keep architectural and training choices aligned across all ladder sizes:

| Component | Guideline |
|---|---|
| Backbone | Keep the same backbone family (e.g., decoder-only Transformer), normalization (e.g., Pre-RMSNorm), positional encoding (e.g., RoPE), and activation (e.g., SwiGLU) |
| Embeddings | Keep embeddings either tied everywhere or untied everywhere |
| Attention | Use the same attention variant (e.g., GQA), keeping KV group ratios and `head_dim` on small models close to the target model |
| Aspect ratio | Scale width and depth together at a consistent ratio ([§4.3](#43-aspect-ratio)) |
| MoE routing | Use the same router, load-balancing method, and shared-expert ratio ([§4.4](#44-moe-structural-scaling)) |
| Training and eval | Match `seq_len` (e.g., 4096), data version, tokenizer, initialization, optimizer (e.g., AdamW), and validation setup ([§3.3](#33-loss-and-token-accounting)) |

### 4.3 Aspect Ratio

At fixed backbone size $N_{\text{body}}$, changing the ratio of hidden width $d$ to layer count $n$ alters the attention compute term $L/d$ ([§3.2](#32-compute)). [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) found pretraining loss largely insensitive to aspect ratio over a wide range, but later work shows aspect ratio affects downstream benchmarks ([Gemstones, 2025](https://arxiv.org/abs/2502.06857) §4.4) and reasoning performance ([GLM-4.5, 2025](https://arxiv.org/abs/2508.06471)). Keep a consistent width-depth scaling rule across the ladder; if you want to compare different depths or head dimensions, running dedicated ablations is recommended.

### 4.4 MoE Structural Scaling

Beyond the general settings in [§4.2](#42-architectural-and-training-consistency), MoE ladders introduce several structural parameters:

- Expert granularity: [Krajewski et al., 2024](https://arxiv.org/abs/2402.07871) study individual expert size as an independent scaling variable and show that matching the dense FFN dimension is generally suboptimal; finer-grained experts widen MoE's advantage over dense models as scale grows.
- Shared experts: fix the count of shared experts and their size ratio relative to routed experts.
- Load balancing: keep the load-balancing mechanism consistent across scales. [Kimi K3, 2026](https://arxiv.org/abs/2607.24653) (Appendix C–D) uses Quantile Balancing (QB), setting per-expert router biases directly from target-load quantiles of router logits (aggregated across microbatches via histograms), which removes the manual bias-update step size required in earlier aux-loss-free methods. Because load-balancing strength affects both loss and expert-parallel throughput, keeping it consistent across the ladder is recommended.
- Capacity factor and token dropping: match capacity factors between training and inference and monitor token-drop rates; DeepSeek-V3 drops zero tokens in both training and inference (§2.1.2).
- Sparsity: ignoring memory and communication costs, [Abnar et al., 2025](https://arxiv.org/abs/2501.12370) find that increasing sparsity (larger $N_{total}$) at fixed training FLOPs steadily lowers pretraining loss; at fixed $N_{total}$, loss follows a U-shaped curve in sparsity whose optimum shifts toward higher sparsity as compute grows. Most downstream tasks track pretraining loss regardless of sparsity, though reading-comprehension tasks such as CoQA and SQuAD favor higher active-parameter fractions. [Ludziejewski et al., 2025](https://arxiv.org/abs/2502.05172) jointly fit expert count, $N_{active}$, and $D$ under memory constraints (up to 2.7B active / 5B total parameters).

### 4.5 Vocabulary Size and Precision

- Vocabulary size: across 33M–3B models, [Tao et al., 2024](https://arxiv.org/abs/2407.13623) show that optimal vocabulary size grows with compute budget and that most open-weight models use smaller-than-optimal vocabularies. Because changing vocabulary size alters both $O(Vd)$ compute and bytes per token, convert cross-vocabulary comparisons to byte-level BPB ([§3.3](#33-loss-and-token-accounting)).
- Training precision: [Kumar et al., 2024](https://arxiv.org/abs/2411.04330) (tested up to 1.7B parameters and 26B tokens) fold numerical precision and quantization into scaling laws: lower-precision training reduces effective parameter count, while higher training TPP makes models more sensitive to post-training quantization. Running ladder experiments directly in the target model's training precision scheme is recommended ([§10.3](#103-aligning-training-frameworks-and-hardware)).

## 5. Experimental Design and Budget

### 5.1 Experimental Grid

A typical $(N, D)$ grid looks like this (sizes and multiples are illustrative):

```text
Tokens D →     0.5×    1×     2×     4×
S   130M        ●      ●      ●      ●
i   520M        ●      ●      ●      ●
z   2.3B        ●      ●      ●      ●
e   8B          ○      ○      ○      ○  ← holdout (excluded from fit)
N   ↓
```

- Fitting points (`●`): small-to-medium runs across multiple token budgets used to fit the scaling law.
- Holdout points (`○`): higher-scale or higher-compute runs withheld from fitting to test extrapolation accuracy.

Choose the layout to match your goal and budget: besides a full grid, common options include fixed-compute IsoFLOP sweeps ([§5.3](#53-token-horizons)) and cheaper L-shape designs ([§8.1](#81-functional-forms)). One pitfall to avoid: if every run sits at a single fixed TPP (say $D=20N$), $N$ and $D$ are perfectly collinear, making it impossible to separate their exponents in a bivariate fit $L(N,D)$.

### 5.2 Model Count, Scale Span, and Extrapolation Factor

| Dimension | Recommendation | Reference |
|---|---|---|
| Number of model sizes | At least 3–4 fitting sizes plus holdouts; more sizes stabilize slope estimates | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) |
| Scale span | Cover the direction of extrapolation; some model families follow power laws across a 34× span | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840) |
| Step ratio between sizes | Geometric spacing (typically $2\times$–$4\times$) | Standard practice |

The extrapolation factor is the ratio between target compute and the largest fitting point. [Delphi](https://openathena.ai/blog/delphi/), for instance, fits on $3\times 10^{18}$–$3\times 10^{20}$ FLOPs and validates on $3\times$–$333\times$ holdouts; [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.2.1 fits on $6\times 10^{18}$–$10^{22}$ FLOPs (40M–16B) and extrapolates to $3.8\times 10^{25}$ FLOPs (roughly $3,800\times$). Larger leaps amplify both functional-form bias and late-stage instability risk; when the target run sits far above the largest holdout, adding an intermediate pilot run before launching the full run is recommended ([§10.1](#101-holdouts-and-joint-validation)).

To save compute, intermediate checkpoints along a single run (e.g., under WSD) can also be included in the fit. Because early loss trajectories are noisy, [Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) find that dropping the first 10% of training or the first 10B tokens noticeably cuts extrapolation error; fix that cutoff beforehand, or pick it on a validation scale split following [Lourie et al., 2026](https://arxiv.org/abs/2608.11859) (fit / validation / test tiers). Under a tight budget, whether to add one larger model size or run extra random seeds at small scales depends on the tradeoff between seed noise and extrapolation span ([Choshen et al., 2025](https://arxiv.org/abs/2410.11840)).

### 5.3 Token Horizons

Set token horizons around the target model's expected TPP. Chinchilla's IsoFLOP approach ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) fixes several compute budgets $C$, trains models across a range of sizes $N$ at each $C$, and fits the minimum of each loss curve to read off $(N_{opt}, D_{opt})$.

{% include figure.liquid
  path='assets/img/pretrain-scaling/chinchilla-isoflop.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 3. Chinchilla IsoFLOP curves. Left: near-quadratic fits of loss vs. log N at fixed compute budgets, where minima give optimal model sizes; center and right: power-law fits of optimal parameters and tokens vs. FLOPs. <a href="https://arxiv.org/abs/2203.15556">Source: Hoffmann et al., 2022, Fig. 3</a>.'
  alt='Chinchilla IsoFLOP plots: near-quadratic loss vs. log parameter curves on the left, and power-law fits of optimal parameter count and token count vs. FLOPs in the center and right.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Across over 400 models (70M–16B, 5B–500B tokens), Chinchilla obtains:

$$N_{opt}\propto C^a,\qquad D_{opt}\propto C^b,\qquad a\approx b\approx0.5$$

meaning compute-optimal training scales model size and training tokens roughly in proportion ($D/N\approx 20$, counting embeddings). Using the same IsoFLOP method, Llama 3 estimated an optimal size of ~402B at $3.8\times 10^{25}$ FLOPs and chose 405B ([Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1).

In practice, adjust token horizons to match the deployment regime:

- Near compute-optimal: 0.5×–4× Chinchilla is common so the grid brackets 1× on both sides; [Fantastic Optimizers](https://arxiv.org/abs/2509.02046v2) uses 1×, 2×, 4×, and 8×.
- Heavy overtraining: [Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2) verify power-law extrapolation up to 32× Chinchilla across 104 models. Testing up to 10,000 TPP across 47 models, [Sardana et al., 2024](https://arxiv.org/abs/2401.00448) find that loss keeps falling, but Chinchilla curves fit only on low-TPP runs overestimate marginal token gains at high TPP. If the target run is heavily overtrained, include high-TPP runs on the ladder.
- Accounting for inference cost: folding serving cost into total cost shifts the optimum toward smaller models trained far past Chinchilla ([Sardana et al., 2024](https://arxiv.org/abs/2401.00448)). Llama 3 405B sits near training compute optimality, whereas 8B and 70B were trained past 15T tokens to cut serving latency and memory ([Llama 3](https://arxiv.org/abs/2407.21783) §1; see [§10.4](#104-wall-clock-speed-and-inference-cost)).
- Data-constrained regime: when target tokens $D$ exceed deduplicated unique tokens $U$, log repetition epochs explicitly and include a repetition decay term when fitting ([§7.3](#73-data-constrained-training-and-repetition)).

### 5.4 MoE Experimental Design

MoE models have more degrees of freedom than dense models, so it is cleanest to split experiments into three separate ladders and fit each on its own:

1. Scale ladder: fix sparsity ratio $S$ and expert size, scale $N_{active}$ and $D$ together, and substitute $N_{active}$ for $N$ in the scaling law;
2. Sparsity ladder: fix $N_{active}$, $D$, and active experts $E_{active}$, varying only total experts $E_{total}$;
3. Granularity ladder: fix $N_{active}$ and $N_{total}$, adjusting expert size and $E_{active}$ inversely ([Krajewski et al., 2024](https://arxiv.org/abs/2402.07871); [Ludziejewski et al., 2025](https://arxiv.org/abs/2502.05172) model expert count, $N_{active}$, and $D$ jointly).

For example, [Kimi K2, 2025](https://arxiv.org/abs/2507.20534) varies $E_{total}$ at fixed $N_{active}$ and training FLOPs to fit a sparsity scaling law: raising the sparsity ratio from 8 to 48 steadily reduces the FLOPs needed to reach a given loss, at the cost of higher cross-node communication and inference memory. Adding iso-$N_{active}$ or iso-FLOP dense baselines alongside the MoE scale ladder also shows how MoE's compute multiplier over dense changes with scale.

## 6. Hyperparameter Search and Training Setup

### 6.1 Search Goals and Procedure

[Lourie et al., 2026](https://arxiv.org/abs/2608.11859) and [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) show that small models are sharply sensitive to hyperparameters: suboptimal settings raise loss and distort the fitted power-law slope, so clean scaling laws hold only along the Fully-Tuned Frontier. Large models, by contrast, exhibit a wide flat optimum (Wide Plateau): on a 156B-A7B model, [Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320v1) scaled LR up or down by $\sqrt{2}$ and raised BSZ by 25%, changing final training loss by no more than $7\times10^{-4}$.

Hyperparameter sensitivity roughly follows $\text{LR} > \text{BSZ} > \text{Schedule} > \text{WD}$, while numerical stability parameters like $\beta_2$ and $\epsilon$ can be fixed once the baseline is stable. Spend search budget primarily on small models and narrow down step by step (sizes are illustrative):

1. Grid search on small models (e.g., 130M): sweep a 2D $(\eta, B)$ grid wide enough that the optimum sits strictly inside the interior rather than on the boundary;
2. Slope check at medium scale (e.g., 500M): run a narrower grid to verify the power-law trend;
3. Local check at larger scales: sample LR within $[\eta_{pred}/\sqrt{2}, \sqrt{2}\eta_{pred}]$ and BSZ within $[B_{opt}/2, 2B_{opt}]$ around the extrapolated values ([§6.3](#63-scaling-laws-for-lr-and-bsz)), expanding only if a boundary point wins.

Picking the lowest loss across noisy trials introduces selection bias and overstates the winning configuration ([Cawley & Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). Re-running the selected configuration with a fresh random seed and fitting on that rerun avoids this bias; also avoid early-stopping candidates based on early loss rankings, which often flip during cooldown ([§6.6](#66-comparing-recipes-fairly)).

### 6.2 Parameterization and Cross-Optimizer Transfer

- $\mu$P and width/depth transfer: $\mu$-Transfer ([Yang et al., 2022](https://arxiv.org/abs/2203.03466v2)) rescales per-layer initializations and LR multipliers so optimal LR transfers zero-shot across width (for non-zero weight decay adjustments, see Appendix G.1.2 of the paper and [Power Lines](https://arxiv.org/abs/2505.13738v2)). Along depth, [Bordelon et al., 2023](https://arxiv.org/abs/2309.16620) scale residual branches by $1/\sqrt{\text{depth}}$ on top of $\mu$P to transfer across both width and depth on ResNets and ViTs; however, [Tensor Programs VI (Yang et al., 2023)](https://arxiv.org/abs/2310.02244) proves that Depth-$\mu$P holds only for single-layer residual blocks and breaks down theoretically for multi-layer nonlinear Transformer blocks. $\mu$P on its own also does not handle transfer across batch size or token horizon $D$. So while $\mu$P is theoretically elegant, industry teams still generally prefer fitting hyperparameter scaling laws empirically on a ladder rather than relying on $\mu$P transfer out of the box.
- Cross-optimizer transfer: when scaling Muon to large LLMs, [Liu et al., 2025](https://arxiv.org/abs/2502.16982) add weight decay and match the RMS norm of orthogonalized updates to AdamW's typical 0.2–0.4 range (using 0.2), allowing Muon to reuse AdamW's tuned LR and WD directly. When architecture and optimizer change together, however, [Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) still observes shifts in optimal LR and BSZ after switching to Muon, requiring a fresh search on small models.

### 6.3 Scaling Laws for LR and BSZ

When a ladder uses only a single fixed $D/N$ ratio, optimal LR and BSZ can be fit directly as power laws of compute $C$:

$$\eta_{opt} = a \cdot C^{-\alpha}, \quad B_{opt} = b \cdot C^{\beta}$$

[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) reports fitted coefficients in this form. Once a ladder spans multiple $D/N$ ratios, $C$ entangles $N$ and $D$, so the two variables need to be separated:

| Formulation | $\eta_{opt}$ | $B_{opt}$ | Scope and Findings |
|---|---|---|---|
| Step Law ([Li et al., 2025](https://arxiv.org/abs/2503.04715v3)) | $c\cdot N^{-\alpha}D^{\beta}$ | $0.58\,D^{0.571}$ | Validated across 3,700+ runs across varying $D/N$; $B_{opt}$ depends mainly on $D$ and barely on $N$ (Appendix A.5) |
| Power Lines ([Bergsma et al., 2025](https://arxiv.org/abs/2505.13738v2)) | Determined via timescale $\tau$ ([§6.4](#64-weight-decay)) | $\propto D^{0.4}$ (nearly independent of $N$) | Connects $(\eta, B, \lambda)$ through a single EMA timescale $\tau$ |
| Token Horizons ([Bjorck et al., 2024](https://arxiv.org/abs/2409.19913)) | $c \cdot N^{-\alpha} D^{-\beta}$ | Fixed BSZ | At fixed $N$ and fixed BSZ, longer horizons $D$ favor a lower peak LR |

Notice that $\eta_{opt}$ has opposite signs on the $D$ exponent in Token Horizons ($D^{-\beta}$) vs. Step Law ($D^{+\beta}$). Their setups differ: Token Horizons holds batch size fixed, whereas Step Law jointly tunes BSZ at each $D$ and scales $B_{opt}\propto D^{0.571}$. Whether batch-size co-scaling fully accounts for the sign flip remains open, so check against holdouts under your own schedule and BSZ rule.

For MoE models, sparsity makes it hard to build tiny MoE ladders, so running full hyperparameter sweeps directly on MoE is expensive. In practice, changing the attention mechanism or switching from dense to MoE barely alters the LR scaling exponents, and optimal LR mostly shifts by a constant factor. A practical compute-saving workflow is to first fit $\eta_{\text{dense}}^{\ast}(N, D)$ on an inexpensive dense ladder, run just a few small MoE points to fit a scalar transfer factor $c_{\text{MoE}}$, and obtain the MoE LR law as $\eta_{\text{MoE}}^{\ast} = c_{\text{MoE}}\,\eta_{\text{dense}}^{\ast}$.

{% include figure.liquid
  path='assets/img/pretrain-scaling/step-law-hyperparameter-validation.png'
  id='fig-step-law-hyperparameter-validation'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 4. Step Law out-of-range test at N = 1B, D = 100B, comparing predicted hyperparameters from several scaling rules against the empirically optimal grid point. Contours are interpolated over 120 (LR, BSZ) runs outside the fitting range under the training recipe and schedule of the paper. <a href="https://arxiv.org/pdf/2503.04715v3#page=1">Source: Predictable Scale: Part I, 2025, v3, Fig. 1 (PDF p. 1)</a>.'
  alt='Step Law Figure 1: loss contours over LR and BSZ at N=1B, D=100B, comparing Step Law, prior hyperparameter rules, and the empirical optimum.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

For batch size, both Step Law and Power Lines find that $B_{opt}$ grows as a power law of training tokens $D$ (exponent ~0.4–0.57) and is largely independent of model size $N$. Runs of different model sizes sharing the same token budget can therefore start with the same BSZ, cutting a search dimension.

Power Lines also measures the critical batch size $B_{crit}\propto D_{min}^{0.5}$ (where $D_{min}$ is the minimum tokens needed to reach a target loss in the small-batch limit). As shown in Figure 5, tokens $D$ and optimization steps $S$ required to hit a given loss form a hyperbola: training at $B=B_{crit}$ costs $2D_{min}$ tokens, and pushing batch size past $B_{crit}$ saves very few steps while burning substantially more compute.

{% include figure.liquid
  path='assets/img/pretrain-scaling/cerebras-bcrit-hyperbola.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 5. Hyperbolic tradeoff between training tokens and steps needed to reach the same target loss on 610M (left) and 1.7B (right) models, colored by loss. B_crit marks the knee between token efficiency and step reduction. <a href="https://arxiv.org/pdf/2505.13738v2#page=6">Source: Bergsma et al., 2025, v2, Fig. 4</a>.'
  alt='Tokens vs. steps curves for 610M and 1.7B models: hyperbolic contours colored by target loss with B_crit marked along the knee.'
  avoid_scaling=true
  zoomable=true
%}

[Schaipp, 2026](https://arxiv.org/abs/2607.01487v1) estimates that the batch-size window incurring at most 5% compute overhead spans roughly $4\times$, or $[B_{opt}/2, 2B_{opt}]$, which is wide enough at medium and large scales for a small local check or rounding to hardware-friendly parallelism multiples. Large runs also sometimes ramp batch size up early in training (for instance, [Llama 3](https://arxiv.org/abs/2407.21783) §3.4.1 steps BSZ from 4M up to 16M tokens, and [Nemotron-4, 2024](https://arxiv.org/abs/2402.16819) uses a similar ramp) to balance early convergence speed and stability.

### 6.4 Weight Decay

Across most industry pretraining runs, weight decay is simply held at a constant (most commonly $\lambda=0.1$) without scaling across $N$ or $D$: for example, [Kimi K2, 2025](https://arxiv.org/abs/2507.20534v2) fixes $\lambda=0.1$ throughout, [OLMo 3, 2025](https://arxiv.org/abs/2512.13961v1) uses a fixed AdamW $\lambda$ excluding embeddings, and [Qwen3, 2025](https://arxiv.org/abs/2505.09388) extrapolates only LR and BSZ while keeping WD constant.

Several academic papers explore WD scaling laws: [Han et al., 2026](https://arxiv.org/abs/2602.11137v2) find that the pretraining-loss-optimal $\lambda_{opt}$ decreases at higher TPP (while noting that slightly larger WD preserves effective weight rank and post-training plasticity), and [Power Lines](https://arxiv.org/abs/2505.13738) ties $(\eta, B, \lambda)$ together via the weight EMA timescale $\tau = B/(\eta\lambda D)$, adjusting $\lambda \propto B/D$ at fixed $\mu$P $\eta$ to keep $\tau_{opt}$ constant. Whether scaling WD actually pays off at frontier scale remains open; in practice, fixing $\lambda=0.1$ on the ladder is usually sufficient.

### 6.5 LR Schedule and Cooldown

Two learning-rate schedules dominate pretraining:

- Cosine schedule: decays smoothly along a cosine curve from warmup down to an end learning rate $\eta_{\text{end}}$, suited to runs with a predetermined step budget.
- WSD (Warmup-Stable-Decay): holds LR constant for most of training before annealing rapidly at the end ([River Valley (Wen et al., 2024)](https://arxiv.org/abs/2410.05192v3) explains the stable and decay phases via loss-landscape geometry; [Tissue et al., 2024](https://arxiv.org/abs/2408.11029v2) and [Wang et al., 2025](https://arxiv.org/abs/2512.13705) study how decay shapes and decay fractions transfer across scales). WSD's practical advantage is that multiple decay branches can fork off a single stable trajectory at different horizons without re-running the trunk.

Three empirical lessons from industry practice matter when choosing $\eta_{\text{end}}$ and the schedule:

- Do not decay $\eta_{\text{end}}$ to 0 during PT: base pretraining is almost always followed by continued pretraining (CPT / CT) and post-training, so keeping $\eta_{\text{end}}$ at a small non-zero value (e.g., $3\times 10^{-5}$) leaves room for subsequent training stages to make progress.
- Fixing absolute $\eta_{\text{end}}$ beats fixing the ratio $\eta_{\text{end}}/\eta_{\text{peak}}$: earlier work (such as [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) and [DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954)) commonly fixed $\eta_{\text{end}}/\eta_{\text{peak}}=0.1$, coupling the end LR to the peak LR. Raising $\eta_{\text{peak}}$ then drags $\eta_{\text{end}}$ up with it, leaving too much gradient noise unaveraged at the end of training and preventing $\eta_{\text{peak}}$ from reaching higher, better values. Fixing an absolute $\eta_{\text{end}}$ (as in [Step Law](https://arxiv.org/abs/2503.04715v3)) unlocks higher optimal $\eta_{\text{peak}}$ and lower final loss.
- Properly tuned Cosine outperforms WSD: at matched $(N, D)$, Cosine and WSD have noticeably different optimal $(\eta, B)$; once $\eta_{\text{end}}$ is fixed and hyperparameters are tuned independently for each schedule, Cosine favors a higher peak LR ($\eta_{\text{cosine}}^{\ast} > \eta_{\text{WSD}}^{\ast}$) and consistently reaches lower final pretraining loss than WSD ([Kimi K3, 2026](https://arxiv.org/abs/2607.24653) §3.2).

If the full-scale run uses either of the following end-of-training steps, mirror them on the ladder as well:

- Cooldown data shift (Continued Training / Cooldown): switching to higher-quality data mixtures during LR decay ([Nemotron-4, 2024](https://arxiv.org/abs/2402.16819); [OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3); [OLMo 3, 2025](https://arxiv.org/abs/2512.13961); [Nemotron 3 Super, 2026](https://arxiv.org/abs/2604.12374)); see [§11](#11-multi-stage-training-ladders).
- Weight averaging: averaging independently fine-tuned branches from a shared checkpoint ([Model Soups](https://arxiv.org/abs/2203.05482v3)) or maintaining a sliding average along a single trajectory ([LAWA](https://arxiv.org/abs/2209.14981); [Sanyal et al., 2024](https://arxiv.org/abs/2306.03241)). On 1.3B and 13B models, [Model Merging (ByteDance Seed, 2025)](https://arxiv.org/abs/2505.12082v3) uses checkpoint averaging (PMA) along the WSD stable phase as a cheap proxy for LR annealing; empirically, however, weight averaging only approximates LR decay and generally does not capture the full loss drop of an actual cooldown.

### 6.6 Comparing Recipes Fairly

Keep three rules in mind when comparing optimizers, architectures, or LR schedules on a ladder:

- Match tuning effort on both sides: [Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) show that comparing a tuned new optimizer against default, untuned AdamW hyperparameters substantially overstates gains; once both sides are properly tuned, the gap shrinks considerably. The same applies when comparing LR schedules: tune hyperparameters separately for each schedule (as in the Cosine vs. WSD comparison in [§6.5](#65-lr-schedule-and-cooldown)).
- Compare at the end of cooldown: Fantastic Optimizers observe that loss rankings during the stable phase frequently flip during LR decay, so mid-run rankings can be misleading.
- Check whether gains decay with scale: in Fantastic Optimizers' 8× Chinchilla runs, matrix-preconditioned optimizers' token-efficiency lead over AdamW shrank from $1.4\times$ at 0.1B to $1.1\times$ at 1.2B (and token-efficiency gains do not automatically translate into wall-clock speedups). Always test candidate improvements across multiple model sizes to see if the gain narrows with scale.

## 7. Data Ladders

I do not come from a data background and my familiarity with data pipelines is limited, so this section draws on public papers to briefly cover the three data questions directly tied to scaling ladders: evaluating new data sources ([§7.1](#71-data-quality-and-source-evaluation)), searching multi-domain mixtures ([§7.2](#72-data-mixture-ladders)), and multi-epoch repetition under data constraints ([§7.3](#73-data-constrained-training-and-repetition)).

### 7.1 Data Quality and Source Evaluation

Comparing pretraining corpora in [DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) highlights two effects:

- Higher-quality data shifts optimal compute allocation toward larger model sizes $N$;
- Fitted scaling law coefficients differ noticeably across corpora and should not be copied across datasets.

Whenever the underlying corpus version changes, re-verify scaling law parameters. In [Kimi K3, 2026](https://arxiv.org/abs/2607.24653) §3.2, for example, updating the architecture, corpus, and training recipe together prompted refitting BSZ, LR, optimal TPP, and aspect ratio.

Routine screening of an individual new data source does not require re-running a full $(N, D)$ grid; three low-cost methods are commonly used:

- Checkpoint micro-annealing: [OLMo 2, 2025](https://arxiv.org/abs/2501.00656) and [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) branch from a mid-to-late trunk checkpoint, mix the candidate source with the base corpus for a short cooldown, and measure downstream shifts directly.
- Small-model single-domain ablations: [Kimi K3](https://arxiv.org/abs/2607.24653) §3.1 runs single-domain ablations on a fixed small model to set per-domain sampling weights.
- Targeted synthetic-data ablations: [Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) mixes synthetic legal reasoning data into [Nemotron 3 Nano, 2025](https://arxiv.org/abs/2512.20848) pretraining, lifting LegalBench average accuracy from 64.6 to 74.7.

### 7.2 Data Mixture Ladders

A data mixture ladder holds model size $N$, token budget $D$, and optimizer hyperparameters fixed while varying sampling weights $\mathbf{w}=(w_1,\ldots,w_m)$ across $m$ data buckets. Combining [Olmix, 2026](https://arxiv.org/abs/2602.12237), [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.4.4, and the [Marin Data Pipeline, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/), a typical workflow looks like this:

1. Define buckets and count unique tokens: partition data by source, domain, or instance-level attributes (e.g., educational value, code, language, safety; [Qwen3, 2025](https://arxiv.org/abs/2505.09388)), and recount deduplicated unique tokens $N_j$ in every bucket whenever filtering thresholds change ([Marin Data Pipeline, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/)). If training shifts mixtures across stages (for example, [Nemotron 3 Super, 2026](https://arxiv.org/abs/2604.12374) emphasizes diversity in the first 80% of its 25T tokens and high-quality sources in the final 20%), optimize each stage's mixture separately ([§11](#11-multi-stage-training-ladders)).
2. Establish simple baselines: compare against natural token-count proportions and epoch-capped [UniMax](https://arxiv.org/abs/2304.09151); [Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) notes that poorly constrained mixture regressions can easily underperform both baselines.
3. Train small proxy models:
   - Proxy size: at 5× Chinchilla horizons, proxy models of 15M parameters and above achieve >0.89 Spearman rank correlation with a 1B target model across mixtures, dropping to 0.73 at 1M ([Olmix](https://arxiv.org/abs/2602.12237)); [OLMo 3](https://arxiv.org/abs/2512.13961) uses 30M proxies trained for 3B tokens.
   - Proportional pool downsampling: because small proxies consume far fewer tokens, sampling directly from the full corpus avoids the multi-epoch repetition that the large run will actually hit. [Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) therefore downsamples each bucket's pool in proportion to the proxy's active parameters so that small runs at 791 tokens/active param experience the exact same repetition epochs as the full model. In my view, keeping repetition epochs consistent across all ladder sizes is very important, especially when studying components like n-gram modules that depend heavily on corpus size, repetition frequency, and memorization/overfitting; without proportional pool scaling, a small model sees well under 1 epoch on the full pool and behaves nothing like a multi-epoch large run.
   - Number of sweeps and model-merging alternatives: fitting a log-linear mixture model typically takes at least $3(m+1)$ proxy runs sampled from a Dirichlet distribution centered on natural proportions. [DeMix, 2026](https://arxiv.org/abs/2602.00747) instead trains one component model per bucket and evaluates arbitrary mixtures via weighted parameter merging, achieving higher rank correlation than retraining small proxies from scratch.
4. Fit and solve under constraints: fitting separate log-linear models per downstream task before aggregating achieves much higher held-out mixture correlation (0.983) than fitting a single aggregate score directly (0.866; [Olmix](https://arxiv.org/abs/2602.12237)). Because overweighting a small bucket causes excessive repetition and drives loss back up ([Marin](https://openathena.ai/blog/marin-data-pipeline-overview/)), enforce per-bucket epoch caps $w_j \le k N_j / R$ ($R$ = target token budget, $k$ = max epochs; [§7.3](#73-data-constrained-training-and-repetition)) alongside a KL penalty toward natural proportions (e.g., $\lambda=0.05$), then verify the resulting mixture against natural sampling and UniMax at medium scale.
5. Incremental mixture reuse after corpus updates: when only a few buckets are added or updated, [Olmix](https://arxiv.org/abs/2602.12237)'s mixture reuse freezes relative weights among unchanged buckets and re-runs only sweeps touching the updated ones. Across 5 update rounds scaling to 64 domains at 1B/100B, this cut proxy runs by 74% while retaining 95% of the gain of a full re-sweep (+11.6% over natural proportions), powering three data iterations in [OLMo 3](https://arxiv.org/abs/2512.13961).

### 7.3 Data-Constrained Training and Repetition

When target training tokens $D$ exceed total unique tokens $U$, training enters the multi-epoch regime. Run global cross-source deduplication before counting epochs: global deduplication in the [Marin Data Pipeline, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/) uncovered heavy overlap across public datasets, with the largest share coming from [Nemotron-CC](https://arxiv.org/abs/2412.02595) and its synthetic rewrites.

Assuming exponentially decaying marginal value from each extra pass, [Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264v5) define the effective data size $D^\prime$:

$$D^{\prime} = U_D + U_D \cdot R_D^{\ast} \left(1 - e^{-R_D / R_D^{\ast}}\right)$$

where $U_D$ is unique tokens, $R_D$ is the number of extra repetitions beyond the first pass ($R_D=0$ for 1 epoch), and $R_D^\ast$ is the repetition half-life parameter (empirically ~4). As $R_D\to\infty$, effective data saturates at $U_D(1+R_D^\ast)$.

{% include figure.liquid
  path='assets/img/pretrain-scaling/muennighoff-epoch-returns.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 6. Diminishing returns of repeated data (4.2B model on 12B unique tokens). The first few epochs nearly match fresh data, after which marginal gains decay and saturate around 40 epochs. <a href="https://arxiv.org/pdf/2305.16264v5#page=1">Source: Muennighoff et al., 2023, v5, Fig. 1 left</a>.'
  alt='Multi-epoch training curve of a 4.2B model on 12B unique tokens: final test loss vs. cumulative tokens and epochs, saturating near 40 epochs.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Key findings on multi-epoch limits and mitigations include:

- Overfitting scales with both $N$ and unique pool size $U$: [Yan et al., 2025](https://arxiv.org/abs/2511.13421v2) show analytically in linear regression that the optimal epoch count grows logarithmically with dataset size; empirically, [Lovelace et al., 2026](https://arxiv.org/abs/2605.01640) and [Xue et al., 2023](https://arxiv.org/abs/2305.13230v2) find that larger $N$, smaller $U$, and more repetitions all worsen overfitting. At fixed $U$, scaling $N$ aggressively worsens memorization and shifts compute-optimal allocation toward smaller $N$.
- Data quality and dropout: [Xue et al., 2023](https://arxiv.org/abs/2305.13230v2) find that higher data quality at fixed unique token count does not prevent multi-epoch overfitting, whereas adding moderate dropout (v2 Table 4) substantially mitigates it.
- Mixing scarce domains into a large general corpus: a small target domain tolerates far more epochs when mixed into a large general background corpus than when trained on alone. Across 2,000+ runs, [Sedova et al., 2026](https://arxiv.org/abs/2605.12715) show positive returns up to 15–20 epochs on scarce target domains in mixtures, with repetition-aware mixture scaling laws extrapolating cleanly from small to large models.
- Quality-tiered epoch caps: when continuing joint pretraining from a late [Kimi K2](https://arxiv.org/abs/2507.20534) checkpoint, [Kimi K2.5, 2026](https://arxiv.org/abs/2602.02276) enforces strict per-source epoch caps; [OLMo 3](https://arxiv.org/abs/2512.13961) (§3.4.4, Appendix A.2.5) upsamples only high-quality buckets up to 7 epochs, outperforming aggressive hard filtering.
- Synthetic rephrasing over raw repetition: on SimpleQA, [Kimi K2, 2025](https://arxiv.org/abs/2507.20534) §2.2 compares repeating raw text 10 epochs (23.76), rephrasing once and training 10 epochs (27.39), and generating 10 distinct rephrasings trained 1 epoch each (28.94); production runs cap knowledge-corpus rephrasing at 2 passes, a practice carried into [Kimi K3](https://arxiv.org/abs/2607.24653) §3.1. Track rephrased tokens separately from raw unique tokens when accounting.

## 8. Fitting Loss Scaling Laws

Before fitting $L(N,D)$, double-check three prerequisites ([Porian et al., 2024](https://arxiv.org/abs/2406.19146)): FLOPs $C$ includes the output head while $N_{\text{body}}$ is logged separately ([§3.1](#31-parameter-counting)), warmup steps scale proportionally with total training tokens ([§4.1](#41-grouping-variables)), and every grid point has been tuned onto the Fully-Tuned Frontier when solving for optimal allocation or comparing methods ([§6.1](#61-search-goals-and-procedure)).

### 8.1 Functional Forms

If all runs share a single fixed $D/N$ ratio, a 1D power law in compute $L(C) = A/C^\gamma + E$ suffices. Predicting across varying $D/N$ ratios or solving for optimal $(N, D)$ allocation requires a 2D surface in $N$ and $D$. The standard starting point is Chinchilla's additive form:

$$L(N,D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + E$$

This additive ansatz assumes $N$ and $D$ act independently ($\partial^2L/\partial N\partial D\equiv0$). Across wide $(N, D)$ grids, however, Chinchilla's residuals form a clear saddle pattern growing toward all four corners (Figure 7 left), and empirical mixed partials remain consistently negative, showing that model capacity and data horizon interact.

Two recent formulations capture this $(N, D)$ coupling:

- [Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972): lets both the coefficient and exponent of the data term vary with $N$, using 9 parameters in total;
- [Skaling (Videau et al., 2026)](https://arxiv.org/abs/2608.07222v1): wraps the additive terms in a single outer coupling exponent $k$, using 6 parameters (recovering Chinchilla at $k=1$, with empirical fits around $k \approx 0.31$–$0.45$):

$$L(N,D)=\left(\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}\right)^{k}+E$$

Skaling adds just one parameter over Chinchilla while preserving closed-form $(N_{opt}, D_{opt})$ solutions; across both the Farseer and SK-Grid benchmarks, 6-parameter Skaling achieves lower interpolation and extrapolation error than both Chinchilla and 9-parameter Farseer ([Skaling](https://arxiv.org/abs/2608.07222) Table 1; Figure 7 center and right), making it a strong default choice.

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-vs-chinchilla-error.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 7. Prediction residuals of Chinchilla vs. Skaling on the same (N, D) grid, where each point is a training run. Left and center: signed percentage error on a shared color scale; Chinchilla shows saddle-shaped corner errors while Skaling stays near zero across the grid. Right: error ratio, where Skaling is more accurate on 76% of runs by a 2.2× median factor. <a href="https://arxiv.org/abs/2608.07222">Source: Videau et al., 2026, Fig. 1</a>.'
  alt='Three (N, D) scatter grids: Chinchilla signed percentage error on the left, Skaling signed percentage error in the center, and their error ratio on the right.'
  avoid_scaling=true
  zoomable=true
%}

A second practical benefit of Skaling is avoiding full-grid sweeps. As shown in Figure 8, running an L-shape layout (many model sizes $N$ at small $D$, plus many token horizons $D$ at small $N$) costs only $1/5$–$1/10$ the compute of a dense grid yet extrapolates into the large-$(N,D)$ corner nearly as accurately as a full grid under Skaling, whereas Chinchilla's additive form degrades sharply on L-shape samples.

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-v1-sampling-evaluation.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 8. Sampling layouts and evaluation regimes in Skaling. Left: Random holdout vs. L-shape sampling (varying N at small D and varying D at small N). Right: interpolation, extrapolation along N or D, and joint extrapolation regions, with D on the horizontal axis and N on the vertical axis. <a href="https://arxiv.org/pdf/2608.07222v1#page=5">Source: Videau et al., 2026, v1, Fig. 4 (PDF p. 5)</a>.'
  alt='Skaling v1 Figure 4: Random vs. L-shape sampling designs on the left, and interpolation plus N-, D-, and joint-extrapolation evaluation regions on the right.'
  avoid_scaling=true
  zoomable=true
%}

### 8.2 Loss Trajectory and Annealing Scaling Laws

[Tissue et al., 2024](https://arxiv.org/abs/2408.11029) model loss along training step $s$ as a functional of the LR schedule:

$$\hat L(s) = L_0 + A \cdot S_1(s)^{-\alpha} - C \cdot S_2(s)$$

where $S_1(s)=\sum_{i\leq s}\eta_i$ is cumulative learning rate area and $S_2(s)$ captures cumulative annealing weighted by a forgetting kernel. This lets you fit entire loss trajectories using intermediate evaluations rather than final endpoints alone, and predicts curves across cosine schedules, WSD, and re-warmed continued training (§4.7 of the paper). Note that the formulation assumes a fixed peak LR, so extrapolating peak LR across scales still relies on the hyperparameter scaling laws in [§6.3](#63-scaling-laws-for-lr-and-bsz).

Personal experience: replacing the single-exponential forgetting kernel in $S_2(s)$ with a power-law kernel $K(\Delta) = (1+\Delta)^{-p}$ generally produces cleaner fits than the paper's original kernel.

### 8.3 Numerical Fitting Details

Fitting power laws with plain nonlinear least squares easily gets trapped in poor local minima: exponents $(\alpha, \beta)$ and prefactors $(A, B)$ are strongly collinear, so minor shifts in initialization or outlier weighting can swing extrapolated $(N_{opt}, D_{opt})$ by several-fold ([Besiroglu et al., 2024](https://arxiv.org/abs/2404.10102); [(Mis)Fitting (Li et al., 2025)](https://arxiv.org/abs/2502.18969)). Standard practice follows Chinchilla: reparameterize prefactors in log space ($a=\log A, b=\log B, e=\log E$), evaluate $\log \hat{L}(N,D)$ via `log-sum-exp`, minimize Huber loss on $\log L$ (e.g., $\delta=10^{-3}$) to dampen outliers, and run multi-start L-BFGS across an initialization grid before picking the lowest-objective solution.

## 9. Downstream Task Prediction

Beyond pretraining loss, teams naturally want to use a pretraining ladder to forecast hero-run scores on downstream benchmarks. I do not work directly on downstream prediction myself, so my perspective here is limited; in practice, however, task losses and discrete accuracies are far noisier and much harder to extrapolate than pretraining loss ([Schaeffer et al., 2024](https://arxiv.org/abs/2406.04391) analyze how distractor probability mass and argmax thresholding add substantial prediction noise on multiple-choice benchmarks). Both academia and industry are still actively experimenting with new approaches, and no single consensus method has emerged yet.

A few representative lines of work illustrate current thinking:

| Approach | Core Idea | Representative Work |
|---|---|---|
| General Loss $\to$ Task Score | Extrapolate pretraining loss from $(N,D)$, then map loss to benchmark score via a monotonic power law or sigmoid | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2); [Delphi](https://openathena.ai/blog/delphi/) |
| $(N,D) \to$ Task NLL $\to$ Acc | Two-stage fit: extrapolate gold-answer negative log-likelihood (NLL) vs. compute, then fit a sigmoid from NLL to accuracy | [Bhagia et al., 2024](https://arxiv.org/abs/2412.04403) (OLMo Task Ladder); [Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1 |
| Difficulty-clustered subsets | Cluster benchmark items by small-model pass rates, extrapolate compute curves on subsets with clean upward signal, then map back to full-benchmark accuracy | [COD (Xu et al., 2026, v4)](https://arxiv.org/pdf/2502.17262v4); [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774v6) |
| Hint-assisted evaluation | Give small models graded hints at eval time to lift them off 0%, jointly fit accuracy over compute and hint strength, and extrapolate unhinted large-model accuracy | [Hinted Scaling Laws, 2026](https://arxiv.org/abs/2611.02270) |

For standard two-stage prediction, [Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1 is a classic example: stage one fits a linear trend from $\log\text{FLOPs}$ to normalized gold-choice NLL on ladder runs at $\le 10^{22}$ FLOPs, and stage two combines ladder checkpoints with Llama 2 models to fit a sigmoid from NLL to accuracy, closely matching 405B performance on ARC-Challenge.

On reasoning benchmarks like MATH, AIME, and GSM8K, the main bottleneck is that small models score near 0% across the board, leaving no upward trend to extrapolate. Two recent methods get around this zero-score floor from opposite sides:

First, **clustering items by difficulty on the benchmark side** ([COD, Xu et al., 2026, v4](https://arxiv.org/pdf/2502.17262v4)): using repeated-sampling pass rates across small models to cluster problems by difficulty, COD drops both zero-score items (0% across all small models) and already-saturated easy items, fits a bi-exponential curve on medium-difficulty clusters with clear monotonic signal to extrapolate to target compute, and maps cluster scores back to full-benchmark accuracy via a learned calibration curve. Across 8 benchmarks, COD predicts 70B accuracy within a 1.55 percentage-point mean absolute error (Figure 9; cross-architecture transfer from dense ladders to MoE roughly doubles error to ~3.11 percentage points).

{% include figure.liquid
  path='assets/img/pretrain-scaling/cod-v4-prediction-accuracy.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 9. COD v4 forecast curves vs. actual 70B scores on MATH and MMLU-pro, comparing COD against Loss-Intermediate, End-to-End(exp), and End-to-end(BNSL). Red points are small ladder models; blue points are 70B ground truth. <a href="https://arxiv.org/pdf/2502.17262v4#page=9">Source: Xu et al., 2026, v4, Fig. 4 MATH and MMLU-pro panels (PDF p. 9)</a>.'
  alt='COD v4 Figure 4 subplots for MATH and MMLU-pro: accuracy vs. compute comparing COD, Loss-Intermediate, End-to-End(exp), and End-to-end(BNSL) against actual 70B scores.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Second, **lowering task difficulty via hints on the evaluation side** ([Hinted Scaling Laws, van Adrichem et al., 2026](https://arxiv.org/abs/2611.02270)): instead of filtering items, it turns the benchmark into a continuum of easier variants parameterized by hint strength $H$ (revealing the first $H$ fraction of reference solution steps in Truncation, randomly unmasking $H$ fraction of solution words in Masking, or interpolating logits with a strong teacher at weight $H$ in Logit mixing), so models that score 0% unhinted produce smooth non-zero accuracy curves (Figure 10 left). Because hint strength $H$ and log compute $C$ trade off nearly linearly in lifting accuracy, fitting a joint floored sigmoid in $(C, H)$ on just the smallest ladder runs and evaluating at $H=0$ forecasts unhinted accuracy at scale. On the Delphi pretraining ladder, fitting only $\le 10^{20}$ FLOPs runs predicts GSM8K accuracy at $10^{23}$ FLOPs within 3 percentage points, compared to a 13 percentage-point miss from unhinted extrapolation (Figure 10 right).

{% include figure.liquid
  path='assets/img/pretrain-scaling/hinted-scaling-laws-delphi.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 10. Overview of Hinted Scaling Laws and its GSM8K forecast on the Delphi pretraining ladder. Top: three hinting schemes (Truncation, Masking, Logit mixing). Bottom: smooth accuracy gains with hint fraction H (left), joint surface fit over compute and H on small runs (center), and H=0 extrapolation hitting the 5,600× larger target run (right). <a href="https://arxiv.org/pdf/2611.02270v1#page=2">Source: van Adrichem et al., 2026, v1, Fig. 1</a>.'
  alt='Hinted Scaling Laws Figure 1: top illustrates Truncation, Masking, and Logit mixing hints; bottom shows GSM8K accuracy vs. hint strength, the joint fit surface, and 5,600x compute extrapolation on the Delphi ladder.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

## 10. Extrapolation Validation

### 10.1 Holdouts and Joint Validation

Place holdout runs along the actual direction of extrapolation (larger $N$, longer $D$, or higher compute $C$) and span multiple extrapolation factors when budget permits (see [§1.1](#11-what-is-a-scaling-ladder), [§5.2](#52-model-count-scale-span-and-extrapolation-factor)). Once a holdout set is used to revise the recipe or functional form, it effectively becomes a validation set, and unbiased error estimates require fresh holdouts. When the target run sits orders of magnitude above the largest holdout, running a medium-scale pilot with the final frozen recipe beforehand checks that the loss trajectory stays inside the predicted band.

Architecture, optimizer, hyperparameters, and data mixtures are usually ablated one at a time, but gains from individual changes are rarely additive ([Marin Follow-up](https://openathena.ai/blog/pretraining-speedup/); [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) Appendix A.2.5). Choices interact: a stronger optimizer can change multi-epoch overfitting speed, and a new data mixture can shift optimal LR. Before locking in the final recipe, combine all winning changes and run a joint validation at the target TPP and repetition count.

### 10.2 Stability Stress Tests

Shallow, short small-model runs rarely surface late-stage numerical overflows, loss spikes, or router collapse that appear at scale. [Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) runs high-LR stress tests on medium-scale models, pushing LR to $2\times$ and $4\times$ the predicted optimum to compare stability head-to-head across candidate architectures or optimizers under elevated optimization pressure.

Short high-LR stress tests still do not replace continuous monitoring late into a long run. In [Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) §2.7, two late-training divergences occurred: the first came from low-precision gradient underflow/overflow in the output layer and was resolved by casting output-layer gradients back to FP32; the cause of the second remained undetermined and was mitigated by starting cooldown early. Track per-layer gradient norms, activation maxima, and loss spikes throughout full-scale runs, plus expert load balance and token-drop rates for MoE.

### 10.3 Aligning Training Frameworks and Hardware

Small ladder runs and large flagship runs often execute on different stacks: small sweeps commonly use FSDP with standard kernels, whereas large runs switch to frameworks like Megatron with hybrid TP/PP/EP parallelism, fused kernels, low-precision training (such as FP8 in [DeepSeek-V3, 2024](https://arxiv.org/abs/2412.19437) §3.3), and frequent checkpoint restarts ([Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.3.4).

In my own experience, even with the exact same architecture, optimizer, and data, running on different training frameworks often produces noticeably different losses and internal training metrics (such as per-layer grad norms and activations); switching GPU generations (for instance, Hopper vs. Blackwell GPUs) can also shift loss by around 0.01.

Ideally, a research ladder should run on the exact same training framework and GPU generation as the full-scale run. When unifying them is not practical, align pretraining loss, validation/test loss, benchmark evaluation losses and scores, and all internal training observables as closely as possible on identical configurations.

### 10.4 Wall-Clock Speed and Inference Cost

Theoretical FLOP reductions do not automatically translate 1-to-1 into wall-clock speedups. The [Marin Follow-up](https://openathena.ai/blog/pretraining-speedup/) distinguishes *theoretical efficiency* from *realized efficiency*, and [Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) likewise benchmarks precision and kernel choices by measured cluster throughput, especially for MoE where cross-node EP all-to-all communication adds zero FLOPs yet directly governs step time.

For models headed into production, factor serving cost into $(N, D)$ selection as well (see [§5.3](#53-token-horizons)). [Sardana et al., 2024](https://arxiv.org/abs/2401.00448) model pretraining cost jointly with prefill and decode serving compute; in practice, combine expected prompt/output lengths, latency targets (TTFT / TPOT), memory limits, and serving quantization precision with measured cluster throughput when choosing model size and token horizon.

## 11. Multi-Stage Training Ladders

Modern LLMs routinely train across multiple stages: general pretraining (PT), followed by high-quality continued training (CT / Mid-training) and long-context extension. Because data mixtures and LR schedules change across stages, scaling law coefficients fit on PT cannot be plugged directly into later stages ([Qwen3, 2025](https://arxiv.org/abs/2505.09388); [OLMo 3, 2025](https://arxiv.org/abs/2512.13961)).

When CT / Mid-training requires quantitative forecasts for hyperparameters, mixture weights, or stage token budgets, build a dedicated CT ladder on top of the PT ladder: initialize each model size from its corresponding PT checkpoint (keeping a non-zero PT $\eta_{\text{end}}$; see [§6.5](#65-lr-schedule-and-cooldown)), fit stage-specific scaling curves, and track gains on target capabilities alongside retention on core general benchmarks.

Keep in mind that CT outcomes depend on the state of the starting checkpoint: two PT checkpoints with similar validation loss can respond differently to the exact same CT recipe if they differ in end LR, prior data repetition counts, or optimizer momentum states. Match how small-scale starting checkpoints are produced, including whether LR re-warms or optimizer states reset at the stage boundary, to the target full-scale pipeline.

As for the final long-context extension stage, its token budget is a tiny fraction of total pretraining and teams do not necessarily build a standalone scaling ladder for it; I also do not work on long-context training or evaluation myself, and it sits outside the main focus of this post, so I skip a detailed discussion here (for long-sequence FLOPs accounting see [§3.2](#32-compute), and for evaluation benchmarks and attention tradeoffs see [RULER, 2024](https://arxiv.org/abs/2404.06654) and [Chen et al., 2025](https://arxiv.org/abs/2503.09579)).

## 12. Practical Checklist

The checklist below is meant only as an empirical rule-of-thumb reference rather than an exhaustive spec. Even at frontier labs, building a scaling ladder inevitably surfaces subtle implementation bugs, and what separates strong executions is usually catching and fixing more bugs faster. Here are the highest-leverage checks distilled from earlier sections:

- [ ] **Separate parameter and FLOPs accounting**: fit $L(N,D)$ using non-embedding backbone parameters $N_{\text{body}}$, while adding both the output head and sequence-length attention terms back into FLOPs $C$ ([§3.1](#31-parameter-counting), [§3.2](#32-compute)).
- [ ] **Scale warmup and eval intervals as fractions of training**: avoid fixed step counts across scales; compare main next-token cross-entropy on a shared validation set, and use BPB across tokenizers ([§3.3](#33-loss-and-token-accounting), [§4.1](#41-grouping-variables), [§4.5](#45-vocabulary-size-and-precision)).
- [ ] **Tune small runs thoroughly and compare after full cooldown**: spend search budget putting small grid points onto the Fully-Tuned Frontier; tune each candidate architecture, optimizer, or schedule independently and judge rankings at the end of LR decay ([§6.1](#61-search-goals-and-procedure), [§6.6](#66-comparing-recipes-fairly)).
- [ ] **LR schedule and MoE LR transfer**: keep a non-zero PT $\eta_{\text{end}}$ for CT, and fix absolute $\eta_{\text{end}}$ rather than fixing $\eta_{\text{end}}/\eta_{\text{peak}}$ ([§6.5](#65-lr-schedule-and-cooldown)); for MoE, fit the LR scaling law on a cheap dense ladder first and estimate the scalar multiplier $c_{\text{MoE}}$ on a few small MoE runs ([§6.3](#63-scaling-laws-for-lr-and-bsz)).
- [ ] **Match data repetition epochs across scales**: for multi-epoch or data-mixture experiments, downsample bucket pools proportionally with model scale and account for repetition decay when fitting ([§7.2](#72-data-mixture-ladders), [§7.3](#73-data-constrained-training-and-repetition)).
- [ ] **Use coupled functional forms and drop noisy early steps**: prefer Skaling for 2D $(N, D)$ fits (paired with L-shape sampling when budget is tight); drop the first ~10% of steps when fitting intermediate checkpoints, and optimize log-space Huber loss with multi-start L-BFGS ([§5.2](#52-model-count-scale-span-and-extrapolation-factor), [§8.1](#81-functional-forms), [§8.3](#83-numerical-fitting-details)).
- [ ] **Multi-tier holdouts, joint validation, and high-LR stress tests**: reserve untouched holdouts along the extrapolation trajectory; validate combined recipe changes together, and stress-test new architectures or optimizers at $2\times / 4\times$ LR ([§10.1](#101-holdouts-and-joint-validation), [§10.2](#102-stability-stress-tests)).
- [ ] **Align training frameworks and GPUs**: run the ladder on the same framework and GPU generation as the target model whenever possible; otherwise align losses, benchmark scores, and internal training observables on matched runs ([§10.3](#103-aligning-training-frameworks-and-hardware)).

## Appendix A. Public Ladder Configurations

### A.1 Public Scale Configurations

Model and token ranges used in representative public scaling-law studies (parameter counting conventions follow each original paper):

| Source | Parameter Range | Token / Compute Range | Reference |
|---|---|---|---|
| OpenAI | Multiple sizes up to 1.5B (non-embedding) | 22M–23B tokens | [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) |
| DeepMind Chinchilla | 70M–16B (400+ models) | 5B–500B tokens | [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) |
| StepFun | Multiple sizes (3,700+ models) | ~100T tokens cumulative across sweeps | [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) |
| Gadre et al. | 11M–6.9B (104 models) | Up to 32× Chinchilla | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540) |
| Fantastic Optimizers | 0.1B–1.2B (4 sizes) | 1×–8× Chinchilla | [Wen et al., 2025](https://arxiv.org/abs/2509.02046) |
| Delphi | IsoFLOP sweeps across sizes and tokens; holdout up to 25B | Fit $3\times 10^{18}$–$3\times 10^{20}$ FLOPs; holdout up to $10^{23}$ FLOPs | [Marin, 2026](https://openathena.ai/blog/delphi/) |
| Llama 3 | 40M–16B | Fit $6\times 10^{18}$–$10^{22}$ FLOPs; target $3.8\times 10^{25}$ FLOPs | [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.2.1 |

Training token horizons of selected open-weight base models (for target-TPP reference):

| Family | Released Model Sizes | Total Training Tokens | References |
|---|---|---|---|
| NVIDIA Nemotron | Nemotron-4: 15B, 340B; Nemotron 3 (Hybrid MoE): Nano 30B-A3B, Super 120B-A12B, Ultra 550B-A55B | Nemotron-4: 8T, 9T; Nemotron 3: Nano / Super 25T each, Ultra 20T | [Nemotron-4 15B](https://arxiv.org/abs/2402.16819); [340B](https://arxiv.org/abs/2406.11704); [Nemotron 3 Nano](https://arxiv.org/abs/2512.20848); [Super](https://arxiv.org/abs/2604.12374); [Ultra](https://arxiv.org/abs/2606.15007) |
| OLMo | OLMo 1: 1B, 7B; OLMo 2: 7B, 13B, 32B; OLMo 3: 7B, 32B | OLMo 1: 2T–2.46T; OLMo 2: up to 5T+; OLMo 3: ~6T (5.9T PT + 100B mid-train + 50B long-context) | [OLMo, 2024](https://arxiv.org/abs/2402.00838v4); [OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3); [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) |
| Llama 3 | 8B, 70B, 405B | 405B: 15.6T tokens (near compute-optimal); 8B/70B: ~15T tokens (heavily overtrained) | [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §1, §3.4 |

### A.2 Fantastic Optimizers Ladder

[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046v2) Tables 2–3 specify a dense ladder built for optimizer comparisons and hyperparameter sweeps: based on Llama 2, all four sizes fix 32 layers, MHA, and `seq_len` 4096, scaling parameter count solely via `hidden_dim`.

| Size | hidden_dim | inter_dim | heads | Token Horizons |
|---|---|---|---|---|
| 130M | 512 | 2,048 | 8 | 1×–8× Chinchilla |
| 300M | 768 | 3,072 | 12 | Same |
| 520M | 1,024 | 4,096 | 16 | Same |
| 1.2B | 1,536 | 6,144 | 24 | Same |

Training data mixes DCLM-baseline, StarCoder V2 Data, and ProofPile 2. Hyperparameters are tuned separately at every $(N, D)$ grid point (e.g., in Table 3, 130M/1× AdamW uses Peak LR 8e-3, WD 0.1, warmup 2000 steps, BSZ 128 sequences or ~0.5M tokens; 520M/1× uses WD 0.2 and BSZ 256).

### A.3 Delphi Ladder

[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) provides a dense ladder designed for end-to-end loss forecasting and extrapolation validation: built on a Qwen 3 decoder-only architecture (MLP ratio 4, `seq_len` 4096) and later extended to MoE ([535B-A23B](https://openathena.ai/blog/pretraining-speedup/)).

Every run uses the AdamH optimizer, WSD scheduling (10% warmup, linear decay to 0 over the final 20%), FP32 master weights with BF16 compute under FSDP, and a mixture of Nemotron-CC, StarCoderData, and ProofPile 2. IsoFLOP sweeps span $3\times 10^{18}$–$3\times 10^{20}$ FLOPs, fitting power laws across 7 IsoFLOP optima using preset hyperparameter rules without per-point grid search, and testing extrapolation across $10^{21}$–$10^{23}$ FLOPs holdouts (up to 25B parameters, or $3\times$–$333\times$ compute extrapolation).

### A.4 Choosing a Starting Point

- For hyperparameter search and optimizer / algorithm comparisons: start from Fantastic Optimizers' $(N \times \text{TPP})$ grid and invest compute in tuning each small grid point onto the Fully-Tuned Frontier.
- For compute allocation and end-to-end loss forecasting: start from Delphi's IsoFLOP layout with rule-based hyperparameters and multi-tier holdouts to verify extrapolation accuracy and stability.
- Match the target model's architecture: both public templates start from dense models (and Fantastic Optimizers fixes 32 MHA layers across sizes); in practice, align aspect ratio and GQA group structure on small models with your target architecture ([§4.2](#42-architectural-and-training-consistency)), and follow [§5.4](#54-moe-experimental-design) and [§6.3](#63-scaling-laws-for-lr-and-bsz) for MoE ladder design and LR transfer.

## References

Grouped by topic, with links back to the sections where each work is cited.

### Scaling Law Foundations, Functional Forms, and Fitting

1. Scaling Laws for Neural Language Models — Kaplan et al., OpenAI, 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)  
   Empirical power laws across 7 orders of magnitude and non-embedding parameter accounting ([§3.1](#31-parameter-counting), [§4.3](#43-aspect-ratio), [§6.5](#65-lr-schedule-and-cooldown), [Appendix A.1](#a1-public-scale-configurations))
2. Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., DeepMind, 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)  
   IsoFLOP sweeps, equal-rate scaling of $N$ and $D$, and the additive power-law form ([§3.1](#31-parameter-counting), [§5.3](#53-token-horizons), [§8.1](#81-functional-forms), [Appendix A.1](#a1-public-scale-configurations))
3. Language models scale reliably with over-training and on downstream tasks — Gadre et al., 2024. [arXiv:2403.08540](https://arxiv.org/abs/2403.08540)  
   Power-law extrapolation up to 32× Chinchilla overtraining and downstream error mapping ([§5.3](#53-token-horizons), [§9](#9-downstream-task-prediction), [Appendix A.1](#a1-public-scale-configurations))
4. Skaling: Chinchilla's Exponents Meet Kaplan's Coupling — Videau et al., FAIR at Meta, 2026. [arXiv:2608.07222](https://arxiv.org/abs/2608.07222)  
   Outer coupling exponent $k$ eliminating saddle-shaped corner residuals, plus L-shape sampling validation ([§8.1](#81-functional-forms))
5. Predictable Scale: Part II, Farseer: A Refined Scaling Law in Large Language Models — Li et al., StepFun & Fudan, NeurIPS 2025. [arXiv:2506.10972](https://arxiv.org/abs/2506.10972)  
   9-parameter coupled formulation and ablations on excluding embeddings during extrapolation ([§3.1](#31-parameter-counting), [§8.1](#81-functional-forms))
6. Scaling Law with Learning Rate Annealing — Tissue et al., 2024. [arXiv:2408.11029](https://arxiv.org/abs/2408.11029)  
   Full loss-trajectory modeling via cumulative LR area and annealing kernels ([§6.5](#65-lr-schedule-and-cooldown), [§8.2](#82-loss-trajectory-and-annealing-scaling-laws))
7. A Hitchhiker's Guide to Scaling Law Estimation — Choshen et al., MIT/IBM, ICML 2025. [arXiv:2410.11840](https://arxiv.org/abs/2410.11840)  
   Early-checkpoint truncation, scale-span requirements, and seed-variance analysis ([§2.3](#23-acceptance-criteria), [§5.2](#52-model-count-scale-span-and-extrapolation-factor))
8. Resolving Discrepancies in Compute-Optimal Scaling of Language Models — Porian et al., NeurIPS 2024. [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)  
   How output-head FLOPs, fixed warmup steps, and undertuned hyperparameters bias scaling exponents ([§3.1](#31-parameter-counting), [§4.1](#41-grouping-variables), [§8](#8-fitting-loss-scaling-laws))
9. Chinchilla Scaling: A Replication Attempt — Besiroglu et al., Epoch AI, 2024. [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)  
   Replication of Chinchilla's fit analyzing sensitivity to initialization and loss objectives ([§8.3](#83-numerical-fitting-details))
10. (Mis)Fitting: A Survey of Scaling Laws — Li et al., 2025. [arXiv:2502.18969](https://arxiv.org/abs/2502.18969)  
    Survey of how omitted fitting objectives and numerical details affect reproducibility ([§8.3](#83-numerical-fitting-details))
11. Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws — Sardana et al., 2024. [arXiv:2401.00448](https://arxiv.org/abs/2401.00448)  
    Inference-aware optimal allocation shifts and overtraining experiments up to 10,000 TPP ([§5.3](#53-token-horizons), [§10.4](#104-wall-clock-speed-and-inference-cost))
12. Gemstones: A Model Suite for Multi-Faceted Scaling Laws — McLeish et al., 2025. [arXiv:2502.06857](https://arxiv.org/abs/2502.06857)  
    Impact of width-to-depth aspect ratio on downstream benchmark performance ([§4.3](#43-aspect-ratio))
13. Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies — Tao et al., NeurIPS 2024. [arXiv:2407.13623](https://arxiv.org/abs/2407.13623)  
    Power-law growth of optimal vocabulary size with training compute ([§4.5](#45-vocabulary-size-and-precision))
14. Scaling Laws for Precision — Kumar et al., 2024. [arXiv:2411.04330](https://arxiv.org/abs/2411.04330)  
    Joint scaling laws for numerical training precision and post-training quantization ([§4.5](#45-vocabulary-size-and-precision))

### Hyperparameter Scaling Laws and Optimizers

{:start="15"}
15. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism — DeepSeek, 2024. [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)  
    Compute-indexed hyperparameter power laws and corpus-quality effects on optimal $N/D$ allocation ([§6.3](#63-scaling-laws-for-lr-and-bsz), [§6.5](#65-lr-schedule-and-cooldown), [§7.1](#71-data-quality-and-source-evaluation))
16. Predictable Scale: Part I, Step Law — Optimal Hyperparameter Scaling Law in Large Language Model Pre-training — Li et al., StepFun, 2025. [arXiv:2503.04715v3](https://arxiv.org/abs/2503.04715v3)  
    Bivariate $\eta_{opt}(N,D)$ and $B_{opt}(D)$ scaling rules fit across 3,700+ models ([§6.1](#61-search-goals-and-procedure), [§6.3](#63-scaling-laws-for-lr-and-bsz), [§6.5](#65-lr-schedule-and-cooldown), [Appendix A.1](#a1-public-scale-configurations))
17. Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training — Bergsma et al., Cerebras, 2025. [arXiv:2505.13738](https://arxiv.org/abs/2505.13738)  
    $B_{opt}\propto D^{0.4}$, hyperbolic $B_{crit}\propto D_{min}^{0.5}$ tradeoffs, and EMA timescale $\tau$ coupling ([§6.2](#62-parameterization-and-cross-optimizer-transfer), [§6.3](#63-scaling-laws-for-lr-and-bsz), [§6.4](#64-weight-decay))
18. Scaling Optimal LR Across Token Horizons — Bjorck et al., Microsoft, 2024. [arXiv:2409.19913](https://arxiv.org/abs/2409.19913)  
    Power-law decay of optimal peak LR with token horizon at fixed $N$ and fixed BSZ ([§6.3](#63-scaling-laws-for-lr-and-bsz))
19. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer ($\mu$-Transfer) — Yang et al., Microsoft, 2022. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)  
    Zero-shot LR transfer across model width under $\mu$P ([§6.2](#62-parameterization-and-cross-optimizer-transfer))
20. Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks — Yang et al., 2023. [arXiv:2310.02244](https://arxiv.org/abs/2310.02244)  
    Depth-$\mu$P for single-layer residual blocks and theoretical limits for multi-layer Transformer blocks ([§6.2](#62-parameterization-and-cross-optimizer-transfer))
21. Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit — Bordelon et al., 2023. [arXiv:2309.16620](https://arxiv.org/abs/2309.16620)  
    $1/\sqrt{\text{depth}}$ residual scaling for width-and-depth hyperparameter transfer on ResNets and ViTs ([§6.2](#62-parameterization-and-cross-optimizer-transfer))
22. Muon is Scalable for LLM Training — Liu et al., Moonshot AI, 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)  
    Matching update RMS norms and adding weight decay so Muon reuses AdamW hyperparameters directly ([§6.2](#62-parameterization-and-cross-optimizer-transfer))
23. Fantastic Pretraining Optimizers and Where to Find Them — Wen et al., Stanford, 2025. [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)  
    Fair optimizer tuning benchmarks, diminishing token-efficiency gains with scale, and cooldown rank flips ([§5.3](#53-token-horizons), [§6.6](#66-comparing-recipes-fairly), [Appendix A.1](#a1-public-scale-configurations), [Appendix A.2](#a2-fantastic-optimizers-ladder))
24. Weight Decay Improves Language Model Plasticity — Han et al., 2026. [arXiv:2602.11137](https://arxiv.org/abs/2602.11137)  
    Decreasing pretraining-optimal WD with TPP and higher WD preserving post-training plasticity ([§6.4](#64-weight-decay))
25. Small-Scale Experiments: Are We There Yet? — Lourie et al., NYU & Meta, 2026. [arXiv:2608.11859](https://arxiv.org/abs/2608.11859)  
    Sharp hyperparameter sensitivity at small scale, the Fully-Tuned Frontier, and fit/validation/test scale splits ([§5.2](#52-model-count-scale-span-and-extrapolation-factor), [§6.1](#61-search-goals-and-procedure))
26. How to Allocate Your Tokens? Scaling Laws with Training Steps and Batch Size — Schaipp, Inria, 2026. [arXiv:2607.01487](https://arxiv.org/abs/2607.01487)  
    Roughly $4\times$-wide near-optimal batch-size window within 5% compute overhead ([§6.3](#63-scaling-laws-for-lr-and-bsz))
27. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation — Cawley & Talbot, JMLR 2010. [JMLR 11](https://www.jmlr.org/papers/v11/cawley10a.html)  
    Selection bias from picking the best run across finite noisy hyperparameter trials ([§6.1](#61-search-goals-and-procedure))

### MoE

{:start="28"}
28. Scaling Laws for Fine-Grained Mixture of Experts — Krajewski et al., 2024. [arXiv:2402.07871](https://arxiv.org/abs/2402.07871)  
    MoE scaling laws treating expert granularity as an independent variable ([§4.4](#44-moe-structural-scaling), [§5.4](#54-moe-experimental-design))
29. Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models — Abnar et al., Apple, 2025. [arXiv:2501.12370](https://arxiv.org/abs/2501.12370)  
    Optimal sparsity laws at fixed FLOPs vs. fixed total parameters and downstream sensitivity ([§4.4](#44-moe-structural-scaling))
30. Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient — Ludziejewski et al., 2025. [arXiv:2502.05172](https://arxiv.org/abs/2502.05172)  
    Joint scaling of expert count, active parameters, and tokens under memory budgets ([§4.4](#44-moe-structural-scaling), [§5.4](#54-moe-experimental-design))

### Data Construction, Mixtures, and Repetition

{:start="31"}
31. Scaling Data-Constrained Language Models — Muennighoff et al., 2023. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)  
    Exponentially decaying effective-token formulation for multi-epoch repetition ([§7.3](#73-data-constrained-training-and-repetition))
32. To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis — Xue et al., 2023. [arXiv:2305.13230](https://arxiv.org/abs/2305.13230)  
    Multi-epoch overfitting across model size $N$ and mitigation via dropout ([§7.3](#73-data-constrained-training-and-repetition))
33. Prescriptive Scaling Laws for Data Constrained Training — Lovelace et al., 2026. [arXiv:2605.01640](https://arxiv.org/abs/2605.01640)  
    Overfitting and compute allocation governed jointly by $N$, unique tokens, and epoch count ([§7.3](#73-data-constrained-training-and-repetition))
34. Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression — Yan et al., 2025. [arXiv:2511.13421](https://arxiv.org/abs/2511.13421)  
    Logarithmic scaling of optimal repetition epochs with dataset size in theory and LLM experiments ([§7.3](#73-data-constrained-training-and-repetition))
35. UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining — Chung et al., ICLR 2023. [arXiv:2304.09151](https://arxiv.org/abs/2304.09151)  
    Epoch-capped domain sampling baseline ([§7.2](#72-data-mixture-ladders))
36. Olmix: A Framework for Data Mixing Throughout LM Development — Chen et al., Allen Institute, ICLR 2026. [arXiv:2602.12237](https://arxiv.org/abs/2602.12237)  
    Proxy sizing, per-task log-linear mixture regression, repetition caps, and incremental mixture reuse ([§7.2](#72-data-mixture-ladders))
37. Scaling Laws for Mixture Pretraining Under Data Constraints — Sedova et al., Apple, 2026. [arXiv:2605.12715](https://arxiv.org/abs/2605.12715)  
    Higher repetition tolerance and cross-scale mixture extrapolation when mixing scarce and general corpora ([§7.3](#73-data-constrained-training-and-repetition))
38. Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training (DeMix) — Li et al., 2026. [arXiv:2602.00747](https://arxiv.org/abs/2602.00747)  
    Evaluating arbitrary data mixtures by merging per-domain component models instead of retraining proxies ([§7.2](#72-data-mixture-ladders))
39. Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset — Su et al., NVIDIA, 2024. [arXiv:2412.02595](https://arxiv.org/abs/2412.02595)  
    Large-scale refined and synthetically rephrased web corpus; major cross-dataset overlap source ([§7.3](#73-data-constrained-training-and-repetition))

### LR Schedules and Cooldown

{:start="40"}
40. Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al., 2024. [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)  
    Loss-landscape geometry explaining the stable and decay phases of WSD ([§6.5](#65-lr-schedule-and-cooldown))
41. Scaling and Transferability of Annealing Strategies in Large Language Model Training — Wang et al., 2025. [arXiv:2512.13705](https://arxiv.org/abs/2512.13705)  
    Cross-scale transfer of annealing shapes and decay fractions ([§6.5](#65-lr-schedule-and-cooldown))
42. Model Merging in Pre-training of Large Language Models — ByteDance Seed, 2025. [arXiv:2505.12082](https://arxiv.org/abs/2505.12082)  
    Approximating cooldown gains via checkpoint weight averaging (PMA) along WSD stable trajectories ([§6.5](#65-lr-schedule-and-cooldown))
43. Model Soups: Averaging Weights of Multiple Fine-tuned Models — Wortsman et al., 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)  
    Weight averaging across independently fine-tuned branches from a shared initialization ([§6.5](#65-lr-schedule-and-cooldown))
44. Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging (LAWA) — Kaddour, 2022. [arXiv:2209.14981](https://arxiv.org/abs/2209.14981)  
    Sliding-window weight averaging over recent checkpoints along a single run ([§6.5](#65-lr-schedule-and-cooldown))
45. Early Weight Averaging meets High Learning Rates for LLM Pre-training — Sanyal et al., COLM 2024. [arXiv:2306.03241](https://arxiv.org/abs/2306.03241)  
    Early sliding weight averaging during high-LR LLM pretraining ([§6.5](#65-lr-schedule-and-cooldown))

### Downstream Prediction and Evaluation

{:start="46"}
46. Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective (COD) — Xu et al., ICLR 2026, v4 (2026-03-09). [arXiv:2502.17262v4](https://arxiv.org/pdf/2502.17262v4)  
    Difficulty-clustered benchmark subsets for extrapolating downstream accuracy ([§9](#9-downstream-task-prediction))
47. Hinted Scaling Laws — van Adrichem et al., Stanford, 2026. [arXiv:2611.02270](https://arxiv.org/abs/2611.02270)  
    Graded eval-time hints lifting small models off 0% and joint $(C, H)$ fits forecasting unhinted accuracy at scale ([§9](#9-downstream-task-prediction))
48. GPT-4 Technical Report — OpenAI, 2023. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)  
    Bucketing HumanEval items by small-model pass rates to forecast full-scale performance ([§9](#9-downstream-task-prediction))
49. Establishing Task Scaling Laws via Compute-Efficient Model Ladders (OLMo Task Ladder) — Bhagia et al., Allen Institute, 2024. [arXiv:2412.04403](https://arxiv.org/abs/2412.04403)  
    Two-stage $(N,D) \to \text{Task NLL} \to \text{Accuracy}$ downstream forecasting ([§9](#9-downstream-task-prediction))
50. Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive? — Schaeffer et al., 2024. [arXiv:2406.04391](https://arxiv.org/abs/2406.04391)  
    How multiple-choice distractor mass and argmax thresholding create downstream accuracy prediction noise ([§9](#9-downstream-task-prediction))
51. RULER: What's the Real Context Size of Your Long-Context Language Models? — Hsieh et al., NVIDIA, 2024. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)  
    Effective context-length benchmark covering multi-hop tracing, aggregation, and QA ([§11](#11-multi-stage-training-ladders))

### Model Technical Reports and Ladder Case Studies

{:start="52"}
52. Delphi: An Open Scaling Suite from 3e18 to 1e23 FLOPs — Marin Team, 2026. [openathena.ai/blog/delphi](https://openathena.ai/blog/delphi/)  
    IsoFLOP sweeps, preset hyperparameter formulas, and multi-tier holdout validation ([§1.1](#11-what-is-a-scaling-ladder), [§2.1](#21-decision-objectives), [§5.2](#52-model-count-scale-span-and-extrapolation-factor), [§9](#9-downstream-task-prediction), [§10.1](#101-holdouts-and-joint-validation), [Appendix A.1](#a1-public-scale-configurations), [Appendix A.3](#a3-delphi-ladder))
53. The Llama 3 Herd of Models — Meta, 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)  
    405B IsoFLOP sizing, small-model overtraining, two-stage downstream prediction, BSZ warmup, and restart recovery ([§5.2](#52-model-count-scale-span-and-extrapolation-factor), [§5.3](#53-token-horizons), [§6.3](#63-scaling-laws-for-lr-and-bsz), [§9](#9-downstream-task-prediction), [§10.3](#103-aligning-training-frameworks-and-hardware), [Appendix A.1](#a1-public-scale-configurations))
54. DeepSeek-V3 Technical Report — DeepSeek, 2024. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)  
    Dropless MoE routing and FP8 mixed-precision training ([§4.4](#44-moe-structural-scaling), [§10.3](#103-aligning-training-frameworks-and-hardware))
55. Nemotron-4 15B Technical Report — NVIDIA, 2024. [arXiv:2402.16819](https://arxiv.org/abs/2402.16819)  
    Batch-size warmup and late-stage high-quality data cooldown ([§6.3](#63-scaling-laws-for-lr-and-bsz), [§6.5](#65-lr-schedule-and-cooldown), [Appendix A.1](#a1-public-scale-configurations))
56. Nemotron-4 340B Technical Report — NVIDIA, 2024. [arXiv:2406.11704](https://arxiv.org/abs/2406.11704)  
    340B training setup and 9T-token horizon reference ([Appendix A.1](#a1-public-scale-configurations))
57. OLMo: Accelerating the Science of Language Models — Groeneveld et al., Allen Institute, 2024. [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)  
    Open pretraining models, corpora, and intermediate checkpoint suite ([Appendix A.1](#a1-public-scale-configurations))
58. 2 OLMo 2 Furious — OLMo Team, Allen Institute, 2025. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)  
    Multi-stage training, micro-annealing source screening, and model souping ([§6.5](#65-lr-schedule-and-cooldown), [§7.1](#71-data-quality-and-source-evaluation), [Appendix A.1](#a1-public-scale-configurations))
59. Olmo 3 — Team Olmo, Allen Institute, 2025. [arXiv:2512.13961](https://arxiv.org/abs/2512.13961)  
    BPB proxy evaluation, Olmix iterations, high-quality bucket upsampling, multi-stage training, and lightweight post-training checks ([§2.2](#22-evaluation-metrics-and-setup), [§6.4](#64-weight-decay), [§6.5](#65-lr-schedule-and-cooldown), [§7.1](#71-data-quality-and-source-evaluation), [§7.2](#72-data-mixture-ladders), [§7.3](#73-data-constrained-training-and-repetition), [§10.1](#101-holdouts-and-joint-validation), [§11](#11-multi-stage-training-ladders), [Appendix A.1](#a1-public-scale-configurations))
60. Qwen3 Technical Report — Qwen Team, 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)  
    Multi-stage hyperparameter prediction and instance-attribute data mixtures ([§6.4](#64-weight-decay), [§7.2](#72-data-mixture-ladders), [§11](#11-multi-stage-training-ladders))
61. On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability — Qwen Team, 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)  
    Wide hyperparameter plateaus at scale, Muon hyperparameter shifts, post-training degradation checks, and high-LR stress tests ([§2.2](#22-evaluation-metrics-and-setup), [§6.1](#61-search-goals-and-procedure), [§6.2](#62-parameterization-and-cross-optimizer-transfer), [§10.2](#102-stability-stress-tests))
62. Kimi K2: Open Agentic Intelligence — Moonshot AI, 2025. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)  
    MoE sparsity scaling law at fixed active parameters and knowledge-corpus rephrasing ablations ([§5.4](#54-moe-experimental-design), [§6.4](#64-weight-decay), [§7.3](#73-data-constrained-training-and-repetition))
63. Kimi K2.5: Visual Agentic Intelligence — Moonshot AI, 2026. [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)  
    Per-source maximum epoch caps during continued joint pretraining ([§7.3](#73-data-constrained-training-and-repetition))
64. Kimi K3: Open Frontier Intelligence — Moonshot AI, 2026. [arXiv:2607.24653](https://arxiv.org/abs/2607.24653)  
    Quantile Balancing (QB), refitting scaling laws after architecture/data updates, tuned Cosine vs. WSD comparison, and knowledge rephrasing ([§4.4](#44-moe-structural-scaling), [§6.5](#65-lr-schedule-and-cooldown), [§7.1](#71-data-quality-and-source-evaluation), [§7.3](#73-data-constrained-training-and-repetition))
65. Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2025. [arXiv:2512.20848](https://arxiv.org/abs/2512.20848)  
    30B-A3B Hybrid MoE architecture and 25T-token pretraining setup ([§7.1](#71-data-quality-and-source-evaluation), [Appendix A.1](#a1-public-scale-configurations))
66. Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2026. [arXiv:2604.12374](https://arxiv.org/abs/2604.12374)  
    120B-A12B model with a two-stage 25T mixture (80% diversity-focused followed by 20% high-quality) ([§6.5](#65-lr-schedule-and-cooldown), [§7.2](#72-data-mixture-ladders), [Appendix A.1](#a1-public-scale-configurations))
67. Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2026. [arXiv:2606.15007](https://arxiv.org/abs/2606.15007)  
    550B-A55B model, FP32 output-gradient fix and early cooldown for late divergences, plus domain synthetic-data ablations ([§7.1](#71-data-quality-and-source-evaluation), [§10.2](#102-stability-stress-tests), [§10.4](#104-wall-clock-speed-and-inference-cost), [Appendix A.1](#a1-public-scale-configurations))
68. Marin: Improving LLM Pretraining Efficiency (MoE Follow-up) — Marin Team, 2026. [openathena.ai/blog/pretraining-speedup](https://openathena.ai/blog/pretraining-speedup/)  
    Distinguishing theoretical vs. realized wall-clock efficiency and validating combined recipe changes ([§10.1](#101-holdouts-and-joint-validation), [§10.4](#104-wall-clock-speed-and-inference-cost), [Appendix A.3](#a3-delphi-ladder))
69. Marin Data Pipeline Overview — Marin Team, 2026. [openathena.ai/blog/marin-data-pipeline-overview](https://openathena.ai/blog/marin-data-pipeline-overview/)  
    Cross-source global deduplication, proportional proxy pool downsampling, and epoch-constrained mixture regression ([§7.2](#72-data-mixture-ladders), [§7.3](#73-data-constrained-training-and-repetition))
70. Phi-4 Technical Report — Microsoft, 2024. [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)  
    Multi-sample evaluation for high-variance downstream benchmarks ([§2.2](#22-evaluation-metrics-and-setup))
71. GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models — Zhipu AI, 2025. [arXiv:2508.06471](https://arxiv.org/abs/2508.06471)  
    Effect of model depth and attention-head configuration on reasoning benchmarks ([§4.3](#43-aspect-ratio))

### Further Reading

Additional references on large-batch training and attention structure:

{:start="72"}
72. Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training — Merrill et al., 2025. [arXiv:2505.23971](https://arxiv.org/abs/2505.23971)
73. An Empirical Model of Large-Batch Training — McCandlish et al., 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
74. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al., Google, 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
75. Cost-Optimal Grouped-Query Attention for Long-Context Modeling — Chen et al., 2025. [arXiv:2503.09579](https://arxiv.org/abs/2503.09579) ([§11](#11-multi-stage-training-ladders))
