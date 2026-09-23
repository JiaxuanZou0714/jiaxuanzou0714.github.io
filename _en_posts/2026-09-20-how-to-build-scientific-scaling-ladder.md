---
layout: post
title: "How to Build a Scientific Scaling Ladder"
date: 2026-09-20 10:00:00
description: "A guide to designing and building a Scaling Ladder, in execution order: decision objectives and acceptance, measurement protocol, scaling rules for dense and MoE, experiment matrix and budget, hyperparameter search, data mixture and data-constrained training, Loss Scaling Law fitting, downstream prediction, extrapolation validation, and the delivery process, synthesizing public literature and engineering practice from Chinchilla, DeepSeek, StepFun, Cerebras, Llama 3, Delphi, and others."
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
source_sha: 9031e4252c5b4640
---

This article is compiled from **<u>public literature and technical reports and contains no confidential content</u>**. It covers how to build a Scaling Ladder: fit a Scaling Law through small-scale experiments, extrapolate the training configuration and performance of the target model, and use this to decide whether to launch the target training run.


---

## 1. Scope and Terminology

### 1.1 Definition of Scaling Ladder

A Scaling Ladder is a set of training runs spanning different parameter counts $N$, training volumes $D$, and any other variables under study. It is used to fit scaling laws and to extrapolate the training configuration, loss, and downstream performance at the target scale. The results give a quantitative basis for choosing model size, training volume, hyperparameters, and candidate designs, and reduce the configuration risk of large-scale training.

{% include figure.liquid
  path='assets/img/pretrain-scaling/marin-delphi-first-ladder.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 1. The first Delphi experiment (Cautious AdamC recipe). The large-scale training run in the right panel deviates from the prediction and diverges. <a href="https://openathena.ai/blog/delphi/#delphi-attempt-1-how-can-scaling-laws-go-wrong">Source: Marin, 2026</a>.'
  alt='Delphi first scaling experiment: the 1e22 FLOPs run has a loss 2.5% higher than predicted, and the 1e23 FLOPs run diverges.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

The recipe in Figure 1 shows no anomaly within the fitting range; at the target scale its loss is higher than predicted and the run diverges. A Ladder must verify extrapolation accuracy before delivery ([§10](#10-extrapolation-validation-and-launch-decisions)).

### 1.2 Typical Scenarios

This article covers three scenarios. Chapters 2–10 describe the common workflow; material specific to MoE and to data-constrained training is given separately within each chapter.

| Scenario | Main Constraints | Dedicated Sections |
|---|---|---|
| Dense | Structural scaling rules and width-depth configuration; used for dense delivery models and as a control for MoE | [§4.3](#43-width-depth-configuration) |
| MoE | Separation of total parameter count and active parameter count; sparsity, expert granularity, routing, and load balancing; actual throughput of expert parallelism | [§3.1](#31-parameter-count-definition), [§4.4](#44-moe-structural-scaling-rules), [§5.6](#56-moe-experimental-axes), [§6.6](#66-moe-training-hyperparameters), [§8.5](#85-moe-fitting), [§10.3](#103-stability-stress-testing) |
| Data-constrained | Available unique tokens fewer than the target training volume; repetition count, data quality, and data mixture jointly determine the achievable loss | [§5.3](#53-training-volume-tiers), [§7.3](#73-data-constrained-training-and-repetition) |

MoE and data-constrained conditions can hold simultaneously. The target TPP can be lower or higher than the compute-optimal ratio; the training volume axis is designed according to the target TPP ([§5.3](#53-training-volume-tiers)).

The following are outside the scope of this article: scaling laws for post-training (SFT/RL) itself, native multimodality, and dedicated protocols for distillation and synthetic data. Post-training appears only as an acceptance condition for pretraining candidates ([§2.2](#22-evaluation-protocol)).

### 1.3 Outputs of the Ladder

| Output | Predicted Target | Section |
|---|---|---|
| Hyperparameter scaling law (LR, BSZ, WD) | Training hyperparameters at the target scale | [§6](#6-hyperparameter-search-and-training-configuration) |
| Loss scaling law | Final training/eval loss | [§8.1](#81-functional-form) |
| Loss curve scaling law | Full training curve and annealing effect | [§8.2](#82-loss-curves-and-annealing-scaling-law) |
| Annealing ratio | Proportion of LR decay in the total training volume | [§6.5](#65-lr-schedule); this article gives only an initial reference value, not a fitting method |
| Data mixture and data repetition scaling law | Optimal mixture; equivalent data volume for multi-epoch training | [§7](#7-data-ladder) |
| MoE sparsity scaling law | Expert configuration at fixed active scale | [§5.6](#56-moe-experimental-axes), [§8.5](#85-moe-fitting) |
| Downstream task scaling law | Benchmark metrics | [§9](#9-downstream-task-prediction) |

### 1.4 Terminology and Notation

| Term | Definition |
|---|---|
| $N_{\text{body}}$ | Transformer backbone parameter count, excluding input embedding and output head ([§3.1](#31-parameter-count-definition)) |
| $N_{total}$, $N_{active}$ | MoE total parameter count (including all experts); parameter count involved in computation per token |
| $D$ | Cumulative training tokens, counted as tokens participating in loss computation ([§3.3](#33-loss-and-token-conventions)) |
| $U$ | Unique tokens |
| $C$ | Training FLOPs, counted according to the actual architecture ([§3.2](#32-compute)) |
| TPP | Tokens per parameter, i.e., $D/N$; when citing external results, note their $N$ convention |
| LR ($\eta$), BSZ ($B$), WD ($\lambda$) | Learning rate, batch size, weight decay. BSZ is counted in tokens; when citing results counted in sequences, note this |
| Holdout | Experimental points excluded from fitting and candidate selection, used only for acceptance |
| Fully-Tuned Frontier | The loss reached by each experimental point when hyperparameters are fully tuned; for the operational definition under limited budget see [§6.1](#61-hyperparameter-search-objective-and-near-optimal-region) |
| Equivalent compute multiplier | The ratio of compute required to reach the same loss, used to convert loss differences into compute differences ([§2.3](#23-acceptance-thresholds-and-decision-rules)) |

## 2. Decision Objectives and Acceptance Criteria

### 2.1 Decision Objectives

A Scaling Ladder can support four types of decisions:

1. Prediction for a fixed recipe: predict the performance of a fixed recipe at the target scale. A valid scaling law does not require exhaustive tuning of every experimental point.
2. Optimal resource allocation: choose the allocation of $N$ and $D$ under budget constraints.
3. Candidate comparison: compare different architectures, optimizers, or data recipes.
4. Hyperparameter prediction: predict hyperparameters across scales or training volumes.

Before building the Ladder, fix the objective type and specify the target model, training stage, budget constraints, deployment conditions for inference, and the size of prediction error that would change the final decision. [Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/) examines performance competitiveness and predictability separately, and the two must be verified separately. Objectives 2–4 require each fitting point to reach the Fully-Tuned Frontier ([§6.1](#61-hyperparameter-search-objective-and-near-optimal-region)); objective 1 runs under the recipe scaling rules.

### 2.2 Evaluation Protocol

- Proxy metrics and final metrics: Proxy metrics (training loss, answer BPB) and final acceptance metrics (downstream benchmark accuracy) should be specified separately. Improvements in loss on small models do not guarantee improvements in downstream metrics at the target scale ([Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) §1).
- Signal discriminability: Each metric must have a discriminable signal at the fitting scale. Small models may be near random level on tasks such as math and code, and some metrics saturate at large scales; neither case can support decisions. For tasks where accuracy cannot discriminate, use continuous proxy metrics such as answer BPB ([OLMo 3, 2025](https://arxiv.org/abs/2512.13961)).
- Cross-scale rank correlation: When proxy metrics are used for candidate selection, they must establish rank correlation with the capability metric at the target scale ([OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3, Appendix A.4). When transfer evidence is insufficient, the applicable scope of the proxy conclusion and what remains to be verified should be stated.
- Representativeness of target capabilities: State which delivery capabilities the evaluation tasks cover, the aggregation method, and the tolerance conditions for degradation in key individual items. High-noise tasks can be sampled more or reported separately ([Phi-4, 2024](https://arxiv.org/abs/2412.08905) §5; [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.3.3–3.3.4).
- Evaluation version control: Record prompt templates, generation settings, scoring methods, and versions, and check for overlap between training and evaluation data.

{% include figure.liquid
  path='assets/img/pretrain-scaling/olmo3-evaluation.png'
  id='fig-olmo3-evaluation'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 2. Math evaluation of OLMo 3: the left and middle panels show bits-per-byte (BPB) on the Easy suite and pass@1 on the Main suite as functions of compute, respectively; small-scale models already show distinguishable differences in BPB, while pass@1 is near zero over the same period. The right panel shows the relationship between the two types of metrics across the models examined; ranking across scales still needs to be validated for the target task. <a href="https://arxiv.org/pdf/2512.13961v1#page=12">Source: Olmo 3, 2025, v1, Fig. 6 (PDF page 12)</a>.'
  alt='The three subplots of Figure 6 in the original OLMo 3 paper: BPB on the math Easy suite as a function of compute, pass@1 on the Main suite as a function of compute, and the relationship between BPB and pass@1.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

If a post-training model (SFT/RL) is delivered, representative candidates should undergo post-training validation under comparable conditions. [Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) reports that two changes each had little impact during pretraining: NoPE exhibited non-terminating generation after post-training, and the sparse-read residual branch degraded in quality after post-training. [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.5.1 performs rapid instruction tuning acceptance after fully annealing candidate mixtures. Post-training acceptance should check delivery metrics such as termination behavior and output length in addition to capability scores.

### 2.3 Acceptance Thresholds and Decision Rules

Acceptance covers four categories, each requiring its own experiments and thresholds (our recommendation):

1. Prediction error of the final metric for a fixed recipe under target conditions;
2. Difference and ranking of candidates at the target scale;
3. Expected loss of a resource allocation relative to feasible alternatives;
4. Stability and key capability constraints.

Acceptance thresholds are determined by the smallest difference that must be resolved between candidates and by the random seed variance, and must be set before observing holdout results. [Choshen et al., 2025](https://arxiv.org/abs/2410.11840) reports that the minimum relative difference driving modeling changes in the literature is about 4%, and random restart fluctuations can reach about 3.5%; Delphi's 0.2%–0.5% is an experimental result for a specific setting and cannot serve as a general acceptance line.

The same relative loss error threshold should not be used across scales. Let the fit be $L(C)=E+A\,C^{-\gamma}$; the equivalent compute multiplier corresponding to a loss difference $\Delta L$ is approximately

$$\ln\frac{C_2}{C_1}\approx\frac{\Delta L}{\gamma\,(L-E)}$$

The closer $L$ is to $E$, the larger the compute multiplier corresponding to the same $\Delta L$. We recommend expressing thresholds as equivalent compute multipliers and also reporting the absolute loss difference.

When comparing two candidates, estimate the uncertainty of their difference under target conditions. Error sources shared by the two candidates partially cancel, so two independent error bars cannot simply be added. For example, if the predicted losses of two candidates differ by 0.2% and the absolute relative error of the existing holdout is 1%, these two numbers are insufficient to judge whether the ranking is reliable.

The uncertainty of the fit should be propagated to the candidate difference, optimal parameter count, and training volume at the target scale; when necessary, report a set of near-optimal feasible configurations and their differences in quality, cost, and stability. When some coefficients are imprecisely estimated, the target decision may still be stable; when the average loss error is small, the optimal resource allocation may also be unstable. [Gemstones, 2025](https://arxiv.org/abs/2502.06857) §4.2 shows that the choice of experimental points can change resource allocation recommendations.

Decisions fall into three categories: select, run additional experiments, or defer because the evidence is insufficient. "Insufficient evidence" is a formal outcome. The decision table should include: candidate configurations, target conditions, expected gains, difference intervals, key constraints, conditions to be verified, and pre-agreed handling rules.

## 3. Measurement Specification

### 3.1 Parameter Count Definition

The parameter count definitions required for fitting $L(N,D)$ and for computing $C$ differ and must be recorded separately.

For fitting $L(N,D)$, this article uses the Transformer backbone parameter count $N_{\text{body}}$, excluding the input embedding and output head. Embedding/head is $O(Vd)$, while the backbone is $O(d^2 n)$; the ratio between the two varies with model size. [Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972v3) Appendix G's ablation shows that the definition including embedding has higher error when extrapolated to 25.1B; [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) also excludes embedding.

When computing $C$, the output head must be included (one $d\times V$ matrix multiplication per token). The input embedding is a table lookup by token ID, and its parameters and memory access cost are recorded separately. Weight tying reduces storage, but the output-head computation must still be counted.

External studies use different definitions, which must be noted when citing: Porian's $N$ includes the output head and excludes the input embedding ([Porian et al., 2024](https://arxiv.org/abs/2406.19146) Table 2); Chinchilla's 20 TPP is based on the total parameter count including embedding ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) Appendix F).

For MoE, each experimental point must record simultaneously:

- $N_{total}$ and $N_{active}$, both under the backbone definition, with embedding and head recorded separately;
- Total number of routed experts $E_{total}$, number of routed experts activated per token $E_{active}$, sparsity $S=E_{total}/E_{active}$;
- Number and size of shared experts, size of a single expert (expert granularity);
- Router and load balancing configuration ([§4.4](#44-moe-structural-scaling-rules)).

### 3.2 Compute

The commonly used $C\approx6ND$ does not count attention matrix operations. Taking standard MHA and a $4d$ FFN as an example, counting by the full $L\times L$ attention matrix:

$$C \approx \left(6 + \frac{L}{d}\right) N_{\text{body}}D + 6VdD$$

$L$ is the sequence length, $d$ is the hidden dimension, and $V$ is the vocabulary size. The $L/d$ term comes from attention, and the last term is the output head. Under the above architecture, $N_{\text{body}}\approx12d^2n$ ($n$ is the number of layers). If the causal attention kernel skips the masked part, the attention term is roughly halved; SwiGLU, GQA, MoE, etc. also change the coefficients. Compute should be counted according to the actual architecture and the actual kernel.

For MoE, the matmul compute is counted by $N_{active}$, with router compute counted separately. The all-to-all communication of expert parallelism is not counted in FLOPs, but it is counted in actual training time ([§10.5](#105-realized-efficiency-and-deployment-constraints)).

### 3.3 Loss and Token Conventions

The following conventions must be fixed within a Ladder and written as checkable definitions (our recommendation):

| Item | What must be specified |
|---|---|
| Primary loss | Main-task cross-entropy; MoE auxiliary loss, z-loss, MTP, and other additional objectives are recorded separately |
| Evaluation distribution | All candidates share a fixed evaluation corpus and domain weights, with metrics for important domains reported separately |
| Aggregation | Aggregate by effective prediction tokens, documents, or tasks; specify the distributed reduction method and the denominator |
| Sequence handling | BOS/EOS, document boundaries, packing, attention mask, position ID, truncation |
| Data volume | Processed tokens, tokens participating in the loss, unique tokens, repeated exposure |
| Randomness | Seeds for parameter initialization, data sampling, and data order are recorded separately; conditions for paired comparisons |
| Tokenizer comparison | Report byte-normalized metrics (e.g., BPB) and actual compute cost on common raw text |

When candidates change the training data mixture, the training loss each computes on its own training distribution also reflects the change in distribution difficulty, so it cannot by itself support a quality ranking; a fixed evaluation distribution must be used.

Evaluation frequency should be set as a fixed fraction of the total number of steps $T$. A fixed step interval gives different numbers of effective eval points for different training lengths, and longer trajectories receive greater weight in fitting; when fitting, the sampling and weighting of eval points must be made explicit ([§8.3](#83-fitting-protocol)).

Deliverables: definitions of the evaluation function and the token counting function, plus a checkable example from a single batch to the global aggregate.

## 4. Baseline Recipe and Scaling Rules

### 4.1 Variable Classification

Variables in a Ladder fall into three categories:

| Variable category | Meaning | Example |
|---|---|---|
| Fixed quantity | All Ladder points take the same value | Architecture, data version, optimizer type, seq_len |
| Scaling rule | Varies with $N$ or $D$ according to a predetermined rule | LR (power law or $\mu$P), BSZ ($D^{0.4}$ as a prior to be verified), warmup |
| Experimental independent variable | Quantity to be fitted or searched | $N$, $D$, the final optimal LR and BSZ |

Warmup must have an explicit scaling rule, and the fraction of total training volume devoted to warmup must be recorded for each experimental point. A fixed number of warmup steps makes its share too large in small-budget experiments, affecting estimates of the compute-optimal exponent ([Porian et al., 2024](https://arxiv.org/abs/2406.19146)). Delphi uses 10% of total training volume as warmup ([Appendix A.3](#a3-delphi-ladder)).

After a change to the architecture, optimizer, data, precision, or tokenizer, the transferability of the original conclusions should first be verified at small scale, and only then should local calibration or a full refit be decided ([§12.3](#123-ladder-maintenance)).

### 4.2 Architecture and Training Configuration Consistency

Within the same model family, apart from the architecture variables explicitly under study, the following properties or scaling rules must be consistent across experimental points:

| Property | Requirement |
|---|---|
| Architecture type | Consistent, e.g., all decoder-only Transformer |
| Normalization | Consistent, e.g., all RMSNorm |
| Positional encoding | Consistent, e.g., all RoPE |
| Activation function | Consistent, e.g., all SwiGLU |
| Embedding sharing | tied / untied kept consistent within the Ladder |
| Attention type | Consistent, e.g., all GQA |
| Width-depth configuration | Scaled according to a predetermined rule ([§4.3](#43-width-depth-configuration)) |
| MoE routing | Routing method, load balancing method, and shared expert ratio consistent or scaled according to a predetermined rule ([§4.4](#44-moe-structural-scaling-rules)) |

Within the same recipe, apart from the variables explicitly under study, the following training configurations are kept consistent:

| Configuration item | Example |
|---|---|
| Sequence length | 4,096 |
| Training data version | Same version |
| Optimizer | AdamW |
| Evaluation set | Same eval set ([§3.3](#33-loss-and-token-conventions)) |
| Evaluation frequency | Set proportionally to training progress |
| Tokenizer, initialization, parameterization, loss statistics conventions | Same version |

The model family and structural scaling rule should have proxy validity for the target model; GQA grouping, head_dim, width-depth configuration, etc. should align with the design direction of the target model.

### 4.3 Width-Depth Configuration

Given a parameter count, the width-depth configuration affects actual compute ([§3.2](#32-compute)). The Ladder should record the structural scaling rule and use the actual $C$. [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) report that given a parameter count, the width-depth ratio has little effect on loss over a fairly wide range; subsequent experiments show that different configurations can produce differences on benchmarks ([Gemstones, 2025](https://arxiv.org/abs/2502.06857) §4.4) and reasoning ability ([GLM-4.5, 2025](https://arxiv.org/abs/2508.06471)). A consistent width-depth scaling rule should be used within the Ladder; when structural choices are involved, add representative controls.

### 4.4 MoE Structural Scaling Rules

In addition to [§4.2](#42-architecture-and-training-configuration-consistency), the MoE Ladder also needs to specify scaling rules for the following configurations:

- Expert granularity: [Krajewski et al., 2024](https://arxiv.org/abs/2402.07871) treat expert granularity as an independent scaling variable. Within the range tested, the common setting where expert size equals dense FFN size is non-optimal under almost all compute budgets, and the advantage of MoE over dense grows with scale.
- Shared experts: their number and size are set at a fixed ratio within the Ladder.
- Load balancing: the auxiliary loss coefficient, or the bias update rate in auxiliary-loss-free methods, is set by a fixed rule within the Ladder. [DeepSeek-V3, 2024](https://arxiv.org/abs/2412.19437) §2.1.2 adjusts the routing bias according to expert load, with a bias update rate of 0.001 for the first 14.3T tokens and 0 for the last 500B tokens. Load balancing strength affects both loss and expert parallelism throughput; changing it is treated as a recipe change.
- Capacity factor and token dropping: the settings for training and inference should be consistent; the proportion of dropped tokens is recorded as a diagnostic quantity. DeepSeek-V3 drops no tokens in either training or inference (§2.1.2).
- Sparsity: [Abnar et al., 2025](https://arxiv.org/abs/2501.12370) found, ignoring memory and communication overhead, that at fixed training compute, increasing sparsity and correspondingly increasing the total parameter count can reduce pretraining loss; at fixed total parameter count, loss varies parabolically with sparsity, and the optimal sparsity increases with model size and training compute. On most downstream tasks, models with similar pretraining loss perform similarly downstream, independent of sparsity; on reading comprehension tasks (e.g., CoQA, SQuAD), denser models perform better. [Ludziejewski et al., 2025](https://arxiv.org/abs/2502.05172) jointly model expert count with active parameter count and training volume, and incorporate memory constraints, with experiments up to 2.7B active parameters and 5B total parameters.

### 4.5 Vocabulary and Numerical Precision

- Vocabulary size: [Tao et al., 2024](https://arxiv.org/abs/2407.13623) found on 33M–3B models that the optimal vocabulary size increases with compute, and most public models have vocabularies that are too small. Changing the vocabulary changes the parameter count and compute of the embedding and head, and also changes the length of text covered by each token; when comparing different vocabularies, loss must be converted to a byte-normalized metric ([§3.3](#33-loss-and-token-conventions)).
- Training precision: [Kumar et al., 2024](https://arxiv.org/abs/2411.04330) incorporate precision into the scaling law: low-precision training reduces the effective parameter count; the larger the training data volume, the greater the loss degradation caused by post-training quantization. The validation range of that paper is within 1.7B parameters and 26B tokens. The Ladder should use the target training precision scheme; a change in the precision scheme is treated as a recipe change and requires re-validation ([§10.4](#104-implementation-consistency-validation)).

## 5. Experimental Design and Budget

### 5.1 Experimental Matrix Structure

Example layout:

```text
Training budget D →  0.5×    1×     2×     4×
Model   130M          ●      ●      ●      ●
size    520M          ●      ●      ●      ●
        2.3B          ●      ●      ●      ●
N       8B            ○      ○      ○      ○  ← holdout (excluded from fitting)
↓
```

- Fitting points: smaller sizes used to fit the Scaling Law.
- Holdout point: the largest size does not participate in fitting and is used only to validate extrapolation accuracy.

The above sizes, training volumes, and full grid layout are all examples. The experimental point layout and the fitting form must be designed jointly:

- Clarify the fitting range and target range of $N$, $D$, and data repetition, and distinguish extrapolation along $N$, along $D$, and joint extrapolation;
- Choose among a full grid, IsoFLOP, or a sparse layout such as L-shape according to the objective in [§2.1](#21-decision-objectives) ([§8.1](#81-functional-form));
- Reserve validation points in advance, and agree on which results trigger supplementary experiments ([§5.5](#55-budget-allocation-and-follow-up-experiments)).

The layout must also be checked for parameter identifiability. Sampling only along a fixed TPP cannot separate the effects of $N$ and $D$. Fitting sensitivity analysis, holding out an entire size, and targeted additional points show whether conclusions depend on where the points were sampled.

### 5.2 Number of Sizes, Span, and Extrapolation Multiplier

| Dimension | Recommendation | Source |
|---|---|---|
| Number of model sizes | Cover the target extrapolation direction, plus a holdout; adding sizes can test fitting stability | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) |
| Size span | Cover the target extrapolation interval; on some model families a 34× span is still usable | [Choshen et al., 2025](https://arxiv.org/abs/2410.11840) |
| Ratio of adjacent sizes | Can increase geometrically, with spacing chosen according to budget | Our recommendation |

The extrapolation multiplier is defined as the ratio of the target compute to the compute of the largest fitting point. Among public examples, [Delphi](https://openathena.ai/blog/delphi/) fits on 3e18–3e20 FLOPs, with holdout covering 3×–333× extrapolation; [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.2.1 fits on 6e18–1e22 FLOPs (40M–16B) and extrapolates to 3.8e25 FLOPs, about 3,800×. The larger the extrapolation multiplier, the higher the share of the total error accounted for by functional-form error and recipe stability issues. When the target scale far exceeds the largest holdout, a medium-scale trial run should be arranged ([§10.6](#106-medium-scale-trial-run)).

### 5.3 Training Volume Tiers

Training volume tiers are designed according to the target TPP, with the compute-optimal ratio as a common reference. The IsoFLOP method of Chinchilla ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)): for each fixed compute budget $C$, train models of different sizes and take the minimum of the loss curve, which gives the optimal $(N, D)$ at that compute.

{% include figure.liquid
  path='assets/img/pretrain-scaling/chinchilla-isoflop.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 3. IsoFLOP results of Chinchilla. Left: an approximate quadratic fit of loss against log N at each compute budget, with the minimum corresponding to the optimal size; middle and right: power-law fits of the optimal parameter count and token count against compute. <a href="https://arxiv.org/abs/2203.15556">Source: Hoffmann et al., 2022, Fig. 3</a>.'
  alt='Chinchilla IsoFLOP curves: the left panel shows an approximate quadratic fit of loss against log parameter count, and the middle and right panels show power-law fits of the optimal parameter count and token count against FLOPs.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Across 400+ models (70M–16B, 5B–500B tokens), Chinchilla obtains:

$$N_{opt}\propto C^a,\qquad D_{opt}\propto C^b,\qquad a\approx b\approx0.5$$

That is, within this range models and data grow approximately proportionally, and the exponents from different fitting methods differ slightly. Chinchilla's $D/N\approx20$ is computed using the total parameter count including embeddings; when converted to $N_{\text{body}}$ the ratio is larger, and the optimal ratio must be verified on one's own recipe. Llama 3 used the same kind of IsoFLOP experiments to obtain an optimal size of about 402B at 3.8e25 FLOPs, and ultimately chose 405B ([Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1).

The training volume tiers must cover the target TPP:

- Example tiers: 0.5×–4× Chinchilla ratio covers the range from under-trained to mildly over-trained, suitable for projects whose target is close to compute-optimal; [Fantastic Optimizers](https://arxiv.org/abs/2509.02046v2) uses four tiers: 1×, 2×, 4×, and 8× Chinchilla ratio.
- Over-training: [Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2) found across 104 models that the power-law exponent of loss with respect to $C$ is similar under different $D/N$, providing empirical evidence for extrapolation up to a 32× Chinchilla ratio. Among the 47 models trained by [Sardana et al., 2024](https://arxiv.org/abs/2401.00448), quality was still improving when TPP increased to 10,000; fitting Chinchilla coefficients using only data at conventional TPP will overestimate the effect of additional tokens at extreme TPP. When the target TPP exceeds the validated range, extrapolation must be confirmed on a holdout close to the target TPP.
- Inference cost: after accounting for inference demand, the optimal size is smaller than the compute-optimal size, and the training volume increases accordingly ([Sardana et al., 2024](https://arxiv.org/abs/2401.00448)). Llama 3's 405B is an approximately compute-optimal size, while the smaller models are trained far beyond compute-optimal in exchange for better performance under the same inference budget ([Llama 3](https://arxiv.org/abs/2407.21783) §1). For handling deployment constraints see [§10.5](#105-realized-efficiency-and-deployment-constraints).
- Data-constrained: when the target $D$ exceeds the available $U$, the training volume axis must also annotate the repetition count, and the fitting form must include a repetition term ([§7.3](#73-data-constrained-training-and-repetition)).

### 5.4 Intermediate Checkpoints, Random Seeds, and Shared Trajectories

Intermediate checkpoints can be included in fitting, but the influence of early-training points must be checked. [Choshen et al., 2025](https://arxiv.org/abs/2410.11840v2) found that excluding early checkpoints reduces prediction error; their truncation setting of the first 10% or first 10B tokens should not be applied directly to all training budgets. The truncation rule must be determined in advance, or the model sizes must be split into fitting, validation, and test segments, with selection performed on the validation segment ([Lourie et al., 2026](https://arxiv.org/abs/2608.11859)). The truncation rule and the warmup rule are determined separately.

Multiple checkpoints along the same trajectory are serially correlated and cannot substitute for independent experiments in the target extrapolation direction ([§8.3](#83-fitting-protocol)). Random seed variance must be incorporated into the experimental design.

WSD branches sharing a stable trajectory should be recorded separately: actual cumulative cost, effective training budget of each branch, and shared starting point. Costs are not double-counted, and branches sharing a prefix cannot be counted as independent runs.

### 5.5 Budget Allocation and Follow-Up Experiments

Search, seeds, evaluation, validation, follow-up experiments, and failed reruns all count toward the total budget. Recommended allocation procedure:

1. First run a small number of trials to estimate seed noise, throughput, and differences among candidates;
2. Based on the trial results, allocate budget for search, fitting, independent replication, final holdout, evaluation, and contingency;
3. State which decision uncertainty each group of experiments is meant to resolve.

Examples of decisions for follow-up experiments:

- When seed fluctuation is the main source of error, add repeated runs;
- When candidate functional forms diverge substantially in predictions in the target region, add experimental points in the corresponding direction;
- When the ranking of candidates changes at longer training budgets, extend representative runs.

Whether it is more valuable to add larger models or to add random seeds for small models depends on the specific conditions ([Choshen et al., 2025](https://arxiv.org/abs/2410.11840)). There is no budget ratio in the public literature that generalizes across projects.

### 5.6 MoE Experimental Axes

The MoE Ladder selects experimental axes according to decision needs, with each axis designed separately:

1. Scale Ladder: fix sparsity and expert granularity, vary $N_{active}$ and $D$;
2. Sparsity Ladder: fix $N_{active}$, $D$, and $E_{active}$, vary only $E_{total}$;
3. Granularity Ladder: fix $N_{active}$ and $N_{total}$, vary individual expert size and number of experts ([Krajewski et al., 2024](https://arxiv.org/abs/2402.07871)).

[Kimi K2, 2025](https://arxiv.org/abs/2507.20534) varied $E_{total}$ under fixed $N_{active}$ and FLOPs to fit a sparsity scaling law: as sparsity increased from 8 to 48, the FLOPs required to reach the same target loss continued to decrease, while communication and inference complexity increased accordingly.

The fitting variables of different axes cannot be mixed. A full cross search is not needed. The Scale Ladder should include dense control points with the same $N_{active}$ or the same $C$, to determine how the gain of MoE over dense changes with scale (our recommendation).

## 6. Hyperparameter Search and Training Configuration

This chapter calibrates fully tuned $(\eta,B,\lambda)$ and expresses them as functions of $N$ and $D$ (or $C$) that can be extrapolated. Fixed-recipe prediction (Objective 1 in [§2.1](#21-decision-objectives)) runs under the recipe scaling rules. §6.1 explains the hyperparameter search objective; §6.2–§6.4 give priors for how each hyperparameter shifts with scale; §6.5–§6.6 cover the LR schedule and MoE hyperparameter settings; §6.7 gives the search procedure; §6.8 the method for comparing recipes; §6.9 the training wrap-up.

### 6.1 Hyperparameter Search Objective and Near-Optimal Region

[Lourie et al., 2026](https://arxiv.org/abs/2608.11859) and [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) show that small models degrade noticeably under suboptimal hyperparameters, and that the Scaling Law only emerges on the Fully-Tuned Frontier; insufficient hyperparameter search changes the shape of the power-law curve, causing bias in predictions at the target scale.

The near-optimal region observed in large-model experiments is wide. [Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320v1) tested multiplying and dividing the LR by $\sqrt{2}$ and increasing BSZ by 25% on 156B-A7B, and the final training loss differed by no more than $7\times10^{-4}$.

Resource allocation principle (Objectives 2–4): concentrate compute on exhaustive search for small models to ensure the fitting points lie on the Fully-Tuned Frontier; for large models, first confirm locally near the extrapolated value, and expand the search if necessary ([§6.7](#67-search-procedure-and-stopping-rules)).

Under a limited budget, "lying on the Fully-Tuned Frontier" uses the following operational definition (our recommendation): the optimum does not lie on the search boundary; under local joint perturbations in the neighborhood of the optimum, the loss change does not exceed a predetermined tolerance, and this tolerance is no smaller than the random seed variance. The conclusion is stated as "reaching the specified near-optimal tolerance within the searched range and budget."

### 6.2 Parameterization and Optimizer Transfer

- Width direction: $\mu$-Transfer ([Yang et al., 2022](https://arxiv.org/abs/2203.03466v2)) supports LR transfer along the width direction; for the case with nonzero WD, see Appendix G.1.2 of the original paper and [Power Lines](https://arxiv.org/abs/2505.13738v2). Experiments along BSZ and training length are limited. It can be combined with hyperparameter power laws.
- Depth direction: [Bordelon et al., 2023](https://arxiv.org/abs/2309.16620) scales the residual branch by $1/\sqrt{\text{depth}}$ and combines it with $\mu$P, observing hyperparameter transfer across width and depth in ResNet and ViT on CIFAR-10 and ImageNet. [Tensor Programs VI (Yang et al., 2023)](https://arxiv.org/abs/2310.02244) gives Depth-$\mu$P for networks with only one layer per residual block; when a residual block contains multiple layers (e.g., Transformer), that paper points out that all infinite-depth parameterizations have limitations. Direct evidence for depth-direction transfer on language models is lacking, so when the width-depth rule changes depth, the hyperparameters must be reconfirmed at a representative scale.
- Optimizer: [Liu et al., 2025](https://arxiv.org/abs/2502.16982) adds WD to Muon and scales the RMS of the update to the 0.2–0.4 range common for AdamW (taking 0.2), so that the LR and WD tuned for AdamW can be reused directly. [Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) found that after changing the architecture and switching to Muon, both the optimal LR and BSZ shifted. After changing the optimizer, one must recalibrate the hyperparameter scaling law, or first verify the transferability of the original coefficients at small scale.

### 6.3 Scaling Law for LR and BSZ

For a single $D/N$ ratio, a power law in $C$ can be used:

$$\eta_{opt} = a \cdot C^{-\alpha}, \quad B_{opt} = b \cdot C^{\beta}$$

The fitting coefficients provided by [DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) can serve as a reference. When covering multiple $D/N$ levels, $N$ and $D$ must be modeled separately:

| Approach | $\eta_{opt}$ | $B_{opt}$ | Applicable conditions and limitations |
|---|---|---|---|
| Step Law ([Li et al., 2025](https://arxiv.org/abs/2503.04715v3)) | $c\cdot N^{-\alpha}D^{\beta}$ | $0.58\,D^{0.571}$ | Can model across $D/N$; validated on 3,700+ models; the assumption that $B_{opt}$ is independent of $N$ passed a regression test (Appendix A.5), but the target range still needs verification |
| Power Lines ([Bergsma et al., 2025](https://arxiv.org/abs/2505.13738v2)) | Jointly constrained by the timescale $\tau$ ([§6.4](#64-weight-decay)) | $\propto D^{0.4}$, with weak dependence on $N$ within the measured range | Approximate fit under the measured recipes; the LR stability and BSZ efficiency range still need to be checked |
| Token Horizons ([Bjorck et al., 2024](https://arxiv.org/abs/2409.19913)) | $c \cdot N^{-\alpha} D^{-\beta}$ | — | With fixed model size and BSZ, the peak LR decays with training length |

In Token Horizons and Step Law, the exponent of $\eta_{opt}$ with respect to $D$ has opposite sign ($D^{-\beta}$ versus $D^{+\beta}$). The experimental setups of the two differ: Token Horizons fixes BSZ, while Step Law jointly searches BSZ for each $D$. Whether this setup difference can fully explain the sign difference has not been verified. One should choose the candidate form in light of one's own configuration and validate it on a holdout.

{% include figure.liquid
  path='assets/img/pretrain-scaling/step-law-hyperparameter-validation.png'
  id='fig-step-law-hyperparameter-validation'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 4. Step Law under the test condition N = 1B, D = 100B, comparing the configuration predicted by the hyperparameter formula with the experimentally determined optimal configuration. The contours are obtained from 120 training experiments with different LR and BSZ combinations; this condition lies outside the fitting range of the paper. The comparison shown is limited to the training recipe and learning rate schedule used in the paper. <a href="https://arxiv.org/pdf/2503.04715v3#page=1">Source: Predictable Scale: Part I, 2025, v3, Fig. 1 (PDF page 1)</a>.'
  alt='Figure 1 of the original Step Law paper: loss contours over LR and BSZ, along with Step Law, other hyperparameter formulas, and the experimentally determined optimal configuration.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

The exponents fitted for $B_{opt}$ in the two works are close, so one can initialize the BSZ transfer rule accordingly and validate it on one's own data.

Power Lines also measures $B_{crit}\propto D_{min}^{0.5}$, where $D_{min}$ is the minimum number of tokens needed to reach the target loss. $B_{crit}$ is the turning point in the trade-off between token efficiency and step count: in that paper's hyperbolic model, at $B=B_{crit}$ reaching the same loss requires about $2D_{min}$ tokens; increasing $B$ further reduces the step count while increasing token and compute cost. Actual training time also depends on hardware utilization and must be measured systematically.

{% include figure.liquid
  path='assets/img/pretrain-scaling/cerebras-bcrit-hyperbola.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 5. Hyperbolic relationship between training tokens and steps required to reach the same target loss, for 610M and 1.7B models on the left and right respectively, with color indicating loss. B_crit marks the trade-off turning point between token efficiency and step count. <a href="https://arxiv.org/pdf/2505.13738v2#page=6">Source: Bergsma et al., 2025, v2, Fig. 4</a>.'
  alt='Two tokens-versus-steps plots for the 610M and 1.7B models: different target losses correspond to different hyperbolas, color indicates loss, and the curves mark B_crit.'
  avoid_scaling=true
  zoomable=true
%}

[Schaipp, 2026](https://arxiv.org/abs/2607.01487v1) defines the near-optimal batch region as the range within about 5% compute loss, with a measured width of about 4×; under that paper's log-symmetric fitting model this corresponds to $[B_{opt}/2,2B_{opt}]$, which can serve as the starting point for local search.

The unit of BSZ and its variation during training also need to be unified:

- When changing the sequence length, keep BSZ in terms of tokens, and at the same time check the changes in attention compute and data packing.
- Some target training runs use a batch size ramp, e.g., Llama 3 405B increases BSZ from 4M tokens to 16M tokens in stages during training ([Llama 3](https://arxiv.org/abs/2407.21783) §3.4.1); Nemotron-4 15B also uses BSZ ramp-up ([Nemotron-4, 2024](https://arxiv.org/abs/2402.16819)). If the Ladder does not include the ramp, its effect on the loss curve and on hyperparameter extrapolation must be verified.

### 6.4 Weight Decay

Several public recipes fix WD: [Kimi K2, 2025](https://arxiv.org/abs/2507.20534v2) uses $\lambda=0.1$; [OLMo 3, 2025](https://arxiv.org/abs/2512.13961v1) uses AdamW with no decay on embeddings; [Qwen3, 2025](https://arxiv.org/abs/2505.09388) does not report WD scaling.

The optimal WD depends on the training volume and the evaluation objective. [Han et al., 2026](https://arxiv.org/abs/2602.11137v2) finds that the WD preferred by pretraining loss decreases as TPP increases, and that stronger WD may benefit post-training plasticity. When reproducing a recipe, follow its WD; when studying fully tuned settings or cross-TPP relationships, the effect of WD should be checked.

Under the AdamW setup, WD can be jointly calibrated with LR and BSZ through the timescale $\tau = B/(\eta\lambda D)$: [Power Lines](https://arxiv.org/abs/2505.13738) fixes $\eta$ under μP when changing $B$, and adjusts $\lambda$ to maintain the optimal $\tau$; LR is also constrained by the maximum stable learning rate ([Power Lines §2.4](https://arxiv.org/abs/2505.13738)).

### 6.5 LR Schedule

Two commonly used schedules:

- Cosine: smooth decay to 0 or a small residual value after a short warmup. The minimum LR may differ across stages in multi-stage training.
- WSD (Warmup-Stable-Decay): facilitates reuse of the stable trajectory and annealing at a chosen budget. [River Valley (Wen et al., 2024)](https://arxiv.org/abs/2410.05192v3) explains the roles of the stable and decay phases under specific loss-geometry assumptions.

An annealing ratio of about 10%–20% can serve as an initial reference ([Tissue et al., 2024](https://arxiv.org/abs/2408.11029v2)), but still needs verification at the target budget. [Wang et al., 2025](https://arxiv.org/abs/2512.13705) studies the cross-scale transfer of annealing strategies.

### 6.6 MoE Training Hyperparameters

The LR and BSZ rules for MoE should be calibrated on the MoE recipe; coefficients from the dense recipe must be verified before reuse (our recommendation). Routing-related hyperparameters, including the load balancing coefficient or bias update rate, the numerical precision of the router, and the capacity factor, are set according to the fixed rules of [§4.4](#44-moe-structural-scaling-rules), and the expert load distribution and token drop ratio are checked at representative scales.

### 6.7 Search Procedure and Stopping Rules

LR and BSZ should be searched first, then the schedule and WD checked; stability hyperparameters such as $\epsilon$ and $\beta_2$ can be fixed after verification at the baseline and representative scales. Coordinate descent can be used to progressively narrow the range (sizes are illustrative):

1. Small model (about 130M) grid search to determine the core interval;
2. Medium model (about 500M) to check the power-law trend;
3. Large model local confirmation: LR is first confirmed within a factor of $\sqrt{2}$ around the extrapolated value, BSZ is first confirmed within $[B_{opt}/2,2B_{opt}]$, and after checking the boundaries, decide whether to expand ([§6.1](#61-hyperparameter-search-objective-and-near-optimal-region), [§6.3](#63-scaling-law-for-lr-and-bsz)).

The search dimension for BSZ can be reduced with the transfer rules of [§6.3](#63-scaling-law-for-lr-and-bsz). Given a selected LR, the optimal timescale $\tau$ can be searched by adjusting WD ([§6.4](#64-weight-decay)), reducing the cost of joint search; equal $\tau$ does not guarantee that different $(\eta, \lambda, B)$ combinations perform equivalently, and the LR stability and BSZ efficiency range still need to be checked.

Selecting the lowest loss from multiple trials with random fluctuations biases the estimated gain of the selected configuration upward ([Cawley & Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)). Handling rules (our recommendation):

- Record the search range, boundary results, local joint perturbation results, repeated-run variance, and budget consumption for each fitting point;
- Separate the evaluation used to select hyperparameters from the evaluation used to report gains; re-verify the final candidate with a new random seed;
- Reserve the holdout at the final scale for acceptance, and exclude it from hyperparameter search;
- List the number of tuning runs and the compute for each candidate separately.

When stopping trials early, do not use early rankings by default to eliminate configurations that may prove effective later; rankings can flip during annealing ([§6.8](#68-recipe-comparison)). Runs that are actively stopped, that diverge algorithmically, that suffer infrastructure failures, and that contain implementation errors must be marked with separate statuses; they cannot be uniformly recorded as some terminal loss, nor deleted without a record ([§12.1](#121-experiment-records)).

### 6.8 Recipe Comparison

When comparing recipes such as optimizers and LR schedules, the following conditions must be met:

- Tune each candidate separately and report its search budget. [Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046) finds that insufficient tuning of the AdamW baseline exaggerates the gains of new optimizers. [Kimi K3, 2026](https://arxiv.org/abs/2607.24653) §3.2 observes that at the same model size and training volume, the optimal peak LR and BSZ of cosine and WSD differ significantly; after separate scaling law searches, the final loss of cosine is consistently lower than that of WSD, so cosine is adopted as the default schedule.
- Compare annealing endpoints under comparable budgets. Fantastic Optimizers observes that loss rankings flip during annealing, so mid-stable-phase rankings cannot directly substitute for final rankings; verified proxy screening is the exception.
- Compare across multiple sizes. In the 8× Chinchilla setting of Fantastic Optimizers, the improvement in token efficiency shrinks as the model grows (from $1.4\times$ at 0.1B to $1.1\times$ at 1.2B), and is not equivalent to the speedup in training time.

### 6.9 Training Wrap-Up

If the delivery pipeline includes the following steps, the corresponding stages of the Ladder should also be executed, or a verified proxy used, to keep the fitting target consistent with the delivery target:

- Continued training: switch the data mixture and decay the LR after main training ([Nemotron-4, 2024](https://arxiv.org/abs/2402.16819); [OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3)). This stage must be included when predicting final performance ([§11.1](#111-staged-ladder)).
- Weight averaging: average weights of checkpoints with shared initialization, including merging after independent fine-tuning ([Model Soups](https://arxiv.org/abs/2203.05482v3)) and sliding-window averaging along the same trajectory ([LAWA](https://arxiv.org/abs/2209.14981); [Sanyal et al., 2024](https://arxiv.org/abs/2306.03241)). [Model Merging (ByteDance Seed, 2025)](https://arxiv.org/abs/2505.12082v3) finds in 1.3B and 13B comparisons that PMA in the WSD stable phase can approach the downstream performance of the annealing endpoint; the merging window and starting point still need verification.

## 7. Data Ladder

The data Ladder fixes the model structure and training configuration, varies the data variables, and uses small-scale experiments to support data decisions at the target scale. This chapter discusses three types of decisions: data source selection ([§7.1](#71-data-quality-and-data-source-evaluation)), mixture ([§7.2](#72-data-mixture-ladder)), and repetition under data-constrained conditions ([§7.3](#73-data-constrained-training-and-repetition)). Data cleaning and filtering pipelines are out of scope.

### 7.1 Data Quality and Data Source Evaluation

[DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954) observed across the compared training corpora:

- Higher-quality corpora correspond to an optimal compute allocation that leans more toward parameter count;
- The scaling law parameters differ significantly across datasets and cannot be reused directly;
- When the recipe and evaluation distribution are controlled, differences in the optimal $N/D$ can help assess data quality.

Therefore the data version must be recorded as a fixed item of the Ladder, and the scaling law parameters must be re-validated after any version change. When Kimi K3 changed its architecture, data, and training recipe together, the team redid the scaling law study and re-tuned BSZ, LR, TPP, and model shape ([Kimi K3, 2026](https://arxiv.org/abs/2607.24653) §3.2).

Evaluating a new data source does not require rerunning the full matrix:

- The micro-annealing of [OLMo 2, 2025](https://arxiv.org/abs/2501.00656) starts from a specified checkpoint, mixes candidate data with general data for short-term annealing, and judges the gain. The conclusions of this method are limited to the starting point and stage of the checkpoint used, and cannot be extrapolated to the data ranking over the full run.
- The per-domain sampling rates of [Kimi K3](https://arxiv.org/abs/2607.24653) were determined by ablations on smaller models (§3.1).
- In an ablation that added the legal-domain synthetic data released with [Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) to Nemotron 3 Nano pretraining, the data raised the average accuracy of the LegalBench proxy evaluation from 64.6 to 74.7.

### 7.2 Data Mixture Ladder

The data mixture Ladder fixes the model structure, parameter count, training volume, and optimizer, and only varies the data mixture vector $\mathbf{w}=(w_1,\ldots,w_k)$. Mixture experiments select the training distribution, and the scale Ladder fits $N$, $D$, and loss on that distribution; when the mixture is adjusted with training volume, the two types of experiments need to iterate. Data selection for pretraining from scratch and for continued training are handled separately, each stating its transfer scope.

The execution flow is as follows. The design values for the proxy experiments come from [Olmix, 2026](https://arxiv.org/abs/2602.12237), using a 1B target model as the reference; the method was used in [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) §3.4.4.

1. Define the mixture variables. Partition data into buckets by attributes such as source and domain, and record the deduplicated available tokens of each bucket; after adjusting filters or thresholds, recheck the capacity and repetition conditions ([Marin data pipeline, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/)). The mixture dimensions can be refined down to instance attributes such as educational value, domain, language, and safety ([Qwen3, 2025](https://arxiv.org/abs/2505.09388)). When the mixture is switched across training stages, design mixture experiments for each stage separately ([§11.1](#111-staged-ladder)): in the 25T-token training of [Nemotron 3 Super, 2026](https://arxiv.org/abs/2604.12374), the first 80% emphasizes diversity and the last 20% emphasizes high-quality data.
2. Set the baselines. The baselines are proportional sampling and [UniMax](https://arxiv.org/abs/2304.09151); [Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) points out that a poorly executed learned mixture can be worse than these two baselines.
3. Design the proxy experiments.
   - Size: at 5× Chinchilla training volume, proxies of 15M and above have a Spearman correlation above 0.89 with the 1B target, while a 1M proxy gives 0.73 ([Olmix](https://arxiv.org/abs/2602.12237)); [OLMo 3](https://arxiv.org/abs/2512.13961) uses a 30M proxy and 3B tokens.
   - Training volume and data pool: scale down in the same proportion as the repetition conditions of the target training. [Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) sets the proxy budget by active parameter count and shrinks each bucket's data pool by the same proportion, so that the proxy's repetition count matches that of the target ladder (791 tokens per active parameter).
   - Number and sampling: the required number of proxies grows linearly with the number of domains $m$, and is no fewer than $3(m+1)$ when using log-linear regression; mixtures are sampled from a Dirichlet distribution centered on the natural distribution, with sparse sampling for topic-level domains and dense sampling for source-level domains.
   - Alternative method: [DeMix, 2026](https://arxiv.org/abs/2602.00747) trains a component model for each candidate dataset and replaces the mixture-trained proxy with a weighted merge of the component models; its ranking consistency is higher than that of proxies trained at small scale.
4. Fit the regression. In [Olmix](https://arxiv.org/abs/2602.12237), the log-linear model gives the best downstream results; different regression model families have their own advantages at different proxy counts, so conclusions in the existing literature are inconsistent. Fit each task separately: the fit correlation on held-out mixtures is 0.983 with per-task fitting and 0.866 with an aggregate metric. The regression form must be able to represent non-monotonic responses: [Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) observes that as a single bucket's weight increases, the loss first decreases, then saturates, and rises again when repetition is excessive. The correlation coefficient between a single bucket's weight and the metric does not represent that bucket's independent effect, because the weights sum to 1; in [Marin](https://openathena.ai/blog/marin-data-pipeline-overview/) this correlation also varies with the evaluation set and with the pretraining versus cooldown stage.
5. Solve for the mixture. The constraint $w_j \le k N_j / R$ ($N_j$ is the available tokens of data bucket $j$, $R$ is the target training tokens, and $k$ is the repetition cap) significantly changes the solved mixture ([§7.3](#73-data-constrained-training-and-repetition)). Exact solving plus a KL regularizer toward the natural distribution ($\lambda=0.05$) works best.
6. Cross-scale confirmation. Before the target training, compare the selected mixture against the baselines on a set of larger models ([Marin](https://openathena.ai/blog/marin-data-pipeline-overview/)). How well this flow transfers to larger models remains to be verified.
7. Re-estimate after data updates. The mixture reuse of [Olmix](https://arxiv.org/abs/2602.12237) preserves the relative proportions of unaffected data buckets and only recomputes the affected part. In a setting with 5 updates, a final 64 domains, and a 1B model trained on 100B tokens, this method improves by 11.6% over the natural distribution, reaches 95% of the gain from full recomputation, and reduces the number of proxies by 74%. The mixture of [OLMo 3](https://arxiv.org/abs/2512.13961) went through 3 rounds of this flow.

Deliverables: a traceable relationship among the data inventory, the candidate mixture table, the feasible-region constraints, the regression acceptance results, the cross-scale confirmation results, and the final sampling configuration.

### 7.3 Data-Constrained Training and Repetition

When data is constrained, the total number of unique tokens $U$ must be written into the experiment matrix as a constraint at the design stage, and the training volume axis must be annotated with cumulative training tokens, unique tokens, and repetition count simultaneously. The deduplication scope must be consistent with the statistical basis of the repetition count: [Marin data pipeline, 2026](https://openathena.ai/blog/marin-data-pipeline-overview/) performs global deduplication across all sources so that the number of epochs can be treated as a controlled variable, where the largest cross-source overlap comes from [Nemotron-CC](https://arxiv.org/abs/2412.02595) and its synthetic rewritten versions.

[Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264v5) assume that the marginal benefit of repeated data decays exponentially and give the effective data volume:

$$D^{\prime} = U_D + U_D \cdot R_D^{\ast} \left(1 - e^{-R_D / R_D^{\ast}}\right)$$

$U_D$ is the number of unique tokens, $R_D$ is the number of additional repetitions ($R_D=0$ means a single epoch), and $R_D^\ast$ is the fitted decay scale. As $R_D\to\infty$, $D^\prime\to U_D(1+R_D^\ast)$. This form describes benefit saturation; if the loss rises again with repetition, an overfitting term must also be modeled.

{% include figure.liquid
  path='assets/img/pretrain-scaling/muennighoff-epoch-returns.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 6. Marginal returns of data repetition (4.2B model, 12B unique tokens). The first few epochs of repetition yield returns close to those of new data, after which marginal returns decline and saturate after roughly 40 epochs. <a href="https://arxiv.org/pdf/2305.16264v5#page=1">Source: Muennighoff et al., 2023, v5, Fig. 1 left</a>.'
  alt='Repetition training curves for a 4.2B model on 12B unique tokens: the horizontal axis is cumulative training tokens and epochs, the vertical axis is final test loss; returns saturate after roughly 40 epochs.'
  avoid_scaling=true
  zoomable=true
%}

- The upper limit on repetition count depends on the data and the recipe. [Yan et al., 2025](https://arxiv.org/abs/2511.13421v2) obtain, under a linear regression assumption, that the optimal repetition count grows logarithmically with the number of samples; [Lovelace et al., 2026](https://arxiv.org/abs/2605.01640) find that a larger parameter count, fewer unique tokens, and more repetitions jointly exacerbate overfitting.
- In the controlled experiments of [Xue et al., 2023](https://arxiv.org/abs/2305.13230v2), overfitting caused by repetition varies mainly with parameter count; increasing the data volume alleviates it, while improving quality at the same data volume does not bring the same improvement. Dropout improves repetition-induced overfitting noticeably (v2 Table 4), and larger models need the dropout rate re-tuned.
- When data is constrained, the choice of model size must also account for the reduced benefit from repetition: with $U$ fixed, increasing $N$ exacerbates overfitting, and the compute-optimal allocation must be re-solved under a fitting form that includes a repetition term.
- When scarce data is mixed with a large amount of general data for training, the tolerable repetition count is higher than for training on a single source. In over 2,000 training runs, [Sedova et al., 2026](https://arxiv.org/abs/2605.12715) find that scarce target-language corpora can be repeated 15–20 times, and that the optimal repetition count depends on the target data volume, compute, and model size; their mixture scaling law with a repetition term can be extrapolated to larger scales after fitting at small scale.
- The maximum repetition count of each data source should be used as a mixture constraint ([§7.2](#72-data-mixture-ladder)). When [Kimi K2.5, 2026](https://arxiv.org/abs/2602.02276) continues joint pretraining from a near-final checkpoint of [Kimi K2](https://arxiv.org/abs/2507.20534), it caps the maximum number of epochs per data source; the quality-aware upsampling of [OLMo 3](https://arxiv.org/abs/2512.13961) repeats only high-quality data, with a maximum of 7 repetitions, and outperforms threshold-based filtering in a simulated data-constrained control (§3.4.4, Appendix A.2.5).
- Rewriting can substitute for part of the repetition. On an early checkpoint, [Kimi K2, 2025](https://arxiv.org/abs/2507.20534) compares three settings on SimpleQA: training on the original text for 10 epochs gives 23.76, rewriting once and training for 10 epochs gives 27.39, and rewriting 10 times and training once each gives 28.94; when generalized to other knowledge corpora, each corpus is rewritten at most 2 times (§2.2). [Kimi K3](https://arxiv.org/abs/2607.24653) follows this rewriting method (§3.1). Rewritten data and original text must be counted separately for unique tokens.

## 8. Loss Scaling Law Fitting

Three settings must be confirmed before fitting ([Porian et al., 2024](https://arxiv.org/abs/2406.19146)):

1. Compute accounting includes the output head, recorded separately from the parameter count basis used for fitting ([§3.1](#31-parameter-count-definition));
2. Warmup is set according to the scaling rule ([§4.1](#41-variable-classification));
3. For Objectives 2–4, each size is searched to the Fully-Tuned Frontier ([§6.1](#61-hyperparameter-search-objective-and-near-optimal-region)); Objective 1 runs under the recipe.

### 8.1 Functional Form

Chinchilla's parameterization is $L(N,D)=E+A/N^\alpha+B/D^\beta$, and the additive form implies $\partial^2L/\partial N\partial D\equiv0$. [Skaling (Videau et al., 2026)](https://arxiv.org/abs/2608.07222v1) found a negative mixed partial derivative on the analyzed data and added an outer exponent $k$:

$$L(N,D)=\left(\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}\right)^{k}+E$$

When $k=1$ it reduces to the Chinchilla form; that paper fits $k\approx0.31$–$0.45$. Skaling has only one more parameter, and the algebraic form of the compute-optimal closed-form solution is unchanged.

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-vs-chinchilla-error.svg'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 7. Prediction residuals of the two forms on the same (N, D) grid, with each point a single training run. Left and middle: signed percentage error, sharing a color scale; the Chinchilla residuals are saddle-shaped and grow toward the four corners, while Skaling is near zero across the whole grid. Right: the ratio of the two errors; Skaling is more accurate on 76% of configurations, with a median ratio of 2.2×. <a href="https://arxiv.org/abs/2608.07222">Source: Videau et al., 2026, Fig. 1</a>.'
  alt='Three (N, D) grid scatter plots: left is the signed percentage error of Chinchilla, middle is the same error for Skaling, right is the ratio of the two errors.'
  avoid_scaling=true
  zoomable=true
%}

[Farseer (Li et al., 2025)](https://arxiv.org/abs/2506.10972) makes both the coefficient and the exponent on the data side depend on $N$, for nine parameters in total. Skaling, with six parameters, has the lowest interpolation and single-axis extrapolation error on both the Farseer and SK-Grid datasets ([Skaling](https://arxiv.org/abs/2608.07222) Table 1). Skaling can be taken as the default candidate and compared with the Chinchilla form on one's own data; refitting may change the compute-optimal ratio and must be validated by holdout.

When $D/N$ is fixed, the one-dimensional approximation $L=G(M)/C^\gamma+E$ can be used (with $M=D/N$); across ratios, $N$ and $D$ still need to be modeled separately.

The layout must be validated together with the functional form. In the two sets of experiments in the Skaling paper, the compute for the L-shape layout is about 1/5–1/10 of the full grid; the Chinchilla additive form shows a clear increase in error on these sparse layouts. When compute is constrained, the L-shape layout can be evaluated.

{% include figure.liquid
  path='assets/img/pretrain-scaling/skaling-v1-sampling-evaluation.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 8. Sampling strategies and evaluation regions of Skaling. Left: Random holds out validation points at random; L-shape covers different N at small D and different D at small N. Right: regions for interpolation, extrapolation along N or D, and joint extrapolation. The horizontal axis is D and the vertical axis is N. <a href="https://arxiv.org/pdf/2608.07222v1#page=5">Source: Videau et al., 2026, v1, Fig. 4 (PDF page 5)</a>.'
  alt='Original Figure 4 from Skaling v1: the left side shows Random and L-shape sampling, and the right side shows the evaluation regions for interpolation and extrapolation along model size, training volume, and both jointly; the horizontal axis is D and the vertical axis is N.'
  avoid_scaling=true
  zoomable=true
%}

### 8.2 Loss Curves and Annealing Scaling Law

[Tissue et al., 2024](https://arxiv.org/abs/2408.11029) expresses loss as a function of step:

$$\hat L(s) = L_0 + A \cdot S_1(s)^{-\alpha} - C \cdot S_2(s)$$

$S_1(s)=\sum_{i\leq s}\eta_i$ is the cumulative learning rate area, and $S_2(s)$ is the cumulative annealing amount with a forgetting kernel. This formula takes the schedule as input, can fit the entire loss curve using multiple eval points from the same trajectory, and can also predict curves under different re-warmup LRs in continued training (§4.7 of the paper). Reusing coefficients across scales or data distributions still needs validation. When the extrapolation span is too large, systematic bias appears and additional experimental points are needed.

### 8.3 Fitting Protocol

The following choices must be fixed and recorded before fitting (our recommendation):

- Fitting target: fit $L$ or $\log L$; fit $E$ or fix $E$.
- Objective function: squared error or a robust loss (such as Huber); the weights for each size, trajectory, and checkpoint.
- Numerical settings: parameter constraints, variable transformations, initialization, multi-start optimization, and convergence checks.
- Outliers: pre-specified handling rules.
- Functional form selection: compare candidate forms on the development validation set; holdout does not participate in selection.
- Uncertainty: the resampling unit for bootstrap must match the dependency structure of the data; multiple branches sharing a prefix cannot be treated as independent runs. Report parameter confidence intervals, prediction intervals for a single future run, and functional form uncertainty separately. Resampling cannot eliminate functional form error or error beyond the validation range.

Parameter fitting and decision computation should be re-verified together: for each valid fit, recompute the target configuration and candidate differences, and observe whether the conclusions are stable ([§2.3](#23-acceptance-thresholds-and-decision-rules)). Reporting only parameter standard errors cannot support candidate selection.

The reproduction by [Besiroglu et al., 2024](https://arxiv.org/abs/2404.10102) shows that the settings for parametric fitting and confidence intervals affect the conclusions; [(Mis)Fitting (Li et al., 2025)](https://arxiv.org/abs/2502.18969) discusses the impact of missing fitting details on reproducibility.

Deliverables: a fitting script that reads the experiment table, outputs residuals, sensitivity analysis, target predictions, and a decision table, together with a complete example using public data or clearly labeled synthetic data.

### 8.4 Fitting Diagnostics and Failure Handling

After fitting is complete, check:

- Residual structure: whether residuals vary systematically with $N$, $D$, or training stage; whether conclusions depend on a few points or a specific functional form.
- Checkpoint correlation: multiple checkpoints from the same trajectory have serial correlation, and the amount of independent information cannot be judged by the number of points ([Delphi](https://openathena.ai/blog/delphi/) performs bootstrap at the IsoFLOP optimum).
- Uncertainty separation: report seed variance, evaluation variance, and fitting uncertainty separately.
- Truncation and validation split: truncation rules are determined in advance or selected using the development validation set; holdout remains for acceptance purposes ([§5.4](#54-intermediate-checkpoints-random-seeds-and-shared-trajectories)).
- Failure handling: when the error is too large, list the experiments to be added and the decisions that cannot currently be made. Explanations without evidence are recorded as "cause unknown".
- Artifacts: recipe version, experiment records, fitting method, prediction intervals, and unresolved issues ([§12.1](#121-experiment-records)).

If the decision involves downstream capabilities, the prediction of task metrics also needs to be validated ([§9](#9-downstream-task-prediction)); the scaling laws may differ across samples.

### 8.5 MoE Fitting

- For a scale ladder with fixed sparsity and granularity, $N_{active}$ can be used in place of $N$ as the fitting independent variable, while also recording $N_{total}$.
- When sparsity or granularity changes, $S$ (or $E_{total}$) and granularity must enter the fitting form as independent variables ([Krajewski et al., 2024](https://arxiv.org/abs/2402.07871); [Ludziejewski et al., 2025](https://arxiv.org/abs/2502.05172)).
- Data from the scale ladder and the sparsity ladder cannot be mixed to fit the same set of coefficients ([§5.6](#56-moe-experimental-axes)).
- Dense control points are fitted separately, to compare how the equivalent compute multiplier of MoE versus dense changes with scale.

## 9. Downstream Task Prediction

### 9.1 Method Routes

| Method Route | Core Idea | Representative Work | Limitation |
|---|---|---|---|
| Loss → Performance | First predict loss or perplexity, then map to downstream metrics | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540v2) (power law of error rate versus perplexity, equivalent to an exponential relationship with cross-entropy loss); [Delphi](https://openathena.ai/blog/delphi/) (sigmoid mapping) | The mapping depends on the task and evaluation protocol |
| Compute or $(N,D)$ → Task NLL → Acc | Two stages: first predict the NLL of the correct answer on the task, then fit the mapping from NLL to accuracy | [Bhagia et al., 2024](https://arxiv.org/abs/2412.04403) (OLMo Task Ladder); [Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1 (sigmoid mapping, extrapolated to 405B) | Inter-task noise and prediction error vary widely |
| End-to-End | Directly model how task metrics change with compute; can be grouped by difficulty | [COD (Xu et al., 2026, v4)](https://arxiv.org/pdf/2502.17262v4) (difficulty feature clustering); [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774v6) (HumanEval difficulty bucketing) | Requires a distinguishable evaluation signal; applicability depends on the method and scoring protocol |

The first stage of Llama 3 uses only scaling law models within 1e22 FLOPs, fitting a linear relationship between the normalized NLL of the correct answer and training FLOPs; the second stage uses both the scaling law model and the Llama 2 model, fitting a sigmoid relationship between NLL and accuracy. On ARC-Challenge, the prediction for 405B is slightly lower than the measured value ([Llama 3](https://arxiv.org/abs/2407.21783) §3.2.1). The IsoFLOP plus sigmoid mapping of [Delphi](https://openathena.ai/blog/delphi/) can serve as a starting point. The pros and cons of different routes need to be compared under the same task and protocol.

### 9.2 COD Framework

The four stages of [COD (Xu et al., 2026, v4)](https://arxiv.org/pdf/2502.17262v4):

1. Clustering: multiple small models sample each question many times, using the average correctness as the difficulty feature, and cluster by difficulty;
2. Fitting: fit $E[\mathrm{Acc}(C)] = g + (1-g) \cdot e^{-aC^{-b}-c}$ for each cluster;
3. Extrapolation: select reliable clusters, plug in the target compute, and take a weighted average by sample count;
4. Mapping: calibrate the mapping curve from the extrapolatable subset to the full evaluation set.

COD v4 achieves an average absolute prediction error of 1.55 percentage points on a 70B model across 8 benchmarks (Table 1).

{% include figure.liquid
  path='assets/img/pretrain-scaling/cod-v4-prediction-accuracy.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block bg-white'
  width='100%'
  caption='Figure 9. COD v4 predicted curves on MATH and MMLU-pro versus 70B measured values, including COD, Loss-Intermediate, End-to-End(exp), and End-to-end(BNSL). Red dots are small-model results, blue dots are measured values of the target model. <a href="https://arxiv.org/pdf/2502.17262v4#page=9">Source: Xu et al., 2026, v4, MATH and MMLU-pro subplots of Fig. 4 (PDF page 9)</a>.'
  alt='The original MATH and MMLU-pro subplots of COD v4 Figure 4: the horizontal axis is compute, the vertical axis is accuracy, comparing the fitting of COD, Loss-Intermediate, End-to-End(exp), and End-to-end(BNSL) and the 70B target prediction.'
  avoid_scaling=true
  zoomable=true
  loading='lazy'
%}

Applicability conditions:

- When there are too few samples, the clustering metrics are unstable; under new architectures or new data distributions, the stability of the difficulty features and the mapping needs to be verified.
- The main pretraining experiments use a constant LR after warmup. The continued training experiments additionally include data changes and annealing, and small models need to match the distribution and TPP of both stages (v4 Appendix D–E).
- v4 uses dense clustering to predict a MoE target with an active parameter count of 32B, with average and maximum absolute errors of 3.11 and 8.11 percentage points, respectively, providing limited cross-architecture evidence (§5.3.1, Table 2).
- CoT already has empirical prediction results, but theory does not yet adequately cover non-unique answers and reasoning paths (Appendix H).

### 9.3 Limits of Predictability

[Schaeffer et al., 2024](https://arxiv.org/abs/2406.04391) analyzed why downstream metrics are hard to predict: multiple-choice accuracy is jointly determined by the probability mass on the correct option and on specific incorrect options, and the step-by-step transformation from loss to accuracy weakens the statistical relationship with compute, while predicting only the probability of the correct answer loses this information.

Therefore, downstream prediction must satisfy the signal distinguishability requirement of [§2.2](#22-evaluation-protocol), and must be validated separately on a holdout close to the target scale. When downstream prediction fails validation, decisions can only rely on loss prediction and proxy metrics, and this limitation must be stated.

## 10. Extrapolation Validation and Launch Decisions

### 10.1 Holdout Validation

A holdout must be set in the target extrapolation direction, and multiple extrapolation multipliers should be arranged according to the budget ([§5.2](#52-number-of-sizes-span-and-extrapolation-multiplier)). [Delphi](https://openathena.ai/blog/delphi/) fits on 3e18–3e20 FLOPs and holds out to 1e23 FLOPs; the first recipe has a loss 2.5% higher than predicted at about 33× extrapolation, and diverges at 333× extrapolation (Figure 1).

The error threshold is predetermined according to [§2.3](#23-acceptance-thresholds-and-decision-rules), and the confidence interval of the extrapolation is reported at the same time. Once holdout results are used to adjust the model or fitting setup, they become development data, and new acceptance requires independent evidence.

### 10.2 Combined Validation

After individual changes pass, the final combination must be validated; the gains of individual items cannot be added directly ([Marin follow-up](https://openathena.ai/blog/pretraining-speedup/); [OLMo 3, 2025](https://arxiv.org/abs/2512.13961) Appendix A.2.5):

- Jointly check the data mixture against training volume and degree of repetition;
- State the transfer scope of conclusions from proxy experiments;
- When combined validation is only completed at small scale, it must not be described as validated at the target scale.

### 10.3 Stability Stress Testing

Small-scale training cannot fully expose loss spikes and gradient anomalies that occur in large-scale training. When the architecture or optimizer changes, stress testing should be added: [Qwen3.8-Next, 2026](https://arxiv.org/abs/2608.30320) uses a medium-scale model and 2×/4× the predicted optimal LR to increase optimization pressure, comparing the old and new recipes under the same pressure.

Stability under a short-term high LR and stability near the target training volume are two different conditions. For key candidates, record gradient norms, activation ranges, and loss spikes; for MoE, also record per-expert load, token drop ratio, and changes in the routing distribution. [Nemotron 3 Ultra, 2026](https://arxiv.org/abs/2606.15007) §2.7 reports two divergences late in training: the first is related to output-layer gradient precision and stabilized after restoring FP32; the cause of the second is unknown, and it was mitigated by starting the anneal early.

### 10.4 Implementation Consistency Validation

Large-scale training and Ladder experiments often use different parallelism schemes, gradient accumulation, fused operators, communication precision, and distributed optimizers. Identical configuration names do not prove that the mathematical operations of training are consistent. Validation steps (our recommendation):

1. Using the same checkpoint and a controlled batch, compare forward, loss, gradients, and a single parameter update;
2. Compare whether the deviation over a short trajectory is within a predetermined tolerance;
3. After changing the parallelism configuration and gradient accumulation, check the effective global batch, reduction method, gradient clipping order, and precision differences;
4. Check whether optimizer state, scheduler, random state, and data loading position are correctly restored when resuming training.

Floating-point reduction introduces numerical differences, so an error tolerance or statistical tolerance should be defined rather than requiring bitwise identity. Repeating or skipping data after resumption, resetting momentum, and schedule shifts must all be recorded in the experiment log. When a new kernel or target-precision path has not undergone the above checks, the applicability of the Ladder conclusions must be restricted. [DeepSeek-V3, 2024](https://arxiv.org/abs/2412.19437) §3.3 sets precision separately for computation, accumulation, and optimizer state; [Llama 3](https://arxiv.org/abs/2407.21783) §3.3.4 describes the fault-recovery requirements in a production training system.

### 10.5 Realized Efficiency and Deployment Constraints

An improvement in theoretical FLOPs does not necessarily yield a training-time improvement of the same magnitude; actual training time must be reported as well ([Marin follow-up](https://openathena.ai/blog/pretraining-speedup/) distinguishes theoretical efficiency from realized efficiency). Precision schemes need to be validated and must not be assumed equivalent ([Nemotron 3 Ultra](https://arxiv.org/abs/2606.15007)).

When selecting a production configuration, the following quantities should be treated as constraints (our recommendation):

- Training side: throughput on the target hardware, parallelism feasibility, memory, effective training time, and the overhead of evaluation, checkpointing, fault recovery, and reruns; for MoE, additionally account for the communication overhead of expert parallelism. When the delivery deadline is a constraint, estimate resource availability separately.
- Deployment side: given the target workload, specify the input and output length distributions, concurrency, latency, and memory constraints, and measure using the actual precision and inference implementation. [Sardana et al., 2024](https://arxiv.org/abs/2401.00448) models the costs of training, input processing, and output generation separately, and its cost-optimal size must be recomputed using one's own measurements.

A research-oriented Ladder that does not involve deployment may omit the deployment side. The deliverable is a comparison table of quality, cost, and constraints under the target hardware and workload, along with the computation process by which the model size and training volume are selected from it.

### 10.6 Medium-Scale Trial Run

When the target scale far exceeds the largest holdout, run a medium-scale trial before launching target training (our recommendation):

- Use the target recipe, target infrastructure, target parallelism scheme, and target precision scheme;
- Freeze the predicted loss curve and its interval for that scale before the run;
- The pass condition is that the loss curve falls within the predicted interval and the stability diagnostics ([§10.3](#103-stability-stress-testing)) show no persistent anomalies;
- If it fails, handle it according to [§12.2](#122-handling-deviations-during-runs) and do not launch target training until the cause is identified.

The scale of the trial run is determined by the budget and the extrapolation multiplier; there is no universal ratio in the public literature.

## 11. Specialized Ladders

### 11.1 Staged Ladder

Different training stages have different data distributions, sequence lengths, and schedules, and the scaling law coefficients may change across stages ([Qwen3, 2025](https://arxiv.org/abs/2505.09388); [OLMo 3, 2025](https://arxiv.org/abs/2512.13961)). For stages whose configuration needs to be predicted or selected, separate Ladders can be built:

1. Pretraining Ladder: start from random initialization, fit $N$, $D$, LR, and BSZ;
2. Mid-training Ladder: start from the corresponding pretraining checkpoint, fit the added token count and LR schedule;
3. Long-context Ladder: start from the corresponding mid-training checkpoint, search over sequence length, RoPE configuration, and LR ([§11.2](#112-long-context-ladder)).

Do not directly reuse scaling law coefficients across stages, and simultaneously record changes in base capabilities and increments in target capabilities.

Checkpoints with similar total loss may differ in domain capability, seen data, optimizer state, and recent LR trajectory. When one checkpoint is fixed to compare subsequent recipes, the conclusions are conditional on that starting point; when extrapolating to other starting points, retain cross-comparisons of representative starting points. Also record (our recommendation):

- Whether stage boundaries reset the optimizer, whether to re-warmup, which data continues to be repeated;
- The allocation of budget across pretraining, mid-training, and long-context stages;
- The confirmation results on end-to-end delivery metrics after combining the locally optimal choices of each stage ([§10.2](#102-combined-validation)).

### 11.2 Long-Context Ladder

The long-context Ladder needs to specify the length distribution for training and evaluation, the position of key information in the sequence, and the cross-document packing method, and simultaneously evaluate whether short-context capabilities degrade and how compute changes. [RULER (Hsieh et al., 2024)](https://arxiv.org/abs/2404.06654v3) shows that passing simple retrieval tests does not imply equivalent capability on tasks such as multi-hop tracing and aggregation; the evaluation set should cover multiple task types.

## 12. Execution Process and Deliverables

### 12.1 Experiment Records

Each run must have a unique identifier and be associated with: code version, configuration, data manifest, tokenizer, evaluation version, initialization and data random seeds, parent checkpoint and shared prefix, hardware and precision scheme.

Run status is divided into: completed, actively stopped, algorithmic anomaly, infrastructure failure, implementation error. A run with a changed configuration gets a new record, and state whether old results are still usable for the current fitting.

### 12.2 Handling Deviations During Runs

When delivering the Ladder, deliver the training trajectories and stage configurations of each experimental point at the same time; when target training deviates from predictions, investigate accordingly. Before running, specify: the evaluation data used for comparison, the prediction interval aligned by training progress, the window for determining sustained deviation, and the diagnostic quantities to save.

A single excursion outside the interval may come from evaluation noise and cannot be directly attributed to scaling law failure. Handling order (our recommendation): first recheck measurements and configuration, then check the data flow, implementation, and hardware, and finally determine whether the recipe needs modification and recalibration. While the cause is unknown, keep the status "cause unknown".

Independent acceptance should be able to regenerate the target predictions from the raw experiment tables, and verify the freezing time of predictions against actual results.

### 12.3 Ladder Maintenance

The Ladder needs continuous maintenance as recipes and infrastructure change (our recommendation):

- Regression Ladder: keep a fixed set of reference configurations, rerun after changes to code, kernels, parallelism, or clusters, and check whether loss is within seed variance;
- Coefficient versions: archive fitted coefficients together with the corresponding recipe version, data version, and fitting script version;
- Refitting conditions: when architecture, optimizer, data version, precision scheme, or tokenizer change, first verify the transferability of the original coefficients at small scale, and when the deviation exceeds the threshold, perform local calibration or full refitting.

### 12.4 Checklist

Design phase:

- [ ] Clarify the applicable scenario ([§1.2](#12-typical-scenarios)), decision objective ([§2.1](#21-decision-objectives)), evaluation protocol ([§2.2](#22-evaluation-protocol))
- [ ] Each metric has a distinguishable signal at the Ladder scale; when delivering a post-training model, arrange SFT/RL acceptance ([§2.2](#22-evaluation-protocol))
- [ ] Determine acceptance thresholds and decision rules before observing the holdout, expressed as equivalent compute multiplier ([§2.3](#23-acceptance-thresholds-and-decision-rules))
- [ ] Distinguish fixed quantities, scaling rules, and experimental independent variables, and specify the warmup scaling rule ([§4.1](#41-variable-classification))
- [ ] Determine the number of sizes and extrapolation multipliers, and set multiple holdout levels ([§5.2](#52-number-of-sizes-span-and-extrapolation-multiplier))
- [ ] Training volume levels cover the target TPP; when data is limited, write in $U$ and repetition count ([§5.3](#53-training-volume-tiers), [§7.3](#73-data-constrained-training-and-repetition))
- [ ] Formulate an experiment plan including trials, search, fitting, acceptance, and reserve budget ([§5.5](#55-budget-allocation-and-follow-up-experiments))
- [ ] MoE: record total parameters, active parameters, and routing configuration, design scale, sparsity, and granularity experiments separately, and set a dense control ([§3.1](#31-parameter-count-definition), [§5.6](#56-moe-experimental-axes))
- [ ] Fix the data version ([§7.1](#71-data-quality-and-data-source-evaluation)); the statistical definitions of deduplication scope and repetition count are consistent ([§7.3](#73-data-constrained-training-and-repetition))
- [ ] When the data mixture is undetermined, arrange mixture experiments and compare with proportional sampling and the UniMax baseline ([§7.2](#72-data-mixture-ladder)); design experiments separately for multi-stage training ([§11.1](#111-staged-ladder))

Configuration phase:

- [ ] Record $N_{\text{body}}$ and $C$ separately, and count FLOPs according to the actual architecture and kernel ([§3.1](#31-parameter-count-definition), [§3.2](#32-compute))
- [ ] Fix the loss, token, and evaluation conventions, and write out counting and aggregation examples ([§3.3](#33-loss-and-token-conventions))
- [ ] Keep architectural attributes and scaling rules consistent within a model family, and fix the MoE routing rule ([§4.2](#42-architecture-and-training-configuration-consistency), [§4.4](#44-moe-structural-scaling-rules))
- [ ] Use the target training precision scheme; when comparing different vocabularies, use byte-normalized metrics ([§4.5](#45-vocabulary-and-numerical-precision))

Hyperparameter search stage:

- [ ] Objectives 2–4: operational definition of searching small models up to the Fully-Tuned Frontier; Objective 1: run according to the recipe ([§6.1](#61-hyperparameter-search-objective-and-near-optimal-region))
- [ ] Recalibrate the hyperparameter rules after changing the optimizer or parameterization ([§6.2](#62-parameterization-and-optimizer-transfer))
- [ ] BSZ is measured in tokens; use $B_{opt}\propto D^{0.4}$ as a prior and verify it; check the effect of batch size ramp ([§6.3](#63-scaling-law-for-lr-and-bsz))
- [ ] Reproduction recipes follow their WD; when studying across TPP, check the effect of WD ([§6.4](#64-weight-decay))
- [ ] Record search coverage, re-verify the final candidate with a new seed, and record failed trials by category ([§6.7](#67-search-procedure-and-stopping-rules))

Fitting stage:

- [ ] Check the FLOPs and parameter count conventions and the warmup rule ([§8](#8-loss-scaling-law-fitting))
- [ ] Compare candidate functional forms ([§8.1](#81-functional-form)); for multi-epoch training, specify unique tokens and repetition count ([§7.3](#73-data-constrained-training-and-repetition))
- [ ] Determine the truncation rule in advance ([§5.4](#54-intermediate-checkpoints-random-seeds-and-shared-trajectories))
- [ ] Fix settings according to the fitting protocol, perform bootstrap according to the dependency structure, and re-verify the decision conclusions ([§8.3](#83-fitting-protocol))
- [ ] Check residual structure, correlations, and sources of uncertainty; when the error is too large, record experiments to be added ([§8.4](#84-fitting-diagnostics-and-failure-handling))

Validation stage:

- [ ] Holdout acceptance according to the predetermined threshold ([§10.1](#101-holdout-validation))
- [ ] Downstream capability decisions require validating task metric predictions ([§9](#9-downstream-task-prediction))
- [ ] After individual changes pass, validate the final combination ([§10.2](#102-combined-validation))
- [ ] Perform stability stress tests after architecture or optimizer changes ([§10.3](#103-stability-stress-testing))
- [ ] Validate consistency between the research implementation and the target implementation ([§10.4](#104-implementation-consistency-validation))
- [ ] Report theoretical FLOPs, actual training time, and deployment constraints simultaneously ([§10.5](#105-realized-efficiency-and-deployment-constraints))
- [ ] When the extrapolation multiplier is large, run a medium-scale trial ([§10.6](#106-medium-scale-trial-run))
- [ ] Save experiment records, fitting coefficient versions, prediction intervals, and unresolved issues ([§12](#12-execution-process-and-deliverables))

## Appendix A. Public Ladder Configurations

### A.1 Public Scale Configurations

Scales and training volumes of the Ladder scans (parameter counts follow each source's convention):

| Source | Model parameter count | Training volume | Reference |
|---|---|---|---|
| OpenAI | Multiple sizes, max 1.5B (non-embedding parameters) | 22M–23B tokens | [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) |
| DeepMind Chinchilla | 70M–16B (400+ models) | 5B–500B tokens | [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) |
| StepFun | Multiple sizes (3,700+ models) | 100T tokens cumulative across all experiments | [Step Law (Li et al., 2025)](https://arxiv.org/abs/2503.04715) |
| Gadre et al. | 11M–6.9B (104 models) | Up to 32× Chinchilla ratio | [Gadre et al., 2024](https://arxiv.org/abs/2403.08540) |
| Fantastic Optimizers | 0.1B–1.2B (4 sizes) | 1×–8× Chinchilla ratio | [Wen et al., 2025](https://arxiv.org/abs/2509.02046) |
| Delphi | Scan size and training volume by compute budget, max holdout 25B | Fit 3e18–3e20 FLOPs, holdout up to 1e23 FLOPs | [Marin, 2026](https://openathena.ai/blog/delphi/) |
| Llama 3 | 40M–16B | Fit 6e18–1e22 FLOPs, target 3.8e25 FLOPs | [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §3.2.1 |

Training volumes of individual released models, as a reference for target TPP (not Ladder scans):

| Source | Model parameter count | Training volume | Reference |
|---|---|---|---|
| NVIDIA Nemotron | 15B, 340B | 8T, 9T tokens | [15B report](https://arxiv.org/abs/2402.16819); [340B report](https://arxiv.org/abs/2406.11704) |
| OLMo | 1B, 7B, 13B, 32B | OLMo 1: 2T–2.46T; OLMo 2: multi-stage budget set per model | [OLMo, 2024](https://arxiv.org/abs/2402.00838v4); [OLMo 2, 2025](https://arxiv.org/abs/2501.00656v3) |
| Llama 3 | 8B, 70B, 405B | 405B: 15.6T tokens; 8B and 70B use similar recipes, with training duration far exceeding compute-optimal | [Llama 3, 2024](https://arxiv.org/abs/2407.21783) §1, §3.4 |

### A.2 Fantastic Optimizers Ladder

[Fantastic Optimizers (Wen et al., 2025)](https://arxiv.org/abs/2509.02046v2) Table 2–3. Dense, Llama 2 architecture, all four sizes fixed at 32 layers, MHA, seq_len 4096. The goal is a fair comparison of optimizers, with emphasis on hyperparameter search.

| Size | hidden_dim | inter_dim | heads | Data ratio |
|---|---|---|---|---|
| 130M | 512 | 2,048 | 8 | 1×–8× Chinchilla |
| 300M | 768 | 3,072 | 12 | Same as above |
| 520M | 1,024 | 4,096 | 16 | Same as above |
| 1.2B | 1,536 | 6,144 | 24 | Same as above |

Table 3 gives a hyperparameter search example for AdamW: Peak LR 8e-3, WD 0.1, warmup 2000 steps, BSZ 128 sequences (seq_len 4096, about 0.5M tokens). This result corresponds to a specific size and ratio; configurations differ across sizes, e.g., 520M/1× uses WD 0.2, BSZ 256. The data is a mixture of DCLM-baseline, StarCoder V2 Data, and ProofPile 2.

### A.3 Delphi Ladder

[Delphi (Marin, 2026)](https://openathena.ai/blog/delphi/). Dense decoder-only, Qwen 3 architecture, MLP ratio 4, seq_len 4096. The goal is to fit an IsoFLOP scaling law and extrapolate to 1e23 FLOPs (25B), with emphasis on end-to-end loss prediction. Later extended to MoE ([535B-A23B](https://openathena.ai/blog/pretraining-speedup/)).

Common settings: AdamH, WSD (10% warmup, 20% decay to 0), f32 parameters with bf16 compute, FSDP. Data is Nemotron-CC, StarCoderData, and ProofPile 2.

Structure: perform an IsoFLOP scan over 3e18–3e20 FLOPs, take the 7 optimal points for fitting; holdout is 1e21–1e23 FLOPs (3×–333× extrapolation). Hyperparameters are set according to recipe rules, without manual per-point search.

### A.4 Selection Recommendations

- When you need to search for optimal hyperparameters and fit a hyperparameter scaling law, refer to the grid design of Fantastic Optimizers.
- When you need end-to-end loss prediction and extrapolation to large scale, refer to the IsoFLOP layout and recipe formulas of Delphi.
- Both configurations are dense starting points; a MoE Ladder must be designed separately per [§5.6](#56-moe-experimental-axes). The model family and architecture scaling rule should have proxy validity for the target model ([§4.2](#42-architecture-and-training-configuration-consistency)).

## References

Grouped by topic, with the corresponding section in the main text noted after each entry.

### Scaling Law Foundations, Functional Forms, and Fitting

1. Scaling Laws for Neural Language Models — Kaplan et al., OpenAI, 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)  
   Empirical power laws spanning 7 orders of magnitude are observed within the measured range; given the parameter count, the width-depth ratio has little effect on loss. See [§3.1](#31-parameter-count-definition), [§4.3](#43-width-depth-configuration)
2. Training Compute-Optimal Large Language Models (Chinchilla) — Hoffmann et al., DeepMind, 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)  
   The IsoFLOP method and results showing that model and data scale at roughly equal rates; additive form $L=E+A/N^\alpha+B/D^\beta$; the parameter count convention of 20 TPP. See [§3.1](#31-parameter-count-definition), [§5.3](#53-training-volume-tiers), [§8.1](#81-functional-form)
3. Language models scale reliably with over-training and on downstream tasks — Gadre et al., 2024. [arXiv:2403.08540](https://arxiv.org/abs/2403.08540)  
   Scaling laws validated on 104 models in the over-trained regime; power-law exponents are close across different $D/N$. See [§5.3](#53-training-volume-tiers), [§9.1](#91-method-routes)
4. Skaling: Chinchilla's Exponents Meet Kaplan's Coupling — Videau et al., FAIR at Meta, 2026. [arXiv:2608.07222](https://arxiv.org/abs/2608.07222)  
   Negative mixed partial derivatives are observed on the analyzed data; an outer exponent $k$ and L-shape sampling are proposed, and their extrapolation performance is validated. See [§8.1](#81-functional-form)
5. Predictable Scale: Part II, Farseer: A Refined Scaling Law in Large Language Models — Li et al., StepFun & Fudan, NeurIPS 2025. [arXiv:2506.10972](https://arxiv.org/abs/2506.10972)  
   A nine-parameter form in which the data-side coefficients and exponents explicitly depend on $N$; ablation on the parameter count convention. See [§3.1](#31-parameter-count-definition), [§8.1](#81-functional-form)
6. Scaling Law with Learning Rate Annealing — Tissue et al., 2024. [arXiv:2408.11029](https://arxiv.org/abs/2408.11029)  
   Expresses loss as a function of the cumulative learning rate area and the annealing amount, allowing the entire curve to be fitted. See [§6.5](#65-lr-schedule), [§8.2](#82-loss-curves-and-annealing-scaling-law)
7. A Hitchhiker's Guide to Scaling Law Estimation — Choshen et al., MIT/IBM, ICML 2025. [arXiv:2410.11840](https://arxiv.org/abs/2410.11840)  
   The use of intermediate checkpoints and early truncation; the value of the number of sizes, extrapolation span, and random seeds; the minimum meaningful difference in the literature. See [§2.3](#23-acceptance-thresholds-and-decision-rules), [§5.2](#52-number-of-sizes-span-and-extrapolation-multiplier), [§5.4](#54-intermediate-checkpoints-random-seeds-and-shared-trajectories), [§5.5](#55-budget-allocation-and-follow-up-experiments)
8. Resolving Discrepancies in Compute-Optimal Scaling of Language Models — Porian et al., NeurIPS 2024. [arXiv:2406.19146](https://arxiv.org/abs/2406.19146)  
   Analyzes the effect of the output head, warmup, and hyperparameter tuning on discrepancies in compute-optimal exponents. See [§3.1](#31-parameter-count-definition), [§4.1](#41-variable-classification), [§8](#8-loss-scaling-law-fitting)
9. Chinchilla Scaling: A Replication Attempt — Besiroglu et al., Epoch AI, 2024. [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)  
   Reproduces Chinchilla's parametric fit and points out issues with the fitting and confidence interval setup. See [§8.3](#83-fitting-protocol)
10. (Mis)Fitting: A Survey of Scaling Laws — Li et al., 2025. [arXiv:2502.18969](https://arxiv.org/abs/2502.18969)  
   Discusses the impact of missing fitting details on reproducibility and conclusions. See [§8.3](#83-fitting-protocol)
11. Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws — Sardana et al., 2024. [arXiv:2401.00448](https://arxiv.org/abs/2401.00448)  
   Incorporates inference cost into resource allocation, separately modeling the costs of training, input processing, and output generation; over-training experiments up to 10,000 TPP. See [§5.3](#53-training-volume-tiers), [§10.5](#105-realized-efficiency-and-deployment-constraints)
12. Gemstones: A Model Suite for Multi-Faceted Scaling Laws — McLeish et al., 2025. [arXiv:2502.06857](https://arxiv.org/abs/2502.06857)  
   The effect of width-depth configuration on benchmarks; the effect of experimental point selection on resource allocation recommendations. See [§2.3](#23-acceptance-thresholds-and-decision-rules), [§4.3](#43-width-depth-configuration)
13. Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies — Tao et al., NeurIPS 2024. [arXiv:2407.13623](https://arxiv.org/abs/2407.13623)  
   The optimal vocabulary size grows with compute. See [§4.5](#45-vocabulary-and-numerical-precision)
14. Scaling Laws for Precision — Kumar et al., 2024. [arXiv:2411.04330](https://arxiv.org/abs/2411.04330)  
   The effect of training precision and post-training quantization on loss. See [§4.5](#45-vocabulary-and-numerical-precision)

### Hyperparameter Scaling Laws and Optimizers

{:start="15"}
15. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism — DeepSeek, 2024. [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)  
   Fits hyperparameter power laws as a function of compute $C$; the optimal resource allocation differs across the corpora tested. See [§6.3](#63-scaling-law-for-lr-and-bsz), [§7.1](#71-data-quality-and-data-source-evaluation)
16. Predictable Scale: Part I, Step Law — Optimal Hyperparameter Scaling Law in Large Language Model Pre-training — Li et al., StepFun, 2025. [arXiv:2503.04715v3](https://arxiv.org/abs/2503.04715v3)  
   $\eta_{opt}=c\,N^{-\alpha}D^{\beta}$, $B_{opt}=d\,D^{\gamma}$ validated on 3,700+ models. See [§6.1](#61-hyperparameter-search-objective-and-near-optimal-region), [§6.3](#63-scaling-law-for-lr-and-bsz)
17. Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training — Bergsma et al., Cerebras, 2025. [arXiv:2505.13738](https://arxiv.org/abs/2505.13738)  
   Under the recipes tested, obtains $B_{opt}\propto D^{0.4}$, $B_{crit}\propto D_{min}^{0.5}$, with weak dependence on $N$; the timescale $\tau=B/(\eta\lambda D)$ is used for joint search. See [§6.2](#62-parameterization-and-optimizer-transfer), [§6.3](#63-scaling-law-for-lr-and-bsz), [§6.4](#64-weight-decay)
18. Scaling Optimal LR Across Token Horizons — Bjorck et al., Microsoft, 2024. [arXiv:2409.19913](https://arxiv.org/abs/2409.19913)  
   With model size and BSZ fixed, peak LR decays with training length. See [§6.3](#63-scaling-law-for-lr-and-bsz)
19. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer ($\mu$-Transfer) — Yang et al., Microsoft, 2022. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)  
   Width-direction hyperparameter transfer under $\mu$P. See [§6.2](#62-parameterization-and-optimizer-transfer)
20. Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks — Yang et al., 2023. [arXiv:2310.02244](https://arxiv.org/abs/2310.02244)  
   Depth-$\mu$P with one layer per residual block; limitations of infinite-depth parameterization when residual blocks contain multiple layers. See [§6.2](#62-parameterization-and-optimizer-transfer)
21. Depthwise Hyperparameter Transfer in Residual Networks: Dynamics and Scaling Limit — Bordelon et al., 2023. [arXiv:2309.16620](https://arxiv.org/abs/2309.16620)  
   Scales the residual branch by $1/\sqrt{\text{depth}}$, observing hyperparameter transfer across depths on ResNet and ViT. See [§6.2](#62-parameterization-and-optimizer-transfer)
22. Muon is Scalable for LLM Training — Liu et al., Moonshot AI, 2025. [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)  
   Adds weight decay to Muon and matches the update RMS, reusing AdamW's hyperparameter settings. See [§6.2](#62-parameterization-and-optimizer-transfer)
23. Fantastic Pretraining Optimizers and Where to Find Them — Wen et al., Stanford, 2025. [arXiv:2509.02046](https://arxiv.org/abs/2509.02046)  
   A fair tuning benchmark across four sizes and the 1×–8× Chinchilla range; token efficiency gains vary with size; ranking flips during annealing. See [§5.3](#53-training-volume-tiers), [§6.8](#68-recipe-comparison), [Appendix A.2](#a2-fantastic-optimizers-ladder)
24. Weight Decay Improves Language Model Plasticity — Han et al., 2026. [arXiv:2602.11137](https://arxiv.org/abs/2602.11137)  
   The WD preferred by pretraining loss decreases as TPP grows; post-training plasticity may benefit from stronger WD. See [§6.4](#64-weight-decay)
25. Small-Scale Experiments: Are We There Yet? — Lourie et al., NYU & Meta, 2026. [arXiv:2608.11859](https://arxiv.org/abs/2608.11859)  
   Small models' sensitivity to hyperparameters and the Fully-Tuned Frontier; sizes are partitioned into fitting, validation, and test segments. See [§5.4](#54-intermediate-checkpoints-random-seeds-and-shared-trajectories), [§6.1](#61-hyperparameter-search-objective-and-near-optimal-region)
26. How to Allocate Your Tokens? Scaling Laws with Training Steps and Batch Size — Schaipp, Inria, 2026. [arXiv:2607.01487](https://arxiv.org/abs/2607.01487)  
   Defines a near-optimal batch interval in terms of compute loss, roughly 4× the width tested. See [§6.3](#63-scaling-law-for-lr-and-bsz)
27. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation — Cawley & Talbot, JMLR 2010. [JMLR 11](https://www.jmlr.org/papers/v11/cawley10a.html)  
   Selection bias from model selection on finite samples. See [§6.7](#67-search-procedure-and-stopping-rules)

### MoE

{:start="28"}
28. Scaling Laws for Fine-Grained Mixture of Experts — Krajewski et al., 2024. [arXiv:2402.07871](https://arxiv.org/abs/2402.07871)  
   Incorporates expert granularity into the scaling law. See [§4.4](#44-moe-structural-scaling-rules), [§5.6](#56-moe-experimental-axes), [§8.5](#85-moe-fitting)
29. Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models — Abnar et al., Apple, 2025. [arXiv:2501.12370](https://arxiv.org/abs/2501.12370)  
   The relationship between sparsity and pretraining loss at fixed compute; the effect of sparsity on downstream transfer. See [§4.4](#44-moe-structural-scaling-rules)
30. Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient — Ludziejewski et al., 2025. [arXiv:2502.05172](https://arxiv.org/abs/2502.05172)  
   Jointly models the number of experts, active parameter count, and training volume, incorporating memory constraints. See [§4.4](#44-moe-structural-scaling-rules), [§8.5](#85-moe-fitting)

### Data Construction, Mixture, and Repetition

{:start="31"}
31. Scaling Data-Constrained Language Models — Muennighoff et al., 2023. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)  
   The equivalent token count formula for repeated data and the diminishing marginal returns curve. See [§7.3](#73-data-constrained-training-and-repetition)
32. To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis — Xue et al., 2023. [arXiv:2305.13230](https://arxiv.org/abs/2305.13230)  
   The effect of parameter count, data volume, and quality on repetition overfitting; the role of dropout. See [§7.3](#73-data-constrained-training-and-repetition)
33. Prescriptive Scaling Laws for Data Constrained Training — Lovelace et al., 2026. [arXiv:2605.01640](https://arxiv.org/abs/2605.01640)  
   Overfitting jointly affected by parameter count, unique tokens, and repetition count. See [§7.3](#73-data-constrained-training-and-repetition)
34. Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression — Yan et al., 2025. [arXiv:2511.13421](https://arxiv.org/abs/2511.13421)  
   The relationship between repetition count and sample size under linear regression assumptions, with LLM experiments provided. See [§7.3](#73-data-constrained-training-and-repetition)
35. UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining — Chung et al., ICLR 2023. [arXiv:2304.09151](https://arxiv.org/abs/2304.09151)  
   A sampling method that caps the maximum repetition count per corpus. See [§7.2](#72-data-mixture-ladder)
36. Olmix: A Framework for Data Mixing Throughout LM Development — Chen et al., Allen Institute, ICLR 2026. [arXiv:2602.12237](https://arxiv.org/abs/2602.12237)  
   Design choices for mixture proxy experiments (proxy size, number of proxies, sampling distribution, regression model and granularity, repetition constraints, solver); mixture reuse after domain updates. See [§7.2](#72-data-mixture-ladder)
37. Scaling Laws for Mixture Pretraining Under Data Constraints — Sedova et al., Apple, 2026. [arXiv:2605.12715](https://arxiv.org/abs/2605.12715)  
   Repetition count in mixture training with scarce data; a mixture scaling law that includes a repetition term. See [§7.3](#73-data-constrained-training-and-repetition)
38. Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training (DeMix) — Li et al., 2026. [arXiv:2602.00747](https://arxiv.org/abs/2602.00747)  
   Replaces training a proxy at a given mixture with weighted merging of component models. See [§7.2](#72-data-mixture-ladder)
39. Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset — Su et al., NVIDIA, 2024. [arXiv:2412.02595](https://arxiv.org/abs/2412.02595)  
   The Common Crawl dataset and its synthetic rewritten version; the largest cross-source overlap in Marin's global deduplication. See [§7.3](#73-data-constrained-training-and-repetition)

### LR Schedule and Training Wrap-up

{:start="40"}
40. Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective — Wen et al., 2024. [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)  
   Explains the stable and decay phases of WSD under specific assumptions about loss geometry and optimization dynamics. See [§6.5](#65-lr-schedule)
41. Scaling and Transferability of Annealing Strategies in Large Language Model Training — Wang et al., 2025. [arXiv:2512.13705](https://arxiv.org/abs/2512.13705)  
   Scaling and cross-scale transferability of annealing strategies. See [§6.5](#65-lr-schedule)
42. Model Merging in Pre-training of Large Language Models — ByteDance Seed, 2025. [arXiv:2505.12082](https://arxiv.org/abs/2505.12082)  
   Checkpoint merging in pretraining; comparison of PMA during the stable phase with the performance at the annealing endpoint. See [§6.9](#69-training-wrap-up)
43. Model Soups: Averaging Weights of Multiple Fine-tuned Models — Wortsman et al., 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)  
   Weight averaging of models obtained by independent fine-tuning from a shared pretraining starting point. See [§6.9](#69-training-wrap-up)
44. Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging (LAWA) — Kaddour, 2022. [arXiv:2209.14981](https://arxiv.org/abs/2209.14981)  
   Sliding-window weight averaging along a single trajectory. See [§6.9](#69-training-wrap-up)
45. Early Weight Averaging meets High Learning Rates for LLM Pre-training — Sanyal et al., COLM 2024. [arXiv:2306.03241](https://arxiv.org/abs/2306.03241)  
   Early weight averaging under high learning rates. See [§6.9](#69-training-wrap-up)

### Downstream Task Prediction and Evaluation

{:start="46"}
46. Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective (COD) — Xu et al., ICLR 2026, v4 (2026-03-09). [arXiv:2502.17262v4](https://arxiv.org/pdf/2502.17262v4)  
    Clustering by difficulty features, selecting predictable clusters, and mapping to the full set; includes dense and MoE target prediction and continued training experiments. See [§9.1](#91-method-routes), [§9.2](#92-cod-framework)
47. GPT-4 Technical Report — OpenAI, 2023. [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)  
    Bucketing HumanEval difficulty by small-model performance, then fitting and extrapolating on the subset. See [§9.1](#91-method-routes)
48. Establishing Task Scaling Laws via Compute-Efficient Model Ladders (OLMo Task Ladder) — Bhagia et al., Allen Institute, 2024. [arXiv:2412.04403](https://arxiv.org/abs/2412.04403)  
    Two-stage downstream prediction: fit task-specific loss from $N,D$, then fit loss to accuracy; noise varies markedly across tasks. See [§9.1](#91-method-routes)
49. Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive? — Schaeffer et al., 2024. [arXiv:2406.04391](https://arxiv.org/abs/2406.04391)  
    Multiple-choice accuracy depends on probability mass on incorrect options, weakening its statistical relationship with compute. See [§9.3](#93-limits-of-predictability)
50. RULER: What's the Real Context Size of Your Long-Context Language Models? — Hsieh et al., NVIDIA, 2024. [arXiv:2404.06654](https://arxiv.org/abs/2404.06654)  
    The gap between simple retrieval tests and performance on multiple long-context task types. See [§11.2](#112-long-context-ladder)

### Model Technical Reports and Ladder Examples

{:start="51"}
51. Delphi: An Open Scaling Suite from 3e18 to 1e23 FLOPs — Marin Team, 2026. [openathena.ai/blog/delphi](https://openathena.ai/blog/delphi/)  
    IsoFLOP fitting and recipe-formula-driven hyperparameters; multi-tier extrapolation holdout; performance competitiveness and predictability validated separately; bootstrap of the IsoFLOP optimum; sigmoid downstream mapping. See [§1.1](#11-definition-of-scaling-ladder), [§2.1](#21-decision-objectives), [§5.2](#52-number-of-sizes-span-and-extrapolation-multiplier), [§8.4](#84-fitting-diagnostics-and-failure-handling), [§9.1](#91-method-routes), [§10.1](#101-holdout-validation), [Appendix A.3](#a3-delphi-ladder)
52. The Llama 3 Herd of Models — Meta, 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)  
    IsoFLOP determines the 405B size; smaller models are overtrained; two-stage downstream prediction; batch size ramp; failure recovery in the training system. See [§5.2](#52-number-of-sizes-span-and-extrapolation-multiplier), [§5.3](#53-training-volume-tiers), [§6.3](#63-scaling-law-for-lr-and-bsz), [§9.1](#91-method-routes), [§10.4](#104-implementation-consistency-validation)
53. DeepSeek-V3 Technical Report — DeepSeek, 2024. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)  
    Auxiliary-loss-free load balancing without dropping tokens (§2.1.2); precision settings for compute, accumulation, and storage in FP8 training (§3.3). See [§4.4](#44-moe-structural-scaling-rules), [§10.4](#104-implementation-consistency-validation)
54. Nemotron-4 15B Technical Report — NVIDIA, 2024. [arXiv:2402.16819](https://arxiv.org/abs/2402.16819)  
    Batch size ramp; continued training at the end of training. See [§6.3](#63-scaling-law-for-lr-and-bsz), [§6.9](#69-training-wrap-up)
55. Nemotron-4 340B Technical Report — NVIDIA, 2024. [arXiv:2406.11704](https://arxiv.org/abs/2406.11704)  
    Training volume of the released model. See [Appendix A.1](#a1-public-scale-configurations)
56. OLMo: Accelerating the Science of Language Models — Groeneveld et al., Allen Institute, 2024. [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)  
    Fully open training data, code, and intermediate checkpoints. See [Appendix A.1](#a1-public-scale-configurations)
57. 2 OLMo 2 Furious — OLMo Team, Allen Institute, 2025. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)  
    Two-stage training; micro-annealing to validate data sources; model souping. See [§6.9](#69-training-wrap-up), [§7.1](#71-data-quality-and-data-source-evaluation)
58. OLMo 3 — Allen Institute, 2025. [arXiv:2512.13961](https://arxiv.org/abs/2512.13961)  
    Effective scale range of evaluation metrics; Olmix mixture pipeline and quality-aware upsampling; no decay for embeddings; staged training and combined validation. See [§2.2](#22-evaluation-protocol), [§6.4](#64-weight-decay), [§7.2](#72-data-mixture-ladder), [§7.3](#73-data-constrained-training-and-repetition), [§10.2](#102-combined-validation), [§11.1](#111-staged-ladder)
59. Qwen3 Technical Report — Qwen Team, 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)  
    Staged prediction of LR and batch size; WD scaling not reported; data mixture along instance-attribute dimensions. See [§6.4](#64-weight-decay), [§7.2](#72-data-mixture-ladder), [§11.1](#111-staged-ladder)
60. On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability — Qwen Team, 2026. [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)  
    Near-optimal region for large models; hyperparameter shift after architecture and optimizer changes; post-training acceptance; stability stress tests. See [§2.2](#22-evaluation-protocol), [§6.1](#61-hyperparameter-search-objective-and-near-optimal-region), [§6.2](#62-parameterization-and-optimizer-transfer), [§10.3](#103-stability-stress-testing)
61. Kimi K2: Open Agentic Intelligence — Moonshot AI, 2025. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)  
    Sparsity scaling law at fixed active scale; fixed WD recipe; controlled comparison of rewriting and repetition. See [§5.6](#56-moe-experimental-axes), [§6.4](#64-weight-decay), [§7.3](#73-data-constrained-training-and-repetition)
62. Kimi K2.5: Visual Agentic Intelligence — Moonshot AI, 2026. [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)  
    Controlling the maximum number of epochs per data source in joint pretraining. See [§7.3](#73-data-constrained-training-and-repetition)
63. Kimi K3: Open Frontier Intelligence — Moonshot AI, 2026. [arXiv:2607.24653](https://arxiv.org/abs/2607.24653)  
    Redoing the scaling law study after recipe changes; comparison of cosine and WSD after separate hyperparameter search; small-model ablations to determine domain sampling rates; reusing K2's rewriting method. See [§6.8](#68-recipe-comparison), [§7.1](#71-data-quality-and-data-source-evaluation), [§7.3](#73-data-constrained-training-and-repetition)
64. Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2026. [arXiv:2604.12374](https://arxiv.org/abs/2604.12374)  
    Two-stage mixture: the first 80% emphasizes diversity, the last 20% emphasizes high-quality data. See [§7.2](#72-data-mixture-ladder)
65. Nemotron 3 Ultra: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning — NVIDIA, 2026. [arXiv:2606.15007](https://arxiv.org/abs/2606.15007)  
    Comparison of late-training divergence against a high-precision branch; ablation of domain synthetic data. See [§7.1](#71-data-quality-and-data-source-evaluation), [§10.3](#103-stability-stress-testing), [§10.5](#105-realized-efficiency-and-deployment-constraints)
66. Marin: MoE and Training Efficiency Follow-up — Marin Team, 2026. [openathena.ai/blog/pretraining-speedup](https://openathena.ai/blog/pretraining-speedup/)  
    Distinguishing theoretical from realized efficiency; comparison across scales and design of combination experiments. See [§10.5](#105-realized-efficiency-and-deployment-constraints), [§10.2](#102-combined-validation)
67. Marin Data Pipeline — Marin Team, 2026. [openathena.ai/blog/marin-data-pipeline-overview](https://openathena.ai/blog/marin-data-pipeline-overview/)  
    Global deduplication and repetition counts; mixture baselines; scaling training volume and data pool proportionally in proxy experiments; confirmation across scales. See [§7.2](#72-data-mixture-ladder), [§7.3](#73-data-constrained-training-and-repetition)
68. Phi-4 Technical Report — Microsoft, 2024. [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)  
    Handling of high-noise evaluation tasks. See [§2.2](#22-evaluation-protocol)
69. GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models — Zhipu AI, 2025. [arXiv:2508.06471](https://arxiv.org/abs/2508.06471)  
    Effect of width-depth configuration on reasoning ability. See [§4.3](#43-width-depth-configuration)

### Further Reading

The following references are related to Ladder design but are not discussed in the main text:

{:start="70"}
70. Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training — Merrill et al., 2025. [arXiv:2505.23971](https://arxiv.org/abs/2505.23971)
71. An Empirical Model of Large-Batch Training — McCandlish et al., 2018. [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)
72. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al., Google, 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
73. Cost-Optimal Grouped-Query Attention for Long-Context Modeling — Chen et al., 2025. [arXiv:2503.09579](https://arxiv.org/abs/2503.09579)
