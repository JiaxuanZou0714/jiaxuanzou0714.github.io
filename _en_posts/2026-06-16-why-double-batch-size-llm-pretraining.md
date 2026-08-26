---
layout: post
title: "Why does the batch size need to be doubled midway through LLM pretraining?"
date: 2026-06-16 14:00:00
description: "Starting from the Double GBS phenomenon of Apertus 70B, we use the gradient noise scale, critical batch size, and calculus of variations to derive the optimal schedule for increasing the batch size midway through LLM pretraining, and validate it on the noisy quadratic model."
tags: [optimization, deep-learning, llm, scaling-law, batch-size]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/why-double-batch-size-llm-pretraining/
ref: why-double-batch-size-llm-pretraining
related_posts: false
---

The training loss curve of Apertus 70B has a vertical line labeled Double GBS: at approximately 4.4T tokens, the global batch size increases from 8.4M to 16.8M tokens, the learning rate remains unchanged, and the loss drops by a small segment.

{% include figure.liquid
  path='assets/img/post-06-16/image.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Apertus 70B loss curve. The red dashed line marks the mid-course Double GBS (global batch size doubled from 8.4M to 16.8M tokens), and the remaining vertical lines indicate data stage switches.'
  zoomable=true
  alt='Apertus 70B loss curve with a Double GBS line'
%}

This article discusses three questions: why the batch size increase occurs late in training, why the loss drops after the increase, and the optimal timing and magnitude of the increase, with numerical verification on the noisy quadratic model.

## 1. Motivation for increasing batch size late in training

Denote the training objective as

$$
L(\theta)=\mathbb E_{x}[\ell(\theta;x)],
$$

The mean of the per-sample gradient $$g(x)=\nabla_\theta \ell(\theta;x)$$ is the true gradient $$\mu=\nabla L(\theta)=\mathbb E[g(x)]$$, and the covariance is

$$
C=\mathbb E[(g(x)-\mu)(g(x)-\mu)^\top].
$$

The mini-batch gradient with batch size $$B$$

$$
\hat g_B=\frac{1}{B}\sum_{i=1}^{B} g(x_i)
$$

is unbiased, and the covariance decays as $$1/B$$:

$$
\operatorname{Cov}(\hat g_B)=\frac{C}{B}.
$$

That is, doubling the batch halves the gradient variance and reduces the standard deviation to $$1/\sqrt 2$$. Define the signal-to-noise ratio and the gradient noise scale

$$
\text{SNR}(B)=\frac{\|\mu\|^2}{\mathbb E\|\hat g_B-\mu\|^2}=\frac{B\|\mu\|^2}{\operatorname{tr}(C)},\qquad
\mathcal G=\frac{\operatorname{tr}(C)}{\|\mu\|^2},
$$

They satisfy $$\text{SNR}(B)=B/\mathcal G$$. Maintaining the signal-to-noise ratio requires $$B\propto \mathcal G$$. Late in training, as the model approaches the low-loss region, $$\|\mu\|$$ decreases while $$\operatorname{tr}(C)$$ does not decrease correspondingly, so $$\mathcal G$$ increases, and the required batch size increases accordingly. This is the first-layer reason why the batch size increase should occur late.

### 1.1 Single-step update analysis

A single SGD step is $$\theta^+=\theta-\eta \hat g_B$$. Expanding $$L(\theta^+)$$ to second order and taking the expectation over sampling,

$$
\mathbb E[L(\theta^+)]\approx L(\theta)-\eta \|\mu\|^2+\frac{\eta^2}{2}\mu^\top H\mu+\frac{\eta^2}{2B}\operatorname{tr}(HC),\qquad H=\nabla^2 L(\theta).
$$

Only the last term $$\frac{\eta^2}{2B}\operatorname{tr}(HC)$$ depends on $$B$$; it is the loss introduced by noise and is proportional to $$1/B$$. Requiring it not to exceed a fixed fraction of the effective descent term $$\eta\|\mu\|^2$$ yields the critical batch size

$$
B_{\text{crit}}\sim\frac{\eta\operatorname{tr}(HC)}{\|\mu\|^2}.
$$

Early in training, $$\|\mu\|^2$$ is large and $$B_{\text{crit}}$$ is small, so an overly large batch only reduces the number of update steps; late in training, $$\|\mu\|^2$$ decreases and $$B_{\text{crit}}$$ increases, and if a small batch is maintained, the loss is dominated by the noise term.

### 1.2 One-dimensional case

Take $$L(\theta)=\tfrac12\lambda\theta^2$$, stochastic gradient $$\hat g_B=\lambda\theta+\xi$$, $$\operatorname{Var}(\xi)=\sigma^2/B$$. From $$\theta_{k+1}=(1-\eta\lambda)\theta_k-\eta\xi_k$$ we obtain

$$
\mathbb E[L_{k+1}]=(1-\eta\lambda)^2\mathbb E[L_k]+\frac{\lambda\eta^2\sigma^2}{2B},
$$

The steady-state loss floor is

$$
L_\infty(B)\approx\frac{\eta\sigma^2}{4B},\qquad L_\infty(2B)\approx\tfrac12 L_\infty(B).
$$

That is, with a fixed learning rate, doubling the batch halves the noise floor; the drop at Double GBS in the figure corresponds to this floor moving down. Combining the above,

$$
\|\mu\|^2 \downarrow \;\Rightarrow\; \mathcal G=\frac{\operatorname{tr}(C)}{\|\mu\|^2}\uparrow \;\Rightarrow\; B_{\text{crit}}\uparrow \;\Rightarrow\; \text{增大 batch}.
$$

The cost is that under a fixed token budget, the number of update steps decreases, so an overly large batch should not be used early.

## 2. Is this practice common

Dynamically increasing the batch size during training (batch ramp / warmup) is a common configuration in large model pretraining, though it is rarely plotted separately in the loss curve. GPT-3's batch size increases linearly from 32k to full batch over the first 4–12B tokens; Llama 3 405B increases in stages from 4M to 8M and then to 16M; OLMo-65B starts at 2M and doubles every 100B tokens up to 16M. The prerequisite for its applicability is large models, synchronous data parallelism, many GPUs, and the importance of parallel efficiency; small models, LoRA/SFT, and CV training typically use a fixed batch size and only adjust the learning rate.

Whether to increase depends on whether the current batch size is close to the critical batch size. The gradient noise scale proposed by McCandlish et al. is used to estimate the "maximum useful batch size" and increases as the loss decreases; the OLMo CBS study also found that CBS rises rapidly early on and then plateaus. However, an overly large batch size hurts token efficiency: under a fixed budget, the number of optimizer steps decreases, and the loss worsens. The critical batch size is the trade-off point between data-parallel efficiency and token efficiency.

## 3. Optimal batch size schedule

One can further ask: what is the form of the optimal schedule, and can it be derived variationally like the optimal learning-rate schedule? The conclusion is that the continuous optimal solution is not linear but a monotonically accelerating clipped power-law; the discrete constraints of hardware approximate it as a few doublings.

### 3.1 Variational form

Let continuous time $$t$$ be the optimizer step, $$b(t)$$ be the batch size per step (i.e., the number of tokens consumed per step), with budget constraint $$\int_0^T b(t)\,dt=D$$. Under the functional scaling law (FSL) approximation, the excess loss decomposes as

$$
\mathcal E[T,b]=A\,T^{-s}+C\int_0^T \frac{K(T-t)}{b(t)}\,dt.
$$

The first term $$A\,T^{-s}$$ reflects signal learning from the number of optimizer steps, and the second term is the cumulative contribution of gradient noise. $$s>0$$ is the source exponent, and $$\beta>1$$ is the capacity exponent.

### 3.2 Fixed T, solve for b(t)

For fixed $$T$$, the optimization problem is

$$
\min_{b(t)}\int_0^T \frac{K(T-t)}{b(t)}\,dt,\qquad \text{s.t.}\ \int_0^T b(t)\,dt=D.
$$

By Cauchy–Schwarz,

$$
\left(\int_0^T \frac{K(T-t)}{b(t)}\,dt\right)\left(\int_0^T b(t)\,dt\right)\ge\left(\int_0^T \sqrt{K(T-t)}\,dt\right)^2,
$$

Equality holds if and only if $$b^*(t)\propto \sqrt{K(T-t)}$$. Substituting the FSL kernel $$K(\tau)\asymp (\tau+1)^{1/\beta-2}$$,

$$
b^*(t)\asymp c\,(T-t+1)^{\frac{1}{2\beta}-1}.
$$

The exponent $$\tfrac{1}{2\beta}-1<0$$, so $$b^*(t)$$ increases monotonically with $$t\to T$$: the continuous optimal solution is accelerating growth, not a linear ramp. Incorporating hardware upper and lower bounds yields the clipped power-law:

$$
b^*(t)=\operatorname{clip}\!\left(c\,(T-t+1)^{\frac{1}{2\beta}-1},\,B_{\min},\,B_{\max}\right).
$$

In practice, "maintain a small batch first, then jump to a large batch" is the discretization of this continuous solution; when only powers of 2 are allowed, it manifests as doublings.

### 3.3 Further optimizing T

Substituting the optimal $$b(t)$$ back,

$$
\mathcal E(T)\asymp A\,T^{-s}+C\,\frac{T^{1/\beta}}{D},
$$

The first-order condition gives

$$
T^*\asymp D^{\frac{\beta}{1+s\beta}},\qquad B_{\max}\asymp D^{\frac{1/2+s\beta}{1+s\beta}}.
$$

This leads to two regimes. For easy tasks ($$s>1-\tfrac1\beta$$), the optimal solution is a slowly increasing power-law throughout, with final loss rate $$\mathcal E_D^*\asymp D^{-\frac{s\beta}{1+s\beta}}$$. For hard tasks ($$s\le 1-\tfrac1\beta$$), the unconstrained solution tends to a very large $$T$$, but the batch size is constrained by a lower bound, so the optimal solution has two segments:

$$
b^*(t)=
\begin{cases}
B_{\min}, & 0\le t<T_1^*,\\[4pt]
B_{\max}(T^*-t+1)^{\frac{1}{2\beta}-1}, & T_1^*\le t\le T^*,
\end{cases}
$$

And the fraction of the growth segment decreases with $$D$$, $$\frac{T^*-T_1^*}{T^*}\asymp D^{-\frac{1-1/\beta-s}{2-1/\beta}}$$. Intuitively, in the hard regime, what is scarce early on is the number of optimizer steps, not low-noise gradients, so one should maintain a small batch for a long time to accumulate steps, and only later use a large batch to reduce noise. FSL calls this shape stable-growth, i.e., the batch-size version of WSD. LLM pretraining falls into this category.

This explains the "double GBS midway": the continuous optimal is monotonically rapidly increasing, but engineering constraints limit the number of available batch sizes, so on the curve it appears as one or two vertical jumps; a single doubling is an engineering approximation of the clipped power-law.

### 3.4 Explicit solutions for two-segment and multi-segment schedules

Restricting to two segments $$B_1\to B_2$$ ($$B_2>B_1$$):

$$
b(t)=
\begin{cases}
B_1,&0\le t<t_s,\\
B_2,&t_s\le t\le T,
\end{cases}
$$

Let $$D_1$$ tokens be consumed before the switch. Substituting back into the objective gives a one-dimensional problem in $$D_1$$, and the interior optimum satisfies

$$
A\,s\,S^{-s-1}=C\left[\frac{K(S)}{B_1}+\frac{K(R)}{B_2}\right],\qquad S=\frac{D_1}{B_1}+\frac{D-D_1}{B_2},\quad R=\frac{D-D_1}{B_2}.
$$

The left-hand side is the signal gain from extending the small batch to accumulate steps, and the right-hand side is the noise accumulation cost; the balance point is the switch. Multi-stage doubling is analogous: take $$B_j=B_{\min}r^j$$ and let each stage boundary fall on the continuous solution, yielding

$$
t_j=T+1-\left(\frac{B_j}{c}\right)^{1/p},\qquad p=\tfrac{1}{2\beta}-1,
$$

Converting to the token axis $$z_j=\int_0^{t_j} b^*(u)\,du$$ gives the timing of each doubling, then align to checkpoint or data stage boundaries.

### 3.5 Empirical Alternatives

The variational solution requires estimating $$s,\beta,K$$ in advance. A more practical approach is to track the critical batch size directly: OLMo's CBS study estimates CBS online, starting from a small batch and doubling once CBS increases, saving about 43% of update steps on OLMo 1B without loss degradation.

Note that "optimal" depends on the objective: final validation loss under a fixed budget, wall-clock to reach a target loss, or accounting for communication and utilization costs, correspond to different optimal $$b(t)$$. The FSL main analysis is based on vanilla SGD with constant learning rate; modern LLMs mostly use AdamW, and the joint learning-rate / batch-size schedule still requires further analysis.

## 4. Numerical Verification on NQM

We verify using the noisy quadratic model (NQM). NQM is a standard proxy model for large-batch theory and is the multidimensional generalization of the one-dimensional case in Section 1.2. Its expected loss satisfies an exact recurrence, requiring no Monte Carlo:

$$
v_{i,k+1}=(1-\eta h_i)^2 v_{i,k}+\frac{\eta^2\sigma_i^2}{B_k},\qquad L_k=\tfrac12\sum_i h_i v_{i,k}.
$$

Expanding to step $$T$$,

$$
L_T=\underbrace{S(T)}_{\text{信号项, 只依赖步数}}+\sum_k\frac{\kappa(T-1-k)}{B_k},\qquad \kappa(j)=\tfrac12\eta^2\sum_i h_i\sigma_i^2(1-\eta h_i)^{2j},
$$

This is isomorphic to the FSL form in Section 3. Hence, under a fixed budget $$D=\sum_k B_k$$, Cauchy–Schwarz directly gives $$B_k^*\propto\sqrt{\kappa(T-1-k)}$$, i.e., the clipped power-law is exact in NQM.

Taking a power-law spectrum, we measure $$s\approx0.49$$ and $$\beta\approx1.96$$, which lie in the hard regime, corresponding to LLMs. With fixed budget and constant learning rate, only the schedule changes; results are as follows.

{% include figure.liquid
  path='assets/img/post-06-16/batch_schedule_experiment.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Comparison under a fixed token budget and constant learning rate. Left: log-log overview of the loss; middle: linear token axis, where the differences among schedules and the sharp drop at the end are clearly visible; right: the corresponding batch schedules, where the optimal solution matches the analytic form $(T-t+1)^{1/2\beta-1}$ and the doubling staircase.'
  zoomable=true
  alt='loss curves and batch schedules for the NQM experiment'
%}

Main results:

- The optimal schedule (red) maintains $$B_{\min}$$ for most of training to accumulate steps, then increases batch size with a power law at the end, dropping loss to about $$1/10$$ of the constant-batch loss, consistent with WSD / optimal lr schedule curves.
- The optimal schedule matches the analytic form $$(T-t+1)^{1/2\beta-1}$$ (black dashed).
- Final loss under the same budget: constant 1.0×, two-stage 3.8×, doubling 9.9×, optimal 10.2×.
- The optimal solution's final loss is already below the noise floor $$L_\infty(B)$$ of the optimal constant batch, which is unattainable at any budget.

Two supporting results: the noise floor is exactly $$\propto 1/B$$, and the kernel $$\kappa(j)$$ is a power law.

{% include figure.liquid
  path='assets/img/post-06-16/batch_schedule_diagnostics.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Left: the noise loss floor is exactly $\propto 1/B$. Right: the noise kernel $\kappa(j)$ is a power-law in the middle segment, and its slope gives $\beta\approx1.96$.'
  zoomable=true
  alt='diagnostics: noise floor and kernel power law'
%}

Code is at `experiments/batch_schedule_nqm.py` (pure NumPy); adjusting the spectral index `S_SRC` and `C_NOISE` switches between easy / hard regimes.

## 5. Verification on a Real Transformer

NQM has an inherent weakness: its expected loss is deliberately constructed to be isomorphic to FSL, so "the optimal schedule wins in NQM" is nearly self-fulfilling. To break this loop, we redo the comparison on a real transformer: the model is no longer tailored to FSL.

Setup: a 45M-parameter standard GPT (6 layers, $$d=512$$, block 1024), trained on FineWeb (GPT-2 tokenizer) with AdamW, learning rate **constant throughout** (unchanged after 8M token linear warmup), fixed total budget **600M tokens**, same initialization and data order, **the only variable is the batch schedule**. Compare four schedules: constant batch 64k / 128k / 512k tokens, and a late-switch doubling ramp (64k→128k→256k→512k).

{% include figure.liquid
  path='assets/img/post-06-16/gpu_batch_schedule.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='45M GPT on FineWeb, under the same 600M token budget and constant lr. Left: val loss vs tokens (the doubling ramp, in red, is lowest throughout); middle: val loss vs optimizer steps (the ramp reaches a lower loss in fewer steps); right: the four batch schedules.'
  zoomable=true
  alt='real transformer batch schedule experiment on FineWeb'
%}

Results (same budget, same lr):

| schedule | final val loss | optimizer steps |
|---|---|---|
| constant 512k | 4.831 | 1145 |
| constant 128k | 4.219 | 4578 |
| constant 64k | 4.145 | 9156 |
| **doubling ramp** | **4.091** | 6182 |

The doubled ramp reaches the final loss of the strongest constant batch (64k) at step 5990 (consuming 520M tokens), **using about 35% fewer optimizer steps**, and continues to decline to 4.091, below all constant batches. This is consistent with OLMo's "43% fewer steps at the same loss" and reflects the mechanisms of §1 and §3 on a real AdamW transformer: small batches accumulate steps early, while large batches reduce noise later.

I must honestly note four points:

1. **The ramp switch points are chosen empirically** (the token proportions for each stage are hand-picked), guided only by the qualitative conclusion of §3 (hard regime → late switch), not by the exact discretization computed from $$z_j=\int_0^{t_j}b^*$$ in §3.4.
2. **Exact determination requires $$\beta$$, but $$\beta$$ is not identifiable here.** Jointly fitting the FSL across three constant-batch curves, the residual is nearly flat in $$\beta$$ (fixing $$\beta$$ at 4 or 8 leaves RMS unchanged), because all three curves are far from the noise floor (final loss is monotone in batch size, signal-dominated), while $$\beta$$ only manifests in the shape of approaching the floor. This limitation is **structural** in real LLM training: no one runs multiple constant-batch sweeps just to fit $$\beta$$, and even a single large training run almost always stops in the signal-dominated regime. This is precisely why §3.5 points to empirical schemes like tracking the critical batch size online.
3. **AdamW $$\neq$$ FSL of vanilla SGD.** Empirically, the gains in the large-batch final stage even exceed the fitted prediction, indicating that the noise–batch relationship under Adam is not fully consistent with SGD theory.
4. **Single seed, single run**, no error bars.

The conclusion of this section is therefore limited but clear: **the late-switch doubling schedule does outperform constant batch size on real transformers, validating the qualitative predictions of §3; however, the "optimal switch point" depends on a $$\beta$$ that is unavailable in practice, and the current ramp remains an empirical choice. The realistic direction for making it principled is not offline fitting, but online adaptation within a single training run (e.g., CBS tracking), which remains an open problem.**

Training code `experiments/bsched_gpt.py`, FSL fitting diagnostics `experiments/fit_fsl.py`.

## Summary

The essence of increasing the batch size later in training is that the noise term $$\frac{\eta^2}{2B}\operatorname{tr}(HC)$$ decays with $$1/B$$, while $$B_{\text{crit}}\sim \eta\operatorname{tr}(HC)/\|\mu\|^2$$ grows with training; with a fixed learning rate, increasing the batch size approximates a learning-rate decay and improves hardware utilization. This practice appears in GPT-3, Llama 3, and OLMo, though it is rarely plotted in the main figures. The optimal schedule is a clipped power-law, which in the hard regime and under discrete constraints for LLMs degenerates into "a long phase of small batch size followed by several doublings late in training"; Apertus's Double GBS is an engineering approximation of this. Both the NQM and a real 45M transformer confirm that this shape outperforms a constant batch size; however, the exact switching point depends on $$\beta$$, which is difficult to obtain in practice, and how to determine it adaptively within a single training run remains an open question.

## References

- Apertus Technical Report: [Apertus: Democratizing Open and Compliant LLMs for Global Language Environments](https://arxiv.org/abs/2509.14233) (source of the 70B loss curve and Double GBS)
- McCandlish, Kaplan, Amodei et al.: [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) (gradient noise scale)
- Smith, Kindermans, Le et al.: [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489)
- Brown et al.: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3's batch ramp)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- OLMo Team: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) (batch size warmup recipe)
- [Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](https://arxiv.org/abs/2505.23971) (batch warmup on OLMo 1B saves about 43% of gradient steps)
- Li, Wang, …, Wu: [Optimal Learning-Rate Schedules under Functional Scaling Laws: Power Decay and Warmup-Stable-Decay](https://arxiv.org/abs/2602.06797) (FSL framework and source / capacity exponents)
- Wang, Li, Zhou, …, Wu: [Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](https://arxiv.org/abs/2602.14208) (batch size version of FSL)

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026doublebatch,
  title={为什么 LLM pretrain 过程中途要把 batch size 翻倍},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/why-double-batch-size-llm-pretraining/}
}
```
