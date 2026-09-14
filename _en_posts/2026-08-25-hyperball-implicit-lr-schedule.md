---
source_sha: 94d74a3dda2b4402
layout: post
title: "Hyperball, effective lr, and the shape of peak-then-decay"
date: 2026-08-25 11:00:00
description: "Addressing two core questions in pretraining learning rate scheduling: the effective learning rate directly governing optimization is dynamically determined by weight norms, and Hyperball eliminates this implicit schedule by constraining weight norms; peak-then-decay corresponds to the optimal solution of the bias–variance trade-off, where shapes satisfying this balance form a set not limited to a specific analytic form."
tags: [deep-learning, lr-schedule, optimizer, spherical-dynamics, scaling-law]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/hyperball-implicit-lr-schedule/
ref: hyperball-implicit-lr-schedule
related_posts: false
---

## TL;DR

Recent pretraining optimization work highlights two empirical phenomena: Hyperball achieves a 20–30% token-equivalent speedup over standard weight decay baselines by constraining the Frobenius norm of weight matrices and their updates to constants [[8]](https://arxiv.org/abs/2606.16899); concurrently, optimal learning rate curves derived from diverse experimental setups consistently exhibit a peak-then-decay profile.

Peak-then-decay refers to a geometric trajectory where the learning rate rises to a peak $\eta_{\max}$ early in training, subsequently decays monotonically, and terminates close to zero, without restricting the decay segment to a specific analytic functional form.

{% include figure.liquid
  path='assets/img/post-08-25/peak_decay_shape.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='85%'
  caption='Examples of peak-then-decay shapes, where the decay segment takes the power-law form $\eta_t\propto(1-t/T)^{\alpha}$, with $\alpha$ equal to 0.5, 1, and 2. $\alpha=0.5$ corresponds to the sqrt shape, and $\alpha=1$ corresponds to linear decay to zero. The three curves have the same peak and endpoint, but the concavity of the decay segment differs.'
  zoomable=true
  alt='peak plus decay learning rate shape illustrated with three power-law exponents'
%}

The power-law formulation in the figure provides convenient parameterization, commonly adopted in manually designed empirical schedules. In the studies discussed below, the refined schedule is constructed pointwise and offline from gradient norm sequences, while the effective step size of Schedule-Free+ is implicitly induced via iterate averaging; both trajectories naturally emerge as peak-then-decay curves without relying on prescribed analytic formulas.

Recent studies indicate that both phenomena are governed by the same underlying physical quantity.

> ##### Core Takeaway
> **The primary quantity governing training progress is the effective learning rate $$\eta_t^\star=\eta_t\lVert U_t\rVert/\lVert W_t\rVert$$.**<br>Continuous growth of the weight norm causes the ratio between nominal and effective learning rates to evolve dynamically, superimposing an implicit decay onto the nominal schedule. By constraining the weight norm to a constant, Hyperball eliminates this implicit decay, functioning primarily as a state-dependent implicit learning rate schedule.
{: .block-tip}

The mechanism leading $\eta^\star$ toward a peak-then-decay profile is independent of the optimizer:

> ##### Shape Criterion
> **Peak-then-decay represents the optimal solution to the bias–variance trade-off.**<br>The peak phase rapidly reduces bias with large step sizes, while the final decay to zero suppresses the accumulation of stochastic gradient variance. The trajectories satisfying this optimal balance form a function set, and shapes within this set yield comparable final performance.
{: .block-tip}

These two judgments establish the central thesis: the effective learning rate $\eta^\star$ is the genuine physical quantity driving optimization, while the balance between bias and variance determines its geometric trajectory.

## 1. Effective Learning Rate Determines Optimization Progress

Parameters followed by normalization layers exhibit scale invariance $\mathcal{L}(\rho W) = \mathcal{L}(W)$. Pure radial scaling leaves the network representation invariant, meaning that the true single-step optimization progress corresponds to the angular rotation of the weight vector on the parameter sphere. Hunyuan ELR designates this metric as the angular update size (AUS) [[1]](https://hy.tencent.com/research/elr):

$$
\mathrm{AUS} := \left\lVert \frac{W_{t+1}}{\lVert W_{t+1}\rVert} - \frac{W_t}{\lVert W_t\rVert}\right\rVert \approx \frac{\eta_t \lVert U_t\rVert}{\lVert W_t \rVert} =: \eta_t^\star
$$

The effective learning rate $\eta_t^\star$ defined on the right-hand side originates from dynamical analyses of weight decay [[9]](https://arxiv.org/abs/2006.08419). It explicitly incorporates the weight norm $\lVert W_t \rVert$. Because the weight norm expands continuously during training, the ratio between the nominal learning rate $\eta_t$ and the effective learning rate $\eta_t^\star$ evolves dynamically, superimposing an implicit schedule on top of the nominal schedule.

The magnitude of this implicit schedule can be observed by comparing three nominal schedules: WSD (peak $3.6\times10^{-3}$), cosine decay, and linear decay (both peaks at $8.8\times10^{-3}$), where the nominal peaks differ by a factor of roughly 2.4.

{% include figure.liquid
  path='assets/img/post-08-25/wsd_cosine_linear_aus.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Bottom right: the learning rate schedules of WSD, cosine, and linear; top right: the corresponding weight norm; bottom left: the corresponding angular update size. Image source: <a href="https://hy.tencent.com/research/elr">Hunyuan ELR</a>.'
  zoomable=true
  alt='weight norm and angular update size under WSD, cosine and linear schedules'
%}

Comparing nominal learning rates against empirical angular update sizes: despite distinct geometric profiles and peak values across the three nominal schedules, their AUS curves align closely after 1000 steps, each displaying a peak followed by monotonic decay. The upper-right panel shows the corresponding weight norm expanding from roughly 30 to 180–280 over the first 2000 steps, absorbing and buffering discrepancies in the nominal schedule.

> ##### Nominal Learning Rate and Effective Learning Rate
> The optimizer explicitly specifies the nominal schedule, but the quantity directly driving network state updates is the angular update size. Continued growth in weight norm introduces an endogenous decay in effective step size, leveling out differences between nominal schedules.
{: .block-tip}

Under a constant nominal learning rate, the shift in effective step size driven by weight norm growth becomes even clearer [[2]](https://arxiv.org/abs/2607.22444):

{% include figure.liquid
  path='assets/img/post-08-25/muonwd_muonh_elr_const.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Left: nominal learning rates of MuonWD and MuonH, with the two curves coinciding at 0.025; right: the corresponding effective learning rates, where MuonWD decays from about $9\times10^{-5}$ to about $1.8\times10^{-5}$, and MuonH remains constant after the initial transient. Image source: <a href="https://arxiv.org/abs/2607.22444">Hyperball May Not Be a Free Lunch</a>, Figure 2(a)(b).'
  zoomable=true
  alt='nominal and effective learning rate of MuonWD and MuonH under a constant schedule'
%}

Under identical nominal learning rates, the effective learning rates of MuonWD and MuonH diverge by roughly a factor of 5 in late training. The effective learning rate decay in MuonWD stems entirely from unconstrained weight norm expansion, whereas MuonH preserves a constant effective learning rate after an initial transient by bounding the weight norm.

## 2. Correspondence Between Effective Learning Rate and Loss Trajectories

Establishing the effective learning rate $\eta^\star$ as the primary analytical target requires verifying whether its trajectory sufficiently determines the loss curve. Two recent independent works evaluated this correspondence from complementary directions.

In Hunyuan ELR's AUS-replay experiment, researchers recorded the step-by-step AUS trajectory of standard Adam or Muon during GPT-2 (124M) pretraining, subsequently setting this recorded AUS trajectory as the nominal learning rate curve for the corresponding Hyperball variants (AdamH, MuonH).

{% include figure.liquid
  path='assets/img/post-08-25/aus_replay.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='First row: AUS curves; second row: validation loss; third row: loss difference. The four columns are results for Adam and Muon on naive nanoGPT and scale-invariant nanoGPT. Image source: <a href="https://hy.tencent.com/research/elr">Hunyuan ELR</a>.'
  zoomable=true
  alt='AUS replay experiment comparing Adam/Muon with their Hyperball variants'
%}

The AUS trajectories and validation loss curves of both groups align closely. On strictly scale-invariant architectures, the difference in validation loss remains within a tight interval of $\pm0.005$ [[1]](https://hy.tencent.com/research/elr).

Another study employed parameter alignment: holding the base optimizer fixed, the nominal learning rate was adjusted pointwise to match target effective learning rates, achieving bidirectional alignment between MuonWD and MuonH trajectories [[2]](https://arxiv.org/abs/2607.22444).

{% include figure.liquid
  path='assets/img/post-08-25/muon_lr_alignment.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Left: MuonH aligned to MuonWD, the green dashed line nearly coincides with MuonWD; right: MuonWD aligned to MuonH, the purple dashed line nearly coincides with MuonH. In both figures, the original curves of MuonWD and MuonH differ in the 500–2000 step range. Image source: <a href="https://arxiv.org/abs/2607.22444">Hyperball May Not Be a Free Lunch</a>, Figure 3(a)(c).'
  zoomable=true
  alt='mutual alignment of MuonWD and MuonH training loss by learning-rate alignment'
%}

Both alignment directions closely reproduce the target loss curves, demonstrating that tuning the nominal learning rate suffices to replicate the other optimizer's convergence behavior. The authors conclude that Hyperball's primary contribution is providing an implicit state-dependent learning rate schedule, rather than offering geometric advantages along its update directions [[2]](https://arxiv.org/abs/2607.22444).

> ##### Equivalence of Effective Learning Rate and Convergence Trajectories
> **Matching effective learning rate trajectories yields nearly identical loss curves.**<br>Performance discrepancies between Hyperball and standard weight-decay optimizers are largely explained by the effective learning rate, indicating that norm constraints primarily serve to counteract implicit step size decay.
{: .block-tip}

This mechanism also clarifies the phased behavior observed in MuonH: early convergence is relatively slower, while late-stage metrics surpass MuonWD [[2]](https://arxiv.org/abs/2607.22444). Under the Hyperball constraint, the absence of norm-induced step size decay keeps late-stage effective step sizes higher.

## 3. Fitting Accuracy of Scaling Laws Under Effective Learning Rate

If the effective learning rate $\eta^\star$ is the primary determinant of convergence, using it as the independent variable should yield higher fitting precision for loss curves than the nominal learning rate $\eta$. When Hunyuan ELR replaced $\eta$ with $\eta^\star$ in the multi-power law (MPL) loss model, both in-sample fitting and cross-schedule predictive errors dropped markedly, and optimal $\eta^\star$ values transferred far more consistently across model width and depth [[1]](https://hy.tencent.com/research/elr).

This improvement arises because the mapping between the nominal learning rate and the loss contains a scaling factor that fluctuates with training steps, model width, and model depth. When fitting scaling laws using $\eta$, shifts in this factor register as fitting error; $\eta^\star$ removes this confounding source of variance, enhancing predictive signal.

A similar relationship governs hyperparameter transfer. Under the Hyperball constraint, the single-step angular displacement satisfies $\Delta\phi_t \approx \eta_t$, allowing the cumulative angular displacement to be expressed as:

$$
\sum_t \Delta\phi_t \approx \int_0^T \eta_t\,\mathrm{d}t
$$

This equation illustrates that the time integral of the learning rate approximates the total angular rotation of the weight vector. Empirically, runs with comparable cumulative learning rates arrive at similar final validation losses [[1]](https://hy.tencent.com/research/elr). Under Hyperball optimization, this integral carries an intuitive geometric interpretation and serves as a reliable alignment target across compute budgets.

Furthermore, on the Frobenius sphere, first-order weight decay vanishes, reducing the two-dimensional hyperparameter search over $(\eta,\lambda)$ to a single dimension. Empirical measurements indicate that optimal learning rates follow a power law against token scale, $\eta^*\propto T^{-0.32}$, matching the scaling exponent reported for AdamW [[3]](https://arxiv.org/abs/2603.28743). The theoretical origin of this exponent has not been identified.

## 4. Origin of the Shape: Bias–Variance Trade-Off

Having established the effective learning rate as the operative quantity, the inquiry shifts to its geometric trajectory. Despite varying experimental setups and analytical derivations, optimal learning rate schedules from four independent lines of work converge on the same profile: an initial ascent to a peak early in training, followed by monotonic decay toward zero.

**Comparison of WSD cooldown profiles.** Evaluating alternative manual cooldown schedules within WSD, `sqrt` ($1-\sqrt{x}$) and adjusted linear decay (`lowered linear 0.7`) achieve comparable performance, with the latter slightly ahead in final perplexity [[4]](https://arxiv.org/abs/2508.01483).

**Schedule-Free+.** Operating without predetermined schedules or tuned learning rate trajectories, its iterate-averaging scheme naturally induces an effective step size that peaks early and decays steadily under a constant nominal learning rate. In long-horizon training, this approach reduces the time required to match baseline validation loss by 31% relative to WSD [[5]](https://arxiv.org/abs/2605.19095).

{% include figure.liquid
  path='assets/img/post-08-25/schedulefree_plus_lr.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='1B model, 1000 tokens per parameter setting. Red is the actual lr of Schedule-Free+ (beta annealed from 0.8 to 0.965), black is linear decay, blue is WSD. Image source: <a href="https://arxiv.org/abs/2605.19095">Schedule-Free+</a>.'
  zoomable=true
  alt='effective learning rate of Schedule-Free+ compared with linear decay and WSD'
%}

**Offline refined schedule.** Schedulers derived offline pointwise from gradient norm sequences spontaneously form peak-then-decay curves across eight distinct tasks spanning vision, language, and recommendation domains [[6]](https://arxiv.org/abs/2310.07831).

{% include figure.liquid
  path='assets/img/post-08-25/refined_schedule.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Results on eight tasks (ImageNet, IWSLT14, GPT, RoBERTa, DLRM, MRI, ViT, RCNN). Left column: gradient norm sequence; middle column: smoothed gradient norm; right column: refined schedule constructed from it, with inset on a logarithmic scale. Image source: <a href="https://arxiv.org/abs/2310.07831">Defazio et al. (2024)</a>, Figure 4.'
  zoomable=true
  alt='gradient norms and refined schedules across eight tasks'
%}

**minus-square-root.** Developed in Hunyuan ELR to match empirical AUS trajectories, this schedule reaches the target validation loss of 3.28 in 3,175 steps on Modded-nanoGPT Track 3 [[1]](https://hy.tencent.com/research/elr); using a power-0.4 schedule, reference [[2]](https://arxiv.org/abs/2607.22444) achieves the same milestone in 3,150 steps. The two geometric profiles align closely, differing by only 25 steps.

Although these four methods arrive at their schedules through distinct methodologies, they converge on the peak-then-decay profile. This agreement stems from two underlying mechanisms: statistical estimation trade-offs and weight norm dynamics.

### 4.1 Balance Between Bias and Variance

Individual stochastic gradient steps influence overall estimation error along two axes: moving parameters away from initialization toward the optimum diminishes model bias, while accumulating gradient noise increases parameter variance. The learning rate modulates the relative contribution of each term.

Early in training, parameters lie far from the stationary region and bias dominates total error, where larger learning rates accelerate bias reduction. Late in training, parameters approach the target basin and variance becomes the leading source of error; shrinking the step size acts statistically as averaging over wider sample windows, damping stochastic noise. The peak-then-decay shape naturally integrates these dual requirements across training.

The relative weights of bias and variance depend on task characteristics, model scale, and training budget. Schedulers that minimize the total error $Bias+Variance$ define a set of functions, which maps to an exponent interval $\alpha$ under power-law parameterization.

{% include figure.liquid
  path='assets/img/post-08-25/bias_variance_shapes.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Bias–variance distribution for each cooldown shape; the dashed line marks the position where $Bias+Variance$ reaches its minimum. Left: only lowered linear shapes compared, as the parameter decreases variance drops and bias rises; right: all nonlinear shapes and some lowered linear shapes, where sqrt and 0.7 fall near the dashed line, while square, cosine, mirror cosine, and linear lie above the line. The horizontal axis ranges differ between the left and right panels. Image source: <a href="https://arxiv.org/abs/2508.01483">Dremov et al. (2025)</a>, Figure 6.'
  zoomable=true
  alt='bias-variance plot for different cooldown shapes'
%}

The illustration captures the inverse trade-off between bias and variance: decreasing the `lowered linear` parameter lowers variance while elevating bias. The right panel shows that both `sqrt` and `lowered linear 0.7` lie near the dashed minimal-error frontier, whereas `square`, `cosine`, `mirror cosine`, and standard `linear` fall above it. Multiple distinct profiles reside close to this minimal frontier [[4]](https://arxiv.org/abs/2508.01483); linear decay to zero [[7]](https://arxiv.org/abs/2502.15938) similarly falls within this near-optimal set.

{% include figure.liquid
  path='assets/img/post-08-25/sqrt_vs_lowered_linear.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='55%'
  caption='Shape comparison of sqrt and lowered linear 0.7 within the cooldown segment (training progress 80%–100%). Image source: <a href="https://arxiv.org/abs/2508.01483">Dremov et al. (2025)</a>, Figure 7.'
  zoomable=true
  alt='comparison of sqrt cooldown shape and lowered linear 0.7'
%}

Comparable performance across differing geometric shapes implies diminishing returns from fine-tuning the exact analytic form of a schedule: empirical results suggest that tuning AdamW's momentum coefficient $\beta_2$ yields loss variations on the same order of magnitude as selecting among candidate cooldown profiles [[4]](https://arxiv.org/abs/2508.01483).

### 4.2 Smoothing Effect of Weight Norm Dynamics on Observed Schedules

In standard training runs without Hyperball constraints, the effective learning rate is jointly determined by the nominal schedule and the spontaneous growth of the weight norm. The minus-square-root schedule targets $\eta^\star$ directly, necessitating an explicit peak-then-decay curve; under Hyperball optimization, weight norms are pinned to constants, eliminating endogenous step size decay and requiring the nominal learning rate $\eta_t$ itself to deliver all necessary attenuation.

Weight norm expansion compresses discrepancies in nominal step sizes, resulting in observed effective learning rate shapes that vary far less than their nominal counterparts. In Section 1, three nominal schedules with distinct peaks and curvatures collapse to nearly identical angular update sizes after 1000 steps, highlighting this buffering effect.

## 5. Summary

This article addresses two central questions in pretraining learning rate scheduling. Regarding the object of scheduling: optimizers modulate the nominal learning rate $\eta_t$, but the true driver of optimization progress is the effective learning rate $\eta_t^\star$, with their relationship scaled dynamically by the weight norm. By holding the weight norm constant, Hyperball eliminates endogenous step size decay, behaving effectively as an implicit state-dependent learning rate schedule. Regarding curve geometry: peak-then-decay represents the optimal resolution of the bias–variance trade-off, lowering model bias early via large steps and suppressing stochastic noise variance late via decay.

The bias–variance trade-off further implies that schedules meeting near-optimal conditions form a function set, placing an upper limit on performance gains from fine-tuning specific analytic forms; concurrently, because the balance shifts with training budget, model capacity, and noise level, optimal decay profiles vary across experimental conditions.

Practical recommendations:

1. Record the effective learning rate $\eta^\star$ or angular update size AUS when benchmarking optimizers and schedules, avoiding misleading artifacts from nominal learning rate scales.
2. Use the effective learning rate $\eta^\star$ or the cumulative angular displacement $\int\eta_t\mathrm{d}t$ as the alignment target when fitting scaling laws or transferring hyperparameters.
3. Because Hyperball optimization removes natural step size decay, nominal learning rates must incorporate an explicit decay phase, with the decay depth bounded to avoid falling behind standard weight decay baselines late in training [[2]](https://arxiv.org/abs/2607.22444).
4. Room for optimization across specific analytic profiles is modest; `sqrt`, `lowered linear 0.7`, and linear decay to zero all inhabit the near-optimal set. Compute budgets are better spent tuning peak learning rates and the momentum parameter $\beta_2$.

Open questions include establishing a quantitative mapping between trajectories induced by adaptive methods and hand-crafted analytic functions, while the theoretical origin of the exponent in $\eta^*\propto T^{-0.32}$ has not been identified.

Related reading: the derivation in the width direction is in ["On the Sphere: μP Scaling of Optimizers with the Hyperball Mechanism"](/en/blog/2026/spherical-hyperball/), the estimation of the update matrix norm is in ["Estimation of the Frobenius Norm of Update Matrices for Adam and Muon Optimizers"](/en/blog/2026/optimizer-update-matrix-norm/), and the schedule in the batch size direction is in ["DASF: A Closed-Loop Batch Size Schedule-Free Method"](/en/blog/2026/schedule-free-effective-batch-size/).

## References

[1] Tencent Hunyuan Pretrain Team (2026). [From LR to ELR: A Better Heuristic for Pretraining Dynamics](https://hy.tencent.com/research/elr).

[2] Xiao, Y., Sun, J., Gao, Z., Wei, Z., Wang, C., Tao, R., Teng, J., & Dai, B. (2026). [Hyperball May Not Be a Free Lunch](https://arxiv.org/abs/2607.22444). arXiv preprint arXiv:2607.22444.

[3] Ren, L., Liu, Y., Shen, Y., & Chen, W. (2026). [Rethinking Language Model Scaling under Transferable Hypersphere Optimization](https://arxiv.org/abs/2603.28743). arXiv preprint arXiv:2603.28743.

[4] Dremov, A., Hägele, A., Kosson, A., & Jaggi, M. (2025). [Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler](https://arxiv.org/abs/2508.01483). Transactions on Machine Learning Research (TMLR), 2025. arXiv preprint arXiv:2508.01483.

[5] Defazio, A. (2026). [Schedule-Free+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models](https://arxiv.org/abs/2605.19095). arXiv preprint arXiv:2605.19095.

[6] Defazio, A., Cutkosky, A., Mehta, H., & Mishchenko, K. (2024). [Optimal Linear Decay Learning Rate Schedules and Further Refinements](https://arxiv.org/abs/2310.07831). arXiv preprint arXiv:2310.07831.

[7] Bergsma, S., Dey, N., Gosal, G., Gray, G., Soboleva, D., & Hestness, J. (2025). [Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs](https://arxiv.org/abs/2502.15938). ICLR 2025. arXiv preprint arXiv:2502.15938.

[8] Wen, K., Dang, X., Lyu, K., Ma, T., & Liang, P. (2026). [Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899). arXiv preprint arXiv:2606.16899.

[9] Wan, R., Zhu, Z., Zhang, X., & Sun, J. (2020). [Spherical Motion Dynamics: Learning Dynamics of Neural Network with Normalization, Weight Decay, and SGD](https://arxiv.org/abs/2006.08419). arXiv preprint arXiv:2006.08419.

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026hyperballimplicitschedule,
  title={Hyperball、effective lr 与峰值加衰减的形状},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/hyperball-implicit-lr-schedule/}
}
```
