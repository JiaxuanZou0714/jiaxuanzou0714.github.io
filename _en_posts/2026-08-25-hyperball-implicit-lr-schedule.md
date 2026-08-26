---
layout: post
title: "Hyperball, effective lr, and the shape of peak-then-decay"
date: 2026-08-25 11:00:00
description: "Two issues. Scheduling object: changes in weight norm superimpose an implicit schedule on top of the base lr schedule; Hyperball removes this layer, and the quantity that matters is the effective lr. Shape: peak-then-decay is the solution to the bias-variance trade-off, where the bias term dominates early and the variance term dominates late; shapes satisfying this balance form a set, not limited to a specific analytic form."
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

Two phenomena have appeared in several recent works. First, Hyperball fixes the Frobenius norm of the weight matrix and its update to a constant, achieving a 20–30% token equivalent speedup over the weight decay baseline [[8]](https://arxiv.org/abs/2606.16899). Second, a batch of methods with different settings all yield optimal learning rate curves that are peak-then-decay.

Peak-then-decay in this paper refers to the following shape: the learning rate rises to a peak $\eta_{\max}$ in the early phase of training, then monotonically decreases, ending close to zero. The functional form of the decay segment is unrestricted.

{% include figure.liquid
  path='assets/img/post-08-25/peak_decay_shape.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='85%'
  caption='Examples of peak-then-decay shapes, where the decay segment takes the power-law form $\eta_t\propto(1-t/T)^{\alpha}$, with $\alpha$ equal to 0.5, 1, and 2. $\alpha=0.5$ corresponds to the sqrt shape, and $\alpha=1$ corresponds to linear decay to zero. The three curves have the same peak and endpoint, but the concavity of the decay segment differs.'
  zoomable=true
  alt='peak plus decay learning rate shape illustrated with three power-law exponents'
%}

The power law form in the figure is convenient for parameterization, and manual schedules often adopt this form. Among the four works listed in Section 4, the refined schedule is constructed pointwise from the gradient norm sequence, and the actual learning rate of Schedule-Free+ is produced by iteration averaging; both curves are peak-then-decay and do not correspond to a specific analytic form.

Several recent works attribute the source of the above two phenomena to the same quantity.

> ##### Core Judgment
> **The quantity that determines training progress is the effective learning rate $$\eta_t^\star=\eta_t\lVert U_t\rVert/\lVert W_t\rVert$$.**<br>The weight norm grows continuously during training, so the conversion factor from $$\eta_t$$ to $$\eta_t^\star$$ changes continuously, superimposing an implicit schedule on top of the set learning rate schedule. Hyperball fixes the weight norm, removing this layer, and its effect is therefore an implicit learning rate schedule.
{: .block-tip}

The reason $\eta^\star$ takes the peak-then-decay shape is independent of the optimizer:

> ##### Shape Criterion
> **Peak-then-decay is the solution to the bias-variance trade-off.**<br>The peak segment reduces bias with larger step sizes, and the end decays to zero to reduce variance. Shapes satisfying this balance form a set, and the performance of shapes within the set is comparable.
{: .block-tip}

The two judgments give the structure of the paper: $\eta^\star$ is the quantity that should be scheduled (Sections 1–3), and the bias-variance trade-off determines its shape (Section 4).

## 1. The Quantity That Matters Is the Effective Learning Rate

Parameters with normalization layers satisfy scale invariance $\mathcal{L}(\rho W) = \mathcal{L}(W)$. Pure radial scaling does not change the network function, so the quantity characterizing single-step progress is the angle through which the weight direction turns. Hunyuan ELR calls it the angular update size (AUS) [[1]](https://hy.tencent.com/research/elr):

$$
\mathrm{AUS} := \left\lVert \frac{W_{t+1}}{\lVert W_{t+1}\rVert} - \frac{W_t}{\lVert W_t\rVert}\right\rVert \approx \frac{\eta_t \lVert U_t\rVert}{\lVert W_t \rVert} =: \eta_t^\star
$$

The right-hand side is the effective learning rate, originating from previous dynamical analyses of weight decay [[9]](https://arxiv.org/abs/2006.08419). It contains $\lVert W_t \rVert$, and the weight norm changes continuously during training, **so the conversion factor from $\eta_t$ to $\eta_t^\star$ changes continuously**. This conversion factor is the implicit schedule in TL;DR.

The magnitude of the conversion can be given by a comparison of three nominal schedules: WSD (peak $3.6\times10^{-3}$), cosine and linear (peak $8.8\times10^{-3}$), with peaks differing by about 2.4 times.

{% include figure.liquid
  path='assets/img/post-08-25/wsd_cosine_linear_aus.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Bottom right: the learning rate schedules of WSD, cosine, and linear; top right: the corresponding weight norm; bottom left: the corresponding angular update magnitude. Image source: <a href="https://hy.tencent.com/research/elr">Hunyuan ELR</a>.'
  zoomable=true
  alt='weight norm and angular update size under WSD, cosine and linear schedules'
%}

Comparing the lower right and lower left: the nominal schedule shapes differ significantly, but after 1000 steps the three angular update magnitude curves basically coincide, all being peak-then-decay. The upper right gives the corresponding weight norm, which grows from about 30 to 180–280 in the first 2000 steps.

> ##### Nominal Learning Rate and Effective Learning Rate
> The adjustment object is the nominal learning rate schedule, and what acts on training is the AUS. Between the two there is a conversion factor that varies with the weight norm.
{: .block-tip}

Under a constant nominal learning rate, the form of this conversion is as follows [[2]](https://arxiv.org/abs/2607.22444).

{% include figure.liquid
  path='assets/img/post-08-25/muonwd_muonh_elr_const.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Left: nominal learning rates of MuonWD and MuonH, with the two curves coinciding at 0.025; right: the corresponding effective learning rates, where MuonWD decays from about $9\times10^{-5}$ to about $1.8\times10^{-5}$, and MuonH remains constant after the initial transient. Image source: <a href="https://arxiv.org/abs/2607.22444">Hyperball May Not Be a Free Lunch</a>, Figure 2(a)(b).'
  zoomable=true
  alt='nominal and effective learning rate of MuonWD and MuonH under a constant schedule'
%}

The nominal learning rates are exactly the same, but the effective learning rates differ by about 5 times. The decay in MuonWD comes from weight norm growth. MuonH fixes the weight norm, and the effective learning rate remains constant after the initial transient.

## 2. Correspondence Between the Effective Learning Rate Trajectory and the Loss Curve

The previous section gave the difference between $\eta^\star$ and $\eta$. As an analysis object, $\eta^\star$ still needs one confirmation: when the $\eta^\star$ trajectory is the same, is the loss curve the same? Two works have examined this from opposite directions.

AUS-replay in Hunyuan ELR: train GPT-2 (124M) with Adam or Muon and record the AUS step by step, then switch to the corresponding Hyperball variant (AdamH, MuonH) and set the recorded AUS trajectory as its learning rate curve.

{% include figure.liquid
  path='assets/img/post-08-25/aus_replay.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='First row: AUS curves; second row: validation loss; third row: loss difference. The four columns are results for Adam and Muon on naive nanoGPT and scale-invariant nanoGPT. Image source: <a href="https://hy.tencent.com/research/elr">Hunyuan ELR</a>.'
  zoomable=true
  alt='AUS replay experiment comparing Adam/Muon with their Hyperball variants'
%}

The AUS curves and loss curves of the two groups basically coincide, and on scale-invariant structures the loss difference is within $\pm0.005$ [[1]](https://hy.tencent.com/research/elr).

The opposite approach is to fix the optimizer and gradually change its learning rate to match the target effective learning rate, aligning MuonWD to the trajectory of MuonH, and also aligning in the reverse direction [[2]](https://arxiv.org/abs/2607.22444).

{% include figure.liquid
  path='assets/img/post-08-25/muon_lr_alignment.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Left: MuonH aligned to MuonWD, the green dashed line nearly coincides with MuonWD; right: MuonWD aligned to MuonH, the purple dashed line nearly coincides with MuonH. In both figures, the original curves of MuonWD and MuonH differ in the 500–2000 step range. Image source: <a href="https://arxiv.org/abs/2607.22444">Hyperball May Not Be a Free Lunch</a>, Figure 3(a)(c).'
  zoomable=true
  alt='mutual alignment of MuonWD and MuonH training loss by learning-rate alignment'
%}

The alignment results in both directions basically coincide with the target curve, and changing the learning rate suffices to reproduce the loss curve of the other optimizer. The paper concludes that the main role of Hyperball is an implicit state-dependent learning rate schedule, and its update direction shows no additional advantage [[2]](https://arxiv.org/abs/2607.22444).

> ##### Verification results in both directions
> **$$\eta^\star$$ When the trajectories are the same, the loss curves basically coincide, and the difference between Hyperball and non-Hyperball optimizers can be explained solely by the learning rate.**<br>This yields the opening judgment: Hyperball removes the implicit schedule that weight norm superimposes on top of the learning rate schedule.
{: .block-tip}

This judgment also corresponds to the staged phenomenon of MuonH: early convergence is slower, and later accuracy is higher than MuonWD [[2]](https://arxiv.org/abs/2607.22444). Under the Hyperball constraint, there is no decay from weight norm growth, and the early actual step size is larger than that of MuonWD.

## 3. Fit accuracy of scaling laws with effective learning rate

If $\eta^\star$ is the quantity that actually takes effect, using it to fit the loss curve should yield higher accuracy than using $\eta$. In the multiplicative power law (MPL) loss model, replacing $\eta$ with $\eta^\star$ in Hunyuan ELR improves both in-sample fit and cross-schedule prediction accuracy, and the transfer reliability of the optimal $\eta^\star$ across model width and depth is also higher [[1]](https://hy.tencent.com/research/elr).

The reason is consistent with Section 1: between $\eta$ and the loss there is a conversion factor that varies with training and with width and depth, and $\eta^\star$ does not include this factor. **When fitting the scaling law with $\eta$ as the independent variable, the variation of this factor is included in the fit error.**

The same relationship applies to hyperparameter transfer. Under the Hyperball constraint, $\Delta\phi_t \approx \eta_t$, so

$$
\sum_t \Delta\phi_t \approx \int_0^T \eta_t\,\mathrm{d}t
$$

That is, the integral of the learning rate equals the total angle rotated by the weight direction. A related phenomenon is that when the cumulative learning rate of two training runs is similar, the final losses are similar [[1]](https://hy.tencent.com/research/elr). Under the Hyperball constraint, this quantity has a corresponding geometric interpretation and can serve as an alignment object for cross-budget transfer.

Another related result: on the Frobenius sphere, weight decay has a first-order effect that vanishes, reducing the two-dimensional search over $(\eta,\lambda)$ to a one-dimensional search. Measurements give the optimal learning rate as a power law in the number of tokens, $\eta^*\propto T^{-0.32}$, consistent with the exponent reported for AdamW [[3]](https://arxiv.org/abs/2603.28743). The theoretical origin of this exponent remains unexplained.

## 4. Source of the shape: bias–variance trade-off

The first three sections identify the quantity being scheduled; this section discusses the shape of that quantity. The settings of the following four works differ, but the resulting learning rate curve shapes are consistent: rising to a peak in the early phase of training, then monotonically decreasing to near zero.

**WSD cooldown shape comparison.** In the cooldown segment, comparing various manual shapes, `sqrt` ($1-\sqrt{x}$) and `lowered linear 0.7` perform comparably, and `lowered linear 0.7` has lower perplexity than `sqrt` [[4]](https://arxiv.org/abs/2508.01483).

**Schedule-Free+.** Without specifying the learning rate value or schedule shape, its actual learning rate curve rises to a peak under a constant nominal learning rate and then decays. This method outperforms the WSD baseline, reducing the time to reach the same loss by 31% in long-horizon settings [[5]](https://arxiv.org/abs/2605.19095).

{% include figure.liquid
  path='assets/img/post-08-25/schedulefree_plus_lr.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='1B model, 1000 tokens per parameter setting. Red is the actual lr of Schedule-Free+ (beta annealed from 0.8 to 0.965), black is linear decay, blue is WSD. Image source: <a href="https://arxiv.org/abs/2605.19095">Schedule-Free+</a>.'
  zoomable=true
  alt='effective learning rate of Schedule-Free+ compared with linear decay and WSD'
%}

**Offline refined schedule.** Constructed offline from the gradient norm sequence, it exhibits the same shape on eight tasks [[6]](https://arxiv.org/abs/2310.07831).

{% include figure.liquid
  path='assets/img/post-08-25/refined_schedule.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Results on eight tasks (ImageNet, IWSLT14, GPT, RoBERTa, DLRM, MRI, ViT, RCNN). Left column: gradient norm sequence; middle column: smoothed gradient norm; right column: refined schedule constructed from it, with inset on a logarithmic scale. Image source: <a href="https://arxiv.org/abs/2310.07831">Defazio et al. (2024)</a>, Figure 4.'
  zoomable=true
  alt='gradient norms and refined schedules across eight tasks'
%}

**minus-square-root.** Hunyuan ELR proposes this based on the observed variation of AUS, reaching the target loss of 3.28 in 3,175 steps on Modded-nanoGPT Track 3 [[1]](https://hy.tencent.com/research/elr). [[2]](https://arxiv.org/abs/2607.22444) uses a power-0.4 schedule on the same track and reaches the target in 3,150 steps. The two shapes are close, with a difference of 25 steps.

The four works produce the shape in different ways, yet the results all fall within the description of peak-then-decay. This phenomenon involves two levels of causes.

### 4.1 Balance of bias and variance

The effect of a single-step update on the final model has two parts. The distance of the parameter from the initial point increases, corresponding to a decrease in bias. Gradient noise accumulates in the parameter, corresponding to an increase in variance. The learning rate determines the ratio of the two parts.

In the early phase of training, the bias term dominates, and a larger learning rate corresponds to faster bias reduction. In the late phase, the variance term dominates, and decaying the learning rate is equivalent to averaging over more updates, corresponding to a decrease in variance. The peak-then-decay shape is a combination of the requirements of these two phases.

The relative weight of bias and variance varies with task, model scale, and training budget. Therefore, the shapes that minimize $Bias+Variance$ form a set, which corresponds to an interval of $\alpha$ under the power-law parameterization. The existence of this set and its variation with conditions correspond to the two inferences in Section 5.

{% include figure.liquid
  path='assets/img/post-08-25/bias_variance_shapes.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='Bias–variance distribution for each cooldown shape, dashed line is the position where $Bias+Variance$ reaches its minimum. Left: only lowered linear shapes compared, as the parameter decreases variance drops and bias rises; right: all nonlinear shapes and some lowered linear shapes, where sqrt and 0.7 fall near the dashed line, while square, cosine, mirror cosine, and linear lie above the line. The horizontal axis ranges differ between the left and right panels. Image source: <a href="https://arxiv.org/abs/2508.01483">Dremov et al. (2025)</a>, Figure 6.'
  zoomable=true
  alt='bias-variance plot for different cooldown shapes'
%}

The left panel shows the inverse relationship between bias and variance: as `lowered linear` decreases, variance decreases and bias increases. The right panel shows the positions of each shape. `sqrt` and `lowered linear 0.7` fall near the minimum line, while `square`, `cosine`, `mirror cosine`, and `linear` lie above the line. There is more than one shape near the minimum line [[4]](https://arxiv.org/abs/2508.01483). The linear decay to zero reported in [[7]](https://arxiv.org/abs/2502.15938) also belongs to this category.

{% include figure.liquid
  path='assets/img/post-08-25/sqrt_vs_lowered_linear.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='55%'
  caption='Shape comparison of sqrt and lowered linear 0.7 within the cooldown segment (training progress 80%–100%). Image source: <a href="https://arxiv.org/abs/2508.01483">Dremov et al. (2025)</a>, Figure 7.'
  zoomable=true
  alt='comparison of sqrt cooldown shape and lowered linear 0.7'
%}

Two curves with different shapes perform comparably. One inference is that there is an upper bound on the tuning gain of the schedule shape: adjusting $\beta_2$ of AdamW yields differences comparable to shape selection [[4]](https://arxiv.org/abs/2508.01483).

### 4.2 Conversion Involving Weight Norm in the Observed Quantity

Without the Hyperball constraint, part of the observed shape comes from the conversion in Section 1, and the rest comes from the schedule design. minus-square-root takes $\eta^\star$ as the design object and requires explicitly writing out that shape. The Hyperball constraint removes the conversion, so $\eta_t$ itself must satisfy the shape requirement.

This conversion corresponds to an observation: the shape differences in the nominal learning rate are compressed after conversion, so the observed differences in the shape of $\eta^\star$ across settings are smaller than the shape differences in the nominal schedule. The AUS curves of the three nominal schedules in Section 1 nearly coincide after 1000 steps, which belongs to this category.

## 5. Summary

The paper is divided into two questions. The first question is about the object of scheduling: the quantity being adjusted is $\eta_t$, and the quantity that takes effect is $\eta_t^\star$. The conversion factor between them is determined by the weight norm and changes during training. Hyperball fixes the weight norm, making the conversion factor constant, so its role is an implicit learning rate schedule. The second question is about shape: peak-then-decay is the solution to the bias-variance trade-off, where the bias term dominates early and the variance term dominates at the end.

The structure of the trade-off yields two inferences. The shapes that satisfy the balance form a set, so there is an upper bound on the tuning gain of shape selection. The relative weight of the two ends is determined by the training budget, model scale, and noise level, so the optimal shape differs across conditions.

Operational conclusions:

1. When comparing different schedules, record the effective learning rate. Three nominal schedules with peak values differing by a factor of 2.4 can correspond to nearly identical AUS curves.
2. When fitting scaling laws and transferring hyperparameters, take $\eta^\star$ or the cumulative angular displacement $\int\eta_t\mathrm{d}t$ as the alignment object.
3. Under the Hyperball constraint, the conversion has been removed, and the shape requirement must be satisfied by $\eta_t$, so a clear decay design is needed. There is an upper bound on the decay magnitude: if too large, the later performance is worse than MuonWD [[2]](https://arxiv.org/abs/2607.22444).
4. The optimizable range of shape selection is limited; `sqrt`, `lowered linear 0.7`, and linear decay to zero perform comparably. Allocate the tuning budget preferentially to the peak learning rate and $\beta_2$.

Two remaining issues. The quantitative correspondence between the shapes produced by adaptive methods and manually specified shapes has not been verified. The theoretical origin of the exponent in $\eta^*\propto T^{-0.32}$ has not been identified.

Related work: the derivation in the width direction is in ["On the Sphere: μP Scaling of Optimizers with the Hyperball Mechanism"](/en/blog/2026/spherical-hyperball/), the estimation of the update matrix norm is in ["Estimation of the Frobenius Norm of Update Matrices for Adam and Muon Optimizers"](/en/blog/2026/optimizer-update-matrix-norm/), and the schedule in the batch size direction is in ["DASF: A Closed-Loop Batch Size Schedule-Free Method"](/en/blog/2026/schedule-free-effective-batch-size/).

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
