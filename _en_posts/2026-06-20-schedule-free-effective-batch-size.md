---
layout: post
title: "DASF: A Closed-Loop Schedule-Free Method for Batch Size"
date: 2026-06-20 12:00:00
description: "This paper proposes DASF (Drift-Aware Schedule-Free): based on the duality of Schedule-Free (iterate averaging ↔ learning rate schedule, gradient averaging ↔ batch size schedule), it uses on-the-fly measured gradient statistics to set the effective batch size online, with no schedule and no tuning, eliminating the cost of training proxy models and fitting scaling laws for batch calibration. On real transformers it matches or exceeds tuned baselines, and provides a falsifiable negative result: under compute-optimal conditions, the optimal effective batch is approximately constant, not growing as √t."
tags: [optimization, deep-learning, llm, schedule-free, batch-size]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/schedule-free-effective-batch-size/
ref: schedule-free-effective-batch-size
related_posts: false
---

## Motivation

In LLM pretraining, setting the batch size and its schedule typically relies on extensive hyperparameter search, or on training a series of small-scale proxy models, fitting a scaling law for the critical batch size (CBS), and then extrapolating to the target scale. Such approaches are costly, and extrapolation may not be reliable: [Merrill et al.](https://arxiv.org/abs/2505.23971) found that the gradient noise scale extrapolated from small models is inconsistent with the actual trend of CBS at the 7B scale.

Therefore, the key to reducing the above cost is not whether the method is "adaptive," but rather to directly use the gradient statistics measured on-the-fly from the target training itself to set the batch size online, avoiding cross-scale extrapolation. This is also a prerequisite for the method to be plug-and-play.

The previous post [“Why double the batch size midway through LLM pretraining”](https://jiaxuanzou0714.github.io/blog/2026/why-double-batch-size-llm-pretraining/)] tuned the **physical batch size**: it used the calculus of variations to derive an optimal schedule of truncated power-law form, but its optimal switching point depends on an exponent that is difficult to obtain in practice. This post instead fixes the physical batch size and uses the intrinsic averaging mechanism of [Schedule-Free](https://arxiv.org/abs/2405.15682) (SF) to set the **effective batch size** online.

The rationale for adopting SF is the following duality: SF averages the iterate points, thereby obtaining implicit learning rate decay; dually, averaging the gradients yields an implicit increase in effective batch size, because variance reduction on gradients is equivalent to increasing the effective batch size.

> ##### Core Duality
> **Iterate averaging ↔ learning rate schedule**<br>**Gradient averaging ↔ batch size schedule**
{: .block-tip}

Both require no preset schedule, because the averaging window automatically widens with $$t$$ ($$c_t=1/t$$), without needing to know the total number of training steps in advance. Based on this, we propose **DASF (Drift-Aware Schedule-Free)**: we turn this duality into an algorithm—using on-the-fly measured gradient drift and noise to close the loop and set the interpolation coefficient $$\beta_t$$ of SF (i.e., the implicit effective batch size), making the effective batch size schedule-free and tuning-free throughout. The rest of this post develops along this duality.

## 1. SF and its momentum form

Time-varying SF is defined by three sequences ($$\Delta_t$$ is the stochastic gradient at the evaluation point $$y_t$$, $$c_t=1/t$$):

$$
\begin{aligned}
x_t&=(1-c_t)\,x_{t-1}+c_t\,z_t,\\
y_t&=(1-\beta_t)\,z_t+\beta_t\,x_t,\\
z_{t+1}&=z_t-\eta_t\,\Delta_t,
\end{aligned}
$$

where $$z$$ is the base optimizer sequence, $$x$$ is the Polyak–Ruppert average of $$z$$, i.e., the model parameters used at inference/deployment, and $$y$$ is the only point where gradients are computed (the point where the forward pass is performed during training). Rewriting it in momentum form ([Through the River](https://arxiv.org/abs/2507.09846) §4.4), let $$m_t:=(x_t-z_{t+1})/\gamma$$:

$$
\begin{aligned}
m_t&=(1-c_t)\,m_{t-1}+\Delta_t,\\
y_{t+1}&=y_t-\gamma\big(\beta\,c_{t+1}\,m_t+(1-\beta)\,\Delta_t\big).
\end{aligned}
$$

$$m_t$$ is a gradient accumulator whose window widens with $$t$$, corresponding to the "gradient averaging" object in the duality in the Motivation; it is already inherent in SF and requires no extra introduction. The gradient actually used at each step is a convex combination of the accumulator $$m_t$$ (stale, low variance) and the fresh gradient $$\Delta_t$$ (full variance), with $$\beta$$ as the combination weight. The next section analyzes $$m_t$$ in the NQM to determine how much effective batch size $$\beta$$ actually provides.

## 2. NQM analysis: four conclusions

Consider the noise quadratic model (NQM), and analyze coordinate-wise along the eigen-directions of the Hessian (scalar): curvature $$h$$, true gradient $$g(\theta)=h\theta$$, single-sample stochastic gradient $$\Delta=h\theta+\epsilon$$, $$\epsilon\sim\mathcal N(0,\sigma^2)$$. The physical batch size $$B$$ reduces the noise variance to $$\sigma^2/B$$, giving the operational definition of effective batch size: the factor by which the noise variance is reduced. The object of analysis is the accumulator from §1

$$
m_t=(1-c_t)\,m_{t-1}+\Delta_t=\sum_{s\le t}w_{t,s}\,\Delta_s,\qquad w_{t,s}=\prod_{r=s+1}^{t}(1-c_r).
$$

**Conclusion 1 (effective batch size on the readout side grows linearly).** When $$c_t=1/t$$, the weights cancel term by term, simplifying to $$w_{t,s}=\prod_{r=s+1}^t\frac{r-1}{r}=\frac{s}{t}$$. The effective sample size of the normalized cumulative gradient (unbiased for the signal) is given by the Kish formula

$$
B_{\text{eff}}(t)=\frac{\big(\sum_s w_{t,s}\big)^2}{\sum_s w_{t,s}^2}=\frac{(t/2)^2}{t/3}=\frac{3}{4}\,t,
$$

That is, the effective sample size of the gradient aggregated by the averaged sequence $$x$$ used for inference is approximately $$3t/4$$, growing linearly with $$t$$. Regarding whether "SF implicitly performs batch warmup," the answer on the readout side is yes.

**Conclusion 2 (effective batch size on the optimization side is limited to a constant by $$\beta$$).** The base step $$z_{t+1}=z_t-\eta\Delta_t$$ uses a single fresh gradient, $$z$$ itself has no variance reduction; variance reduction exists only in the averaged sequence $$x$$. But what truly determines the loss decrease is neither $$z$$ nor $$x$$, but the sequence of evaluation points $$y$$ where gradients are computed. Its equivalent update is $$y_{t+1}=y_t-\gamma\,G^{\text{eff}}_t$$, and the effective gradient actually used at each step is a convex combination of the accumulator $$m_t$$ and the fresh gradient $$\Delta_t$$:

$$
G^{\text{eff}}_t=\beta\,c_{t+1}\,m_t+(1-\beta)\,\Delta_t.
$$

Next, compute the noise of $$G^{\text{eff}}_t$$. Let the single-sample gradient be $$\Delta_s=h\theta_s+\epsilon_s$$, where the noises $$\epsilon_s$$ are mutually independent, each with variance $$\sigma^2$$; the accumulator expands as $$m_t=\sum_{s\le t}w_{t,s}\Delta_s$$, $$w_{t,s}=s/t$$ per Conclusion 1. Substituting, the noise part $$\xi_t$$ of $$G^{\text{eff}}_t$$ is a linear combination of these independent noises $$\{\epsilon_s\}$$:

$$
\xi_t=\beta c_{t+1}\sum_{s\le t}\tfrac{s}{t}\,\epsilon_s\;+\;(1-\beta)\,\epsilon_t\;=\;\sum_{s\le t}a_s\,\epsilon_s,
$$

The two terms come from the accumulator $$m_t$$ and the fresh gradient $$\Delta_t$$, respectively. Here $$a_s$$ refers to the weight multiplied when noise $$\epsilon_s$$ enters $$\xi_t$$; it has two cases: the current noise $$\epsilon_t$$ appears in both terms (weight $$w_{t,t}=1$$ within $$m_t$$, weight $$1-\beta$$ in the fresh term), while past noise $$\epsilon_s\,(s<t)$$ enters only via $$m_t$$, so

$$
a_t=(1-\beta)+\beta c_{t+1},\qquad a_s=\beta c_{t+1}\,\tfrac{s}{t}\ \ (s<t).
$$

By independence, the variance equals the sum of squared coefficients $$\mathrm{Var}(\xi_t)=\sigma^2\sum_{s}a_s^2$$, i.e.,

$$
\frac{\mathrm{Var}(\xi_t)}{\sigma^2}=\big[(1-\beta)+\beta c_{t+1}\big]^2+\beta^2 c_{t+1}^2\sum_{s<t}\Big(\frac{s}{t}\Big)^2\ \xrightarrow{\,t\to\infty\,}\ (1-\beta)^2.
$$

Basis for taking the limit: $$c_{t+1}\sim1/t\to0$$, the accumulator part $$\big(\sim\beta^2c_{t+1}^2\cdot t/3\big)$$ tends to zero, but the fresh term injects a proportion $$(1-\beta)$$ of unreduced single-sample noise at each step, constituting a variance lower bound that does not decay with $$t$$. Therefore, the actual effective batch size $$B_{\text{eff}}^{\text{realized}}=\sigma^2/\mathrm{Var}(\xi_t)\to 1/(1-\beta)^2$$ of the $$y$$ trajectory is constant ($$\beta=0.9$$ corresponds to $$100$$, $$\beta=0.98$$ corresponds to $$2500$$). That is, vanilla SF with constant $$\beta$$ does not provide a growing batch size to the optimizer: the variance reduction in Conclusion 1 acts only on $$x$$, which is used solely for inference, while the optimization trajectory $$y$$ is limited to a constant by $$\beta$$. This relationship holds under SGD; under AdamW preconditioning, $$B_{\text{eff}}$$ is the *nominal* effective batch size derived from $$\beta$$, and all values reported below are this nominal value.

> ##### Correction under time-varying β
> The momentum rewrite in §1 takes $$\beta$$ constant. For general time-varying $$\beta_t$$, directly expanding $$y_{t+1}=(1-\beta_{t+1})z_{t+1}+\beta_{t+1}x_{t+1}$$ (substituting $$x_{t+1}=(1-c_{t+1})x_t+c_{t+1}z_{t+1}$$, $$z_{t+1}=z_t-\gamma\Delta_t$$) gives
>
> $$
> y_{t+1}=y_t-\gamma\big[(1-\beta_t)\Delta_t+(\beta_t-\beta_{t+1}+\beta_{t+1}c_{t+1})\,m_t\big],
> $$
>
> Only when $$\beta_{t+1}=\beta_t=\beta$$ does it reduce to the previous form $$G^{\text{eff}}_t=(1-\beta)\Delta_t+\beta c_{t+1}m_t$$. Therefore, $$B_{\text{eff}}=1/(1-\beta_t)^2$$ in this paper is a quasi-static approximation under time-varying $$\beta_t$$, requiring $$\lvert\beta_{t+1}-\beta_t\rvert\ll c_{t+1}$$, i.e., $$\beta_t$$ changes slower than the averaging rate. DASF's $$\beta_t$$ is driven by EMA-smoothed statistics and changes gently, largely satisfying this condition; if $$\beta_t$$ undergoes rapid jumps, the extra $$(\beta_t-\beta_{t+1})\,m_t$$ term will alter the sign and variance structure of the cumulative gradient term, requiring separate handling.
{: .block-tip}

From Conclusion 2, to make the optimization trajectory also benefit from batch size growth, we must set $$\beta_t\to1$$; the problem thus reduces to scheduling $$\beta$$. At the same time, $$\beta$$ plays a dual role: it determines both the effective batch size and the stability threshold ([Through the River](https://arxiv.org/abs/2507.09846) introduces a parameter $$C$$ to decouple the two).

**Conclusion 3 ($$\beta_t\to1$$ unlocks growth, $$\rho$$ is the growth exponent).** Aligning the actual effective batch size with the target $$B^\ast(t)$$ yields $$\beta_t=1-1/\sqrt{B^\ast(t)}$$. [AMUSE](https://arxiv.org/abs/2605.22432) (Kim et al. 2026) makes $$1-\beta_t$$ decay asymptotically as $$t^{-\rho}$$ (its Eq.5 implementation includes warmup-hold, and only for large $$t$$ does it follow this power law; see experiments). Hence for large $$t$$, $$B_{\text{eff}}^{\text{realized}}(t)\propto t^{2\rho}$$. That is, the exponent $$\rho$$ that AMUSE uses to ensure stability is precisely the growth exponent of the implicit batch size; $$\rho=1/4$$ gives $$B_{\text{eff}}\propto\sqrt t$$, consistent with the CBS scaling measured by [Merrill et al.](https://arxiv.org/abs/2505.23971). AMUSE interprets $$\beta_t\uparrow$$ as moving the evaluation point from the fast sequence $$z$$ to the averaged sequence $$x$$ to suppress oscillation, while from the batch size perspective the same operation gradually shuts off fresh noise injection, allowing the growth from accumulation to manifest; the two are equivalent and give $$\rho$$ a falsifiable meaning.

**Conclusion 4 (staleness gives a cube-root upper bound).** The $$y$$ trajectory uses a window average over the past ~$$W=B_{\text{eff}}$$ gradients to estimate the current true gradient $$G(y_t)$$. But it is not an i.i.d. batch of the same size; rather, it is a time average along the moving trajectory, so widening the window requires trading off two opposing errors.

First, variance. Averaging $$W$$ gradient samples reduces noise by a factor of $$W$$ according to the law of large numbers:

$$
V(W)\approx\frac{\sigma^2}{W}\qquad(\text{窗越宽越小}).
$$

Second, staleness bias. The gradients in the window are computed at past points $$y_s\,(s<t)$$, while the iterate keeps moving. Suppose the gradient drifts at rate $$\dot g$$ per step (in NQM, $$\dot g=h\dot y$$, where $$h$$ is curvature and $$\dot y$$ is iterate velocity). The window's weight centroid lags the current step by about $$W/2$$ steps (for $$c_t=1/t$$, the centroid is at $$2t/3$$, lagging by $$t/3$$, same order), so the averaged gradient systematically deviates from the current true gradient:

$$
\text{bias}\approx\dot g\cdot\frac{W}{2}\ \Longrightarrow\ \text{bias}^2\approx\frac{\dot g^2 W^2}{4}\qquad(\text{窗越宽越大}).
$$

The sum of the two is the mean squared error; differentiating with respect to $$W$$ and setting to zero gives the optimal window $$W^\ast$$:

$$
\begin{aligned}
\mathrm{MSE}(W)&=\frac{\sigma^2}{W}+\frac{\dot g^2 W^2}{4},\\
\frac{d\,\mathrm{MSE}}{dW}&=-\frac{\sigma^2}{W^2}+\frac{\dot g^2 W}{2}=0
\ \Longrightarrow\ W^3=\frac{2\sigma^2}{\dot g^2}
\ \Longrightarrow\ W^\ast=\Big(\frac{2\sigma^2}{\dot g^2}\Big)^{1/3}.
\end{aligned}
$$

The useful accumulation window (i.e., the upper bound on useful effective batch size) is determined by the cube root of the ratio of gradient noise $$\sigma^2$$ to the square of gradient drift rate $$\dot g^2$$. Its structure is similar to the noise scale $$\sigma^2/g^2$$, but the denominator is the rate of change of the gradient $$\dot g$$ rather than the gradient itself, because staleness depends on how fast the gradient changes, not its magnitude; both quantities can be measured on the fly. Meanwhile, $$c_t=1/t$$ gives linear growth $$W\sim t$$, far exceeding this cube-root optimum, so vanilla SF over-accumulates on the readout side, and $$B_{\text{eff}}\propto t$$ will exceed CBS in the late phase.

Synthesizing the four conclusions: SF's $$1/t$$ averaging inherently provides linearly growing effective batch size on the readout side (Conclusion 1), but the optimization side is limited to a constant by $$\beta$$ (Conclusion 2); liberating the optimization side requires setting $$\beta_t\to1$$, which is exactly what AMUSE does, with its $$\rho$$ being the growth exponent and $$\rho=1/4$$ reproducing Merrill's $$\sqrt t$$ (Conclusion 3); and this growth has a cube-root upper bound determined by staleness (Conclusion 4). Therefore, no new optimizer is needed: the appropriate schedule-free batch size method is to retain AMUSE's structure, only changing $$\beta_t$$ from an open-loop $$t^{-\rho}$$ to a closed-loop driven by on-the-fly measured gradient statistics—targeting the $$W^\ast$$ of Conclusion 4 and setting $$\beta_t$$ via the conversion of Conclusion 3, which is DASF in the next section.

> ##### Approximations used in the analysis
> The above analysis relies on several approximations: NQM (quadratic, additive constant-variance noise), per-eigen-direction (cross-spectrum aggregation requires spectral weighting), local linearization of drift, and quasi-static treatment of $$B_{\text{eff}}$$ when reading $$c_t$$. These affect only constants, not scaling (exponents such as $$t^{2\rho}$$ and cube root remain stable).
{: .block-tip}

## 3. DASF: drift-aware closed-loop controller

From Conclusions 3–4, the controller's form is determined: take the $$W^\ast$$ of Conclusion 4 as the target effective batch size, and convert via the $$\beta_t=1-1/\sqrt{B^\ast}$$ of Conclusion 3. Aggregating the cube-root law from coordinate-wise form ($$\sigma^2\to\operatorname{tr}\Sigma$$, $$\dot g^2\to\lVert\dot G\rVert^2$$) and substituting gives the setpoint

$$
1-\beta_t=\Big(\frac{\lVert\dot G\rVert^2}{2\,\operatorname{tr}\Sigma}\Big)^{1/6}.
$$

Exponent $$1/6=\tfrac13\times\tfrac12$$: $$\tfrac13$$ comes from the bias-variance optimal window of Conclusion 4, and $$\tfrac12$$ from the $$B_{\text{eff}}=1/(1-\beta)^2$$ conversion of Conclusions 2–3. Among these, $$\operatorname{tr}\Sigma$$, $$\lVert G\rVert^2$$, and $$\lVert\dot G\rVert^2$$ are all estimated on the fly during training without touching the data pipeline (method below). Compared to AMUSE, this controller eliminates three manual choices—the heuristic warmup-hold, the empirical exponent $$\rho$$, and the initial $$\beta_1$$: DASF's $$\beta_t$$ over the entire trajectory (including the early small-batch segment) is determined pointwise from on-the-fly measured $$\dot G$$ and $$\operatorname{tr}\Sigma$$, rather than a preset curve—this is its key simplification relative to AMUSE.

The setpoint depends on gradient drift $$\dot G$$ rather than gradient magnitude $$G$$, which is the core of this design. If one directly used the noise scale $$\operatorname{tr}\Sigma/\lVert G\rVert^2$$ as the setpoint, then as the signal converges, $$\lVert G\rVert\to0$$, the noise scale diverges, pushing $$\beta$$ above $$0.99$$ and increasing the effective batch size to tens of thousands, causing over-accumulation and even divergence. A smaller gradient does not imply entering the noise-dominated regime. The drift-aware setpoint is based on the gradient drift rate: when the signal is rapidly decreasing, $$\dot G$$ is large, so the batch size remains small, avoiding excessive lag. This is mechanistically consistent with [Merrill](https://arxiv.org/abs/2505.23971)'s conclusion that the noise scale is unreliable.

**On-the-fly estimation.** DASF requires only two quantities: noise $$\operatorname{tr}\Sigma$$ and drift $$\lVert\dot G\rVert^2$$—it does not need gradient magnitude $$\lVert G\rVert^2$$. $$\operatorname{tr}\Sigma$$ uses the [McCandlish](https://arxiv.org/abs/1812.06162) two-point method: split a physical batch into micro-batches of size $$B_{\text{small}}$$, and from the mean squared norm of a single micro-batch $$\overline{\lVert g_{\text{small}}\rVert^2}$$ and the squared norm of the full batch $$\lVert G_{\text{big}}\rVert^2$$, eliminating $$\lVert G\rVert^2$$ via $$\mathbb E\lVert g_B\rVert^2=\lVert G\rVert^2+\operatorname{tr}\Sigma/B$$, we obtain

$$
\operatorname{tr}\Sigma=\frac{\overline{\lVert g_{\text{small}}\rVert^2}-\lVert G_{\text{big}}\rVert^2}{1/B_{\text{small}}-1/B_{\text{big}}}.
$$

Drift $$\lVert\dot G\rVert^2$$ is taken as the difference between full-batch gradients of adjacent $$k$$ steps, $$\lVert G_t-G_{t-k}\rVert^2$$, then subtracting the noise floor $$2\operatorname{tr}\Sigma/B_{\text{big}}$$ (the noise variance of the difference of two independent noisy gradients). Both quantities are EMA-smoothed to denoise; the probe only splits the same physical batch to read, consuming no extra tokens.

It should be emphasized: what needs to be estimated for $$\lVert G\rVert^2$$ is the noise-scale setpoint $$\operatorname{tr}\Sigma/\lVert G\rVert^2$$ for comparison, not DASF. Moreover, $$\lVert G\rVert^2$$ must use the two-point unbiased estimate $$\tfrac{B_{\text{big}}\lVert G_{\text{big}}\rVert^2-B_{\text{small}}\overline{\lVert g_{\text{small}}\rVert^2}}{B_{\text{big}}-B_{\text{small}}}$$—directly using $$\lVert G_{\text{big}}\rVert^2$$ would overestimate $$\operatorname{tr}\Sigma/B_{\text{big}}$$, and this bias is largest in the late low-signal phase (exactly the regime where the noise-scale scheme operates). DASF bypasses this most difficult and unreliable quantity.

> ##### Constants in the formula
> The exponent $$1/6$$ is determined by theory; the constants in the formula (the factor 2 and the omitted window shape, physical batch size normalization factors) only shift the absolute level of $$\beta$$ without changing the functional form. Also, since the $$1/6$$ power is scale-insensitive, directly substituting measured quantities places $$\beta$$ in the appropriate range.
{: .block-tip}

## 4. Experiments

The setup is strictly aligned with the companion paper: standard GPT (FineWeb, GPT-2 tokenizer), SF-AdamW, constant learning rate throughout, same initialization and data order. The only variable is $$\beta_t$$. We compare four $$\beta_t$$ schedules (all with fixed physical batch and vanilla SF), whose respective $$1-\beta_t$$ and implied effective batch size $$B_{\text{eff}}=1/(1-\beta_t)^2$$ (converted via Conclusion 2) are:

- Constant $$\beta$$: $$1-\beta_t$$ and $$B_{\text{eff}}$$ are both constants.
- AMUSE (open-loop): $$1-\beta_t\propto t^{-\rho}$$, so $$B_{\text{eff}}\propto t^{2\rho}$$.
- Closed-loop noise scale: $$1-\beta_t=\sqrt{\lVert G\rVert^2/\operatorname{tr}\Sigma}$$, so $$B_{\text{eff}}=\operatorname{tr}\Sigma/\lVert G\rVert^2$$ (i.e., the noise scale itself).
- Closed-loop drift-aware: $$1-\beta_t=(\lVert\dot G\rVert^2/2\operatorname{tr}\Sigma)^{1/6}$$, so $$B_{\text{eff}}=(2\operatorname{tr}\Sigma/\lVert\dot G\rVert^2)^{1/3}$$ (i.e., the $$W^\ast$$ of Conclusion 4).

The first two are baselines (constant, open-loop power law), and the latter two are the closed-loop setpoints of this paper. AMUSE is implemented according to its Eq.5—holding $$\beta_1$$ during warmup, then $$1-\beta_t=(1-\beta_1)\big(\tfrac{T_0-1}{t-1}\big)^\rho$$ ($$T_0$$ is the number of warmup steps, asymptotically $$\propto t^{-\rho}$$)—and is a well-tuned strong baseline: $$\rho$$ and $$\beta_1$$ are optimized via grid search (both scales use $$\rho=0.6$$, $$\beta_1=0.4$$), so the comparison is against its optimally tuned version, while drift-aware requires no tuning throughout. Two scales: 45M / 600M and 117M / 2.5B tokens.

{% include figure.liquid
  path='assets/img/post-06-20/bsf_compare.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='45M / 600M tokens. Four panels: $B_\text{eff}$ (top left), validation loss (top right), $\beta_t$ (bottom left), measured noise scale (bottom right). drift-aware (red, i.e., DASF, tuning-free) achieves the lowest validation loss, slightly better than the manually tuned const $\beta=0.9$, and better than AMUSE and const $\beta=0.98$.'
  zoomable=true
  alt='45M transformer comparison of beta schedules on FineWeb'
%}

The bottom-right subplot illustrates the necessity of the drift-aware setpoint: facing the same diverging noise scale, the noise-scale scheme over-accumulates and diverges, while the drift-aware scheme switches to $$\dot G$$, remains stable, and achieves the lowest validation loss. This is consistent with Merrill's argument about the unreliability of the noise scale.

{% include figure.liquid
  path='assets/img/post-06-20/bsf_big_compare.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='117M / 2.5B tokens (modded-nanogpt). drift-aware reproduces the advantage (lowest validation loss); its $B_\text{eff}$ (top right red line) bottoms out at 25 and only rebounds to about 43, fluctuating between 27 and 52 in the second half, never entering the batch amplification phase; AMUSE shows an unstable validation loss rebound at about 1750M.'
  zoomable=true
  alt='117M transformer comparison showing effective batch stays small'
%}

Results are consistent across both scales:

| | 45M / 600M | 117M / 2.5B |
| :--- | :---: | :---: |
| drift-aware (tuning-free) val | **4.235** | **3.643** |
| const $$\beta=0.9$$ | 4.275 | 3.659 |
| AMUSE-style open-loop | significantly worse | 3.990 |
| drift-aware's $$B_{\text{eff}}$$ | ~14–37 | ~27–52 |
{: .table .table-striped .table-sm}

DASF matches or exceeds tuned AMUSE and the optimal manually tuned constant at both scales, achieving plug-and-play with no proxy model calibration. However, its advantage does not come from a larger implicit batch—DASF's $$B_{\text{eff}}$$ (about 40) is smaller than const $$\beta=0.9$$ (100); in this regime a smaller effective batch is actually better, and DASF's value lies in automatically calibrating to an appropriate (small) value without over-accumulation.

## 5. Conclusion and Method Positioning

As a method, this paper uses SF's $$\beta$$ as an online control quantity for effective batch size, closed-loop via the drift-aware cube-root law, yielding DASF, a tuning-free effective batch size setter: at 45M and 117M it matches or slightly outperforms manually tuned constants, clearly outperforms open-loop AMUSE, and avoids the divergence of the noise-scale scheme.

As a falsifiable scientific finding, the benefit of increasing batch size never appears: even at 117M / 2.5B / compute-optimal, $$B_{\text{eff}}$$ is only on the order of tens, and the advantage does not grow with training. More precisely, under the compute-optimal regime the optimal effective batch size is small and approximately constant, rather than growing as $$\sqrt t$$—when training at the [Chinchilla](https://arxiv.org/abs/2203.15556) ratio, the gradient changes rapidly and continuously, and the cube-root formula of Conclusion 4 always gives a small batch size. This also explains why constant $$\beta$$ SF is already sufficient in this regime.

On positioning: with a fixed physical batch size, $$\beta$$ only changes the variance of each step's gradient, not the number of optimization steps; the main source of benefit in [Merrill](https://arxiv.org/abs/2505.23971)-style batch warmup is the additional optimization steps (step-count channel) provided by a smaller physical batch early on under the same token budget, which $$\beta$$ is structurally unable to access. Therefore, DASF is a setter for effective batch size (variance), not a scheduler for physical batch size—that is, between Route A (changing physical batch, with step-count benefits but system overhead) and Route B (changing effective batch, plug-and-play), it is the upper bound of Route B. To realize growth benefits, one must move to a regime with $$B^\ast$$ substantial changes: heavy overtraining, LR cooldown, or letting the physical batch itself vary.

## References

1. A. Defazio, X. Yang, H. Mehta, K. Mishchenko, A. Khaled, and A. Cutkosky. "The Road Less Scheduled." *NeurIPS*, 2024. [arXiv:2405.15682](https://arxiv.org/abs/2405.15682).
2. S. McCandlish, J. Kaplan, D. Amodei, and OpenAI Dota Team. "An Empirical Model of Large-Batch Training." [arXiv:1812.06162](https://arxiv.org/abs/1812.06162), 2018.
3. W. Merrill, S. Arora, D. Groeneveld, and H. Hajishirzi. "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training." *NeurIPS*, 2025. [arXiv:2505.23971](https://arxiv.org/abs/2505.23971).
4. M. Song, B. Baek, K. Ahn, and C. Yun. "Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training." *NeurIPS*, 2025. [arXiv:2507.09846](https://arxiv.org/abs/2507.09846).
5. J. Kim, B. Shin, J. Yun, B. Baek, M. Song, and C. Yun. "AMUSE: Anytime Muon with Stable Gradient Evaluation." [arXiv:2605.22432](https://arxiv.org/abs/2605.22432), 2026.
6. D. Morwani et al. "Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants." [arXiv:2502.02431](https://arxiv.org/abs/2502.02431), 2025.
7. J. Hoffmann et al. "Training Compute-Optimal Large Language Models." [arXiv:2203.15556](https://arxiv.org/abs/2203.15556), 2022.

*Companion paper: [Why double the batch size midway through LLM pretraining](https://jiaxuanzou0714.github.io/blog/2026/why-double-batch-size-llm-pretraining/) (calculus of variations solution for physical batch size scheduling).*

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026sfbatch,
  title={DASF：一种闭环的 batch size schedule-free 方法},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/schedule-free-effective-batch-size/}
}
```
