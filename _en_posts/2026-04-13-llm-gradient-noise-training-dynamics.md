---
source_sha: 3ebdcd862da8319f
layout: post
title: "In the LLM context, how does noise in the gradient affect training dynamics?"
date: 2026-04-14 00:00:00
description: "A discussion of gradient noise in the late stages of LLM pretraining, and why blockwise normalized updates look more like constraining the update magnitude than correcting the gradient direction."
tags: [optimization, deep-learning, llm, scaling-law]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/llm-gradient-noise-training-dynamics/
ref: llm-gradient-noise-training-dynamics
related_posts: false
---

## Introduction

In the later stages of LLM pretraining, the effective signal in the stochastic gradient weakens, but the variance from mini-batch sampling remains. As a result, many updates appear to be dominated by noise rather than signal. We refer to this situation as the noise-dominated regime.

This is also the reason I want to revisit optimizers based on row normalization and block normalization. According to traditional optimization intuition, an optimizer should ideally estimate the gradient direction, curvature, or second-order structure more accurately; but normalized updates are crude—they directly discard the gradient magnitude.

So we can ask a different question:

> When the gradient is mainly controlled by noise, what part of the training dynamics do normalized updates actually change?

My conclusion is conservative: at least for blockwise $$L_2$$ normalization, it does not improve the per-step direction accuracy, nor does it extract additional information from the noise. What it does is constrain the update magnitude, preventing the noise scale in the raw gradient from directly propagating to the parameter update. In the multi-block heteroscedastic LLM setting, this manifests as an implicit blockwise learning rate adjustment.

## 1. Problem Formulation

### 1.1 A Minimal Model for Stochastic Optimization

Start with a minimal model. Partition the parameters into $$m$$ blocks, and denote the $$r$$-th block as $$x_r \in \mathbb{R}^{d_r}$$. The stochastic gradient on this block is written as

$$
g_r = s_r + \xi_r = s_r + \sigma_r z_r
$$

where $$s_r := \nabla_r F(x)$$ is the true gradient, $$z_r$$ is zero-mean noise ($$\mathbb{E}[z_r \mid x] = 0$$), and $$\sigma_r$$ controls the noise scale for this block. The corresponding local signal-to-noise ratio is defined as

$$
\rho_r := \frac{\lVert s_r \rVert_2}{\sigma_r \sqrt{d_r}}
$$

In the following, we mainly consider the case $$\rho_r \ll 1$$, i.e., when the noise is much larger than the signal.

### 1.2 A Unified Form of Normalized Updates: Steepest Descent under a Conjugate Norm

Many blockwise-normalized, row-normalized, and fractional-power gradient updates can be written in the following form:

$$
u_r = \frac{\operatorname{sign}(g_r) \odot \lvert g_r \rvert^p}{\lVert g_r \rVert_{p+1}^p}
$$

This expression can also be derived from the perspective of steepest descent. Given a norm constraint on the step, we ask how much the linear term can be reduced at most:

$$
v^* = \arg\min_{\lVert v \rVert_q \le 1} \langle g_r, v \rangle
$$

By Hölder's inequality, for conjugate exponents satisfying $$\frac{1}{p+1} + \frac{1}{q} = 1$$, we have

$$
\lvert\langle g_r, v \rangle\rvert \le \lVert g_r \rVert_{p+1} \lVert v \rVert_q
$$

When equality holds, $$\lvert v_{r,i} \rvert^q \propto \lvert g_{r,i} \rvert^{p+1}$$. Combining with $$\lVert v \rVert_q = 1$$ and $$q = \frac{p+1}{p}$$, we obtain

$$
\lvert v_{r,i} \rvert = \frac{\lvert g_{r,i} \rvert^p}{\lVert g_r \rVert_{p+1}^p}
$$

The negative sign is chosen to decrease the inner product; if we move the negative sign of the update direction into the optimization step, we get the $$u_r$$ above. Two common special cases are:

- $$p = 1$$: blockwise $$L_2$$ normalization;
- $$p = 0$$: element-wise sign update, i.e., Sign SGD.

## 2. An Intuition to Clarify: Is Normalization Correcting the Direction?

A natural reading of normalization is that it projects the gradient onto the unit sphere, removes the noise in the magnitude, and therefore makes the direction more accurate.

This reading is incorrect, at least for $$L_2$$ normalization.

Taking blockwise $$L_2$$ normalization as an example, the update is $$u_r = \frac{g_r}{\lVert g_r \rVert_2}$$. Its cosine similarity with the true gradient $$s_r$$ is

$$
\cos(u_r, s_r) = \frac{\langle u_r, s_r \rangle}{\lVert u_r \rVert_2 \lVert s_r \rVert_2} = \frac{\langle g_r, s_r \rangle}{\lVert g_r \rVert_2 \lVert s_r \rVert_2} = \cos(g_r, s_r)
$$

That is, before and after blockwise $$L_2$$ normalization, the angle with the true gradient is exactly the same. It merely scales the same direction to unit length; it does not make the direction closer to $$s_r$$.

Of course, this conclusion only applies to $$p=1$$. For Sign SGD or general power updates, each coordinate undergoes a nonlinear transformation, and the direction typically changes. In that case, the update can be understood as steepest descent under a different geometry; whether the direction is better depends on the relationship between gradient, noise, and curvature, and we cannot simply claim that it is closer to the true gradient.

So, if even $$L_2$$ normalization, which leaves the direction unchanged, can be useful in some training scenarios, the explanation should not stop at "the direction is more accurate." The more pertinent question is: once noise enters the update, how does its magnitude affect the dynamics?

## 3. Homogeneity of Degree Zero Limits the Update Magnitude

### 3.1 Two Properties

Abstract the normalized update as $$u_r = \phi_r(g_r)$$. The mappings of interest here typically satisfy two properties:

1. Odd symmetry: $$\phi_r(-g) = -\phi_r(g)$$;
2. Homogeneity of degree zero: for any $$c > 0$$, $$\phi_r(cg) = \phi_r(g)$$.

The second property is important. If the input is multiplied by a positive scalar, the output remains unchanged. Thus, no matter how large the noise scale $$\sigma_r$$ is, as long as it is just an overall scaling of the magnitude, the normalized update magnitude will not grow linearly with it.

This is not denoising. Normalization does not recover the signal from noise, nor does it improve the signal-to-noise ratio; it also discards magnitude information. It merely constrains the output update to a fixed scale. The simplest one-dimensional example is

$$
u = \operatorname{sign}(s + \sigma z)
$$

The variance of this $$u$$ is always bounded, whereas the noise variance of the raw gradient $$g = s + \sigma z$$ grows with $$\sigma^2$$.

### 3.2 Why the First-Order Response Weakens with Noise

Next, we examine the following question: if $$g_r = s_r + \sigma_r z_r$$ and $$\sigma_r z_r$$ is much larger than $$s_r$$, how much of a response to the signal $$s_r$$ does $$\phi_r(g_r)$$ retain?

First, note a technical limitation. The following uses the Jacobian and Taylor expansion, so strictly speaking it applies to smooth normalization mappings, such as $$L_2$$ normalization. Non-smooth mappings like Sign require subgradients or distributional derivatives. We first consider the smooth case, as it already explains the main scaling relationships.

For any nonzero vector $$g$$ and small perturbation $$h$$, using homogeneity of degree zero:

$$
\phi_r(cg+\varepsilon h) = \phi_r\left(c\left(g+\frac{\varepsilon}{c}h\right)\right) = \phi_r\left(g+\frac{\varepsilon}{c}h\right)
$$

Taking the directional derivative with respect to $$\varepsilon$$ at 0, we obtain

$$
J_{\phi_r}(cg)h = \frac{1}{c} J_{\phi_r}(g)h
$$

Therefore

$$
J_{\phi_r}(cg) = \frac{1}{c} J_{\phi_r}(g)
$$

The second derivative is similar: $$\nabla^2\phi_r(cg) = \frac{1}{c^2}\nabla^2\phi_r(g)$$.

Now expand around the noise point $$\sigma_r z_r$$:

$$
u_r = \phi_r(\sigma_r z_r) + J_{\phi_r}(\sigma_r z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

Substituting the homogeneity and the Jacobian scaling law:

$$
u_r = \phi_r(z_r) + \frac{1}{\sigma_r} J_{\phi_r}(z_r)s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

Now take the expectation over the noise. If $$z_r$$ is symmetric about the origin and $$\phi_r$$ is odd, then $$\mathbb{E}[\phi_r(z_r)] = 0$$. Hence

$$
\mathbb{E}[u_r \mid x] = \frac{1}{\sigma_r} A_r s_r + \mathcal{O}\left(\frac{\lVert s_r \rVert^2}{\sigma_r^2}\right)
$$

where $$A_r := \mathbb{E}[J_{\phi_r}(z_r)]$$, determined solely by the noise distribution and the normalization scheme. The zeroth-order term of the covariance is

$$
\operatorname{Cov}(u_r \mid x) = \operatorname{Cov}(\phi_r(z_r)) + \mathcal{O}\left(\frac{\lVert s_r \rVert}{\sigma_r}\right)
$$

Let $$B_r := \operatorname{Cov}(\phi_r(z_r))$$. In the high-noise regime, the update can be written as

$$
u_r \approx \frac{1}{\sigma_r} A_r s_r + \zeta_r, \quad \mathbb{E}[\zeta_r \mid x] = 0, \quad \operatorname{Cov}(\zeta_r \mid x) \approx B_r
$$

This expression will be used throughout. The normalized update has two parts: an effective drift attenuated by $$1/\sigma_r$$, and residual noise whose magnitude does not grow with $$\sigma_r$$. Note that the signal term is attenuated by $$1/\sigma_r$$ as well. Normalization does not amplify the signal; it merely bounds the output noise magnitude.

As a consistency check, consider $$L_2$$ normalization. If $$z_r \sim \mathcal{N}(0, I_{d_r})$$ and the noise is isotropic, rotational symmetry gives $$A_r = a_{d_r} I_{d_r}$$, where $$a_{d_r} \sim d_r^{-1/2}$$. Hence

$$
\mathbb{E}[u_r \mid x] \approx \frac{a_{d_r}}{\sigma_r} s_r
$$

Also $$\mathbb{E}\lVert u_r \rVert_2^2 = 1$$. The drift shrinks with $$1/\sigma_r$$, while the second moment stays fixed.

## 4. Dynamical Consequences: Stability and Error Floor

The above gives the statistical form of the update itself. Now we place it back into the optimization dynamics. First, we examine how large a learning rate the single-step decrease condition allows, and then the steady-state error under a local quadratic model.

### 4.1 Global Single-Step Decrease: Maximum Stable Step Size

Suppose the objective $$F$$ is $$L$$-smooth. By the descent lemma, for an update $$x_r^+ = x_r - \eta u_r$$ we have

$$
F(x^+) \le F(x) - \eta \sum_{r=1}^m \langle s_r, u_r \rangle + \frac{L}{2} \eta^2 \sum_{r=1}^m \lVert u_r \rVert_2^2
$$

Consider SGD first, i.e., $$u_r = s_r + \sigma_r z_r$$. Taking expectations:

$$
\mathbb{E}[F(x^+) \mid x] \le F(x) - \eta \sum_{r=1}^m \lVert s_r \rVert_2^2 + \frac{L}{2} \eta^2 \left( \sum_{r=1}^m \lVert s_r \rVert_2^2 + \sum_{r=1}^m d_r \sigma_r^2 \right)
$$

In the noise-dominated regime, the step size that guarantees expected decrease must roughly satisfy

$$
\eta_{\text{SGD}} \lesssim \frac{2 \sum_{r} \lVert s_r \rVert_2^2}{L \sum_{r} d_r \sigma_r^2}
$$

This restriction comes from the $$d_r \sigma_r^2$$ in the quadratic term.

Now consider blockwise $$L_2$$ normalization. Then $$\lVert u_r \rVert_2^2 = 1$$, so the quadratic term reduces to the constant $$m$$. Substituting the drift approximation from Section 3:

$$
\mathbb{E}[F(x^+) \mid x] \le F(x) - \eta \sum_{r=1}^m \frac{a_{d_r}}{\sigma_r} \lVert s_r \rVert_2^2 + \frac{L}{2} \eta^2 m + \mathcal{O}\left(\eta \sum_{r=1}^m \frac{\lVert s_r \rVert_2^3}{\sigma_r^2}\right)
$$

Ignoring higher-order terms, the step size upper bound is approximately

$$
\eta_{\text{norm}} \lesssim \frac{2 \sum_{r} a_{d_r} \lVert s_r \rVert_2^2 / \sigma_r}{L m}
$$

Here a direct difference emerges: the stable step size for SGD shrinks with $$1/\sigma^2$$, while for normalized updates it shrinks roughly with $$1/\sigma$$. The larger the noise, the more pronounced the difference.

However, this should not be overinterpreted. In a single-block, homoscedastic model, a larger stable step size does not imply a larger single-step decrease. For normalized methods, the single-step decrease at the optimal step size is approximately $$a_d^2 \lVert s \rVert^4 / (L\sigma^2)$$; for SGD, it is approximately $$\lVert s \rVert^4 / (Ld\sigma^2)$$. Since $$a_d^2 \sim 1/d$$, they are of the same order.

Thus, in this simplest model, normalization mainly changes the stability trade-off: the usable learning rate range is wider, but it does not automatically yield larger per-step progress. The real difference appears later, in the multi-block heteroscedastic setting.

### 4.2 Local Convergence: Steady-State Error Floor

Now consider the local quadratic model:

$$
F(x) = \frac{1}{2} \sum_{r=1}^m \lambda_r \lVert x_r \rVert_2^2
$$

At this point $$s_r = \lambda_r x_r$$.

For the normalized update, the approximate dynamics of the $$r$$-th block are

$$
x_{r, t+1} = \left( 1 - \eta \frac{a_{d_r} \lambda_r}{\sigma_r} \right) x_{r, t} - \eta \zeta_{r, t}
$$

Taking the expectation of the mean squared error and using $$\mathbb{E}[\zeta_{r,t}] = 0$$ to eliminate the cross term:

$$
\mathbb{E}\lVert x_{r, t+1} \rVert_2^2 \approx \left( 1 - 2\eta \frac{a_{d_r} \lambda_r}{\sigma_r} \right) \mathbb{E}\lVert x_{r, t} \rVert_2^2 + \eta^2
$$

At steady state, solving gives

$$
\mathbb{E}\lVert x_{r, \infty} \rVert_2^2 \approx \frac{\eta^2}{2\eta \frac{a_{d_r} \lambda_r}{\sigma_r}} = \frac{\eta \sigma_r}{2 a_{d_r} \lambda_r} \sim \mathcal{O}\left(\frac{\eta \sigma_r \sqrt{d_r}}{\lambda_r}\right)
$$

The SGD recursion is $$x_{r, t+1} = (1 - \eta \lambda_r) x_{r, t} - \eta \xi_{r, t}$$, with corresponding steady-state variance

$$
\mathbb{E}\lVert x_{r, \infty} \rVert_2^2 \approx \frac{\eta d_r \sigma_r^2}{2\lambda_r} \sim \mathcal{O}\left(\frac{\eta d_r \sigma_r^2}{\lambda_r}\right)
$$

The trade-off here is clear. Local contraction slows down because the contraction rate changes from $$\eta \lambda_r$$ to $$\eta a_{d_r} \lambda_r / \sigma_r$$. The larger the noise, the slower the pull-back. On the other hand, the steady-state error drops from the $$\sigma_r^2$$ level to the $$\sigma_r$$ level, and the dimensional factor also weakens.

This is still not an improvement in direction estimation. It merely controls the magnitude of the stochastic update, so long-term fluctuations do not inflate with the raw noise variance as they do in SGD.

## 5. From Theory to Practice: Heteroscedasticity and Implicit Blockwise Adaptive Learning Rate

### 5.1 Why We Must Look at Cross-Block Noise Differences

In a single-block homoscedastic model, the optimal single-step decrease is of the same order for normalization and for SGD. So why is normalization still valuable in LLM training?

Because the real model is not a single block, nor is it homoscedastic. The noise levels of different parameter blocks can differ greatly. For example, the embedding layer may have large gradient variance because token access is sparse; some intermediate layers receive denser signal, so their noise scale sits in a different regime. In the notation above, $$\sigma_r$$ can differ significantly across blocks.

On the other hand, training pipelines typically still use a single global learning rate $$\eta$$. Of course, layer-wise learning rates are possible, but the most common setup in the main pipeline remains a shared schedule. Thus the global learning rate must accommodate all parameter blocks simultaneously.

Putting these two facts together makes the problem concrete: if the noise levels of different blocks differ greatly, how should a global $$\eta$$ be chosen? The SGD learning rate is constrained by the highest-noise block; normalization automatically gives different effective step sizes to different blocks.

### 5.2 Normalization Is Equivalent to a Stochastic Blockwise Learning Rate

For blockwise $$L_2$$ normalization, the update of the $$r$$-th block can be directly rewritten as

$$
x_r^+ = x_r - \eta \frac{g_r}{\lVert g_r \rVert_2} = x_r - \left(\frac{\eta}{\lVert g_r \rVert_2}\right) g_r
$$

That is, it is equivalent to using a stochastic effective step size

$$
\eta_r^{\text{eff}} = \frac{\eta}{\lVert g_r \rVert_2}
$$

to perform SGD. In the high-noise regime, $$\lVert g_r \rVert_2 \approx \sigma_r \sqrt{d_r}$$, so

$$
\eta_r^{\text{eff}} \approx \frac{\eta}{\sigma_r \sqrt{d_r}}
$$

This is implicit inverse-noise-scale weighting. High-noise blocks get smaller effective step sizes; low-noise blocks get larger effective step sizes.

This and the magnitude saturation of Section 3 are two ways of writing the same phenomenon. Magnitude saturation looks at the scale of the output update; the stochastic blockwise learning rate rewrites the normalized update back into the SGD form.

### 5.3 Why the Global Learning Rate Is Constrained by the Highest-Noise Block

Let the maximum noise standard deviation be $$\sigma_{\max} := \max_r \sigma_r$$. If we want the steady-state mean squared error $$\mathbb{E}\lVert x_{r,\infty} \rVert_2^2$$ of every parameter block to stay below the threshold $$\varepsilon$$, the global learning rate must satisfy the most stringent block constraint.

For SGD, according to the steady-state formula in Section 4.2, we need

$$
\frac{\eta d_r \sigma_r^2}{2\lambda_r} \lesssim \varepsilon
$$

to hold for all blocks. Therefore,

$$
\eta \lesssim \min_r \frac{2\lambda_r \varepsilon}{d_r \sigma_r^2} \propto \frac{1}{\sigma_{\max}^2}
$$

Once the learning rate is constrained by the highest-noise block, the contraction rate of all blocks becomes $$\eta \lambda_r$$. Low-noise blocks could have used larger step sizes, but the shared learning rate slows them all down together.

For normalized updates, the steady-state condition becomes

$$
\frac{\eta \sigma_r}{2 a_{d_r} \lambda_r} \lesssim \varepsilon
$$

Therefore

$$
\eta \lesssim \min_r \frac{2 a_{d_r} \lambda_r \varepsilon}{\sigma_r} \propto \frac{1}{\sigma_{\max}}
$$

Looking only at the global upper bound, the suppression caused by the worst-noise block has been relaxed from $$1/\sigma_{\max}^2$$ to $$1/\sigma_{\max}$$.

Also note that the effective contraction rate is not the same for all blocks. For block $$r$$ it is

$$
\text{rate}^{\text{norm}}_r = \eta \frac{a_{d_r} \lambda_r}{\sigma_r}
$$

It varies with $$1/\sigma_r$$. Low-noise blocks contract quickly, high-noise blocks contract slowly. More precisely, because $$\sigma_r$$ is the standard deviation, this is inverse-noise-scale weighting, not strict inverse-variance weighting.

### 5.4 Practical Implications for LLM Training

Translating the above model back to LLM training, I would interpret it as follows:

1. If the gradient variance of certain layers or parameter blocks is particularly high, row normalization/block normalization will automatically reduce their effective step size. Embedding-related parameters, for instance, then need not rely entirely on a hand-tuned smaller learning rate.

2. In the noise-dominated regime, the stable learning rate for SGD-like updates is limited by high-noise blocks, otherwise loss spikes become likely. The stability condition for normalized updates contracts with $$1/\sigma$$ rather than $$1/\sigma^2$$, so the usable range is wider.

3. Mixed-precision training introduces additional quantization noise and may also amplify noise differences across parameter blocks. Normalization methods are less sensitive to such heteroscedasticity.

These claims all rely on the simplified model in this article and cannot replace real large-model experiments. But the model does suggest one line of explanation: the value of normalized optimizers may not lie in more accurate direction estimation, but in the way they change the effective learning rate of parameter blocks with different noise scales.

## 6. Summary

Under the model where intra-block noise is approximately isotropic, blockwise $$L_2$$ normalization can be understood as

$$
\text{SGD with } \eta_r^{\text{eff}} = \frac{\eta}{\lVert g_r \rVert}
$$

It does not improve single-step direction accuracy, nor does it increase information content. What it does is constrain the update magnitude to a fixed scale, so the noise magnitude in the raw gradient is not directly transmitted to the parameter update.

Homogeneity of degree zero prevents the output magnitude from growing with the input noise scale. The cost is also clear: the effective drift weakens with $$1/\sigma_r$$.

In the single-block homoscedastic model, the steady-state error drops from $$\mathcal{O}(\sigma^2)$$ to $$\mathcal{O}(\sigma)$$, and the maximum stable step size is relaxed from $$\mathcal{O}(1/\sigma^2)$$ to $$\mathcal{O}(1/\sigma)$$. However, the single-step decrease at the optimal step size remains of the same order, so it cannot be described as a free speedup.

In the multi-block heteroscedastic model, normalization automatically assigns each block an effective step size of $$\eta_r^{\text{eff}} \propto 1/\sigma_r$$, preventing the global learning rate from being completely dominated by high-noise blocks.

Therefore, in noise-dominated training dynamics, I prefer to understand the normalized update as a form of magnitude control. It does not make the direction estimate more accurate, but it makes the update magnitude more stable. This property manifests as a stability trade-off in single-block models, and in the multi-block heteroscedastic structure of LLMs, it resembles a usable adaptive mechanism.

## 7. Numerical Verification

Below we use 5 sets of Monte Carlo experiments to check the scaling relations above.

Notation is as follows:

- Noise standard deviation: $$\sigma$$;
- Update vector: SGD takes $$u=g$$, normalized takes $$u=g/\lVert g \rVert_2$$;
- Drift strength: $$\lvert \mathbb{E}\langle u,s \rangle \rvert$$;
- Noise strength: $$\operatorname{tr}(\operatorname{Cov}(u))$$;
- Maximum stable learning rate: under the local quadratic model $$F(x)=\frac{1}{2}\lVert x \rVert_2^2$$, the $$\eta_{\max}$$ estimated from the expected single-step decrease condition.

### 7.1 Direction Invariance (Normalization Does Not Correct the Single-Step Direction)

{% include figure.liquid
  path='assets/img/post-04-14/01_direction_invariance.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Direction Invariance Verification'
  zoomable=true
  alt='Direction Invariance Verification'
%}

The figure has two panels.

Left panel (Direction Cosine Is Preserved): $$\cos(g,s)$$ on the x-axis, $$\cos(g/\lVert g \rVert,s)$$ on the y-axis. If normalization does not change the single-step direction, the scatter points should lie along the diagonal $$y=x$$.

Right panel (Numerical Difference Distribution): shows the distribution of $$\Delta:=\lvert \cos(g,s)-\cos(g/\lVert g \rVert,s) \rvert$$, with the y-axis on a logarithmic scale. If the direction is preserved, $$\Delta$$ should be concentrated near 0.

In the results, the scatter points mostly lie on $$y=x$$, and $$\max \Delta=1.11\times10^{-16}$$. This is consistent with Section 2: for $$L_2$$ normalization, the direction is not corrected.

### 7.2 Magnitude Saturation and Drift Scaling

{% include figure.liquid
  path='assets/img/post-04-14/02_noise_compression.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Magnitude Saturation and Drift Scaling'
  zoomable=true
  alt='Magnitude Saturation and Drift Scaling'
%}

This figure also has two panels.

Left panel (Drift Scaling in Noise-Dominated Regime): examines the drift term. The x-axis is $$\sigma$$, the y-axis is $$\lvert \mathbb{E}\langle u,s \rangle \rvert$$, both on a logarithmic scale. The main check is whether the curve approaches $$\sigma^{-1}$$.

Right panel (Noise Amplification vs Noise Compression): examines noise strength. The x-axis is $$\sigma$$, the y-axis is $$\operatorname{tr}(\operatorname{Cov}(u))$$. SGD should approach $$\sigma^2$$, while normalization should approach a horizontal line.

The results are fairly clean. The trace of the SGD noise covariance grows with $$\sigma^2$$, with a slope of approximately $$2.00$$; the covariance trace of the normalized update is approximately constant, with a slope of approximately $$0.00$$. Meanwhile, the normalized drift term decays with $$1/\sigma$$, with a slope of approximately $$-0.99$$. This is consistent with the claims in Section 3: the signal response weakens, but the output noise magnitude is bounded as well.

### 7.3 Maximum Stable Step Size Scaling

{% include figure.liquid
  path='assets/img/post-04-14/03_eta_scaling.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Maximum Stable Step Size Scaling'
  zoomable=true
  alt='Maximum Stable Step Size Scaling'
%}

This figure checks the learning rate scaling from Section 4.1.

The x-axis is $$\sigma$$, the y-axis is the estimated $$\eta_{\max}$$, both on a logarithmic scale. Dots are empirical estimates; the dashed and dotted lines are theoretical curves and reference power laws ($$\sigma^{-2}$$, $$\sigma^{-1}$$).

Read the slopes. SGD is approximately $$-1.99$$, corresponding to $$\eta_{\max}\propto 1/\sigma^2$$; normalization is approximately $$-0.99$$, corresponding to $$\eta_{\max}\propto 1/\sigma$$.

### 7.4 Steady-State Error Floor Scaling

{% include figure.liquid
  path='assets/img/post-04-14/04_steady_state_scaling.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Steady-State Error Floor Scaling'
  zoomable=true
  alt='Steady-State Error Floor Scaling'
%}

This figure corresponds to the steady-state error floor in Section 4.2.

The x-axis is $$\sigma$$, and the y-axis is the steady-state $$\mathbb{E}\lVert x \rVert_2^2$$, both on a logarithmic scale. The two reference dashed lines correspond to the $$\sigma^2$$ and $$\sigma$$ scalings, respectively.

The parallelism between the curves and the reference lines is consistent with the derivation: SGD is approximately $$\mathcal{O}(\sigma^2)$$ with a slope of about $$2.00$$; normalized updates are approximately $$\mathcal{O}(\sigma)$$ with a slope of about $$1.00$$.

### 7.5 Heteroscedasticity and Implicit Blockwise Adaptive Learning Rate

{% include figure.liquid
  path='assets/img/post-04-14/05_heteroscedastic_blocks.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Heteroscedasticity and Implicit Blockwise Adaptive Learning Rate'
  zoomable=true
  alt='Heteroscedasticity and Implicit Blockwise Adaptive Learning Rate'
%}

This figure examines blockwise adaptation in the heteroscedastic setting.

The left panel (Global eta Under Heteroscedastic Noise) plots the blockwise error $$\mathbb{E}\lVert x_r \rVert_2^2$$ against the iteration count, with the y-axis on a logarithmic scale. The four curves correspond to the trajectories of SGD/normalized updates on low-noise and high-noise blocks.

The right panel (Implicit Inverse-Variance Weighting) plots the effective contraction coefficient $$\kappa_r:=\mathbb{E}\langle u_r,s_r \rangle/\lVert s_r \rVert_2^2$$. The x-axis is the block category, divided into low-noise and high-noise.

The experimental setup is $$\sigma_{\text{low}}=0.5,\ \sigma_{\text{high}}=4.0$$. Under normalized updates, the effective contraction coefficient for the low-noise block is approximately $$7.49$$ times that of the high-noise block, close to the ratio of standard deviations $$\sigma_{\text{high}}/\sigma_{\text{low}} = 8$$. This supports the explanation in Section 5: what we see here is inverse-noise-scale weighting, not strict inverse-variance weighting.

These 5 experiments examine, respectively, direction invariance, drift scaling, covariance saturation, maximum stable step size, the steady-state error floor, and blockwise adaptation under heteroscedasticity. They do not prove that real LLM training necessarily behaves this way, but they do show that the simplified model above is numerically self-consistent.

## References

[1] Shazeer, N., & Stern, M. (2018). [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://proceedings.mlr.press/v80/shazeer18a.html). In _Proceedings of the 35th International Conference on Machine Learning (ICML 2018)_, _Proceedings of Machine Learning Research_, 80, 4596-4604.

[2] Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., & Bernstein, J. (2024). [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/).

[3] Deng, S., Ouyang, Z., Pang, T., Liu, Z., Jin, R., Yu, S., & Yang, Y. (2026). [RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](https://arxiv.org/abs/2603.20527). _arXiv preprint_ arXiv:2603.20527.

[4] Gu, Y., & Xie, Z. (2026). [Mano: Restriking Manifold Optimization for LLM Training](https://arxiv.org/abs/2601.23000). _arXiv preprint_ arXiv:2601.23000.

[5] Wang, M., Wang, J., Zhang, J., Wang, W., Pei, P., Cai, X., E, W., & Wu, L. (2025). [GradPower: Powering Gradients for Faster Language Model Pre-Training](https://arxiv.org/abs/2505.24275). _arXiv preprint_ arXiv:2505.24275.

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026noise-training-dynamics,
  title={在 LLM 语境下，梯度里的噪声会如何影响 training dynamics？},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/noise-training-dynamics/}
}
```
