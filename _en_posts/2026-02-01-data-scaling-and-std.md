---
layout: post
title: "How to align data scaling curves under different initialization magnitudes"
date: 2026-02-01 10:00:00
description: "We study the relationship between the empirical slope of data scaling and the initialization std, and propose a simple method to align data scaling curves under different initialization magnitudes."
tags: [scaling-law]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:                      # 目录配置
  sidebar: left           # 侧边栏目录 (left/right)
lang: en
permalink: /en/blog/2026/data-scaling-and-std/
ref: data-scaling-and-std
related_posts: false
---
## Phenomenon
In our previous blog ([Can We Derive Scaling Law From First Principles]({% post_url 2025-12-30-scaling-law %})), we discussed how scaling laws arise. In the preprint, we added an experimental section with a series of figures showing data scaling curves under different $\alpha$. For example, one figure shows the relationship between loss and datasize in the data-limited regime.

{% include figure.liquid
    path="assets/img/post-02-01/data-scaling-0p01.png"
    class="img-fluid rounded z-depth-1 mx-auto d-block"
    width="50%"
    zoomable=true
    alt="Scaling curve of loss versus data size under data-limited conditions when initialization std=0.01" 
%}

To ensure feature learning, in this figure we set the initialization std=0.01. But what happens if we increase the std to 0.05? The results are as follows.

{% include figure.liquid
    path="assets/img/post-02-01/data-scaling-0p05.png"
    class="img-fluid rounded z-depth-1 mx-auto d-block"
    width="50%"
    zoomable=true
    alt="Data scaling curve when initialization std=0.05, showing a shift compared to std=0.01" 
%}

We can see that with std=0.05, the data scaling curve shifts. What if we increase it further, say to 0.1? The results are as follows.

{% include figure.liquid
    path="assets/img/post-02-01/data-scaling-0p1.png"
    class="img-fluid rounded z-depth-1 mx-auto d-block"
    width="50%"
    zoomable=true
    alt="Data scaling curves at initialization std=0.1, where the empirical line deviates further from the theoretical prediction"
%}
As the initialization std increases, the empirical line gradually deviates from the theoretical prediction. What if we plot the empirical slope as a function of std? The results are as follows.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/post-02-01/fix_lr.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true 
            alt="Empirical slope versus initialization std at a fixed learning rate"
            caption="Empirical slope versus initialization std" 
            id="fig:std-slope"
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/post-02-01/all-curves-fix.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true 
            alt="Data scaling curves at a fixed learning rate and different initialization stds"
            caption="Data scaling curves under different initialization stds" 
            id="fig:all-curves-fix"
        %}
    </div>
</div>

This result is quite interesting. There could be many possible reasons, such as the std being too large causing the model to enter the lazy learning regime; or the increased std leading to optimization instability (this possibility is debatable, because I tested different numbers of epochs and the curve in [the above figure](#fig:std-slope)] is still stably reproduced). But we can set aside the cause for now and first consider how to align the data scaling curves under different initialization sizes.


## Preliminaries
We consider a two-layer ReLU network whose forward computation is written as $g(W_1,W_2;x)=W_2\phi(W_1 x)$, where $\phi=\mathrm{ReLU}$, and it satisfies positive homogeneity: for any $c>0$,

$$
\phi(cu)=c\,\phi(u).
$$

The input $x\in\mathbb R^K$ is set to be one-hot or a normalized vector satisfying $\lVert x\rVert^2=1$. The hidden layer first obtains the pre-activation $h=W_1 x$, then through ReLU obtains the activation $a=\phi(h)=\mathrm{ReLU}(h)$, and finally outputs $y=W_2 a\in\mathbb R^K$. The hidden layer width is denoted by $N$. The weight matrices $W_1,W_2$ are both initialized independently and identically distributed according to $\mathcal N(0,\sigma^2)$. For any matrix $A\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$ with elements having mean $0$ and variance $\sigma^2$, the expectation of the squared Frobenius norm satisfies $\mathbb E[\lVert A\rVert_F^2]=d_{\mathrm{out}}\cdot d_{\mathrm{in}}\cdot\sigma^2$, so in magnitude $\lVert A\rVert_F \propto \sigma$.

## Attempt 1 (Failed)

With a fixed learning rate lr, increasing the initialization std means that although the absolute step size remains unchanged, the step size relative to $\lVert W\rVert_F$ becomes smaller, and too small weight updates can cause the model to fall into the lazy learning regime. From this perspective, we need to compute the relative update ratio R:

$$
\mathcal{R} = \frac{\lVert\Delta W\rVert_F}{\lVert W\rVert_F} = \frac{\lVert\eta \cdot \nabla W\rVert_F}{\lVert W\rVert_F}
$$

This is the denominator part, which is straightforward.
For the second layer weight $W_2 \in \mathbb{R}^{K \times N}$:

$$
\mathbb{E}[\lVert W_2\rVert_F^2] = K \cdot N \cdot \sigma^2
$$

Taking the root mean square as the magnitude estimate:

$$
\lVert W_2\rVert_F \approx \sqrt{KN} \cdot \sigma \propto \sigma
$$

Next, we estimate $\lVert\nabla W_2\rVert_F$. Each element $h_j$ of the pre-activation $h$ follows $\mathcal{N}(0,\sigma^2)$. $\text{Var}(h_j) = \sigma^2$. For the activation layer output a: $a_j = \text{ReLU}(h_j)$. Since ReLU sets the negative half-axis to zero, the second moment is halved:

$$\mathbb{E}[a_j^2] = \frac{1}{2} \mathbb{E}[h_j^2] = \frac{1}{2} \sigma^2$$

Then the expectation of the squared norm of the activation vector $\lVert a\rVert^2 = \sum_{j=1}^N a_j^2$ is:

$$\mathbb{E}[\lVert a\rVert^2] = N \cdot \frac{1}{2}\sigma^2 \implies \lVert a\rVert \propto \sqrt{N}\sigma$$

---

For the second layer output y: $y_k = \sum_{j=1}^N W_{2, kj} a_j$. Assuming $W_2$ and $a$ are independent, and $\mathbb{E}[W_2]=0$, then 

$$\text{Var}(y_k) = \sum_{j=1}^N \text{Var}(W_{2, kj} a_j) = \sum_{j=1}^N \mathbb{E}[W_{2, kj}^2] \mathbb{E}[a_j^2]$$

Substituting the known terms:

$$\text{Var}(y_k) = N \cdot (\sigma^2) \cdot (\frac{1}{2}\sigma^2) = \frac{N}{2} \sigma^4$$

Then $\lVert y\rVert \approx \sqrt{\frac{N}{2}} \sigma^2 \propto \sqrt{N} \sigma^2$

---
The loss function is $L = \frac{1}{2} \lVert y - t\rVert^2$. The gradient formula is:

$$
\nabla_{W_2} L = (y - t) \cdot a^T.
$$

When $\sigma$ is large, the initial output $\lVert y\rVert$ is much larger than the target $\lVert t\rVert$ (which in our setting is one-hot, with magnitude 1). Therefore, the error term $\epsilon = y - t \approx y$. Now compute the norm of the gradient matrix $\nabla_{W_2} L$:

$$
\lVert\nabla_{W_2} L\rVert_F = \lVert(y - t) a^T\rVert_F = \lVert y - t\rVert_2 \cdot \lVert a\rVert_2.
$$

Substituting the result from step 2, $\lVert y - t\rVert \approx \lVert y\rVert \propto \sqrt{N} \sigma^2$, and $\lVert a\rVert \propto \sqrt{N} \sigma$. Multiplying gives the gradient magnitude:

$$
\lVert\nabla W_2\rVert_F \approx (\sqrt{N} \sigma^2) \cdot (\sqrt{N} \sigma) = N \sigma^3.
$$

---

Now we substitute the above results into the relative update ratio formula:

$$
\mathcal{R} = \frac{\lVert\Delta W_2\rVert_F}{\lVert W_2\rVert_F} = \frac{\eta \lVert\nabla W_2\rVert_F}{\lVert W_2\rVert_F}
$$

Substituting the magnitude relations:

$$
\mathcal{R} \approx \frac{\eta \cdot (N \sigma^3)}{\sqrt{KN} \cdot \sigma}.
$$

Simplifying (ignoring the constant $K$):

$$
\mathcal{R} \propto \frac{\eta \sigma^3}{\sigma} = \eta \sqrt{N} \sigma^2.
$$


To keep the relative update ratio constant, we need $\mathcal{R} = \text{Constant}$, i.e.:

$$
\eta \sigma^2 = C \implies \eta \propto \frac{1}{\sqrt{N} \sigma^2}.
$$

Since in our experiments $N$ is fixed and only the initialization std is varied, we only need $\eta \propto \frac{1}{\sigma^2}$. So I immediately implemented this adjustment in the code. The results are as follows
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/post-02-01/ada_lr.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true 
            alt="Empirical slope versus initialization std under adaptive learning rate"
            caption="Empirical slope versus initialization std" 
            id="fig:std-slope"
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/post-02-01/all-curves-ada.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true 
            alt="Data scaling curves under adaptive learning rate and different initialization stds"
            caption="Data scaling curves under different initialization stds" 
            id="fig:all-curves-ada"
        %}
    </div>
</div>

Let's put them in the same plot for comparison:
{% include figure.liquid 
    path="assets/img/post-02-01/compare-fix-ada.png" 
    class="img-fluid rounded z-depth-1" 
    zoomable=true 
    alt="Comparison of empirical slopes between fixed and adaptive learning rates"
    caption="Comparison of empirical slopes with and without adaptive learning rate" 
    id="fig:compare-fix-ada"
%}

We can see that the adaptive learning rate $\eta \propto \frac{1}{\sigma^2}$ does mitigate the effect of initialization std on the empirical slope of the scaling curve to some extent. But when the initialization std is small, it is still hard to mitigate. I think the reason might be that the approximation
$$
\lVert\nabla W_2\rVert_F \approx  N \sigma^3
$$ holds only when $\sigma$ is large (see the derivation above), so in the small initialization std regime, $\eta \propto \frac{1}{\sigma^2}$ may not be correct.


## Why Attempt 1 Failed (probably)


Besides the regime where the approximation holds, we can analyze it this way. Under the setting in our paper, writing the parameters as $W=\sigma U$, the risk corresponding to the model is


$$
\mathcal L_\sigma(U)=\sum_{k=1}^K p_k\underbrace{\big(\sigma^2 g(U;e_k)-1\big)^2}_{q_k}.
$$


Note that the target "1" is fixed and does not scale with $\sigma$ (in most cases, the true label also does not scale with $\sigma$), and this step already breaks scale invariance.


Performing gradient descent on $W$


$$
W^{t+1}=W^t-\eta \nabla_W \mathcal L(W^t),
$$


Switching to $U^t=W^t/\sigma$, we can derive the exact update rule


$$
U^{t+1}
=U^t-2\eta\sum_{k=1}^K p_k\big(\sigma^2 g(U^t;e_k)-1\big)\,\nabla_U g(U^t;e_k).
\tag{★}
$$


There is a factor that cannot be eliminated:


$$
(\sigma^2 g(U^t;e_k)-1).
$$


It depends on the current $U^t$ and the sample $k$, and is not a global constant. In general, it is impossible to achieve alignment with a scalar $\eta(\sigma)$. $\eta\propto 1/\sigma^2$ might be slightly better in some regimes (because it aligns with a dominant scale, such as the leading NTK term). But it cannot turn training for all $\sigma$ into the same problem; there will still be systematic drift and instability.


## Attempt 2 (Successful)

What exactly needs to be done to align the data scaling curves for all $\sigma$?


To cancel out the $\sigma^2$ scaling that automatically appears in the output during the forward pass, we set

$$
f_\sigma(x;W)=\frac{1}{\sigma^2}W_2\phi(W_1x),
\qquad W_i\sim\mathcal N(0,\sigma^2),
$$

and set the learning rate $\eta_\sigma\propto\sigma^2$.

If we write $W=\sigma U$, then we have

$$
\begin{aligned}
f_\sigma(x;\sigma U)
&=\frac1{\sigma^2}(\sigma U_2)\phi((\sigma U_1)x)\\
&=\frac1{\sigma^2}(\sigma U_2)\cdot(\sigma\phi(U_1x))\\
&=U_2\phi(U_1x)=:f_1(x;U).
\end{aligned}
$$


The right-hand side contains no $\sigma$ at all.




Let the objective function


$$
\mathcal J_\sigma(W)=\sum_{k=1}^K p_k\big(f_\sigma(e_k;W)-1\big)^2,
\quad f_\sigma(x;W)=\sigma^{-2}W_2\phi(W_1x).
$$


Let $U=W/\sigma$. Then it is easy to obtain:


1. The loss function itself does not contain $\sigma$:


$$
\mathcal J_\sigma(\sigma U)=\mathcal J_1(U).
$$


2. Gradient scaling (chain rule):


$$
\nabla_W\mathcal J_\sigma(\sigma U)=\frac1\sigma \nabla_U\mathcal J_1(U).
$$


3. Gradient descent trajectory:

Define $U^t:=W^t/\sigma$, then

$$
U^{t+1}=\frac{W^{t+1}}{\sigma}
=\frac{W^t}{\sigma}-\frac{\eta_\sigma}{\sigma}\nabla_W\mathcal J_\sigma(W^t)
=U^t-\frac{\eta_\sigma}{\sigma}\nabla_W\mathcal J_\sigma(\sigma U^t).
$$

Substituting $\nabla_W\mathcal J_\sigma(\sigma U^t)=\frac{1}{\sigma}\nabla_U\mathcal J_1(U^t)$, we get

$$U^{t+1}
=U^t-\frac{\eta_\sigma}{\sigma}\cdot\frac{1}{\sigma}\nabla_U\mathcal J_1(U^t)
=U^t-\frac{\eta_\sigma}{\sigma^2}\nabla_U\mathcal J_1(U^t).
$$

This is the key point: we need to choose $\eta_\sigma=\sigma^2\eta_0$ to obtain

$$
U^{t+1}=U^t-\eta_0\nabla_U\mathcal J_1(U^t),
$$



> The generalization to momentum SGD is also strictly equivalent.

In summary, we need to set

$$
f_\sigma(x;W_1,W_2)\;:=\;\frac{1}{\sigma^2}\,W_2\,\phi(W_1x).
$$


Initialization:


$$
W_1^{(0)},W_2^{(0)} \stackrel{\text{i.i.d.}}{\sim}\mathcal N(0,\sigma^2).
$$


Learning rate selection:


$$
\eta_\sigma = \sigma^2 \eta_0.
$$

Only in this way can the loss-$D$ curves under different initialization std be aligned. The experimental results are shown below.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/post-02-01/recons.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true 
            alt="Empirical slope versus initialization std after alignment"
            caption="Empirical slope versus initialization std after alignment" 
            id="fig:std-slope"
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/post-02-01/all-curves-recon.png" 
            class="img-fluid rounded z-depth-1" 
            zoomable=true 
            alt="Data scaling curves under different initialization stds after alignment"
            caption="Data scaling curves under different initialization stds after alignment" 
            id="fig:all-curves-recon"
        %}
    </div>
</div>

This result can be easily generalized to multi-layer ReLU networks. Suppose the network satisfies:


- $h_0=x$
- $h_\ell=\phi(W_\ell h_{\ell-1})$, $\ell=1,\dots,K-1$
- $f(x;W)=W_K h_{K-1}$


And at initialization, all layers $W_\ell\sim \mathcal N(0,\sigma^2)$. Define the normalized model output:


$$
\tilde f_\sigma(x;W)=\frac{1}{\sigma^K}f(x;W).
$$

As long as we choose:


$$
\eta_\sigma=\sigma^2 \eta_0
$$

we can align both the forward and backward processes.

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026scaling,
  title={如何对齐不同初始化大小下的 Data scaling 曲线},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/data-scaling-and-std/}
}
```