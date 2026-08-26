---
layout: post
title: "Tensor Programs (Part 1): From the Spectral Conditions of Feature Learning to μP"
date: 2026-02-14 17:00:00
description: "This article introduces the entry paper of Greg Yang's Tensor Programs series—A Spectral Condition for Feature Learning—deriving the scaling conditions required for feature learning from the perspective of spectral norm, and from this re-deriving the Maximal Update Parametrization (μP)."
tags: [deep-learning, tensor-programs, muP, feature-learning]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/spectral-condition-feature-learning/
ref: spectral-condition-feature-learning
related_posts: false
---


> This is the first article in the Tensor Programs series guide. The entire series aims to introduce readers to the [Tensor Programs](https://thegregyang.com/) research program initiated by Greg Yang—an ambitious framework seeking to provide a unified mathematical foundation for width limits, feature learning, and hyperparameter transfer in deep learning. This article selects the entry-level paper *A Spectral Condition for Feature Learning* [1], recommended by Greg Yang himself, as the starting point. If you wish to continue along this line to see how it leads to the complete $\mu$P scaling derivation, you can proceed to [Tensor Programs (Part 2): From Tensor Programs to μP](/en/blog/2026/tensor-programs-mup-intuition/); if you prefer an alternative derivation that bypasses the probabilistic formalization of Tensor Programs and instead takes a geometric route, you can consult [On the Hypersphere: From Spherical Dynamics to μP](/en/blog/2026/spherical-dynamics-mup/).

## 0. Why Tensor Programs?

The core magic of deep learning lies in feature learning: models can automatically learn hierarchical, semantically rich representations from raw data. This ability allows neural networks to surpass traditional kernel methods, making the success of LLMs possible.

However, most current theoretical research still relies on the NTK mathematical framework. NTK describes the behavior of neural networks in the infinite-width limit, treating the network as a fixed kernel function whose predictions are essentially a kernel-weighted sum of the training samples (i.e., kernel regression, or linear regression on the random features at initialization).

$$
f(\boldsymbol{x}) = \sum_{i=1}^N \alpha_i K_{\text{NTK}}(\boldsymbol{x}, \boldsymbol{x}_i), \quad \text{其中 } K_{\text{NTK}}(\boldsymbol{x}, \boldsymbol{x}_i) = \langle \nabla f(\boldsymbol{x}; \boldsymbol{\theta}_0), \nabla f(\boldsymbol{x}_i; \boldsymbol{\theta}_0) \rangle.
$$

The drawback of the NTK framework is that, in its setting, the trained weights remain near initialization (also called lazy learning, because this is what allows a first-order Taylor expansion approximation). In this case, the neural network is effectively performing linear regression on the random features obtained at initialization, and the features do not change throughout training, which is completely detached from the feature learning that occurs in real scenarios.

The reality is that we need feature learning, and at the same time we are scaling up model width, but if we adopt the NTK setting, we will lose feature learning capability (falling into the lazy learning regime). This raises a crucial question: **when we want to scale up to wider models, how can we maintain the model's feature learning capability?**

This is precisely the core question that the Tensor Programs line of research seeks to answer. Tensor Programs is not merely a simple parametrization trick; it is a grand mathematical framework aimed at precisely characterizing the behavior of neural network computations in the infinite-width limit. Within this framework, Greg Yang and others derived a parametrization scheme that preserves feature learning in the infinite-width limit—namely, the renowned Maximal Update Parametrization (μP).

In other words, μP is merely a byproduct of the great mathematical framework of Tensor Programs (albeit an extremely useful one, as it allows us to tune hyperparameters on small models and zero-shot transfer them to large models, i.e., hyperparameter transfer). To truly understand the essence of μP, we need to return to more fundamental mathematical laws. The paper [1] introduced in this article provides exactly such an entry point, offering an intuitive understanding of the feature learning preservation condition through spectral norm, and showing how μP can be derived from this preservation condition.

---

## 1. What is Feature Learning?

Consider an $$L$$-layer MLP. Let the hidden representation of layer $$\ell$$ be $$\boldsymbol{h}_\ell(\boldsymbol{x}) \in \mathbb{R}^{n_\ell}$$, and its change after one step of gradient descent be $$\Delta \boldsymbol{h}_\ell(\boldsymbol{x})$$. We want the neural network to undergo feature learning; in plain terms, we want the features to change significantly, so we define feature learning as follows.

> Feature Learning: for all hidden layers $$\ell$$, as the width of each layer $$n_\ell \to \infty$$:
> $$
> \|\boldsymbol{h}_\ell(\boldsymbol{x})\|_2 = \Theta(\sqrt{n_\ell}), \qquad \|\Delta \boldsymbol{h}_\ell(\boldsymbol{x})\|_2 = \Theta(\sqrt{n_\ell}).
> $$

In other words, the "typical magnitude" of each component of the feature vector is $$\Theta(1)$$ (i.e., constant order, not varying with network width $$n$$), which is reasonable because common activation functions are designed to accept inputs of order $$O(1)$$ and produce outputs of order $$O(1)$$. Meanwhile, the components of the update during training are also of order $$\Theta(1)$$. This is also reasonable—larger updates would explode as width increases (causing training instability and divergence), while smaller updates would vanish as width increases (causing features to remain unchanged, i.e., lazy learning).

---

## 2. Spectral Norm vs. Frobenius Norm

Before discussing scaling, we need to distinguish between two matrix norms. For an $$m \times n$$ matrix $$\boldsymbol{A}$$:

- Spectral norm: $$\|\boldsymbol{A}\|_* = \max_{\|\boldsymbol{v}\|_2=1} \|\boldsymbol{A}\boldsymbol{v}\|_2 = \sigma_{\max}(\boldsymbol{A})$$.
- Frobenius norm: $$\|\boldsymbol{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$$.

The relationship between them is $$\|\boldsymbol{A}\|_* \leq \|\boldsymbol{A}\|_F \leq \sqrt{\mathrm{rank}(\boldsymbol{A})} \cdot \|\boldsymbol{A}\|_*$$.

The key difference is that for an iid Gaussian random matrix $$\boldsymbol{A} \in \mathbb{R}^{m \times n}$$ (with element variance $$\sigma^2$$),

$$
\|\boldsymbol{A}\|_F \approx \sigma \sqrt{mn}, \qquad \|\boldsymbol{A}\|_* \approx \sigma(\sqrt{m} + \sqrt{n}).
$$

The derivation can be found in Su Jianlin's blog ([Fast estimation of the spectral norm of random matrices](https://spaces.ac.cn/archives/11335)]).

The Frobenius norm essentially measures the "total energy" of all elements in a matrix, and it grows linearly with the matrix's dimension; in contrast, the spectral norm measures the maximum amplification factor of the matrix as a linear operator, more directly reflecting the behavior of $$\boldsymbol{A}\boldsymbol{v}$$. From this perspective, since neural networks contain a large number of linear operations of the form $$\boldsymbol{A}\boldsymbol{v}$$, the spectral norm seems to be a more reasonable metric.

This difference also explains why scaling schemes based on the Frobenius norm or element-wise magnitude may yield different conclusions from those based on the spectral norm.

---

## 3. Deriving the Spectral Condition from Feature Learning

With the objective definition of feature learning and the tool of spectral norm in hand, we can begin to derive: what conditions exactly must be satisfied to guarantee that feature learning occurs?

### 3.1 Forward Propagation: $$\scriptsize\|\boldsymbol{h}_\ell(\boldsymbol{x})\|_2 = \Theta(\sqrt{n_\ell})$$

Consider a simplified setting first: an $$L$$-layer linear MLP: $$\boldsymbol{h}_\ell(\boldsymbol{x}) = \boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}(\boldsymbol{x})$$, with input satisfying $$\|\boldsymbol{x}\|_2 = \Theta(\sqrt{n_0})$$.

Our goal is to have $$\|\boldsymbol{h}_\ell\|_2 = \Theta(\sqrt{n_\ell})$$. Each layer does $$\boldsymbol{h}_\ell = \boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}$$, and submultiplicativity of the spectral norm tells us:

$$
\|\boldsymbol{h}_\ell\|_2 = \|\boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}\|_2 \leq \|\boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_{\ell-1}\|_2.
$$

Assuming $$\|\boldsymbol{h}_{\ell-1}\|_2 = \Theta(\sqrt{n_{\ell-1}})$$ already holds (induction hypothesis), to have $$\|\boldsymbol{h}_\ell\|_2 = \Theta(\sqrt{n_\ell})$$, we need the "amplification factor" of the linear operator to be exactly:

$$
\|\boldsymbol{W}_\ell\|_* = \Theta\!\left(\sqrt{\frac{n_\ell}{n_{\ell-1}}}\right).
$$

This is the constraint on the spectral norm of the weight matrices that follows naturally from the numerical stability requirement of feature learning.

But an upper bound alone is not enough—the inequality above only says the spectral norm cannot be too large, otherwise features would explode. Is the submultiplicative upper bound tight? Could it be that $$\|\boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}\|_2 \ll \|\boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_{\ell-1}\|_2$$, causing features to vanish layer by layer? Actually not, and we have the following observation.

> Claim 1: For randomly initialized weight matrices (Gaussian or semi-orthogonal), when fan-out $$\geq$$ fan-in, i.e., $$n_\ell \geq n_{\ell-1}$$, we have
> $$
> \|\boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}\|_2 = \Theta(\|\boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_{\ell-1}\|_2).
> $$

This can be verified for Gaussian initialization using the law of large numbers: $$\|\boldsymbol{W}_\ell \boldsymbol{h}\|^2 = \sum_{i=1}^{n_\ell} (\sum_j W_{ij} h_j)^2$$, each $$\sum_j W_{ij} h_j$$ has variance $$\sigma_\ell^2 \|\boldsymbol{h}\|^2$$, and summing over $$n_\ell$$ independent terms concentrates around the expectation $$n_\ell \sigma_\ell^2 \|\boldsymbol{h}\|^2$$. Combined with $$\|\boldsymbol{W}_\ell\|_* \approx \sigma_\ell(\sqrt{n_\ell} + \sqrt{n_{\ell-1}}) = \Theta(\sigma_\ell \sqrt{n_\ell})$$ (when $$n_\ell \geq n_{\ell-1}$$), this guarantees that the lower and upper bounds are of the same order.

Thus, the submultiplicative upper bound is tight under random initialization, and $$\|\boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$ is a necessary and sufficient condition.

### 3.2 Gradient updates: $$\scriptsize\|\Delta \boldsymbol{h}_\ell\|_2 = \Theta(\sqrt{n_\ell})$$

Feature learning requires not only that the initial features have the correct magnitude, but also that the feature updates during training, $$\Delta \boldsymbol{h}_\ell$$, are $$\Theta(\sqrt{n_\ell})$$. A similar derivation gives constraints on $$\Delta \boldsymbol{W}_\ell$$.

For gradient descent with batch size 1, the weight update for layer $$\ell$$ is:

$$
\Delta \boldsymbol{W}_\ell = -\eta_\ell \nabla_{\boldsymbol{W}_\ell} \mathcal{L}.
$$

where $$\eta_\ell$$ is the learning rate for layer $$\ell$$. The question now is: what does $$\nabla_{\boldsymbol{W}_\ell} \mathcal{L}$$ look like?

Let the backpropagated signal be $$\boldsymbol{\delta}_\ell = \partial \mathcal{L} / \partial \boldsymbol{h}_\ell \in \mathbb{R}^{n_\ell}$$. Since forward propagation is $$\boldsymbol{h}_\ell = \boldsymbol{W}_\ell \boldsymbol{h}_{\ell-1}$$, applying the chain rule to the $$(i,j)$$-th element of the weight matrix:

$$
\frac{\partial \mathcal{L}}{\partial W_{\ell,ij}} = \frac{\partial \mathcal{L}}{\partial h_{\ell,i}} \cdot \frac{\partial h_{\ell,i}}{\partial W_{\ell,ij}} = \delta_{\ell,i} \cdot h_{\ell-1,j}.
$$

In matrix form, this is $$\nabla_{\boldsymbol{W}_\ell} \mathcal{L} = \boldsymbol{\delta}_\ell \boldsymbol{h}_{\ell-1}^\top$$. Substituting back into the update formula:

$$
\Delta \boldsymbol{W}_\ell = -\eta_\ell \boldsymbol{\delta}_\ell \boldsymbol{h}_{\ell-1}^\top.
$$

This is a rank-one matrix (the outer product of two vectors). Since it is rank-one, its spectral norm equals its Frobenius norm:

$$
\|\Delta \boldsymbol{W}_\ell\|_* = \|\Delta \boldsymbol{W}_\ell\|_F = \eta_\ell \|\boldsymbol{\delta}_\ell\|_2 \cdot \|\boldsymbol{h}_{\ell-1}\|_2.
$$

Here there is a very nice property.
> Claim 2: Since the right singular vector of $$\Delta \boldsymbol{W}_\ell$$ is exactly $$\boldsymbol{h}_{\ell-1}$$, we have
> $$
> \|\Delta \boldsymbol{W}_\ell \cdot \boldsymbol{h}_{\ell-1}\|_2 = \|\Delta \boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_{\ell-1}\|_2.
> $$

Submultiplicativity holds with equality here—the gradient update and the input features are perfectly aligned. This shows that the gradient update during training is exactly along the direction that most affects the current features.

Therefore, to have $$\|\Delta \boldsymbol{h}_\ell\|_2 = \|\Delta \boldsymbol{W}_\ell \cdot \boldsymbol{h}_{\ell-1}\|_2 = \Theta(\sqrt{n_\ell})$$, combined with $$\|\boldsymbol{h}_{\ell-1}\|_2 = \Theta(\sqrt{n_{\ell-1}})$$, we need:

$$
\|\Delta \boldsymbol{W}_\ell\|_* = \Theta\!\left(\sqrt{\frac{n_\ell}{n_{\ell-1}}}\right).
$$

### 3.3 Summary: Spectral Scaling Condition

Combining the derivations from both directions, we obtain the core result of the paper:

> Spectral Scaling Condition: for each layer $$\ell = 1, \ldots, L$$, require
>
> $$
> \|\boldsymbol{W}_\ell\|_* = \Theta\!\left(\sqrt{\frac{n_\ell}{n_{\ell-1}}}\right), \qquad \|\Delta \boldsymbol{W}_\ell\|_* = \Theta\!\left(\sqrt{\frac{n_\ell}{n_{\ell-1}}}\right).
> $$

The meaning of this condition is now very clear: the weight matrix $$\boldsymbol{W}_\ell \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$$, as a linear operator mapping $$n_{\ell-1}$$-dimensional vectors to $$n_\ell$$-dimensional vectors, its "amplification factor" (spectral norm) needs to exactly match the dimension ratio of input to output. If it is too large, features explode; if too small, features vanish or learning stalls.

### 3.4 Generalization: From Toy Model to Real Networks

Although the above derivation is based on simplified assumptions, the original paper [1] proves that the conclusions of the Spectral Scaling Condition hold in a broader setting.

- Nonlinear activation functions: After adding a nonlinear activation $$\boldsymbol{h}'_\ell = \phi(\boldsymbol{h}_\ell)$$, as long as the activation function does not change the magnitude of the feature vectors (i.e., satisfies $$\|\boldsymbol{h}'_\ell\|_2 = \Theta(\|\boldsymbol{h}_\ell\|_2)$$), then $$\Delta \boldsymbol{W}_\ell$$ remains a rank-one matrix and satisfies the perfect alignment property $$\|\Delta \boldsymbol{W}_\ell \boldsymbol{h}'_{\ell-1}\|_2 = \|\Delta \boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}'_{\ell-1}\|_2$$. Therefore, the conclusions from the linear case apply completely.

- Batch size > 1: When $$B > 1$$, the update $$\Delta \boldsymbol{W}_\ell = \frac{1}{B} \sum \Delta \boldsymbol{W}_\ell^{(i)}$$ is no longer a rank-one matrix and cannot perfectly align with all input vectors. However, as long as $$B$$ is independent of width $$n$$ and the update terms do not cancel each other maliciously, we still have alignment in the scaling sense:

    $$
    \|\Delta \boldsymbol{W}_\ell \boldsymbol{h}_\ell(\boldsymbol{x}_i)\|_2 = \Theta(\|\Delta \boldsymbol{W}_\ell\|_* \cdot \|\boldsymbol{h}_\ell(\boldsymbol{x}_i)\|_2)
    $$

    This suffices to ensure that the Spectral Scaling Condition remains valid. Interestingly, the paper observes that even with large batch sizes, the update matrix maintains a numerically low-rank structure.

    {% include figure.liquid
        path="assets/img/post-02-14/low_rank.png"
        class="img-fluid rounded z-depth-1 mx-auto d-block"
        width="auto"
        zoomable=true
        alt="The numerical low-rank structure of the update matrix"
    %}

- Multi-step training: The evolution of gradients depends on the two properties "correct spectral norm magnitude" and "correct feature propagation magnitude." The paper points out that as long as the update does not cancel the initial weights extremely perfectly ($$\scriptsize\|\boldsymbol{W} + \Delta \boldsymbol{W}\|_* = \Theta(\|\boldsymbol{W}\|_* + \|\Delta \boldsymbol{W}\|_*)$$), then the weights after one update will maintain the above properties. By induction, feature learning continues to hold in subsequent training steps.

- Adaptive optimizers (Adam): For optimizers like Adam that process gradients element-wise, the paper proves in the appendix that when the width is large, element-wise nonlinear processing preserves the Frobenius norm of the matrix (up to a constant factor), and the gradient still exhibits properties similar to outer products of independent vectors, so the conclusions apply as well.

---

## 4. From Spectral Conditions to μP

To satisfy the Spectral Scaling Condition, the most direct method is to apply spectral normalization to weights and gradients. For example, we can enforce:

$$
\boldsymbol{W}_\ell = \sigma \sqrt{\frac{n_\ell}{n_{\ell-1}}} \frac{\boldsymbol{W}'_\ell}{\|\boldsymbol{W}'_\ell\|_*}, \qquad \Delta \boldsymbol{W}_\ell = -\eta \sqrt{\frac{n_\ell}{n_{\ell-1}}} \frac{\nabla_{\boldsymbol{W}_\ell} \mathcal{L}}{\|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_*}.
$$

Although this method can quickly verify the theory, computing the spectral norm (largest singular value) of large matrices is extremely expensive and infeasible in practical training.

Fortunately, we do not need to explicitly compute the spectral norm. The paper shows that by analyzing the scaling laws of random matrices, one can choose appropriate layer-wise initialization variances $$\sigma_\ell$$ and learning rates $$\eta_\ell$$ to automatically satisfy the Spectral Scaling Condition. This is the essence of μP.

### 4.1 Initialization Scaling

Assume $$\boldsymbol{W}_\ell = \sigma_\ell \boldsymbol{W}'_\ell$$, where the elements of $$\boldsymbol{W}'_\ell$$ are iid standard normal. By random matrix theory:

$$
\|\boldsymbol{W}_\ell\|_* \approx \sigma_\ell (\sqrt{n_\ell} + \sqrt{n_{\ell-1}}).
$$

To have $$\|\boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$, we need:

$$
\sigma_\ell = \Theta\!\left(\frac{\sqrt{n_\ell / n_{\ell-1}}}{\sqrt{n_\ell} + \sqrt{n_{\ell-1}}}\right) = \Theta\!\left(\frac{1}{n_{\ell-1}}\right) \quad \text{（当隐藏层等宽 $n_\ell = n$ 时）}.
$$

### 4.2 Learning Rate Scaling



How to determine the learning rate $$\eta_\ell$$ to satisfy $$\|\Delta \boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$? The key challenge here is to determine the scaling of the gradient $$\|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_*$$.

We can derive this by performing a first-order Taylor expansion of the loss function $$\mathcal{L}$$.
Each gradient update $$\Delta \boldsymbol{W}_\ell$$ aims to cause a change in the output $$\Delta \boldsymbol{h}_L(\boldsymbol{x})$$, which in turn causes a change in the loss of order $$\Theta(1)$$ ($$\Delta \mathcal{L} = \Theta(1)$$).

Using the properties of the trace inner product, the loss change can be approximated as:

$$
\Delta \mathcal{L} \approx \langle \Delta \boldsymbol{W}_\ell, \nabla_{\boldsymbol{W}_\ell} \mathcal{L} \rangle = \Theta(\|\Delta \boldsymbol{W}_\ell\|_F \cdot \|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_F) = \Theta(\|\Delta \boldsymbol{W}_\ell\|_* \cdot \|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_*).
$$

Here we use our observation under the low-rank update: since the matrix is approximately rank-one (or low-rank), its Frobenius norm is of the same order as its spectral norm.

Substituting our desired $$\Delta \mathcal{L} = \Theta(1)$$ and the Spectral Scaling Condition $$\|\Delta \boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$, we can directly solve for the scaling of the gradients:

$$
\|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_* = \Theta(\sqrt{n_{\ell-1} / n_\ell}).
$$

Since $$\Delta \boldsymbol{W}_\ell = -\eta_\ell \nabla_{\boldsymbol{W}_\ell} \mathcal{L}$$, to satisfy $$\|\Delta \boldsymbol{W}_\ell\|_* = \Theta(\sqrt{n_\ell / n_{\ell-1}})$$, the learning rate must be set to:

$$
\eta_\ell = \frac{\|\Delta \boldsymbol{W}_\ell\|_*}{\|\nabla_{\boldsymbol{W}_\ell} \mathcal{L}\|_*} = \Theta\left(\frac{n_\ell}{n_{\ell-1}}\right).
$$

This gives an intuitive explanation for the μP learning rate scaling: for a standard $$n_\ell = n$$ hidden layer, the learning rate should be $$\Theta(1)$$; for the output layer (assuming $$n_L=1$$), the learning rate should be $$\Theta(1/n)$$.

### 4.3 Spectral Parametrization

Combining the derived initialization and learning rate results, the paper summarizes the Spectral Parametrization, which is one of the main contributions of the paper.

> If the initialization scaling and learning rate for each layer $\ell$ are chosen as follows, then the Spectral Scaling Condition holds and feature learning is achieved:
> $$
> \sigma_\ell = \Theta \left( \frac{1}{\sqrt{n_{\ell-1}}} \min \left\{ 1, \sqrt{\frac{n_\ell}{n_{\ell-1}}} \right\} \right), \qquad \eta_\ell = \Theta \left( \frac{n_\ell}{n_{\ell-1}} \right).
> $$

This unified formula covers all layers:
- For hidden layers (typically $n_\ell \approx n_{\ell-1}$), $\sigma_\ell = \Theta(1/n_{\ell-1})$, $\eta_\ell = \Theta(1)$.
- For the output layer ($n_\ell \ll n_{\ell-1}$, e.g., $n_L=1$), $\sigma_\ell = \Theta(1/n_{\ell-1})$, $\eta_\ell = \Theta(1/n_{\ell-1})$.

This is fully consistent with the μP table (Table 3 of Yang et al., 2021). In other words, the spectral scaling condition provides an equivalent but more intuitive derivation of μP.

---

## 5. Comparison with Other Parametrizations

### 5.1 Standard Parametrization（SP）

The mainstream Kaiming/Xavier/LeCun initializations use $$\sigma_\ell = \Theta(1/\sqrt{n_{\ell-1}})$$, with width-independent learning rates.

This means:

$$
\|\boldsymbol{W}_\ell\|_* \approx \frac{1}{\sqrt{n_{\ell-1}}} (\sqrt{n_\ell} + \sqrt{n_{\ell-1}}) = \Theta\!\left(1 + \sqrt{\frac{n_\ell}{n_{\ell-1}}}\right).
$$

When $$n_\ell \gg n_{\ell-1}$$, this is much larger than $$\sqrt{n_\ell / n_{\ell-1}}$$, but more critically, SP has an overly large spectral norm at the output layer (fan-out $$\ll$$ fan-in), which can cause the output to diverge as width increases.

SP uses a fixed learning rate (width-independent), which is actually too small for wide hidden layers—the spectral norm of the update decays with width, leading to insufficient feature learning.

### 5.2 Neural Tangent Parametrization（NTP）

NTP parametrizes weights as $$\boldsymbol{W}_\ell / \sqrt{n_{\ell-1}}$$, with a width-independent learning rate. It can be verified that this is equivalent to $$\sigma_\ell = \Theta(1/\sqrt{n_{\ell-1}})$$, $$\eta_\ell = \Theta(1/n_{\ell-1})$$.

The output layer's $$\sigma_L$$ is $$\sqrt{n_{L-1}}$$ times larger than in μP, which amplifies the gradients of all intermediate layers through backpropagation; however, the overly small learning rate $$\eta_\ell = 1/n_{\ell-1}$$ pushes this amplification back down. The final result is:

$$
\|\Delta \boldsymbol{W}_\ell\|_* \propto \frac{\sqrt{n_{L-1}}}{n_{\ell-1}} \to 0 \quad (n \to \infty).
$$

The spectral norm of the weight update decays to zero with width—this is the hallmark of lazy learning / kernel regime: features freeze, and the network behavior degenerates to NTK.

### 5.3 Summary

| Scheme | Initialization $$\sigma_\ell$$ (hidden layer) | Learning rate $$\eta_\ell$$ (hidden layer) | Feature Learning? |
| --- | --- | --- | --- |
| SP | $$1/\sqrt{n}$$ | $$\Theta(1)$$ | ✗ |
| NTP | $$1/\sqrt{n}$$ | $$1/n$$ | ✗ |
| μP / Spectral | $$1/n$$ | $$\Theta(1)$$ | ✓ |
{: .table .table-striped}


---

## 6. Experimental Verification

To verify the above theoretical derivations, the paper conducts experiments on MLPs of varying widths. The figure below shows the scaling behavior of internal features and weight changes under NTP and μP scaling. The horizontal axis is the network width $$n$$, and the vertical axis is the feature change and weight change.

{% include figure.liquid
    path="assets/img/post-02-14/experiments.png"
    class="img-fluid rounded z-depth-1 mx-auto d-block"
    width="auto"
    zoomable=true
    alt="Training performance under different parameterization schemes"
%}

We can observe:
1. Feature change (left plot): Under μP scaling, the feature change $$\frac{\|\boldsymbol{h}_2(\boldsymbol{x}) - \boldsymbol{h}_2^0(\boldsymbol{x})\|_2}{\|\boldsymbol{h}_2^0(\boldsymbol{x})\|_2}$$ remains at a constant order $$\Theta(1)$$, independent of width; whereas under NTP scaling, the feature change decays as $$n^{-1/2}$$ with increasing width. This means that under NTP scaling, as the model becomes wider, feature learning gradually diminishes and eventually degenerates into the Lazy Regime.
2. Weight change (right plot): Under μP scaling, the spectral norm change of the weights $$\frac{\|\boldsymbol{W}_2 - \boldsymbol{W}_2^0\|_*}{\|\boldsymbol{W}_2^0\|_*}$$ also does not decay with width (remaining $$\Theta(1)$$), while under NTP scaling it decays significantly.

This confirms that only μP is able to maintain non-trivial feature learning in the large-width limit.

---

## 7. How to Understand the Unique Maximal Scaling

Spectral Scaling Condition gives the unique maximal scaling.

Specifically, if any $$\|\boldsymbol{W}_\ell\|_*$$ or $$\|\Delta \boldsymbol{W}_\ell\|_*$$ exceeds $$\Theta(\sqrt{n_\ell / n_{\ell-1}})$$, training diverges as width increases. Conversely, overly small scaling leads to insufficient feature learning, or even falls into the lazy learning regime. μP (spectral condition) is precisely the unique solution that makes feature learning as sufficient as possible in every layer.

This is also the origin of the word "maximal" in "maximal update parametrization."

---

## 8. Relationship with the Tensor Programs Series

This paper presents a method for deriving μP using basic linear algebra, bypassing the Tensor Programs formalism. However, it should be noted that the true power of Tensor Programs lies in its universality: it can handle arbitrary architectures (not just MLPs), arbitrary optimizers (not just SGD), and the entire training process (not just a single step).

| Problem | This Paper's Method | Tensor Programs |
| --- | --- | --- |
| Applicable architectures | MLP (generalizable) | Any architecture expressible in TP |
| Applicable optimizers | SGD (generalizable to Adam) | Any adaptive optimizer |
| Training steps | One step → multiple steps | Infinite steps (limit theorems) |
| Derivation difficulty | Basic linear algebra | Requires Master Theorem |

In the subsequent articles of this series, we will progressively delve into the formal framework of Tensor Programs, understanding how the Master Theorem provides the infinite-width limit for computations in arbitrary neural networks. Continuing from this article, the most natural next read is [“Tensor Programs (Part 2): From Tensor Programs to μP”]](/en/blog/2026/tensor-programs-mup-intuition/); if you are more concerned with geometric alignment under the RMSNorm architecture, you can also jump directly to [“On the Sphere: From Spherical Dynamics to μP”]](/en/blog/2026/spherical-dynamics-mup/).

---

## References

[1] Bernstein, J., Newhouse, L., Lee, J., Yang, G. (2024). A Spectral Condition for Feature Learning. *arXiv preprint arXiv:2310.17813*.

[2] Yang, G. & Hu, E. J. (2021). Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks. *ICML 2021*. arXiv:2011.14522.

[3] Yang, G., Hu, E. J., Babuschkin, I., et al. (2022). Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. *arXiv preprint arXiv:2203.03466*.

[4] Yang, G., Schnabel, T., Li, Z., & Du, S. S. (2023). Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks. *arXiv preprint arXiv:2310.02244*.

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026spectral,
  title={Tensor Programs (一)：从Feature Learning 的谱条件到 μP},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/spectral-condition-feature-learning/}
}
```
