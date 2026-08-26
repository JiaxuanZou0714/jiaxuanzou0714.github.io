---
layout: post
title: "On the Hypersphere: μP Scaling of Optimizers with the Hyperball Mechanism"
date: 2026-03-07 10:24:00
description: "Starting from the first principles of continuous-time spherical dynamics, this article explores how the intrinsic dependence of weight norms disrupts hyperparameter alignment, and rigorously derives the underlying mathematical mechanisms by which various Hyperball optimizer variants achieve feature-space alignment."
tags: [deep-learning, spherical-dynamics, muP, optimizer]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/spherical-hyperball/
ref: spherical-hyperball
related_posts: false
---

In modern neural network training, transferring hyperparameters across model scales is always a core issue. For architectures with normalization (e.g., RMSNorm), the key is no longer the parameters themselves, but the evolution of features on the hypersphere [[2]](https://arxiv.org/abs/2006.08419).

Since normalized features satisfy $\lVert z \rVert_2 = \sqrt{n}$, when aligning across widths, there is no need to worry about the feature norm itself; the components of $z$ already satisfy $\lvert z_i \rvert = \Theta(1)$. We only need to ensure:
> Ensure that the evolution rate of the normalized feature $z$, $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$, remains at a stable magnitude.

Below, we first explain the problems with standard optimizers, then derive the scaling laws for the Hyperball family of optimizers proposed by Wen et al. [[1]](https://tinyurl.com/muonh). If you haven't read the previous article in this geometric line, you can first read ["On the Hypersphere: From Spherical Dynamics to μP"](/en/blog/2026/spherical-dynamics-mup/); that article establishes the basic correspondence between spherical dynamics and $\mu$P learning rate scaling in RMSNorm architectures, without the Hyperball constraint.

## 1. Basic Setup and Continuous-Time Spherical Mapping

Let the hidden width of the network be $n$. The coordinate and norm magnitudes of key variables are as follows.

The input satisfies $\lVert x \rVert_2^2 = \Theta(n)$, with coordinate components of magnitude $x_j = \Theta(1)$.

Define the unnormalized feature $y_t = W_t x$. Under the assumption that the input directions are isotropically dispersed in the subspace of $W_t$, the squared $L_2$ norm of $y_t$ is:

$$
\lVert y_t \rVert_2^2 = x^T W_t^T W_t x = \Theta\left(\frac{\lVert x \rVert_2^2}{n} \text{Tr}(W_t^T W_t)\right)
$$

From $\lVert x \rVert_2^2 = \Theta(n)$ and $\text{Tr}(W_t^T W_t) = \lVert W_t \rVert_F^2$, we obtain:

$$
\lVert y_t \rVert_2^2 = \Theta\left(\frac{n}{n} \lVert W_t \rVert_F^2\right) = \Theta(\lVert W_t \rVert_F^2)
$$

Thus:

$$
\lVert y_t \rVert_2 = \Theta(\lVert W_t \rVert_F)
$$

After applying RMSNorm, the feature passed backward is:

$$
z = \sqrt{n}\frac{y_t}{\lVert y_t \rVert_2}
$$

The corresponding Jacobian matrix is:

$$
J = \frac{\partial z}{\partial y_t} = \frac{\sqrt{n}}{\lVert y_t \rVert_2}P_y, \quad P_y = I - \frac{y_t y_t^T}{\lVert y_t \rVert_2^2}
$$

where $P_y$ is the orthogonal projection onto the hyperplane normal to $y_t$. Introducing the update matrix $U_t$, the spherical dynamics of the feature can be written as:

$$
\frac{dz}{dt} = \frac{\sqrt{n}}{\lVert y_t \rVert_2} P_y \frac{dy_t}{dt} = \frac{\sqrt{n}}{\Theta(\lVert W_t \rVert_F)} P_y \left(-\eta U_t x\right)
$$

Requiring the coordinate components to align to $\Theta(1)$ yields the general alignment equation without norm constraints:

$$
\eta = \frac{\Theta(\lVert W_t \rVert_F)}{\sqrt{n} \lvert (P_y (U_t x))_i \rvert}
$$

## 2. Intrinsic Radius Dependence and Dynamic Balance Dilemma of Standard Optimizers [[2]](https://arxiv.org/abs/2006.08419)

From the above, it is clear that what truly determines the spherical angular velocity of the feature is not the base learning rate $\eta$, but the effective spherical step size $\eta_{\mathrm{eff}}^{(i)}(t)$:

$$
\eta_{\mathrm{eff}}^{(i)}(t) := \eta \frac{\sqrt{n} \lvert (P_y (U_t x))_i \rvert}{\lVert W_t \rVert_F}
$$

Here, the Frobenius norm of the weight matrix $\lVert W_t \rVert_F$ acts as the intrinsic radius controlling the spherical angular velocity. For standard optimizers with decoupled weight decay, this radius evolves according to a difference equation. Consider the stochastic gradient descent update with weight decay coefficient $\lambda$:

$$
W_{t+1} = W_t - \eta \left( \frac{\partial \mathcal{L}}{\partial W_t} + \lambda W_t \right)
$$

Taking the squared Frobenius norm on both sides of the above and expanding:

$$
\lVert W_{t+1} \rVert_F^2 = (1 - \eta \lambda)^2 \lVert W_t \rVert_F^2 - 2 \eta (1 - \eta \lambda) \left\langle W_t, \frac{\partial \mathcal{L}}{\partial W_t} \right\rangle + \eta^2 \left\lVert \frac{\partial \mathcal{L}}{\partial W_t} \right\rVert_F^2
$$

The normalization mechanism endows the weight matrix with scale invariance [[2]](https://arxiv.org/abs/2006.08419); the network output does not change with the weight magnitude, so the gradient tensor is orthogonal to the current weight tensor, i.e., $\langle W_t, \frac{\partial \mathcal{L}}{\partial W_t} \rangle = 0$. Thus the cross term is zero:

$$
\lVert W_{t+1} \rVert_F^2 = (1 - \eta \lambda)^2 \lVert W_t \rVert_F^2 + \eta^2 \left\lVert \frac{\partial \mathcal{L}}{\partial W_t} \right\rVert_F^2
$$

To eliminate the dependence of the gradient norm on the current weight norm, introduce the unit gradient $\tilde{G}_t = \lVert W_t \rVert_F \frac{\partial \mathcal{L}}{\partial W_t}$. Substituting and taking the square root on both sides, and performing a Taylor expansion under $\eta \lambda \ll 1$, we obtain the master equation for the evolution of the intrinsic radius in a single step:

$$
\lVert W_{t+1} \rVert_F \approx \lVert W_t \rVert_F - \lambda \eta \lVert W_t \rVert_F + \frac{\eta^2}{2 \lVert W_t \rVert_F^3} \lVert \tilde{G}_t \rVert_F^2
$$

At the stationary state, the expectation of the intrinsic radius remains unchanged, i.e., $\mathbb{E}[\lVert W_{t+1} \rVert_F] = \mathbb{E}[\lVert W_t \rVert_F]$. Let $L = \mathbb{E}[\lVert \tilde{G}_t \rVert_F^2 \mid W_t]$ be the expected squared unit gradient norm. Setting the increment to zero yields the asymptotic limit $w^*$:

$$
w^* = \sqrt[4]{\frac{L\eta}{2\lambda}}
$$

To analyze the dependence of $w^*$ on width $n$, we first need the magnitude of the norm of the unit gradient $\tilde{G}_t$. Let the gradient of the loss with respect to the normalized feature $z$ be $g_z$. By the chain rule, the gradient received by the unnormalized feature $y_t$ is $\frac{\partial \mathcal{L}}{\partial y_t} = \frac{\sqrt{n}}{\lVert y_t \rVert_2} P_y g_z$. Then, taking the gradient with respect to the weight tensor $W_t$ and substituting into the definition of $\tilde{G}_t$, combined with the identity $\lVert y_t \rVert_2 = \frac{\lVert x \rVert_2}{\sqrt{n}} \lVert W_t \rVert_F$ under the isotropy assumption, we have:

$$
\tilde{G}_t = \lVert W_t \rVert_F \frac{\sqrt{n}}{\frac{\lVert x \rVert_2}{\sqrt{n}} \lVert W_t \rVert_F} (P_y g_z) x^T = \frac{n}{\lVert x \rVert_2} (P_y g_z) x^T
$$

Its squared Frobenius norm is:

$$
\lVert \tilde{G}_t \rVert_F^2 = \frac{n^2}{\lVert x \rVert_2^2} \lVert P_y g_z \rVert_2^2 \lVert x \rVert_2^2 = n^2 \lVert P_y g_z \rVert_2^2
$$

Assume the upstream normalized gradient satisfies $(g_z)_i = \Theta(1)$, then $\lVert g_z \rVert_2^2 = \Theta(n)$. Since $P_y$ is an orthogonal projection onto a hyperplane, it only attenuates the norm by a factor of $\Theta(1)$, so $\lVert P_y g_z \rVert_2^2 = \Theta(n)$. Thus, the expected squared unit gradient norm $L$ scales as:

$$
L = \mathbb{E}[\lVert \tilde{G}_t \rVert_F^2 \mid W_t] = n^2 \Theta(n) = \Theta(n^3)
$$

Substituting the magnitude of $L$ into $w^*$, and setting the weight decay coefficient $\lambda = \Theta(1)$, we obtain the intrinsic radius scaling at the stationary point:

$$
\lVert W_t \rVert_F = w^* = \Theta\left(\sqrt[4]{n^3 \eta}\right) = \Theta(n^{3/4} \eta^{1/4})
$$

To satisfy the alignment criterion $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$, consider the stochastic gradient descent update direction $U_t = \frac{\partial \mathcal{L}}{\partial W_t} = \frac{\sqrt{n}}{\lVert y_t \rVert_2} (P_y g_z) x^T$. Its action on the input feature $x$ is:

$$
U_t x = \frac{\sqrt{n}}{\lVert y_t \rVert_2} (P_y g_z) x^T x = \frac{\sqrt{n} \lVert x \rVert_2^2}{\lVert y_t \rVert_2} (P_y g_z)
$$

Substituting $\lVert x \rVert_2^2 = \Theta(n)$, $\lVert y_t \rVert_2 = \Theta(\lVert W_t \rVert_F)$, and using $P_y^2 = P_y$:

$$
P_y (U_t x) = \frac{n \sqrt{n}}{\Theta(\lVert W_t \rVert_F)} (P_y g_z)
$$

Thus, the magnitude of the absolute value of the coordinate component is $\lvert (P_y (U_t x))_i \rvert = \Theta\left(\frac{n \sqrt{n}}{\lVert W_t \rVert_F}\right)$. Substituting into the feature evolution alignment equation:

$$
\frac{\eta \sqrt{n}}{\lVert W_t \rVert_F} \Theta\left(\frac{n \sqrt{n}}{\lVert W_t \rVert_F}\right) = \Theta\left(\frac{\eta n^2}{\lVert W_t \rVert_F^2}\right) = \Theta(1)
$$

Substituting the intrinsic radius scaling at the stationary point $\lVert W_t \rVert_F = \Theta(n^{3/4} \eta^{1/4})$ into the above constraint:

$$
\frac{\eta n^2}{(n^{3/4} \eta^{1/4})^2} = \frac{\eta n^2}{n^{3/2} \eta^{1/2}} = \eta^{1/2} n^{1/2} = \Theta(1)
$$

Solving yields the learning rate scaling law required for cross-scale alignment:

$$
\eta = \Theta\left(\frac{1}{n}\right)
$$

Substituting back into the intrinsic radius expression, we obtain the norm magnitude of the system in the dynamical equilibrium state:

$$
w^* = \Theta\left(n^{3/4} (n^{-1})^{1/4}\right) = \Theta(n^{1/2}) = \Theta(\sqrt{n})
$$

This shows that if the system can instantaneously reach dynamical equilibrium, then when $\eta = \Theta(1/n)$, the intrinsic weight norm of standard optimization will automatically converge to $\Theta(\sqrt{n})$, consistent with the magnitude corresponding to standard initialization variance. Its core insight is consistent with mup.

However, in practical engineering, relying on this mechanism to approach the natural stationary point to maintain hyperparameter alignment encounters three problems:

1. Convergence delay: $w^*$ is an asymptotic limit, and the weight norm does not instantly reach the equilibrium point. In the early training phase, $\lVert W_t \rVert_F$ has not yet converged to $\Theta(\sqrt{n})$, so the effective step size on the hypersphere is incorrect.
2. Imbalance upon learning rate changes: Modern training commonly uses multi-stage learning rate schedules. Once $\eta$ decays, the corresponding equilibrium point $w^*$ immediately changes, but the weight norm takes time to catch up with the new equilibrium point. During this transition period, the alignment condition does not hold.
3. Orthogonality assumption does not always hold: The above derivation relies on $\langle W_t, \frac{\partial \mathcal{L}}{\partial W_t} \rangle = 0$, i.e., the gradient is strictly orthogonal to the weights. However, in networks with residual connections, this condition is not strictly satisfied; the cross term is nonzero, and the weight norms of each layer drift independently, preventing unified alignment.

## 3. Hyperball constraint and unified master equation

To fundamentally eliminate the interference of the intrinsic radius $\lVert W_t \rVert_F$ on the dynamics, the Hyperball mechanism proposed by Wen et al. [[1]](https://tinyurl.com/muonh) explicitly introduces the tangent space projection operator $\Pi_W$, constraining the weight update direction to the hypersphere with the initial norm as radius.

In the continuous-time limit of the discrete update rule, the weight dynamics equation is:

$$
\frac{dW}{dt} = -\eta \lVert W_0 \rVert_F \Pi_W\left(\frac{u_t}{\lVert u_t \rVert_F}\right)
$$

Under standard initialization, the matrix element sampling variance is $\frac{1}{n}$, so the initial constant $R = \lVert W_0 \rVert_F = \Theta(\sqrt{n})$. The tangent space projection operator $\Pi_W$ ensures that the Frobenius norm of the weight matrix remains constant at any time $t$:

$$
\lVert W_t \rVert_F = \lVert W_0 \rVert_F = \Theta(\sqrt{n})
$$

Therefore, the dynamic denominator becomes an invariant in time and width:

$$
\lVert y_t \rVert_2 = \Theta(\lVert W_t \rVert_F) = \Theta(\sqrt{n})
$$

Using the linearity property of the projection operator $P_y(\Pi_W(A)x) = P_y(Ax)$, and substituting back into the Jacobian matrix from Section 1, the prefactor simplifies to the constant $\frac{\sqrt{n}}{\Theta(\sqrt{n})} = \Theta(1)$. Thus, we obtain the unified master equation governing the dynamics of all Hyperball variants:

$$
\frac{dz}{dt} = -\eta \Theta(\sqrt{n}) \frac{1}{\lVert u_t \rVert_F} P_y (u_t x)
$$

For $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$ to hold, the learning rate $\eta$ must satisfy:

$$
\eta = \frac{\lVert u_t \rVert_F}{\Theta(\sqrt{n}) \lvert (P_y (u_t x))_i \rvert}
$$

| | without hyperball | with Hyperball |
| :--- | :--- | :--- |
| learning rate constraint | $\eta = \Theta\left(\frac{\lVert W_t \rVert_F}{\sqrt{n} \lvert (P_y (U_t x))_i \rvert}\right)$ | $\eta = \Theta\left(\frac{\lVert u_t \rVert_F}{\sqrt{n} \lvert (P_y (u_t x))_i \rvert}\right)$ |
{: .table .table-striped .table-sm .w-auto .mx-auto style="font-size: 0.8em;"}


> This transformation decouples the alignment of hyperparameters from the convergence state of the system. Without Hyperball, to obtain the correct spherical angular velocity, the network must rely on weight decay and gradient orthogonality to reach the equilibrium state $\lVert W_t \rVert_F = \Theta(\sqrt{n})$; with Hyperball, $\lVert u_t \rVert_F$ is explicitly placed in the numerator of the learning rate, so the setting of $\eta$ depends only on the gradient structure of the current update step, and is no longer constrained by the historical trajectory of the weight norm.

## 4. Alignment derivations for specific Hyperball variants

Below we analyze the update characteristics of different optimizers under specific assumptions and give their respective alignment learning rates.

### 4.1 Alignment derivation for SGDH

Assume the upstream gradient $g = \nabla_z L$ has coordinate components of magnitude $\Theta(1)$. The base gradient update matrix is $u_t = \Theta(1) (P_y g) x^T$. Since $\lVert P_y g \rVert_2 = \Theta(\sqrt{n})$ and $\lVert x \rVert_2 = \Theta(\sqrt{n})$, we have:

$$
\lVert u_t \rVert_F = \Theta(1) \lVert P_y g \rVert_2 \lVert x \rVert_2 = \Theta(n)
$$

Now consider the action of the update matrix on the input feature vector:

$$
u_t x = \Theta(1) (P_y g) (x^T x) = \Theta(1) (P_y g) \Theta(n) = \Theta(n) P_y g
$$

After applying the orthogonal projection operator:

$$
P_y (u_t x) = P_y (\Theta(n) P_y g) = \Theta(n) P_y g
$$

By the premise assumption $\lvert (P_y g)_i \rvert = \Theta(1)$, we have $\lvert (P_y (u_t x))_i \rvert = \Theta(n)$. Substituting into the master equation:

$$
\eta = \frac{\Theta(n)}{\Theta(\sqrt{n}) \Theta(n)} = \Theta\left(\frac{1}{\sqrt{n}}\right)
$$

### 4.2 Alignment derivation for AdamH

Here we only retain the result of the assumption $\lVert u_t \rVert_F = \Theta(n)$ shared by AdamH and MuonH below. A more detailed Frobenius norm estimate can be found in ["Frobenius Norm Estimates for the Update Matrices of Adam and Muon Optimizers"](/en/blog/2026/optimizer-update-matrix-norm/), which devotes a separate section to this step.

After extracting the sign matrix of the gradient, the update matrix $u_t$ contains $n^2$ elements with absolute value $1$, so its norm is:

$$
\lVert u_t \rVert_F = \sqrt{n^2} = n = \Theta(n)
$$

When momentum or large batches are introduced, the signs of the update matrix and the coordinate distribution of the input vector $x$ are approximately independent. By the central limit theorem, the linear combination of $n$ independent terms gives $\lvert (u_t x)_i \rvert = \Theta(\sqrt{n})$, so $\lvert (P_y (u_t x))_i \rvert = \Theta(\sqrt{n})$. Substituting into the master equation:

$$
\eta = \frac{\Theta(n)}{\Theta(\sqrt{n}) \Theta(\sqrt{n})} = \Theta(1)
$$

### 4.3 Alignment derivation for MuonH and its isotropy advantage

In the actual engineering implementation of Muon, the orthogonalized update is further adjusted by a learning rate so that the root mean square magnitude of the update across different matrix shapes is consistent with standard optimizers.

Under this setting, the leading order of the Frobenius norm of the update matrix is:

$$
\lVert u_t \rVert_F = \Theta(n)
$$

Now consider the action of this update matrix on the current input feature $x$. Due to orthogonalization and root mean square alignment, after tangent space projection, the typical magnitude of each coordinate component is:

$$
\lvert (P_y (u_t x))_i \rvert = \Theta(\sqrt{n})
$$

Substituting into the master equation yields the optimal learning rate:

$$
\eta = \frac{\Theta(n)}{\Theta(\sqrt{n}) \Theta(\sqrt{n})} = \Theta(1)
$$

This shows that the root cause of drift in unconstrained Muon is that the intrinsic radius cannot automatically align across different widths; after MuonH enforces $\lVert W_t \rVert_F = \Theta(\sqrt{n})$, the leading term aligns to a constant learning rate.

Further comparing MuonH and AdamH, MuonH often provides more precise cross-scale alignment. The reason is that even with the Hyperball constraint, Adam's update matrix $u_t$ still relies on elementwise adaptive normalization, so the projected component $\lvert (P_y (u_t x))_i \rvert$ retains residual anisotropy; whereas Muon's orthogonalization flattens the singular value structure of the update matrix, making its angular action on the current feature more isotropic. Therefore, after the norm is explicitly fixed, MuonH has smaller residual alignment error across different widths.

## 5. Summary of Global Scaling Laws

Based on the above master equation in feature space, the learning rates $\eta$ required by the Hyperball family of optimizers to achieve feature angular velocity alignment under different geometric and statistical assumptions are summarized as follows:

| Hyperball optimizer variant | Update matrix norm $\lVert u_t \rVert_F$ | Required $\eta$ |
| :--- | :--- | :--- |
| SGDH | $\Theta(n)$ | $\Theta(1/\sqrt{n})$ |
| AdamH | $\Theta(n)$ | $\Theta(1)$ |
| MuonH | $\Theta(n)$ | $\Theta(1)$ |
{: .table .table-striped .table-sm .w-auto .mx-auto style="font-size: 0.8em;"}

## 6. Conclusion

Traditional optimization relies on the intrinsic weight norm to find natural equilibrium points, but this mechanism is deeply coupled with network width, scheduling strategy, and model architecture, so it cannot guarantee cross-scale consistency of spherical angular velocity when model scale changes. Hyperball eliminates this intrinsic dependence through geometric projection constraints on the hypersphere, simplifying the Jacobian prefactor to a scalar constant. The derivation shows that only by taking $\lvert \left(\frac{dz}{dt}\right)_i \rvert = \Theta(1)$ as the unified alignment criterion and cutting off the coupling between the intrinsic weight norm and hyperparameters can the scaling law of optimizer hyperparameters be clearly characterized. If you wish to further complete the assumptions on the update matrix norm used by AdamH / MuonH in this article, you can continue reading ["Frobenius Norm Estimates of the Update Matrices of Adam and Muon Optimizers"](/en/blog/2026/optimizer-update-matrix-norm/).

## References

[1] Wen, K., Dang, X., Lyu, K., Ma, T., & Liang, P. (2025). Fantastic Pretraining Optimizers and Where to Find Them 2.1: Hyperball Optimization. https://tinyurl.com/muonh

[2] Wan, R., Zhu, Z., Zhang, X., & Sun, J. (2020). Spherical Motion Dynamics: Learning Dynamics of Neural Network with Normalization, Weight Decay, and SGD. arXiv preprint arXiv:2006.08419. https://arxiv.org/abs/2006.08419

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026sphericalhyperball,
  title={球面之上：带有 Hyperball 机制的优化器的 μP 缩放},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={[https://jiaxuanzou0714.github.io/blog/2026/spherical-hyperball/](https://jiaxuanzou0714.github.io/blog/2026/spherical-hyperball/)}
}
```