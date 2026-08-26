---
layout: post
title: "On the Hypersphere: From Spherical Dynamics to μP"
date: 2026-03-05 08:47:35
description: "This article departs from the probabilistic framework of Tensor Programs and, from the perspective of continuous-time spherical dynamics, rigorously derives how to achieve alignment of networks of different sizes by aligning the dynamics on the hypersphere in network architectures that apply RMSNorm."
tags: [deep-learning, spherical-dynamics, muP, rmsnorm]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: en
permalink: /en/blog/2026/spherical-dynamics-mup/
ref: spherical-dynamics-mup
related_posts: false
---

In modern large-model architectures, RMSNorm strips the influence of the weight vector's norm on the final output, so the effect of the gradient is essentially to change the direction of the weights. When deriving scaling laws for neural networks, the traditional Tensor Programs theory relies on microscopic statistics of coordinate systems for a large number of random variables. This article provides a completely different path: under a setting with strict scale invariance, we equivalently map the dynamics of the entire linear layer to motion in the tangent space of a hypersphere, and then rigorously derive the hyperparameter scaling rule required by μP. If you want to see the derivation along the Tensor Programs route, you can refer to [Tensor Programs (Part 1): From the Spectral Conditions of Feature Learning to μP](/en/blog/2026/spectral-condition-feature-learning/) and [Tensor Programs (Part 2): From Tensor Programs to μP](/en/blog/2026/tensor-programs-mup-intuition/); this article can be viewed as a geometric derivation path parallel to them.
> The core insight is that, when using rmsnorm, the object to be aligned should be the evolution rate of features on the hypersphere (i.e., angular velocity).

## 1. Scale Invariance and the Natural Orthogonality of the Gradient

Let the input feature be $x \in \mathbb{R}^n$, satisfying $\|x\|_2^2 = \Theta(n)$, i.e., having coordinate components of $\Theta(1)$. Let the hidden layer weight matrix be $W \in \mathbb{R}^{n \times n}$. The unnormalized linear mapping output (pre-activation) is $y = Wx \in \mathbb{R}^n$.

In the setting where RMSNorm is applied after the linear mapping, the normalized feature $z$ passed to the subsequent network is:

$$
z = \text{RMSNorm}(y) = \sqrt{n} \frac{y}{\|y\|_2}
$$

> This indicates that the feature vector $z$ is strictly constrained to the hypersphere $\mathbb{S}^{n-1}(\sqrt{n})$ of radius $\sqrt{n}$.

Let the loss function be $L(z)$. Since the pre-activation output passes through RMSNorm, the model output is invariant for any scalar $c > 0$, so the loss function has strict scale invariance with respect to the weight matrix:

$$
L(cW) = L(W)
$$

Differentiate both sides of the above equation with respect to the scalar $c$. Using the chain rule for multivariate composite functions, the derivative of the left-hand side is:

$$
\frac{d}{dc} L(cW) = \sum_{i=1}^n \sum_{j=1}^n \frac{\partial L(cW)}{\partial (cW_{ij})} \frac{\partial (cW_{ij})}{\partial c} = \sum_{i=1}^n \sum_{j=1}^n \frac{\partial L(cW)}{\partial (cW_{ij})} W_{ij} = \langle \nabla_{cW} L(cW), W \rangle_F
$$

where $\langle \cdot, \cdot \rangle_F$ denotes the Frobenius inner product of matrices. The right-hand side $L(W)$ is a constant independent of $c$, so its derivative with respect to $c$ is strictly $0$. Therefore:

$$
\langle \nabla_{cW} L(cW), W \rangle_F = 0
$$

Setting $c=1$, we obtain:

$$
\langle \nabla_W L(W), W \rangle_F = 0
$$

This shows that the Euclidean gradient $\nabla_W L(W)$ is orthogonal to the weight matrix $W$ at every point. Under continuous-time gradient flow, with learning rate $\eta$, the weight update rate is:

$$
\frac{dW}{dt} = - \eta \nabla_W L(W)
$$

Consider the time derivative of the squared Frobenius norm of the weight matrix, $\|W\|_F^2 = \langle W, W \rangle_F$:

$$
\frac{d}{dt} \|W\|_F^2 = \frac{d}{dt} \langle W, W \rangle_F = 2 \langle W, \frac{dW}{dt} \rangle_F = 2 \langle W, - \eta \nabla_W L(W) \rangle_F = - 2 \eta \langle W, \nabla_W L(W) \rangle_F = 0
$$

Since the derivative of the squared norm is strictly zero, the norm of the weights $\|W\|_F$ remains strictly constant during continuous optimization.

> Therefore, for scale-invariant weights $W$, ordinary gradient flow is naturally equivalent to Riemannian gradient flow on the hypersphere.

## 2. Spherical Mapping and the Jacobian Matrix

Let the upstream gradient scalar be $g = \nabla_z L \in \mathbb{R}^n$, assuming the coordinate magnitude of $g$ is $\Theta(1)$. First, compute the Jacobian matrix $J \in \mathbb{R}^{n \times n}$ from the normalized feature $z$ to the unnormalized feature $y$. Given $z = \sqrt{n} \frac{y}{\|y\|_2}$, consider the partial derivative of its $i$-th component with respect to the $j$-th component $y_j$:

$$
z_i = \sqrt{n} \frac{y_i}{( \sum_{k=1}^n y_k^2 )^{1/2}}
$$

Using the quotient rule for differentiation:

$$
\frac{\partial z_i}{\partial y_j} = \sqrt{n} \frac{\frac{\partial y_i}{\partial y_j} \|y\|_2 - y_i \frac{\partial \|y\|_2}{\partial y_j}}{\|y\|_2^2}
$$

where the component derivative is $\frac{\partial y_i}{\partial y_j} = \delta_{ij}$ (equal to $1$ when $i=j$, otherwise $0$). The derivative of the norm with respect to the component is:

$$
\frac{\partial \|y\|_2}{\partial y_j} = \frac{1}{2} \left( \sum_{k=1}^n y_k^2 \right)^{-1/2} (2 y_j) = \frac{y_j}{\|y\|_2}
$$

Substituting the above terms into the derivative formula yields:

$$
\frac{\partial z_i}{\partial y_j} = \sqrt{n} \frac{\delta_{ij} \|y\|_2 - y_i \frac{y_j}{\|y\|_2}}{\|y\|_2^2} = \frac{\sqrt{n}}{\|y\|_2} \left( \delta_{ij} - \frac{y_i y_j}{\|y\|_2^2} \right)
$$

Converting the above element-wise partial derivative results into matrix form, the Kronecker delta $\delta_{ij}$ corresponds to the identity matrix $I$, and the term $y_i y_j$ corresponds to the outer product matrix $y y^T$. Therefore, the Jacobian matrix $J$ is strictly equal to:

$$
J = \frac{\partial z}{\partial y} = \frac{\sqrt{n}}{\|y\|_2} \left( I - \frac{y y^T}{\|y\|_2^2} \right) = \frac{\sqrt{n}}{\|y\|_2} P_y
$$

where $P_y = I - \frac{y y^T}{\|y\|_2^2}$ is the orthogonal projection operator that projects Euclidean space vectors onto the hyperplane with $y$ as the normal vector. By the chain rule, the gradient of the loss with respect to $y$ is:

$$
\nabla_y L = J^T g = \frac{\sqrt{n}}{\|y\|_2} P_y g
$$

Next, compute the Euclidean gradient of the loss function with respect to the weight matrix $W$:

$$
\nabla_W L = (\nabla_y L) x^T = \left( \frac{\sqrt{n}}{\|y\|_2} P_y g \right) x^T
$$

Under continuous-time gradient flow, examine how the weight update feeds back into the feature $y$ in the forward pass. The dynamics equation for the unnormalized feature $y$ is:

$$
\frac{dy}{dt} = \frac{dW}{dt} x = - \eta \left( \frac{\sqrt{n}}{\|y\|_2} P_y g \right) x^T x
$$

Since the input features satisfy $\|x\|_2^2 = \Theta(n)$, the inner product $x^T x = \|x\|_2^2$. Substituting into the above equation yields:

$$
\frac{dy}{dt} = - \eta \Theta(n) \frac{\sqrt{n}}{\|y\|_2} P_y g
$$

## 3. Initialization Scaling (same approach as Tensor Programs)

The ordinary differential equation itself describes the rate of change of the system state and cannot directly internalize the scale of the system at the initial time. Therefore, the logic for deriving the initialization scaling rule in this section is highly consistent with the theoretical basis of the Tensor Programs framework. We need to rely on the law of large numbers for random variables to establish the initialization scaling.

In order for the model to extract and transmit meaningful features in the forward pass, and to prevent singularity or degeneration in the backward pass (eliminating the influence of $n$), the coordinates of the pre-activation vector must be maintained at the order of $\Theta(1)$.

Let the weights $W_{ij}$ be independent and identically distributed with mean $0$ and variance $\sigma_w^2$. Compute the variance of the unnormalized output $y_i$:

$$
\mathbb{E}[y_i^2] = \sum_{j=1}^n \mathbb{E}[W_{ij}^2] x_j^2 = n \sigma_w^2 \Theta(1)
$$

To satisfy the initial condition of geometric stability that the pre-activation vector coordinates are $\Theta(1)$, we must enforce $n \sigma_w^2 = \Theta(1)$.

> From this, the variance of the initialization must strictly satisfy:
> $$
> \sigma_w^2 = \Theta\left(\frac{1}{n}\right)
> $$

Under this well-defined initialization boundary condition, the squared norm of the entire vector concentrates probabilistically at $\|y\|_2^2 = \Theta(n)$.

## 4. Learning Rate Scaling (Spherical Dynamics Perspective)

After determining the initialization boundary condition consistent with Tensor Programs in the previous section, this section returns to the novel spherical dynamics perspective to strictly derive the scaling rule for the learning rate. To achieve hyperparameter alignment across models of different widths, the core requirement is that, regardless of how large the width $n$ grows, the coordinate-wise rate of change of the normalized feature $z$ on the hypersphere, $\frac{dz}{dt}$, must remain $\Theta(1)$.

Using the Jacobian matrix, map the dynamics of the unnormalized feature back to the hypersphere:

$$
\frac{dz}{dt} = J \frac{dy}{dt} = \left( \frac{\sqrt{n}}{\|y\|_2} P_y \right) \left( - \eta \Theta(n) \frac{\sqrt{n}}{\|y\|_2} P_y g \right)
$$

Extract the constant scalar and use the idempotence of the projection operator $P_y^2 = P_y$ to simplify and obtain the final spherical dynamics equation:

$$
\frac{dz}{dt} = - \eta \Theta(n) \frac{n}{\|y\|_2^2} P_y g
$$

Substitute $\|y\|_2^2 = \Theta(n)$ determined by the initialization boundary condition:

$$
\frac{dz}{dt} = - \eta \Theta(n) \frac{n}{\Theta(n)} P_y g = - \eta \Theta(n) P_y g
$$

Since the coordinate magnitude of the gradient projection $P_y g$ on the tangent space is $\Theta(1)$, to ensure strict alignment of feature learning across different widths $n$ and to keep the coordinate-wise rate of change of $\frac{dz}{dt}$ stable at $\Theta(1)$, it must hold that:

$$
\eta \cdot \Theta(n) = \Theta(1)
$$

> This strictly proves that for hidden layers applying RMSNorm, the learning rate $\eta$ must be inversely proportional to the width $n$, i.e., $\eta = \Theta(1/n)$.

It is easy to prove that, from the spherical dynamics perspective, aligning $\frac{dz}{dt}$ is equivalent to aligning the angular velocity $\frac{d\theta}{dt}$ of $z$.

## 5. Summary

The derivation path of spherical dynamics avoids the cumbersome process of probabilistic limits on matrix elements and directly exploits the spherical structure and Jacobian projection brought by RMSNorm. It proves that $\eta = \Theta(1/n)$ is the unique solution to ensure consistent angular velocity on the hypersphere for networks of different sizes. Continuing along this geometric route, one can next read ["On the Sphere: μP Scaling for Optimizers with the Hyperball Mechanism"](/en/blog/2026/spherical-hyperball/); that article further discusses how the scaling rules for different optimizers change after introducing the Hyperball constraint.

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026sphericaltomup,
  title={球面之上：从球面动力学到 μP},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/spherical-dynamics-mup/}
}
```