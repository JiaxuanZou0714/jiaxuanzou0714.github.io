---
layout: post
title: "从球面动力学到 μP"
date: 2026-03-05 08:47:35
description: "本文脱离 Tensor Programs 的概率论框架，从连续时间的球面动力学视角，严格推导在应用 RMSNorm 的网络架构中，如何通过对齐超球面上的动力学来实现大小网络的对齐。"
tags: [deep-learning, spherical-dynamics, muP, rmsnorm]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

在现代大模型架构中，RMSNorm 使得权重向量的模长对最终输出的影响被剥离，梯度的作用本质上是改变权重的方向。在推导神经网络的缩放法则时，传统的 Tensor Programs 理论依赖于对大量随机变量的坐标系微观统计。本文将提供一个完全不同的路径：在具有严格尺度不变性的设定下，将整个线性层的动力学等价映射为在超球面上的切空间运动，进而严谨推导出 μP 所要求的超参数缩放规律。
> 核心的insight是，在使用rmsnorm的情况下，对齐的对象应该是特征在球面上的演化速率（即角速度）。

## 1. 尺度不变性与梯度的自然正交

设定输入特征 $x \in \mathbb{R}^n$，且满足 $\|x\|_2^2 = \Theta(n)$，换言之拥有 $\Theta(1)$ 的坐标分量。设隐藏层权重矩阵为 $W \in \mathbb{R}^{n \times n}$。未归一化的线性映射输出（预激活）为 $y = Wx \in \mathbb{R}^n$。

在将 RMSNorm 作用于线性映射之后的设定下，传递给后续网络的归一化特征 $z$ 为：

$$
z = \text{RMSNorm}(y) = \sqrt{n} \frac{y}{\|y\|_2}
$$

> 这表明特征向量 $z$ 被严格约束在半径为 $\sqrt{n}$ 的超球面 $\mathbb{S}^{n-1}(\sqrt{n})$ 上。

设损失函数为 $L(z)$。由于预激活输出通过 RMSNorm 后，对于任意标量 $c > 0$ 模型的输出不变，损失函数对于权重矩阵具有严格的尺度不变性：

$$
L(cW) = L(W)
$$

对上述等式两边的标量 $c$ 求导。利用多元复合函数的链式法则，左式的导数为：

$$
\frac{d}{dc} L(cW) = \sum_{i=1}^n \sum_{j=1}^n \frac{\partial L(cW)}{\partial (cW_{ij})} \frac{\partial (cW_{ij})}{\partial c} = \sum_{i=1}^n \sum_{j=1}^n \frac{\partial L(cW)}{\partial (cW_{ij})} W_{ij} = \langle \nabla_{cW} L(cW), W \rangle_F
$$

其中 $\langle \cdot, \cdot \rangle_F$ 表示矩阵的 Frobenius 内积。右式 $L(W)$ 为与 $c$ 无关的常数，其对 $c$ 的导数严格为 $0$。因此：

$$
\langle \nabla_{cW} L(cW), W \rangle_F = 0
$$

令 $c=1$，可得：

$$
\langle \nabla_W L(W), W \rangle_F = 0
$$

这表明欧氏梯度 $\nabla_W L(W)$ 与权重矩阵 $W$ 处处正交。在连续时间的梯度流下，设学习率为 $\eta$，权重的更新速率为：

$$
\frac{dW}{dt} = - \eta \nabla_W L(W)
$$

考察权重矩阵 Frobenius 范数平方 $\|W\|_F^2 = \langle W, W \rangle_F$ 随时间的导数：

$$
\frac{d}{dt} \|W\|_F^2 = \frac{d}{dt} \langle W, W \rangle_F = 2 \langle W, \frac{dW}{dt} \rangle_F = 2 \langle W, - \eta \nabla_W L(W) \rangle_F = - 2 \eta \langle W, \nabla_W L(W) \rangle_F = 0
$$

由于范数平方的导数严格为零，权重的模长 $\|W\|_F$ 在连续优化过程中严格保持常数。

> 因此对于满足尺度不变的权重$W$，普通的梯度流天然等同于超球面上的黎曼梯度流。

## 2. 球面映射与雅可比矩阵

设上游传回的梯度标量为 $g = \nabla_z L \in \mathbb{R}^n$，假设 $g$ 的坐标大小为 $\Theta(1)$。首先计算从归一化特征 $z$ 到未归一化特征 $y$ 的雅可比矩阵 $J \in \mathbb{R}^{n \times n}$。已知 $z = \sqrt{n} \frac{y}{\|y\|_2}$，考察其第 $i$ 个分量对第 $j$ 个分量 $y_j$ 的偏导数：

$$
z_i = \sqrt{n} \frac{y_i}{( \sum_{k=1}^n y_k^2 )^{1/2}}
$$

利用除法求导法则：

$$
\frac{\partial z_i}{\partial y_j} = \sqrt{n} \frac{\frac{\partial y_i}{\partial y_j} \|y\|_2 - y_i \frac{\partial \|y\|_2}{\partial y_j}}{\|y\|_2^2}
$$

其中分量求导 $\frac{\partial y_i}{\partial y_j} = \delta_{ij}$（当 $i=j$ 时为 $1$，否则为 $0$）。模长对分量的导数为：

$$
\frac{\partial \|y\|_2}{\partial y_j} = \frac{1}{2} \left( \sum_{k=1}^n y_k^2 \right)^{-1/2} (2 y_j) = \frac{y_j}{\|y\|_2}
$$

将上述项代入导数公式可得：

$$
\frac{\partial z_i}{\partial y_j} = \sqrt{n} \frac{\delta_{ij} \|y\|_2 - y_i \frac{y_j}{\|y\|_2}}{\|y\|_2^2} = \frac{\sqrt{n}}{\|y\|_2} \left( \delta_{ij} - \frac{y_i y_j}{\|y\|_2^2} \right)
$$

将上述逐元素的偏导数结果转换为矩阵形式，克罗内克函数 $\delta_{ij}$ 对应单位矩阵 $I$，项 $y_i y_j$ 对应外积矩阵 $y y^T$。因此雅可比矩阵 $J$ 严格等于：

$$
J = \frac{\partial z}{\partial y} = \frac{\sqrt{n}}{\|y\|_2} \left( I - \frac{y y^T}{\|y\|_2^2} \right) = \frac{\sqrt{n}}{\|y\|_2} P_y
$$

其中 $P_y = I - \frac{y y^T}{\|y\|_2^2}$ 是将欧氏空间向量投影到以 $y$ 为法向量的超平面的正交投影算子。通过链式法则，损失函数关于 $y$ 的梯度为：

$$
\nabla_y L = J^T g = \frac{\sqrt{n}}{\|y\|_2} P_y g
$$

接下来计算损失函数关于权重矩阵 $W$ 的欧氏梯度：

$$
\nabla_W L = (\nabla_y L) x^T = \left( \frac{\sqrt{n}}{\|y\|_2} P_y g \right) x^T
$$

在连续时间梯度流下，考察权重更新如何反作用于前向传播中的特征 $y$。未归一化特征 $y$ 的动力学方程为：

$$
\frac{dy}{dt} = \frac{dW}{dt} x = - \eta \left( \frac{\sqrt{n}}{\|y\|_2} P_y g \right) x^T x
$$

由于输入特征满足 $\|x\|_2^2 = \Theta(n)$，内积 $x^T x = \|x\|_2^2$。代入上式得到：

$$
\frac{dy}{dt} = - \eta \Theta(n) \frac{\sqrt{n}}{\|y\|_2} P_y g
$$

## 3. 初始化缩放（与Tensor Programs思路相同）

常微分方程本身描述的是系统状态的变化率，无法直接内生出系统在初始时刻的尺度。因此，本节推导初始化缩放规律的逻辑，在理论依据上与 Tensor Programs 框架高度契合。我们需要借助随机变量的大数定律，确立初始化缩放。

为了使模型在前向传播中能够提取并传递有意义的特征，防止反向传播出现奇异性或退化（剔除$n$的影响），预激活向量的坐标必须维持在 $\Theta(1)$ 的量级。

设权重 $W_{ij}$ 独立同分布于均值为 $0$、方差为 $\sigma_w^2$ 的分布。计算未归一化输出 $y_i$ 的方差：

$$
\mathbb{E}[y_i^2] = \sum_{j=1}^n \mathbb{E}[W_{ij}^2] x_j^2 = n \sigma_w^2 \Theta(1)
$$

为了满足预激活向量坐标为 $\Theta(1)$ 这一几何稳定性的初始条件，我们必须强制要求 $n \sigma_w^2 = \Theta(1)$。

> 由此可严格解出初始化的方差必须遵循：
> $$
> \sigma_w^2 = \Theta\left(\frac{1}{n}\right)
> $$

在这一良定义的初始化边界条件下，整个向量的模长平方在概率上集中于 $\|y\|_2^2 = \Theta(n)$。

## 4. 学习率缩放（球面动力学视角）

在上一节确定了与 Tensor Programs 相一致的初始化边界条件后，本节回到全新的球面动力学视角，严格推导学习率的缩放规律。为了实现不同宽度模型间的超参数对齐，核心要求是无论宽度 $n$ 趋向于多大，归一化特征 $z$ 在球面上的演化速度 $\frac{dz}{dt}$ 的坐标变化率必须保持为 $\Theta(1)$。

利用雅可比矩阵，将未归一化特征的动力学映射回球面：

$$
\frac{dz}{dt} = J \frac{dy}{dt} = \left( \frac{\sqrt{n}}{\|y\|_2} P_y \right) \left( - \eta \Theta(n) \frac{\sqrt{n}}{\|y\|_2} P_y g \right)
$$

提取常数标量并利用投影算子的幂等性 $P_y^2 = P_y$，化简得到最终的球面动力学方程：

$$
\frac{dz}{dt} = - \eta \Theta(n) \frac{n}{\|y\|_2^2} P_y g
$$

代入由于初始化边界条件确定的 $\|y\|_2^2 = \Theta(n)$：

$$
\frac{dz}{dt} = - \eta \Theta(n) \frac{n}{\Theta(n)} P_y g = - \eta \Theta(n) P_y g
$$

由于切空间上的梯度投影 $P_y g$ 的坐标大小为 $\Theta(1)$，为了保证特征学习在不同宽度 $n$ 下严格对齐，使得 $\frac{dz}{dt}$ 的坐标变化率稳定在 $\Theta(1)$，必须满足：

$$
\eta \cdot \Theta(n) = \Theta(1)
$$

> 由此严格证明，对于应用了 RMSNorm 的隐层，其学习率 $\eta$ 必须与宽度 $n$ 成反比，即 $\eta = \Theta(1/n)$。

容易证明的是，在球面动力学视角下，对齐了 $\frac{dz}{dt}$ 等价于对齐了 $z$ 的角速度 $\frac{d\theta}{dt}$。

## 5. 总结

球面动力学的推导路径规避了对矩阵元素进行概率极限推导的繁杂过程，直接利用了 RMSNorm 带来的球面结构和雅可比投影。它证明了 $\eta = \Theta(1/n)$ 是保证不同宽度网络在球面上具有一致角速度的唯一解。

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026sphericaltomup,
  title={从球面动力学到 μP},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/spherical-dynamics-mup/}
}
```