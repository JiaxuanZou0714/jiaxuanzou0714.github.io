---
layout: post
title: "Loss Spike 是否能够逐点预测？一个三次模型下的动力系统视角"
date: 2026-06-14 12:00:00
description: "用一个三次 loss 的 toy model 看 loss spike：宏观趋势也许能预测，但单个 spike 的精确时刻未必能够预测。"
tags: [optimization, chaos, edge-of-stability, loss-landscape]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

> TL;DR：训练虽然是确定性的，但这不等于每个 loss spike 都适合逐点预测。一个局部三次 loss 在 GD 下可以严格化成 Logistic map；当步长跨过稳定边界后，状态对初值极敏感，loss 只是这个底层状态的 observable。因此更合理的目标是预测训练所在 regime、spike 的统计特征和宏观 envelope，而不是每个 spike 的精确时刻。

## 1. 问题：确定性不等于逐点可预测

训练大模型时，loss 曲线经常在下降过程中抖一下。一个自然的念头是：训练是确定性的，每个 spike 原则上都应该能预测。这在逻辑上没问题，但"有因果"和"适合逐点预测"是两回事。下面用一个极小的 toy model 看清楚这个区别。

## 2. 从三次 loss 到 Logistic map

取一个局部三次 loss：

$$L(x) = L_* + \frac{\lambda}{2}x^2 + \frac{\gamma}{6}x^3$$

对它做梯度下降：

$$x_{t+1} = x_t - \eta L'(x_t) = (1-\eta\lambda)x_t - \frac{\eta\gamma}{2}x_t^2$$

这里有个细节值得注意：三次 loss 求导后梯度是二次的，所以 GD 递推是一个二次映射，而不是三次。

记 $a = 1 - \eta\lambda$，$b = \eta\gamma/2$，对状态做仿射换元 $y_t = Ax_t + B$，解方程可得

$$A = \frac{\eta\gamma/2}{1+\eta\lambda}, \quad B = \frac{\eta\lambda}{1+\eta\lambda}$$

在这个变量下，递推严格变成标准 Logistic map：

$$y_{t+1} = r\, y_t(1-y_t), \quad r = 1 + \eta\lambda$$

这不是近似，是严格等价。

## 3. Edge of Stability 与倍周期分叉

### 3.1 稳定边界

局部稳定性要求 $\lvert 1 - \eta\lambda \rvert < 1$，临界步长 $\eta_c = 2/\lambda$ 对应 $r = 3$，恰好是 Logistic map 第一次 period-doubling 的位置——固定点失稳、出现非平凡周期行为。在这个 toy model 里，Edge of Stability 对应的正是这一步，而不是"已经混沌了"。

### 3.2 全混沌情形

继续增大到 $r = 4$（对应 $\eta = 3/\lambda$）是个干净的全混沌情形。令 $y_t = \sin^2(\pi s_t)$，递推退化为 doubling map：

$$s_{t+1} = 2s_t \pmod{1}$$

每迭代一步，等于把 $s_0$ 的二进制展开向右移一位——第 $t$ 步的状态取决于初值里第 $t$ 位之后的内容。初值误差 $\varepsilon$ 经过 $t$ 步放大到 $2^t \varepsilon$，可预测时长大约只有 $\log_2(1/\varepsilon)$ 步。

## 4. Loss 是 observable，不是 Logistic map 本身

上面的推导对象是状态 $x_t$（或变换后的 $y_t$），不是 loss。两者的关系是：

$$\ell_t = L(x_t) = h(y_t), \quad y_{t+1} = r\, y_t(1-y_t)$$

因为三次函数 $L(x)$ 不是一一对应的，同一个 $\ell_t$ 可能来自多个不同位置，所以一般不存在封闭的 $\ell_{t+1} = G(\ell_t)$。Loss curve 是底层混沌状态的一个 observable，继承了它的敏感性，但本身不形成一维递推。直接说"loss 满足 Logistic map"是说错了。

## 5. 结论：预测 regime，而不是执着于每个 spike

单个 spike 的精确时刻对初值靠后的二进制位敏感，预测代价随步数指数增长。但训练落在哪个 regime（稳定区、EoS 附近、超临界区）、spike 的频率和幅度分布、以及 loss 的宏观 envelope，对初值扰动要稳定得多。

真实训练比这个一维模型复杂得多，但这里的教训很清楚：哪怕在一维确定性的三次 loss 下，GD 就能产生复杂动力学。把逐点预测每个 spike 当成目标，可能从一开始就设错了问题。

## 6. 数值验证

下面做一个最小数值实验来检查上面的三个结论。取

$$
\lambda=1,\quad \gamma=2,\quad L_*=0.
$$

于是

$$
L(x)=\frac{1}{2}x^2+\frac{1}{3}x^3,\quad x_{t+1}=(1-\eta)x_t-\eta x_t^2.
$$

对每个给定的 $$r$$，令 $$\eta=r-1$$，并使用上文的仿射换元

$$
y_t=A x_t+B,\quad A=B=\frac{\eta}{1+\eta}.
$$

实验里先固定同一个 $$y_0=0.123456789$$，再由 $$x_0=(y_0-B)/A$$ 反解出对应的 GD 初值，然后直接迭代原始的 $$x_t$$ 递推。

{% include figure.liquid
  path="assets/img/post-06-14/loss-spike-dynamics.png"
  class="img-fluid mx-auto d-block"
  width="100%"
  max-width="1050px"
  sizes="(min-width: 1200px) 1050px, 95vw"
  zoomable=true
  alt="三次 loss 梯度下降与 Logistic map 动力学的数值验证"
  caption="三次 loss 梯度下降与 Logistic map 动力学的数值验证。"
%}

图中的三栏分别对应上面的三个推导环节：

- (a) 比较不同 $$r=1+\eta$$ 下的 loss 轨迹。$$r<3$$ 时收敛到稳定点；跨过 $$r=3$$ 后出现非单调振荡；$$r=4$$ 时进入强烈的混沌区。
- (b) 把 $$r=4$$ 的 GD 轨迹换元到 $$y_t$$ 后画出 return map。散点落在 $$y_{t+1}=4y_t(1-y_t)$$ 上，一步残差只有浮点误差量级。
- (c) 在 doubling coordinate $$s_t$$ 中比较两个相差 $$10^{-10}$$ 的初值。距离先按 $$2^t$$ 放大，直到饱和。

这组实验的作用不是模拟真实大模型，而是确认 toy model 里的逻辑链条是闭合的：三次 loss 下的 GD 递推确实等价于 Logistic map；$$r=3$$ 对应第一次稳定性破缺；而在 $$r=4$$ 时，初值误差在 $$s_t$$ 坐标里近似每步翻倍。因此，单个 spike 的逐点时刻会很快依赖初值中极靠后的有效位数，这也正是它不适合作为长期预测目标的原因。

## 参考文献

[1] Robert M. May. "Simple mathematical models with very complicated dynamics." _Nature_ 261, 459-467 (1976). <https://doi.org/10.1038/261459a0>

[2] Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet Talwalkar. "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability." ICLR 2021. arXiv:2103.00065. <https://openreview.net/forum?id=jh-rTtvkGeM>, <https://arxiv.org/abs/2103.00065>

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026loss_spikes_chaos,
  title={Loss Spike 是否能够逐点预测？一个三次模型下的动力系统视角},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/loss-spikes-chaos/}
}
```
