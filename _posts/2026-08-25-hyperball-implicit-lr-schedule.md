---
layout: post
title: "Hyperball、effective lr 与峰值加衰减的形状"
date: 2026-08-25 11:00:00
description: "两个问题。调度对象：weight norm 的变化在 base lr schedule 之上叠加一层隐式调度，Hyperball 移除的是这一层，起作用的量为 effective lr。形状：峰值加衰减为 bias–variance trade-off 的解，早期偏差项占主导、末期方差项占主导，满足该平衡的形状构成一个集合，不限于特定的解析形式。"
tags: [deep-learning, lr-schedule, optimizer, spherical-dynamics, scaling-law]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: zh-CN
---

## TL;DR

近期几项工作中出现两项现象。第一项，Hyperball 将权重矩阵及其更新的 Frobenius 范数固定为常数，相对 weight decay 基线取得 20–30% 的 token 等效提速 [[8]](https://arxiv.org/abs/2606.16899)。第二项，设置互不相同的一批方法产出的最优 lr 曲线均为峰值加衰减。

峰值加衰减在本文中指以下形状：lr 在训练前段上升到峰值 $\eta_{\max}$，随后单调下降，末端接近零。衰减段的函数形式不限。

{% include figure.liquid
  path='assets/img/post-08-25/peak_decay_shape.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='85%'
  caption='峰值加衰减形状的示例，衰减段取幂律形式 $\eta_t\propto(1-t/T)^{\alpha}$，$\alpha$ 分别为 0.5、1、2。$\alpha=0.5$ 对应 sqrt 形状，$\alpha=1$ 对应线性衰减到零。三条曲线的峰值与终点相同，衰减段的凹凸方向不同。'
  zoomable=true
  alt='peak plus decay learning rate shape illustrated with three power-law exponents'
%}

图中的幂律形式便于参数化，人工指定的调度多采用此类形式。第 4 节列出的四项工作中，refined schedule 由梯度范数序列逐点构造，Schedule-Free+ 的实际 lr 由迭代平均产生，两者的曲线均为峰值加衰减，不对应特定的解析式。

近期几项工作将上述两项现象的来源归于同一个量。

> ##### 核心判断
> **决定训练进展的量为 effective lr $$\eta_t^\star=\eta_t\lVert U_t\rVert/\lVert W_t\rVert$$。**<br>weight norm 在训练中持续增长，因此 $$\eta_t$$ 到 $$\eta_t^\star$$ 的折算系数持续变化，在设定的 lr schedule 之上叠加一层隐式调度。Hyperball 固定 weight norm，移除的是这一层，其作用因此是隐式学习率调度。
{: .block-tip}

$\eta^\star$ 取峰值加衰减形状的原因与优化器无关：

> ##### 形状的判据
> **峰值加衰减为 bias–variance trade-off 的解。**<br>峰值段以较大的步长降低偏差，末端衰减到零以降低方差。满足该平衡的形状构成一个集合，集合内各形状的表现相当。
{: .block-tip}

两项判断给出全文结构：$\eta^\star$ 为应当被调度的量（第 1–3 节），bias–variance trade-off 决定其形状（第 4 节）。

## 1. 起作用的量为 effective lr

含归一化层的参数满足尺度不变性 $\mathcal{L}(\rho W) = \mathcal{L}(W)$。纯径向伸缩不改变网络函数，因此刻画单步进展的量为权重方向转过的角度。混元 ELR 将其称为角更新幅度（AUS）[[1]](https://hy.tencent.com/research/elr)：

$$
\mathrm{AUS} := \left\lVert \frac{W_{t+1}}{\lVert W_{t+1}\rVert} - \frac{W_t}{\lVert W_t\rVert}\right\rVert \approx \frac{\eta_t \lVert U_t\rVert}{\lVert W_t \rVert} =: \eta_t^\star
$$

右端为 effective lr，源自此前对 weight decay 的一系列动力学分析 [[9]](https://arxiv.org/abs/2006.08419)。其中含 $\lVert W_t \rVert$，而 weight norm 在训练过程中持续变化，**因此 $\eta_t$ 到 $\eta_t^\star$ 的折算系数持续变化**。该折算系数即 TL;DR 中的隐式调度。

折算的幅度可由三种名义调度的对照给出：WSD（峰值 $3.6\times10^{-3}$）、cosine 与 linear（峰值 $8.8\times10^{-3}$），峰值相差约 2.4 倍。

{% include figure.liquid
  path='assets/img/post-08-25/wsd_cosine_linear_aus.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='右下：WSD、cosine、linear 三条 lr schedule；右上：对应的 weight norm；左下：对应的角更新幅度。图片来源：<a href="https://hy.tencent.com/research/elr">混元 ELR</a>。'
  zoomable=true
  alt='weight norm and angular update size under WSD, cosine and linear schedules'
%}

对比右下与左下：名义调度形状差异明显，1000 步之后三条角更新幅度曲线基本重合，均为峰值后衰减。右上给出对应的 weight norm，其在前 2000 步由约 30 增长到 180–280。

> ##### 名义 lr 与 effective lr
> 调节对象为名义 lr schedule，作用于训练的为 AUS。二者之间存在一层随 weight norm 变化的折算系数。
{: .block-tip}

在恒定名义 lr 下，该折算的形态如下 [[2]](https://arxiv.org/abs/2607.22444)。

{% include figure.liquid
  path='assets/img/post-08-25/muonwd_muonh_elr_const.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='左：MuonWD 与 MuonH 的名义 lr，两条曲线重合于 0.025；右：对应的 effective lr，MuonWD 由约 $9\times10^{-5}$ 衰减至约 $1.8\times10^{-5}$，MuonH 在初始瞬态后保持恒定。图片来源：<a href="https://arxiv.org/abs/2607.22444">Hyperball May Not Be a Free Lunch</a>，Figure 2(a)(b)。'
  zoomable=true
  alt='nominal and effective learning rate of MuonWD and MuonH under a constant schedule'
%}

名义 lr 完全相同，effective lr 相差约 5 倍。MuonWD 的衰减来自 weight norm 增长。MuonH 固定了 weight norm，effective lr 在初始瞬态后保持恒定。

## 2. effective lr 轨迹与损失曲线的对应关系

上一节给出 $\eta^\star$ 与 $\eta$ 的差异。$\eta^\star$ 作为分析对象还需确认一项：$\eta^\star$ 轨迹相同时损失曲线是否相同。两篇工作从相反方向检验了这一项。

混元 ELR 的 AUS-replay：用 Adam 或 Muon 训练 GPT-2（124M）并逐步记录 AUS，再换为对应的 Hyperball 变体（AdamH、MuonH），将记录的 AUS 轨迹设为其 lr 曲线。

{% include figure.liquid
  path='assets/img/post-08-25/aus_replay.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='第一行：AUS 曲线；第二行：验证损失；第三行：损失差值。四列分别为 Adam 与 Muon 在朴素 nanoGPT 与尺度不变 nanoGPT 上的结果。图片来源：<a href="https://hy.tencent.com/research/elr">混元 ELR</a>。'
  zoomable=true
  alt='AUS replay experiment comparing Adam/Muon with their Hyperball variants'
%}

两组的 AUS 曲线与损失曲线基本重合，在尺度不变结构上损失差值在 $\pm0.005$ 范围内 [[1]](https://hy.tencent.com/research/elr)。

反方向的做法是固定优化器、逐步改变其 lr 以匹配目标 effective lr，把 MuonWD 对齐到 MuonH 的轨迹，以及反向对齐 [[2]](https://arxiv.org/abs/2607.22444)。

{% include figure.liquid
  path='assets/img/post-08-25/muon_lr_alignment.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='左：MuonH 对齐到 MuonWD，绿色虚线与 MuonWD 基本重合；右：MuonWD 对齐到 MuonH，紫色虚线与 MuonH 基本重合。两图中 MuonWD 与 MuonH 的原始曲线在 500–2000 步区间存在差距。图片来源：<a href="https://arxiv.org/abs/2607.22444">Hyperball May Not Be a Free Lunch</a>，Figure 3(a)(c)。'
  zoomable=true
  alt='mutual alignment of MuonWD and MuonH training loss by learning-rate alignment'
%}

两个方向的对齐结果均与目标曲线基本重合，改变 lr 即可复现另一优化器的损失曲线。该文的结论为，Hyperball 的主要作用是隐式的状态相关学习率调度，其更新方向未表现出额外优势 [[2]](https://arxiv.org/abs/2607.22444)。

> ##### 两个方向的检验结果
> **$$\eta^\star$$ 轨迹相同时损失曲线基本重合，Hyperball 与非 Hyperball 优化器的差异可由 lr 单独解释。**<br>由此得到开篇的判断：Hyperball 移除的是 weight norm 叠加在 lr schedule 之上的隐式调度。
{: .block-tip}

该判断同时对应 MuonH 的分阶段现象：早期收敛较慢，后期正确率高于 MuonWD [[2]](https://arxiv.org/abs/2607.22444)。Hyperball 约束下不存在 weight norm 增长带来的衰减，早期实际步长大于 MuonWD。

## 3. effective lr 对 scaling law 的拟合精度

若 $\eta^\star$ 为实际生效的量，用它拟合损失曲线的精度应高于 $\eta$。混元 ELR 在多重幂律（MPL）损失模型中把 $\eta$ 替换为 $\eta^\star$ 后，样本内拟合与跨调度预测的精度均更高，最优 $\eta^\star$ 跨模型宽度与深度迁移的可靠性也更高 [[1]](https://hy.tencent.com/research/elr)。

原因与第 1 节一致：$\eta$ 与损失之间存在一层随训练变化、且随宽度深度变化的折算系数，$\eta^\star$ 不含该系数。**以 $\eta$ 为自变量拟合 scaling law 时，该系数的变化计入拟合误差。**

同一关系适用于超参迁移。Hyperball 约束下 $\Delta\phi_t \approx \eta_t$，于是

$$
\sum_t \Delta\phi_t \approx \int_0^T \eta_t\,\mathrm{d}t
$$

即 lr 的积分等于权重方向转过的总角度。一项相关现象为，两次训练的累积 lr 相近时最终损失相近 [[1]](https://hy.tencent.com/research/elr)。该量在 Hyperball 约束下具有对应的几何含义，可作为跨预算迁移时的对齐对象。

另一项相关结果：Frobenius 球面上 weight decay 一阶失效，$(\eta,\lambda)$ 的二维搜索降为一维。实测给出最优 lr 随 token 数的幂律 $\eta^*\propto T^{-0.32}$，与 AdamW 上报告的指数一致 [[3]](https://arxiv.org/abs/2603.28743)。该指数的理论来源原因未查明。

## 4. 形状的来源：bias–variance trade-off

前三节确定了被调度的量，本节讨论该量的形状。以下四项工作的设置互不相同，产出的 lr 曲线形状一致：在训练前段上升到峰值，随后单调下降到接近零。

**WSD cooldown 形状比较。** 在 cooldown 段比较各人工形状，`sqrt`（$1-\sqrt{x}$）与 `lowered linear 0.7` 表现相当，`lowered linear 0.7` 的困惑度低于 `sqrt` [[4]](https://arxiv.org/abs/2508.01483)。

**Schedule-Free+。** 不指定 lr 取值与调度形状，其实际 lr 曲线在恒定名义 lr 下上升到峰值、随后衰减。该方法优于 WSD 基线，长周期设置下达到同等损失所需时间减少 31% [[5]](https://arxiv.org/abs/2605.19095)。

{% include figure.liquid
  path='assets/img/post-08-25/schedulefree_plus_lr.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='1B 模型、1000 tokens per parameter 设置。红色为 Schedule-Free+（beta 由 0.8 退火到 0.965）的实际 lr，黑色为 linear decay，蓝色为 WSD。图片来源：<a href="https://arxiv.org/abs/2605.19095">Schedule-Free+</a>。'
  zoomable=true
  alt='effective learning rate of Schedule-Free+ compared with linear decay and WSD'
%}

**离线 refined schedule。** 由梯度范数序列离线构造，在八个任务上均呈现同一形状 [[6]](https://arxiv.org/abs/2310.07831)。

{% include figure.liquid
  path='assets/img/post-08-25/refined_schedule.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='八个任务（ImageNet、IWSLT14、GPT、RoBERTa、DLRM、MRI、ViT、RCNN）上的结果。左列：梯度范数序列；中列：平滑后的梯度范数；右列：由此构造的 refined schedule，内嵌图为对数刻度。图片来源：<a href="https://arxiv.org/abs/2310.07831">Defazio et al. (2024)</a>，Figure 4。'
  zoomable=true
  alt='gradient norms and refined schedules across eight tasks'
%}

**minus-square-root。** 混元 ELR 依据实测 AUS 的变化规律提出，在 Modded-nanoGPT Track 3 上以 3,175 步达到目标损失 3.28 [[1]](https://hy.tencent.com/research/elr)。[[2]](https://arxiv.org/abs/2607.22444) 在同一 track 上用 power-0.4 调度以 3,150 步达到该目标。两者的形状接近，步数差 25 步。

四项工作的形状产出方式互不相同，结果均落在峰值加衰减的描述内。该现象涉及两个层面的原因。

### 4.1 偏差与方差的平衡

单步更新对最终模型的影响分为两部分。参数与初始点的距离增大，对应偏差下降。梯度噪声在参数中累积，对应方差上升。lr 决定两部分的比例。

训练早期偏差项占主导，较大的 lr 对应更快的偏差下降。训练末期方差项占主导，衰减 lr 等价于对更多次更新做平均，对应方差下降。峰值加衰减的形状为这两段要求的组合。

偏差与方差的相对权重随任务、模型规模、训练预算变化。因此使 $Bias+Variance$ 取到最小值的形状构成一个集合，在幂律参数化下对应 $\alpha$ 的一个区间。该集合的存在与其随条件的变化，对应第 5 节的两项推论。

{% include figure.liquid
  path='assets/img/post-08-25/bias_variance_shapes.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='各 cooldown 形状的 bias–variance 分布，虚线为 $Bias+Variance$ 取到最小值的位置。左：仅比较 lowered linear 形状，参数减小时方差下降、偏差上升；右：全部非线性形状与部分 lowered linear 形状，其中 sqrt 与 0.7 落在虚线附近，square、cosine、mirror cosine、linear 位于虚线上方。左右两图横轴范围不同。图片来源：<a href="https://arxiv.org/abs/2508.01483">Dremov et al. (2025)</a>，Figure 6。'
  zoomable=true
  alt='bias-variance plot for different cooldown shapes'
%}

左图给出偏差与方差的反向关系：`lowered linear` 的参数减小时方差下降、偏差上升。右图给出各形状的位置。`sqrt` 与 `lowered linear 0.7` 落在最小线附近，`square`、`cosine`、`mirror cosine`、`linear` 位于线上方。落在最小线附近的形状数量不止一个 [[4]](https://arxiv.org/abs/2508.01483)。[[7]](https://arxiv.org/abs/2502.15938) 报告的线性衰减到零同样属于此类。

{% include figure.liquid
  path='assets/img/post-08-25/sqrt_vs_lowered_linear.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='55%'
  caption='cooldown 段（训练进度 80%–100%）内 sqrt 与 lowered linear 0.7 的形状对照。图片来源：<a href="https://arxiv.org/abs/2508.01483">Dremov et al. (2025)</a>，Figure 7。'
  zoomable=true
  alt='comparison of sqrt cooldown shape and lowered linear 0.7'
%}

两条形状不同的曲线表现相当。一项推论是调度形状的调参收益存在上界：调整 AdamW 的 $\beta_2$ 带来的差异与形状选择相当 [[4]](https://arxiv.org/abs/2508.01483)。

### 4.2 观测量中含 weight norm 的折算

不施加 Hyperball 约束时，观察到的形状有一部分来自第 1 节的折算，其余部分来自调度设计。minus-square-root 以 $\eta^\star$ 为设计对象，需要显式写出该形状。Hyperball 约束移除了折算，因此 $\eta_t$ 本身必须满足形状要求。

该折算对应一项观测结果：名义 lr 的形状差异经过折算后被压缩，因此不同设置下观察到的 $\eta^\star$ 形状差异小于名义调度的形状差异。第 1 节三种名义调度的 AUS 曲线在 1000 步后基本重合属于此类。

## 5. 小结

全文分为两个问题。第一个问题为调度对象：调节的量为 $\eta_t$，起作用的量为 $\eta_t^\star$，两者之间的折算系数由 weight norm 决定并随训练变化。Hyperball 固定 weight norm，使折算系数为常数，其作用因此是隐式学习率调度。第二个问题为形状：峰值加衰减为 bias–variance trade-off 的解，早期偏差项占主导，末期方差项占主导。

trade-off 的结构给出两项推论。满足平衡的形状构成一个集合，因此形状选择的调参收益存在上界。两端的相对权重由训练预算、模型规模、噪声水平决定，因此不同条件下的最优形状不同。

操作层面的结论：

1. 比较不同调度时记录 effective lr。峰值相差 2.4 倍的三种名义调度可对应基本重合的 AUS 曲线。
2. 拟合 scaling law 与迁移超参时，对齐对象取 $\eta^\star$ 或累积角位移 $\int\eta_t\mathrm{d}t$。
3. Hyperball 约束下折算已被移除，形状要求需由 $\eta_t$ 满足，因此需配合明确的衰减设计。衰减幅度存在上界：幅度过大时后期表现低于 MuonWD [[2]](https://arxiv.org/abs/2607.22444)。
4. 形状选择的可优化范围有限，`sqrt`、`lowered linear 0.7`、线性衰减到零的表现相当。调参预算优先分配给峰值 lr 与 $\beta_2$。

两项遗留问题。自适应方法产出的形状与人工指定形状之间的定量对应关系尚未验证。$\eta^*\propto T^{-0.32}$ 中指数的理论来源原因未查明。

相关内容：宽度方向的推导见 [《球面之上：带有 Hyperball 机制的优化器的 μP 缩放》]({% post_url 2026-03-06-spherical-hyperball %})，更新矩阵范数估计见 [《Adam 与 Muon 优化器更新矩阵的 Frobenius 范数估计》]({% post_url 2026-03-08-optimizer-update-matrix-norm %})，batch size 方向的调度见 [《DASF：一种闭环的 batch size schedule-free 方法》]({% post_url 2026-06-20-schedule-free-effective-batch-size %})。

## 参考文献

[1] Tencent Hunyuan Pretrain Team (2026). [From LR to ELR: A Better Heuristic for Pretraining Dynamics](https://hy.tencent.com/research/elr).

[2] Xiao, Y., Sun, J., Gao, Z., Wei, Z., Wang, C., Tao, R., Teng, J., & Dai, B. (2026). [Hyperball May Not Be a Free Lunch](https://arxiv.org/abs/2607.22444). arXiv preprint arXiv:2607.22444.

[3] Ren, L., Liu, Y., Shen, Y., & Chen, W. (2026). [Rethinking Language Model Scaling under Transferable Hypersphere Optimization](https://arxiv.org/abs/2603.28743). arXiv preprint arXiv:2603.28743.

[4] Dremov, A., Hägele, A., Kosson, A., & Jaggi, M. (2025). [Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler](https://arxiv.org/abs/2508.01483). Transactions on Machine Learning Research (TMLR), 2025. arXiv preprint arXiv:2508.01483.

[5] Defazio, A. (2026). [Schedule-Free+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models](https://arxiv.org/abs/2605.19095). arXiv preprint arXiv:2605.19095.

[6] Defazio, A., Cutkosky, A., Mehta, H., & Mishchenko, K. (2024). [Optimal Linear Decay Learning Rate Schedules and Further Refinements](https://arxiv.org/abs/2310.07831). arXiv preprint arXiv:2310.07831.

[7] Bergsma, S., Dey, N., Gosal, G., Gray, G., Soboleva, D., & Hestness, J. (2025). [Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs](https://arxiv.org/abs/2502.15938). ICLR 2025. arXiv preprint arXiv:2502.15938.

[8] Wen, K., Dang, X., Lyu, K., Ma, T., & Liang, P. (2026). [Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899). arXiv preprint arXiv:2606.16899.

[9] Wan, R., Zhu, Z., Zhang, X., & Sun, J. (2020). [Spherical Motion Dynamics: Learning Dynamics of Neural Network with Normalization, Weight Decay, and SGD](https://arxiv.org/abs/2006.08419). arXiv preprint arXiv:2006.08419.

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026hyperballimplicitschedule,
  title={Hyperball、effective lr 与峰值加衰减的形状},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/hyperball-implicit-lr-schedule/}
}
```
