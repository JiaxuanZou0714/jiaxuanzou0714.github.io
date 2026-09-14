---
layout: post
title: "Hyperball、effective lr 与峰值加衰减的形状"
date: 2026-08-25 11:00:00
description: "探讨预训练学习率调度的两个核心问题：优化器实际调节的有效学习率由权重范数动态决定，Hyperball 通过约束权重范数消除该隐式调度；峰值加衰减形状对应偏差与方差权衡的最优解，满足该平衡的形状构成一个集合，不限于特定的解析形式。"
tags: [deep-learning, lr-schedule, optimizer, spherical-dynamics, scaling-law]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
lang: zh-CN
---

## TL;DR

近期预训练优化领域呈现两项显著的实证现象：Hyperball 将权重矩阵及其更新的 Frobenius 范数固定为常数，相对权重衰减（weight decay）基线取得 20–30% 的 token 等效加速 [[8]](https://arxiv.org/abs/2606.16899)；同时，实验设置各异的多项独立研究得出的最优学习率曲线，均呈现峰值加衰减形态。

峰值加衰减指学习率在训练前期上升至峰值 $\eta_{\max}$、随后单调衰减且末端接近零的几何形态，衰减段不限定特定解析函数形式。

{% include figure.liquid
  path='assets/img/post-08-25/peak_decay_shape.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='85%'
  caption='峰值加衰减形状的示例，衰减段取幂律形式 $\eta_t\propto(1-t/T)^{\alpha}$，$\alpha$ 分别为 0.5、1、2。$\alpha=0.5$ 对应 sqrt 形状，$\alpha=1$ 对应线性衰减到零。三条曲线的峰值与终点相同，衰减段的凹凸方向不同。'
  zoomable=true
  alt='peak plus decay learning rate shape illustrated with three power-law exponents'
%}

图中的幂律形式便于参数化，人工指定的经验调度多采用此类形式。而在后文讨论的研究中，refined schedule 由梯度范数序列逐点离线构造，Schedule-Free+ 的实际有效学习率由迭代平均机制隐式诱导，两者的曲线均自然呈现峰值加衰减形态，并不依赖先验的解析公式。

近期研究表明，上述两项现象均由同一个物理量所决定。

> ##### 核心判断
> **决定训练进展的关键物理量为有效学习率 $$\eta_t^\star=\eta_t\lVert U_t\rVert/\lVert W_t\rVert$$。**<br>权重范数在训练中持续增长，导致名义学习率与有效学习率的比值动态变化，并在设定的基础学习率之上引入隐式衰减。Hyperball 通过约束权重范数为常数消除了该隐式衰减，其实质机制体现为状态相关的隐式学习率调度。
{: .block-tip}

有效学习率 $\eta^\star$ 呈现峰值加衰减形态的机理独立于具体优化器：

> ##### 形状的判据
> **峰值加衰减对应偏差与方差权衡（bias–variance trade-off）的最优解。**<br>训练前期以较大步长快速缩减偏差，末期通过将步长衰减至零以抑制梯度方差累积。满足该最优平衡条件的调度形状构成一个函数集合，集合内部不同衰减构型的最终性能相当。
{: .block-tip}

这两项判断构成了本文的核心逻辑：有效学习率 $\eta^\star$ 是决定模型优化的真实物理量，偏差与方差的权衡决定了其几何形态。

## 1. 决定优化进展的有效学习率

含归一化层的网络参数满足尺度不变性 $\mathcal{L}(\rho W) = \mathcal{L}(W)$。纯径向伸缩不改变网络表示函数，因此表征单步优化进展的物理量为权重方向在参数球面上转过的角度。混元 ELR 研究将其定义为角更新幅度（angular update size, AUS）[[1]](https://hy.tencent.com/research/elr)：

$$
\mathrm{AUS} := \left\lVert \frac{W_{t+1}}{\lVert W_{t+1}\rVert} - \frac{W_t}{\lVert W_t\rVert}\right\rVert \approx \frac{\eta_t \lVert U_t\rVert}{\lVert W_t \rVert} =: \eta_t^\star
$$

等式右端定义的有效学习率 $\eta_t^\star$ 源自对权重衰减的一系列动力学分析 [[9]](https://arxiv.org/abs/2006.08419)。该式显式包含权重范数 $\lVert W_t \rVert$；由于权重范数在训练过程中持续增长，名义学习率 $\eta_t$ 与有效学习率 $\eta_t^\star$ 的比值动态演化，构成了叠加在名义调度之上的隐式学习率调度。

该隐式调度的影响幅度可通过对比三种名义调度进行观察：WSD 调度（峰值 $3.6\times10^{-3}$）、余弦衰减与线性衰减（峰值均为 $8.8\times10^{-3}$），两组设定的名义峰值相差约 2.4 倍。

{% include figure.liquid
  path='assets/img/post-08-25/wsd_cosine_linear_aus.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='右下：WSD、cosine、linear 三条 lr schedule；右上：对应的 weight norm；左下：对应的角更新幅度。图片来源：<a href="https://hy.tencent.com/research/elr">混元 ELR</a>。'
  zoomable=true
  alt='weight norm and angular update size under WSD, cosine and linear schedules'
%}

对比名义学习率与实际角更新幅度：尽管三条名义调度曲线形态与峰值差异显著，但在 1000 步后，三者的角更新幅度曲线高度重合，均呈现峰值后持续衰减的趋势。右上方图表显示对应的权重范数在前 2000 步由约 30 增长至 180–280，吸收并平滑了名义学习率的设定差异。

> ##### 名义学习率与有效学习率
> 优化器显式控制名义学习率调度，而直接作用于网络参数状态的是角更新幅度。权重范数的持续增长使有效步长呈现内生衰减，削弱了不同名义调度之间的实际差异。
{: .block-tip}

在恒定名义学习率设置下，权重范数引起的有效步长变化更为直观 [[2]](https://arxiv.org/abs/2607.22444)：

{% include figure.liquid
  path='assets/img/post-08-25/muonwd_muonh_elr_const.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='左：MuonWD 与 MuonH 的名义 lr，两条曲线重合于 0.025；右：对应的 effective lr，MuonWD 由约 $9\times10^{-5}$ 衰减至约 $1.8\times10^{-5}$，MuonH 在初始瞬态后保持恒定。图片来源：<a href="https://arxiv.org/abs/2607.22444">Hyperball May Not Be a Free Lunch</a>，Figure 2(a)(b)。'
  zoomable=true
  alt='nominal and effective learning rate of MuonWD and MuonH under a constant schedule'
%}

在名义学习率完全相同的条件下，MuonWD 与 MuonH 的有效学习率在训练后期相差约 5 倍。MuonWD 的有效学习率衰减完全来自权重范数的自发增长；MuonH 则因约束了权重范数，其有效学习率在经历初始瞬态后保持恒定。

## 2. 有效学习率轨迹与损失曲线的对应关系

将有效学习率 $\eta^\star$ 确立为分析对象的前提，在于验证其轨迹能否充分决定模型的损失轨迹。近期两项独立工作从正反两个方向检验了这一对应关系。

在混元 ELR 的重放实验（AUS-replay）中，研究人员首先记录标准 Adam 或 Muon 在 GPT-2（124M）训练过程中的逐步 AUS 轨迹，随后在对应的 Hyperball 变体（AdamH、MuonH）中将该 AUS 轨迹直接设为其名义学习率曲线。

{% include figure.liquid
  path='assets/img/post-08-25/aus_replay.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='第一行：AUS 曲线；第二行：验证损失；第三行：损失差值。四列分别为 Adam 与 Muon 在朴素 nanoGPT 与尺度不变 nanoGPT 上的结果。图片来源：<a href="https://hy.tencent.com/research/elr">混元 ELR</a>。'
  zoomable=true
  alt='AUS replay experiment comparing Adam/Muon with their Hyperball variants'
%}

两组设定的 AUS 轨迹与验证损失曲线基本重合，在严格满足尺度不变性的网络结构上，两者验证损失的差值保持在 $\pm0.005$ 的极小区间内 [[1]](https://hy.tencent.com/research/elr)。

另一项工作则采取参数对齐方式：在固定基础优化器的前提下，逐点微调名义学习率以匹配目标有效学习率，分别完成 MuonWD 对齐至 MuonH 轨迹以及 MuonH 对齐至 MuonWD 轨迹的双向实验 [[2]](https://arxiv.org/abs/2607.22444)。

{% include figure.liquid
  path='assets/img/post-08-25/muon_lr_alignment.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='左：MuonH 对齐到 MuonWD，绿色虚线与 MuonWD 基本重合；右：MuonWD 对齐到 MuonH，紫色虚线与 MuonH 基本重合。两图中 MuonWD 与 MuonH 的原始曲线在 500–2000 步区间存在差距。图片来源：<a href="https://arxiv.org/abs/2607.22444">Hyperball May Not Be a Free Lunch</a>，Figure 3(a)(c)。'
  zoomable=true
  alt='mutual alignment of MuonWD and MuonH training loss by learning-rate alignment'
%}

双向对齐的实验结果均与目标曲线高度贴合，仅调整名义学习率即可精确复现另一优化器的损失演化过程。该项研究据此得出结论：Hyperball 的主要机制在于提供了一种隐式的状态相关学习率调度，其梯度更新方向并未展现额外的几何优势 [[2]](https://arxiv.org/abs/2607.22444)。

> ##### 有效学习率与收敛轨迹的等价性
> **当有效学习率轨迹相同时，模型损失曲线高度重合。**<br>Hyperball 与常规带权重衰减优化器之间的性能差异可由有效学习率轨迹充分解释，权重范数约束的本质作用在于消除了参数尺度增长带来的隐式步长衰减。
{: .block-tip}

该机制同时解释了 MuonH 表现出的阶段性特征：早期收敛相对较慢，后期指标优于 MuonWD [[2]](https://arxiv.org/abs/2607.22444)。在 Hyperball 约束下不存在权重范数增长带来的自然衰减，其有效步长在后期维持在相对较高水平。

## 3. 有效学习率对 scaling law 的拟合精度

若有效学习率 $\eta^\star$ 是决定收敛状态的核心变量，以其为自变量拟合损失曲线的精度应显著优于名义学习率 $\eta$。混元 ELR 在多重幂律（MPL）损失模型中引入 $\eta^\star$ 代替 $\eta$ 后，不仅样本内拟合与跨调度预测的误差明显降低，最优 $\eta^\star$ 跨越模型宽度与深度的迁移稳定性也显著提升 [[1]](https://hy.tencent.com/research/elr)。

该现象的机理在于名义学习率与最终损失之间包含受训练进程、模型宽度与深度调控的比例变化。以名义学习率 $\eta$ 为自变量拟合 scaling law 时，参数尺度演化引入的波动被直接计入拟合误差；而有效学习率 $\eta^\star$ 剥离了该外生变量，提升了泛化预测的信噪比。

类似规律同样适用于超参数迁移过程。在 Hyperball 约束下，单步角位移满足 $\Delta\phi_t \approx \eta_t$，因此累积角位移可表示为：

$$
\sum_t \Delta\phi_t \approx \int_0^T \eta_t\,\mathrm{d}t
$$

此式表明学习率的时间积分对应权重向量转过的总角位移。实证研究发现，当不同训练任务的累积学习率相近时，模型最终损失亦保持高度接近 [[1]](https://hy.tencent.com/research/elr)。该积分量在 Hyperball 约束下具备严格的几何意义，可作为跨训练预算迁移时的基准对齐量。

此外，在 Frobenius 球面约束下一阶权重衰减失效，使得原本关于学习率与权重衰减 $(\eta,\lambda)$ 的二维超参搜索简化为一维搜索。实验给出的最优学习率随训练 token 规模呈现幂律关系 $\eta^*\propto T^{-0.32}$，与 AdamW 上报告的缩放指数相符 [[3]](https://arxiv.org/abs/2603.28743)。该指数的理论来源原因未查明。

## 4. 形状的来源：bias–variance trade-off

在明确有效学习率作为优化分析的实际对象后，核心问题转向有效学习率曲线的几何形态。尽管实验环境与推导方式各异，以下四项独立研究得出的最优学习率曲线均展现出相同的形态特征：训练前期上升至峰值，随后单调衰减并趋近于零。

**WSD 衰减段形状对比。** 在 WSD 调度的退火（cooldown）阶段对比不同人工设定形态，`sqrt`（$1-\sqrt{x}$）与参数调整后的线性衰减 `lowered linear 0.7` 表现相当，后者在困惑度指标上略优于前者 [[4]](https://arxiv.org/abs/2508.01483)。

**Schedule-Free+。** 该方法无需人工预设学习率取值与调度形态，在恒定名义学习率下，其迭代平均机制自然诱导出前期上升至峰值、后期持续衰减的实际有效步长曲线。该方法在长周期训练中达到同等损失所需时间相比 WSD 基线减少 31% [[5]](https://arxiv.org/abs/2605.19095)。

{% include figure.liquid
  path='assets/img/post-08-25/schedulefree_plus_lr.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='1B 模型、1000 tokens per parameter 设置。红色为 Schedule-Free+（beta 由 0.8 退火到 0.965）的实际 lr，黑色为 linear decay，蓝色为 WSD。图片来源：<a href="https://arxiv.org/abs/2605.19095">Schedule-Free+</a>。'
  zoomable=true
  alt='effective learning rate of Schedule-Free+ compared with linear decay and WSD'
%}

**离线 refined schedule。** 基于梯度范数序列逐点离线构造的最优调度，在跨越视觉、语言与推荐等八个不同任务中均自发形成峰值加衰减形态 [[6]](https://arxiv.org/abs/2310.07831)。

{% include figure.liquid
  path='assets/img/post-08-25/refined_schedule.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='八个任务（ImageNet、IWSLT14、GPT、RoBERTa、DLRM、MRI、ViT、RCNN）上的结果。左列：梯度范数序列；中列：平滑后的梯度范数；右列：由此构造的 refined schedule，内嵌图为对数刻度。图片来源：<a href="https://arxiv.org/abs/2310.07831">Defazio et al. (2024)</a>，Figure 4。'
  zoomable=true
  alt='gradient norms and refined schedules across eight tasks'
%}

**minus-square-root。** 混元 ELR 依据实测角更新幅度演化规律提出的负平方根调度，在 Modded-nanoGPT Track 3 上以 3,175 步达到目标损失 3.28 [[1]](https://hy.tencent.com/research/elr)；文献 [[2]](https://arxiv.org/abs/2607.22444) 在同一基准上采用 power-0.4 调度以 3,150 步达到目标。两者的几何形态接近，步数差异仅为 25 步。

上述四项研究获得调度形态的路径互不相同，但最终结论均收敛于峰值加衰减。这一收敛性可由统计估计误差与权重尺度动力学两个维度进行解释。

### 4.1 偏差与方差的平衡

随机梯度下降中的单步更新对模型估计误差产生两方面作用：参数脱离初始点向最优解移动，降低模型偏差；随机梯度噪声在迭代轨迹中持续累积，推高估计方差。学习率大小调节两者的相对比例。

在训练早期，参数远离收敛区间，偏差项主导总误差，较大的学习率能够加速偏差下降；在训练末期，参数接近目标区域，方差项逐渐主导误差，降低学习率在统计上相当于扩展更新采样的有效平均窗口，从而有效降低方差。峰值加衰减形态正是兼顾这两个阶段误差控制的自然组合。

偏差项与方差项的相对比例受任务性质、模型规模与训练预算调控。使总误差 $Bias+Variance$ 取到最小值的调度形态构成一个函数集合，在幂律形式参数化下对应指数 $\alpha$ 的一个区间。

{% include figure.liquid
  path='assets/img/post-08-25/bias_variance_shapes.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='100%'
  caption='各 cooldown 形状的 bias–variance 分布，虚线为 $Bias+Variance$ 取到最小值的位置。左：仅比较 lowered linear 形状，参数减小时方差下降、偏差上升；右：全部非线性形状与部分 lowered linear 形状，其中 sqrt 与 0.7 落在虚线附近，square、cosine、mirror cosine、linear 位于虚线上方。左右两图横轴范围不同。图片来源：<a href="https://arxiv.org/abs/2508.01483">Dremov et al. (2025)</a>，Figure 6。'
  zoomable=true
  alt='bias-variance plot for different cooldown shapes'
%}

图示直观展现了偏差与方差的反向权衡关系：随着 `lowered linear` 参数的减小，方差下降伴随着偏差上升。右侧图表显示，`sqrt` 与 `lowered linear 0.7` 均落在总误差最小线附近，而 `square`、`cosine`、`mirror cosine` 及标准 `linear` 则位于最小线上方。落在最小线附近的形状构成包含多种非线性构型的集合 [[4]](https://arxiv.org/abs/2508.01483)；文献 [[7]](https://arxiv.org/abs/2502.15938) 报告的直接线性衰减至零亦属于该近优集合。

{% include figure.liquid
  path='assets/img/post-08-25/sqrt_vs_lowered_linear.png'
  class='img-fluid rounded z-depth-1 mx-auto d-block'
  width='55%'
  caption='cooldown 段（训练进度 80%–100%）内 sqrt 与 lowered linear 0.7 的形状对照。图片来源：<a href="https://arxiv.org/abs/2508.01483">Dremov et al. (2025)</a>，Figure 7。'
  zoomable=true
  alt='comparison of sqrt cooldown shape and lowered linear 0.7'
%}

两条几何构型不同的衰减曲线表现相当，表明调度曲线具体解析形式的调优收益存在上限：实证表明调整 AdamW 的动量系数 $\beta_2$ 所带来的损失差异与精细挑选调度形状基本处于同一量级 [[4]](https://arxiv.org/abs/2508.01483)。

### 4.2 权重范数动态对观测调度的平滑效应

在未施加 Hyperball 约束的标准训练中，实际生效的有效学习率由显式调度函数与权重范数的自然演化共同塑造。minus-square-root 调度直接以 $\eta^\star$ 作为设计对象，因而需要显式设定峰值加衰减形态；而在 Hyperball 优化中，权重范数被固定为常数，消除了范数膨胀带来的内生步长衰减，名义学习率 $\eta_t$ 本身必须显式满足衰减要求。

权重范数的增长对步长差异具有压缩平滑效应，导致在不同名义调度下观察到的有效学习率形态差异，显著小于名义调度曲线本身的差异。第 1 节中三种峰值与几何构型各异的名义调度在 1000 步后角更新幅度高度趋同，体现了该平滑效应的作用。

## 5. 小结

本文讨论了预训练学习率调度的两个核心问题。在调度对象维度，优化器显式控制名义学习率 $\eta_t$，而决定优化进展的核心物理量是有效学习率 $\eta_t^\star$；二者的比例关系由权重范数动态调控。Hyperball 约束固定了权重范数，其实质作用在于消除权重范数增长带来的内生衰减，从而表现为一种隐式的学习率调度。在曲线形态维度，峰值加衰减形态对应偏差与方差权衡的最优解，前期以大步长降低模型偏差，末期以衰减步长抑制随机噪声方差。

偏差与方差的权衡结构进一步表明：满足近优平衡条件的调度形态构成一个函数集合，特定解析形式的调优收益存在上限；同时两者的相对比重由训练预算、模型容量与数据噪声水平共同决定，因而不同训练环境下的最优衰减形态存在差异。

实践中的参考建议：

1. 对比不同优化器与调度策略时记录有效学习率 $\eta^\star$ 或角更新幅度 AUS，避免受名义学习率的尺度假象误导。
2. 拟合 scaling law 与迁移超参数时，以有效学习率 $\eta^\star$ 或累积角位移 $\int\eta_t\mathrm{d}t$ 作为对齐基准。
3. 在 Hyperball 优化中，由于去除了权重范数引入的自然衰减，名义学习率必须配合明确的衰减规划，且衰减幅度存在上限以避免后期表现劣于带权重衰减的基准优化器 [[2]](https://arxiv.org/abs/2607.22444)。
4. 调度曲线具体解析构型的调优空间相对有限，`sqrt`、`lowered linear 0.7` 与线性衰减至零均处于近优集合内，调参预算应优先投入峰值学习率与动量参数 $\beta_2$。

尚未完全解决的问题包括：自适应方法诱导的步长形态与人工指定解析函数之间的定量对应关系尚未建立；$\eta^*\propto T^{-0.32}$ 关系式中标度指数的理论来源原因未查明。

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
