---
layout: post
title: "为什么 LLM pretrain 过程中途要把 batch size 翻倍"
date: 2026-06-16 14:00:00
description: "从 Apertus 70B 的 Double GBS 现象出发，用梯度噪声尺度、critical batch size 与变分法，推导 LLM 预训练中途增大 batch size 的最优 schedule，并在 noisy quadratic model 上验证。"
tags: [optimization, deep-learning, llm, scaling-law, batch-size]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

Apertus 70B 的训练 loss 曲线上有一条标注为 Double GBS 的竖线：训练约 4.4T tokens 时，global batch size 由 8.4M 增至 16.8M tokens，learning rate 不变，loss 随之下降一小段。

{% include figure.liquid
  path='assets/img/post-06-16/image.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Apertus 70B 的 loss 曲线。红色虚线为中途的 Double GBS（global batch size 从 8.4M 翻到 16.8M tokens），其余竖线为数据阶段切换。'
  zoomable=true
  alt='Apertus 70B loss curve with a Double GBS line'
%}

本文讨论三个问题：为什么增大 batch 发生在训练后期、增大后 loss 为何下降、以及最优的增大时机与幅度，并在 noisy quadratic model 上做数值验证。

## 1. 后期增大 batch 的动机

记训练目标为

$$
L(\theta)=\mathbb E_{x}[\ell(\theta;x)],
$$

单样本梯度 $$g(x)=\nabla_\theta \ell(\theta;x)$$ 的均值为真实梯度 $$\mu=\nabla L(\theta)=\mathbb E[g(x)]$$，协方差为

$$
C=\mathbb E[(g(x)-\mu)(g(x)-\mu)^\top].
$$

batch size 为 $$B$$ 的 mini-batch 梯度

$$
\hat g_B=\frac{1}{B}\sum_{i=1}^{B} g(x_i)
$$

无偏，协方差按 $$1/B$$ 衰减：

$$
\operatorname{Cov}(\hat g_B)=\frac{C}{B}.
$$

即 batch 翻倍使梯度方差减半、标准差降至 $$1/\sqrt 2$$。定义信噪比与 gradient noise scale

$$
\text{SNR}(B)=\frac{\|\mu\|^2}{\mathbb E\|\hat g_B-\mu\|^2}=\frac{B\|\mu\|^2}{\operatorname{tr}(C)},\qquad
\mathcal G=\frac{\operatorname{tr}(C)}{\|\mu\|^2},
$$

二者满足 $$\text{SNR}(B)=B/\mathcal G$$。维持信噪比要求 $$B\propto \mathcal G$$。训练后期模型趋近低损区，$$\|\mu\|$$ 减小而 $$\operatorname{tr}(C)$$ 不同步减小，故 $$\mathcal G$$ 增大，所需 batch 随之增大。这是增大 batch 应发生在后期的第一层原因。

### 1.1 单步更新分析

SGD 单步为 $$\theta^+=\theta-\eta \hat g_B$$。对 $$L(\theta^+)$$ 作二阶展开并对采样取期望，

$$
\mathbb E[L(\theta^+)]\approx L(\theta)-\eta \|\mu\|^2+\frac{\eta^2}{2}\mu^\top H\mu+\frac{\eta^2}{2B}\operatorname{tr}(HC),\qquad H=\nabla^2 L(\theta).
$$

仅末项 $$\frac{\eta^2}{2B}\operatorname{tr}(HC)$$ 依赖 $$B$$，为噪声引入的损耗，与 $$1/B$$ 成正比。要求其不超过有效下降项 $$\eta\|\mu\|^2$$ 的固定比例，得到临界 batch size

$$
B_{\text{crit}}\sim\frac{\eta\operatorname{tr}(HC)}{\|\mu\|^2}.
$$

训练早期 $$\|\mu\|^2$$ 大、$$B_{\text{crit}}$$ 小，过大的 batch 仅减少更新步数；后期 $$\|\mu\|^2$$ 减小、$$B_{\text{crit}}$$ 增大，若维持小 batch 则 loss 受噪声项主导。

### 1.2 一维情形

取 $$L(\theta)=\tfrac12\lambda\theta^2$$，随机梯度 $$\hat g_B=\lambda\theta+\xi$$，$$\operatorname{Var}(\xi)=\sigma^2/B$$。由 $$\theta_{k+1}=(1-\eta\lambda)\theta_k-\eta\xi_k$$ 得

$$
\mathbb E[L_{k+1}]=(1-\eta\lambda)^2\mathbb E[L_k]+\frac{\lambda\eta^2\sigma^2}{2B},
$$

稳态 loss floor 为

$$
L_\infty(B)\approx\frac{\eta\sigma^2}{4B},\qquad L_\infty(2B)\approx\tfrac12 L_\infty(B).
$$

即 learning rate 固定时 batch 翻倍使噪声 floor 减半；图中 Double GBS 处的下降即对应该 floor 的下移。综合上述，

$$
\|\mu\|^2 \downarrow \;\Rightarrow\; \mathcal G=\frac{\operatorname{tr}(C)}{\|\mu\|^2}\uparrow \;\Rightarrow\; B_{\text{crit}}\uparrow \;\Rightarrow\; \text{增大 batch}.
$$

代价是固定 token 预算下更新步数减少，故早期不宜采用过大 batch。

## 2. 这一做法是否常见

训练中动态增大 batch（batch ramp / warmup）是大模型预训练的常见配置，只是较少单独绘入 loss 曲线。GPT-3 的 batch 在前 4–12B tokens 由 32k 线性增至 full batch；Llama 3 405B 分阶段由 4M 增至 8M 再至 16M；OLMo-65B 自 2M 起每 100B tokens 翻倍至 16M。其适用前提是模型大、采用同步数据并行、GPU 数多、并行效率重要；小模型、LoRA/SFT、CV 训练通常固定 batch 而仅调整 learning rate。

是否增大取决于当前是否接近 critical batch size。McCandlish 等提出的 gradient noise scale 用于估计「最大有用 batch size」，且随 loss 下降而增大；OLMo 的 CBS 研究亦发现 CBS 早期快速上升、随后趋于平台。但 batch 过大会损失 token efficiency：固定预算下更新步数减少，loss 反而变差。critical batch size 即数据并行效率与 token efficiency 的折中点。

## 3. 最优 batch size schedule

进一步可问：最优 schedule 的形式是什么，能否如最优 learning rate schedule 一样由变分导出。结论是连续最优解并非线性，而是单调加速的截断幂律（clipped power-law）；硬件的离散约束将其近似为若干次翻倍。

### 3.1 变分形式

设连续时间 $$t$$ 为 optimizer step，$$b(t)$$ 为每步 batch（即每步消耗的 token 数），预算约束 $$\int_0^T b(t)\,dt=D$$。在 functional scaling law（FSL）近似下，excess loss 分解为

$$
\mathcal E[T,b]=A\,T^{-s}+C\int_0^T \frac{K(T-t)}{b(t)}\,dt.
$$

首项 $$A\,T^{-s}$$ 反映更新步数带来的信号学习，次项为梯度噪声的累积贡献。$$s>0$$ 为 source exponent，$$\beta>1$$ 为 capacity exponent。

### 3.2 固定 T 求 b(t)

固定 $$T$$ 时优化问题为

$$
\min_{b(t)}\int_0^T \frac{K(T-t)}{b(t)}\,dt,\qquad \text{s.t.}\ \int_0^T b(t)\,dt=D.
$$

由 Cauchy–Schwarz，

$$
\left(\int_0^T \frac{K(T-t)}{b(t)}\,dt\right)\left(\int_0^T b(t)\,dt\right)\ge\left(\int_0^T \sqrt{K(T-t)}\,dt\right)^2,
$$

等号成立当且仅当 $$b^*(t)\propto \sqrt{K(T-t)}$$。代入 FSL 核 $$K(\tau)\asymp (\tau+1)^{1/\beta-2}$$，

$$
b^*(t)\asymp c\,(T-t+1)^{\frac{1}{2\beta}-1}.
$$

指数 $$\tfrac{1}{2\beta}-1<0$$，故 $$b^*(t)$$ 随 $$t\to T$$ 单调增大：连续最优解为加速增长，而非线性 ramp。计入硬件上下限即得截断幂律：

$$
b^*(t)=\operatorname{clip}\!\left(c\,(T-t+1)^{\frac{1}{2\beta}-1},\,B_{\min},\,B_{\max}\right).
$$

实践中「先维持小 batch、后跳至大 batch」即该连续解的离散化；当仅允许 2 的幂时表现为翻倍。

### 3.3 进一步优化 T

将最优 $$b(t)$$ 回代，

$$
\mathcal E(T)\asymp A\,T^{-s}+C\,\frac{T^{1/\beta}}{D},
$$

一阶条件给出

$$
T^*\asymp D^{\frac{\beta}{1+s\beta}},\qquad B_{\max}\asymp D^{\frac{1/2+s\beta}{1+s\beta}}.
$$

由此分为两种 regime。easy task（$$s>1-\tfrac1\beta$$）下最优解为全程缓增的幂律，最终 loss rate $$\mathcal E_D^*\asymp D^{-\frac{s\beta}{1+s\beta}}$$。hard task（$$s\le 1-\tfrac1\beta$$）下无约束解倾向极大的 $$T$$，但 batch 受下限约束，最优解分为两段：

$$
b^*(t)=
\begin{cases}
B_{\min}, & 0\le t<T_1^*,\\[4pt]
B_{\max}(T^*-t+1)^{\frac{1}{2\beta}-1}, & T_1^*\le t\le T^*,
\end{cases}
$$

且增长段占比随 $$D$$ 减小，$$\frac{T^*-T_1^*}{T^*}\asymp D^{-\frac{1-1/\beta-s}{2-1/\beta}}$$。直观上，hard task 早期稀缺的是更新步数而非低噪声梯度，故应长期维持小 batch 以累积步数，后期再以大 batch 降噪。FSL 称此形态为 stable-growth，即 batch 版的 WSD。LLM 预训练属于此类。

由此「中途 Double GBS」得到解释：连续最优为单调快速增长，工程上受限于少数可用 batch size，在曲线上即表现为一两次竖直跳变；单次翻倍是截断幂律的工程近似。

### 3.4 两段式与多段式的显式解

限制为两段 $$B_1\to B_2$$（$$B_2>B_1$$）：

$$
b(t)=
\begin{cases}
B_1,&0\le t<t_s,\\
B_2,&t_s\le t\le T,
\end{cases}
$$

设切换前消耗 $$D_1$$ 个 token，回代目标函数得关于 $$D_1$$ 的一维问题，内点最优满足

$$
A\,s\,S^{-s-1}=C\left[\frac{K(S)}{B_1}+\frac{K(R)}{B_2}\right],\qquad S=\frac{D_1}{B_1}+\frac{D-D_1}{B_2},\quad R=\frac{D-D_1}{B_2}.
$$

左端为延长小 batch 以累积步数的信号收益，右端为噪声累积代价，二者平衡处即切换点。多段翻倍同理：取 $$B_j=B_{\min}r^j$$，令各段边界落于连续解，得

$$
t_j=T+1-\left(\frac{B_j}{c}\right)^{1/p},\qquad p=\tfrac{1}{2\beta}-1,
$$

换算至 token 轴 $$z_j=\int_0^{t_j} b^*(u)\,du$$ 即得各次翻倍的时机，再对齐至 checkpoint 或数据阶段边界。

### 3.5 经验替代方案

变分解需预先估计 $$s,\beta,K$$。更易落地的方法是直接追踪 critical batch size：OLMo 的 CBS 研究在线估计 CBS，自小 batch 起步，待 CBS 增大后翻倍，在 OLMo 1B 上节省约 43% 的更新步数而不损 loss。

需注意「最优」依赖于目标函数：固定预算的最终 validation loss、达到目标 loss 的 wall-clock、或计入通信与利用率成本，对应不同的最优 $$b(t)$$。FSL 的主分析基于 vanilla SGD 与常数 learning rate；现代 LLM 多用 AdamW，联合的 learning-rate / batch-size schedule 仍需进一步分析。

## 4. NQM 上的数值验证

采用 noisy quadratic model（NQM）验证。NQM 是大 batch 理论的标准代理模型，亦为第 1.2 节一维情形的多维推广，其期望 loss 满足精确递推，无需 Monte Carlo：

$$
v_{i,k+1}=(1-\eta h_i)^2 v_{i,k}+\frac{\eta^2\sigma_i^2}{B_k},\qquad L_k=\tfrac12\sum_i h_i v_{i,k}.
$$

展开至第 $$T$$ 步，

$$
L_T=\underbrace{S(T)}_{\text{信号项, 只依赖步数}}+\sum_k\frac{\kappa(T-1-k)}{B_k},\qquad \kappa(j)=\tfrac12\eta^2\sum_i h_i\sigma_i^2(1-\eta h_i)^{2j},
$$

与第 3 节的 FSL 形式同构。故固定预算 $$D=\sum_k B_k$$ 下，Cauchy–Schwarz 直接给出 $$B_k^*\propto\sqrt{\kappa(T-1-k)}$$，即截断幂律在 NQM 中为精确解。

取幂律谱，测得 $$s\approx0.49$$、$$\beta\approx1.96$$，位于 hard regime，对应 LLM。固定预算与常数 learning rate、仅改变 schedule，结果如下。

{% include figure.liquid
  path='assets/img/post-06-16/batch_schedule_experiment.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='固定 token 预算与常数 learning rate 下的对比。左：loss 的 log-log 总览；中：线性 token 横轴，各 schedule 的差异与末段骤降清晰可见；右：对应 batch schedule，最优解与解析式 $(T-t+1)^{1/2\beta-1}$、翻倍阶梯一致。'
  zoomable=true
  alt='loss curves and batch schedules for the NQM experiment'
%}

主要结果：

- 最优 schedule（红）在大部分训练中维持 $$B_{\min}$$ 以累积步数，末段以幂律增大 batch，loss 骤降至常数 batch 的约 $$1/10$$，与 WSD / 最优 lr schedule 的曲线一致。
- 最优 schedule 与解析式 $$(T-t+1)^{1/2\beta-1}$$（黑虚线）吻合。
- 同预算下最终 loss：常数 1.0×、两段式 3.8×、翻倍 9.9×、最优 10.2×。
- 最优解的最终 loss 已低于最优常数 batch 的噪声 floor $$L_\infty(B)$$，后者在任意预算下均无法达到。

两项支撑结果：噪声 floor 精确 $$\propto 1/B$$，核函数 $$\kappa(j)$$ 为幂律。

{% include figure.liquid
  path='assets/img/post-06-16/batch_schedule_diagnostics.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='左：噪声 loss floor 精确 $\propto 1/B$。右：噪声核 $\kappa(j)$ 在中段为幂律，斜率给出 $\beta\approx1.96$。'
  zoomable=true
  alt='diagnostics: noise floor and kernel power law'
%}

代码见 `experiments/batch_schedule_nqm.py`（纯 NumPy）；调整谱指数 `S_SRC`、`C_NOISE` 可在 easy / hard regime 间切换。

## 5. 真实 transformer 上的验证

NQM 有一个固有弱点：它的期望 loss 被刻意构造成与 FSL 同构，所以「最优 schedule 在 NQM 里胜出」近乎自证。为打破这一循环，在真实 transformer 上重做对比：模型不再为 FSL 量身定做。

设置：一个 45M 参数的标准 GPT（6 层，$$d=512$$，block 1024），在 FineWeb（GPT-2 tokenizer）上用 AdamW 训练，learning rate **全程恒定**（8M token 线性 warmup 后不变），固定总预算 **600M tokens**，同一初始化与数据顺序，**唯一变量是 batch schedule**。对比四条：常数 batch 64k / 128k / 512k tokens，以及一条 late-switch 翻倍 ramp（64k→128k→256k→512k）。

{% include figure.liquid
  path='assets/img/post-06-16/gpu_batch_schedule.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='45M GPT 在 FineWeb 上、同 600M token 预算与常数 lr 下的对比。左：val loss vs tokens（翻倍 ramp 红线全程最低）；中：val loss vs optimizer steps（ramp 以更少步数达到更低 loss）；右：四条 batch schedule。'
  zoomable=true
  alt='real transformer batch schedule experiment on FineWeb'
%}

结果（同预算、同 lr）：

| schedule | 最终 val loss | optimizer steps |
|---|---|---|
| 常数 512k | 4.831 | 1145 |
| 常数 128k | 4.219 | 4578 |
| 常数 64k | 4.145 | 9156 |
| **翻倍 ramp** | **4.091** | 6182 |

翻倍 ramp 在第 5990 步（消耗 520M tokens）即达到最强常数 batch（64k）的最终 loss，**少用约 35% 的 optimizer step**，并继续下降到 4.091，低于所有常数 batch。这与 OLMo「同 loss 省 43% step」一致，也是 §1、§3 的机制在真实 AdamW transformer 上的体现：小 batch 在前累积步数，大 batch 在后降低噪声。

须诚实标注四点：

1. **ramp 的切换点是经验选取的**（各档 token 占比手取），仅由 §3 的定性结论（hard regime → late switch）给出方向，不是 §3.4 用 $$z_j=\int_0^{t_j}b^*$$ 算出的精确离散化。
2. **精确化需要 $$\beta$$，而 $$\beta$$ 在此不可辨识。** 从三条常数 batch 曲线联合拟合 FSL，残差对 $$\beta$$ 几乎是平的（把 $$\beta$$ 固定在 4 或 8，RMS 不变），原因是三条都远未接近噪声 floor（最终 loss 随 batch 单调，信号主导），而 $$\beta$$ 只在逼近 floor 的形状里显形。这一限制在真实 LLM 训练中是**结构性**的：没人会为拟 $$\beta$$ 而跑多条常数 sweep，单次大训练也几乎总停在信号主导区。正因如此，§3.5 才指向在线 track critical batch size 这类经验方案。
3. **AdamW $$\neq$$ FSL 的 vanilla SGD。** 经验上大 batch 末段的收益甚至超过该拟合的预测，说明 Adam 下的噪声–batch 关系与 SGD 理论并不完全一致。
4. **单 seed、单次跑**，无误差棒。

因此这一节的结论是有限而明确的：**late-switch 翻倍 schedule 在真实 transformer 上确实优于常数 batch，验证了 §3 的定性预测；但「最优切换点」依赖一个实践中拿不到的 $$\beta$$，目前的 ramp 仍是经验选择。把它 principled 化的现实方向不是离线拟合，而是单次训练内的在线自适应（如 CBS tracking），这仍是开放问题。**

训练代码 `experiments/bsched_gpt.py`，FSL 拟合诊断 `experiments/fit_fsl.py`。

## 小结

后期增大 batch 的本质，是噪声损耗项 $$\frac{\eta^2}{2B}\operatorname{tr}(HC)$$ 随 $$1/B$$ 衰减、而 $$B_{\text{crit}}\sim \eta\operatorname{tr}(HC)/\|\mu\|^2$$ 随训练增大；固定 learning rate 下增大 batch 近似于一次 learning-rate decay，并提升硬件利用率。该做法见于 GPT-3、Llama 3、OLMo，仅较少绘入主图。最优 schedule 为截断幂律，在 LLM 的 hard regime 与离散约束下退化为「长期小 batch + 后期若干次翻倍」；Apertus 的 Double GBS 即其工程近似。NQM 与一个真实 45M transformer 都印证了这一形态优于常数 batch；但精确的切换点依赖实践中难以获得的 $$\beta$$，如何在单次训练内自适应地确定它，仍是开放问题。

## 参考资料

- Apertus 技术报告：[Apertus: Democratizing Open and Compliant LLMs for Global Language Environments](https://arxiv.org/abs/2509.14233)（70B loss 曲线与 Double GBS 的出处）
- McCandlish, Kaplan, Amodei et al.：[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)（gradient noise scale）
- Smith, Kindermans, Le et al.：[Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489)
- Brown et al.：[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)（GPT-3 的 batch ramp）
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- OLMo Team：[2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)（batch size warmup 配方）
- [Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](https://arxiv.org/abs/2505.23971)（OLMo 1B 上 batch warmup 省约 43% gradient step）
- Li, Wang, …, Wu：[Optimal Learning-Rate Schedules under Functional Scaling Laws: Power Decay and Warmup-Stable-Decay](https://arxiv.org/abs/2602.06797)（FSL 框架与 source / capacity 指数）
- Wang, Li, Zhou, …, Wu：[Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](https://arxiv.org/abs/2602.14208)（FSL 的 batch size 版本）

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026doublebatch,
  title={为什么 LLM pretrain 过程中途要把 batch size 翻倍},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/why-double-batch-size-llm-pretraining/}
}
```
