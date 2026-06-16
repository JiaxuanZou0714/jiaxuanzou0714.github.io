---
layout: post
title: "为什么 LLM pretrain 过程中途要把 batch size 翻倍"
date: 2026-06-16 14:00:00
description: "从 Apertus 70B loss 曲线上那条 Double GBS 竖线出发，用梯度噪声、critical batch size，再到泛函/变分求最优 batch size schedule，把中途翻倍这件事讲清楚。"
tags: [optimization, deep-learning, llm, scaling-law, batch-size]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

前几天翻 Apertus 的技术报告，70B 这张 loss 曲线让我盯了很久。

{% include figure.liquid
  path='assets/img/post-06-16/image.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='Apertus 70B 的 loss 曲线。红色虚线是中途的 Double GBS（global batch size 从 8.4M 翻到 16.8M tokens），其余几条竖线是数据阶段的切换。'
  zoomable=true
  alt='Apertus 70B loss curve with a Double GBS line'
%}

图上有一条红色竖虚线，标着 Double GBS。就在那个位置，global batch size 从 8.4M tokens 翻到了 16.8M，learning rate 没动，loss 还顺势往下掉了一小节。报告给的理由很朴素：训练后期把 batch 调大，效果类似于把 learning rate 调小，既帮助收敛，又能多用机器、提高硬件利用率，而且这次翻倍没有引起 loss spike，也没回滚。

我还是第一次见有人把 batch 翻倍直接画进主图。背后的数学其实不难，我想算一遍：为什么偏偏后期翻、翻完 loss 为什么会掉、到底该在哪一步翻多少。

## 1. 后期把 batch 调大，到底在调什么

先把记号立起来。训练目标是

$$
L(\theta)=\mathbb E_{x}[\ell(\theta;x)],
$$

单样本梯度记成 $$g(x)=\nabla_\theta \ell(\theta;x)$$，真实梯度是它的均值 $$\mu=\nabla L(\theta)=\mathbb E[g(x)]$$，而梯度的方差结构是

$$
C=\mathbb E[(g(x)-\mu)(g(x)-\mu)^\top].
$$

batch size 取 $$B$$ 时，mini-batch 梯度

$$
\hat g_B=\frac{1}{B}\sum_{i=1}^{B} g(x_i)
$$

是无偏的，$$\mathbb E[\hat g_B]=\mu$$，但它的协方差随 $$B$$ 反比缩小：

$$
\operatorname{Cov}(\hat g_B)=\frac{C}{B}.
$$

核心就这一句：batch 翻倍，梯度估计的方差减半，标准差只剩原来的 $$1/\sqrt 2$$，每一步都更贴近真实梯度的方向。

写成信噪比看得更清楚。定义

$$
\text{SNR}(B)=\frac{\|\mu\|^2}{\mathbb E\|\hat g_B-\mu\|^2}=\frac{B\|\mu\|^2}{\operatorname{tr}(C)},
$$

再定义 gradient noise scale

$$
\mathcal G=\frac{\operatorname{tr}(C)}{\|\mu\|^2},
$$

就有一个很干净的关系

$$
\text{SNR}(B)=\frac{B}{\mathcal G}.
$$

想维持同样的梯度信噪比，batch 就得跟着 noise scale 走，$$B\propto \mathcal G$$。

关键在于 $$\mathcal G$$ 自己会变。训练到后期，模型靠近一个局部低损区，真实梯度 $$\|\mu\|$$ 越来越小；可数据本身带来的梯度方差 $$\operatorname{tr}(C)$$ 不会同步缩小。于是 $$\mathcal G$$ 变大，维持信噪比所需要的 batch 也跟着变大。这是「为什么是后期才翻」的第一层解释。

### 1.1 放进一步更新里看

信噪比只是个直觉，得放进一步更新里才算数。SGD 一步是

$$
\theta^+=\theta-\eta \hat g_B,
$$

对 $$L(\theta^+)$$ 做二阶泰勒展开，再对 mini-batch 的随机性取期望，得到

$$
\mathbb E[L(\theta^+)]\approx L(\theta)-\eta \|\mu\|^2+\frac{\eta^2}{2}\mu^\top H\mu+\frac{\eta^2}{2B}\operatorname{tr}(HC),
$$

其中 $$H=\nabla^2 L(\theta)$$。前三项都跟 batch 无关，真正被 $$B$$ 控制的是最后那项

$$
\frac{\eta^2}{2B}\operatorname{tr}(HC).
$$

这是随机噪声带来的 loss 损耗，和 $$1/B$$ 成正比。batch 翻倍，这一项直接减半。

要让这项噪声损耗压不过有用的下降项 $$\eta\|\mu\|^2$$，写成 $$\frac{\eta^2}{2B}\operatorname{tr}(HC)\le \rho\,\eta\|\mu\|^2$$，整理出一个临界 batch：

$$
B_{\text{crit}}\sim\frac{\eta\operatorname{tr}(HC)}{\|\mu\|^2}.
$$

训练早期 $$\|\mu\|^2$$ 大，$$B_{\text{crit}}$$ 小，这时硬上大 batch 只是白白减少更新步数；后期 $$\|\mu\|^2$$ 变小，$$B_{\text{crit}}$$ 涨上来，你还守着早期那个小 batch，梯度噪声就显得太大，loss 会更多地被噪声项卡住。

### 1.2 缩到一维

降到一维，这事就一眼看明白了。取 $$L(\theta)=\tfrac12\lambda\theta^2$$，随机梯度 $$\hat g_B=\lambda\theta+\xi$$，$$\mathbb E[\xi]=0$$，$$\operatorname{Var}(\xi)=\sigma^2/B$$。更新 $$\theta_{k+1}=(1-\eta\lambda)\theta_k-\eta\xi_k$$ 给出

$$
\mathbb E[L_{k+1}]=(1-\eta\lambda)^2\mathbb E[L_k]+\frac{\lambda\eta^2\sigma^2}{2B}.
$$

稳态下，噪声撑起来的 loss floor 是

$$
L_\infty(B)\approx\frac{\eta\sigma^2}{4B},\qquad L_\infty(2B)\approx\tfrac12 L_\infty(B).
$$

learning rate 不动、batch 翻倍，噪声造成的收敛 loss floor 大致砍半。回头看那张图，Double GBS 处 loss 掉的那一小节，差不多就是这个 floor 在往下挪，不是什么玄学。

整条链子连起来就是

$$
\text{后期 } \|\mu\|^2 \downarrow \;\Rightarrow\; \mathcal G=\frac{\operatorname{tr}(C)}{\|\mu\|^2}\uparrow \;\Rightarrow\; B_{\text{crit}}\uparrow \;\Rightarrow\; \text{该把 batch 调大}.
$$

代价是同样的 token 数下 optimizer step 变少，所以早期不该贪大 batch，后期才划算。

## 2. 这是常规操作，还是个例

算明白之后我反而更好奇：这种中途翻 batch 的事到底常不常见，为什么我以前没怎么注意到。

结论是，动态调大 batch（batch ramp / batch warmup）在大模型预训练里是常见 recipe 之一，但「中途单独放一个 Double GBS、还画在 loss 曲线上」确实少见。所以第一次见觉得新鲜，很正常。

几个例子就够说明它不是个例。GPT-3 就不是固定 batch 的，它的 batch size 从 32k tokens 线性涨到 full batch，ramp 集中在前 4–12B tokens。Llama 3 405B 也是分段调大，早期用小 batch 稳住，再为效率往上加，4M → 8M → 16M tokens，报告说这套 recipe 很稳，loss spike 很少。OLMo-65B 用的是 batch size warmup，大约从 2M tokens 起，每 100B tokens 翻一倍，直到 16M 左右；而更小的 1B、7B 反而用固定的 4M tokens。

共同点是，模型够大、同步数据并行、训练 token 多、卡多、并行效率要紧的时候，才值得去动 batch。普通深度学习教程、小模型、LoRA/SFT、CV 训练里，大多数人还是固定 batch、只调 learning rate schedule。

至于为什么容易错过它，我觉得主要是两点。公开的 loss 曲线通常不标这些细节，batch schedule 多半藏在训练附录里，不会单独画一条竖线，Apertus 把它摆到主图上本身就少见。再加上 batch ramp 常发生在训练很早期（GPT-3 是前 4–12B tokens），而完整训练动辄几百 B 到几 T tokens，这条 ramp 在整段曲线里太短，不留意就过去了。

要不要这么做，关键是当前是不是已经接近、甚至低于 critical batch size。McCandlish 那套 gradient noise scale 就是用来估「最大有用 batch size」的，并且指出 noise scale 会随 loss 下降而变大，正好对应第 1 节的结论。后来 OLMo 关于 critical batch size 的工作也发现，CBS 在初始化附近很小，训练早期快速上升，之后趋于平台，这就是「先小 batch、后大 batch」的经验依据。但 batch 也不能无脑加大，太大就会牺牲 token efficiency，同样 token 预算下 optimizer step 变少，loss 反而可能更差。critical batch size 本质就是数据并行效率和 token 效率之间的那个折中点。

## 3. 能不能直接解出最优的 batch size schedule

既然临界 batch 会随训练变化，那自然的下一个问题是：最优的 batch size schedule 长什么样？是线性增加，还是中途翻倍？能不能像求最优 learning rate schedule 那样，用泛函/变分直接解出来？

能，而且已经有挺直接的理论版本。先把结论摆这：连续最优解通常不是线性，而是一条单调加速增长的 clipped power-law；真实系统受 GPU 数、pipeline、梯度累积、checkpoint 这些离散约束，才把它压成「几次翻倍」或者「几段固定 batch」。

### 3.1 变分建模

令连续时间 $$t$$ 表示 optimizer step，$$b(t)$$ 是第 $$t$$ 步的 global batch size，也就是每步吃掉的 token 数，总 token 预算是

$$
\int_0^T b(t)\,dt = D.
$$

在 functional scaling law（FSL）的近似下，最终的 excess loss 可以拆成两部分：

$$
\mathcal E[T,b]=A\,T^{-s}+C\int_0^T \frac{K(T-t)}{b(t)}\,dt.
$$

第一项 $$A\,T^{-s}$$ 是「optimizer step 越多，信号学得越够」，第二项是 SGD 梯度噪声累计出的 loss 贡献，batch 越大、噪声越小。这里 $$s>0$$ 是 source exponent，控制信号学习的快慢；$$\beta>1$$ 是 capacity exponent，控制噪声遗忘的快慢。

### 3.2 固定 T，先解 b(t)

先固定总步数 $$T$$，只优化 $$b(t)$$，问题就是

$$
\min_{b(t)}\int_0^T \frac{K(T-t)}{b(t)}\,dt,\qquad \text{s.t.}\ \int_0^T b(t)\,dt=D.
$$

一个 Cauchy–Schwarz 就够了：

$$
\left(\int_0^T \frac{K(T-t)}{b(t)}\,dt\right)\left(\int_0^T b(t)\,dt\right)\ge\left(\int_0^T \sqrt{K(T-t)}\,dt\right)^2,
$$

等号当且仅当

$$
b^*(t)\propto \sqrt{K(T-t)}.
$$

FSL 模型里核函数 $$K(\tau)\asymp (\tau+1)^{1/\beta-2}$$，代进去得到

$$
b^*(t)\asymp c\,(T-t+1)^{\frac{1}{2\beta}-1}.
$$

因为指数 $$\frac{1}{2\beta}-1<0$$，$$t$$ 越靠近 $$T$$，$$b^*(t)$$ 越大。也就是说，连续最优 batch 是单调往上加速的，根本不是线性 ramp。

加上硬件上下限 $$B_{\min}\le b(t)\le B_{\max}$$，KKT 给出的就是一条被截断的幂律：

$$
b^*(t)=\operatorname{clip}\!\left(c\,(T-t+1)^{\frac{1}{2\beta}-1},\,B_{\min},\,B_{\max}\right).
$$

真实训练里那种「先稳着一个 batch、后面跳到更大 batch」的样子，就是这条连续曲线被离散化的结果。系统只允许少数几个 batch size，平滑增长就变成几次跳变；如果只允许 2 的幂，自然就成了翻倍。

### 3.3 再把 T 也优化掉

把固定 $$T$$ 的最优 $$b(t)$$ 回代，噪声项的量级变成 $$(T+1)^{1/\beta}/D$$，于是

$$
\mathcal E(T)\asymp A\,T^{-s}+C\,\frac{T^{1/\beta}}{D}.
$$

一阶条件给出

$$
T^*\asymp D^{\frac{\beta}{1+s\beta}},\qquad B_{\max}\asymp D^{\frac{1/2+s\beta}{1+s\beta}}.
$$

到这里会分出两个 regime。

easy task（$$s>1-\tfrac1\beta$$）下，最优解是一条缓慢增长的 power-law batch schedule，全程随训练往上抬，最终 loss rate

$$
\mathcal E_D^*\asymp D^{-\frac{s\beta}{1+s\beta}}.
$$

hard task（$$s\le 1-\tfrac1\beta$$）下，无约束解想要非常大的 $$T$$，也就是想用非常小的平均 batch 去换更多 step，但硬件不让 batch 无限小，至少有 $$b(t)\ge B_{\min}$$。于是最优解被压成两段：

$$
b^*(t)=
\begin{cases}
B_{\min}, & 0\le t<T_1^*,\\[4pt]
B_{\max}(T^*-t+1)^{\frac{1}{2\beta}-1}, & T_1^*\le t\le T^*,
\end{cases}
$$

而且增长段的占比随 $$D$$ 变小，

$$
\frac{T^*-T_1^*}{T^*}\asymp D^{-\frac{1-1/\beta-s}{2-1/\beta}}.
$$

直觉上，hard task 早期最缺的是 optimizer step，而不是低噪声梯度，所以该长时间用小 batch 多攒 step，后期再用大 batch 降噪。FSL 把这种形状叫 stable-growth，可以看成 batch 版的 WSD。LLM 预训练更接近这一档。

这下「中途突然 Double GBS」就讲通了：真实最优是单调快速增长，但工程上只能用少数几个 batch size，画到 loss 曲线上就是一两次竖直跳变。一次翻倍不是数学最优的完整形态，而是 clipped power-law / late-switch schedule 的工程近似。

### 3.4 两段式的显式解

如果干脆限制成两段，

$$
b(t)=
\begin{cases}
B_1,&0\le t<t_s,\\
B_2,&t_s\le t\le T,
\end{cases}
\qquad B_2>B_1,
$$

切换点是可以算的。设切换前消耗 $$D_1$$ 个 token，则 $$t_s=D_1/B_1$$，$$T=\tfrac{D_1}{B_1}+\tfrac{D-D_1}{B_2}$$，代回 FSL objective 得到一个关于 $$D_1$$ 的一维问题。内点最优的一阶条件是

$$
A\,s\,S^{-s-1}=C\left[\frac{K(S)}{B_1}+\frac{K(R)}{B_2}\right],\qquad S=\frac{D_1}{B_1}+\frac{D-D_1}{B_2},\quad R=\frac{D-D_1}{B_2}.
$$

左边是继续用小 batch 多换 optimizer step 带来的 signal 收益，右边是噪声累积的代价，两边平衡时就该切。

多段翻倍也一样。允许 $$B_j=B_{\min}r^j$$（比如 $$r=2$$），先解连续解 $$b^*(t)=c(T-t+1)^p$$，$$p=\tfrac{1}{2\beta}-1<0$$，再让每段边界满足 $$B_j=c(T-t_j+1)^p$$，于是

$$
t_j=T+1-\left(\frac{B_j}{c}\right)^{1/p}.
$$

把这些 $$t_j$$ 换算到 token 轴 $$z_j=\int_0^{t_j} b^*(u)\,du$$，就知道大概消耗多少 token 时该翻一次 batch。再把 $$z_j$$ 对齐到 checkpoint、数据阶段或者集群重配的边界，就是工程上能直接用的 schedule。

### 3.5 一个更经验、但好落地的做法

变分解漂亮，但要先知道 $$s,\beta,K$$。更省事的替代是直接 track critical batch size。OLMo 那篇 CBS 的工作就是直接估 CBS，发现它早期快速上升、之后趋于平台，于是建议从小 batch 起步，等 CBS 够大了再翻 batch。他们用这个方法训 OLMo 1B，省了约 43% 的 gradient step，最终 loss 没有变差。它不像 FSL 那样给出闭式的 $$b^*(t)$$，但更容易落地：从 checkpoint 拉 branched run，测一下当前最大安全 batch，再决定翻不翻。

最后提醒一句，所有「最优」都得先说清目标。你可以优化固定 token 预算下的最终 validation loss，也可以优化到达目标 loss 的 wall-clock，还可以把通信开销、GPU 利用率、checkpoint 成本一起塞进去，目标不同，最优 $$b(t)$$ 就不同。而且 FSL 的主分析建立在 vanilla SGD 加常数 learning rate 上，现代 LLM 多用 AdamW，joint 的 learning-rate / batch-size schedule 还需要更细的处理。

## 4. 在 noisy quadratic 上验证一下

公式推完，我想亲眼看看这条变分最优 schedule 到底比默认常数 batch 好多少。

最干净的验证场是 noisy quadratic model（NQM），大 batch / critical batch size 这套理论一直拿它当代理模型，它也正是第 1.2 节那个一维例子的多维版。它的好处是期望 loss 有精确递推，不用 Monte Carlo：

$$
v_{i,k+1}=(1-\eta h_i)^2 v_{i,k}+\frac{\eta^2\sigma_i^2}{B_k},\qquad L_k=\tfrac12\sum_i h_i v_{i,k}.
$$

把它展开到第 $$T$$ 步，最终 loss 恰好是

$$
L_T=\underbrace{S(T)}_{\text{信号项, 只依赖步数}}+\sum_k\frac{\kappa(T-1-k)}{B_k},\qquad \kappa(j)=\tfrac12\eta^2\sum_i h_i\sigma_i^2(1-\eta h_i)^{2j},
$$

跟第 3 节的 $$A\,T^{-s}+C\int K(T-t)/b(t)\,dt$$ 同构。所以固定预算 $$D=\sum_k B_k$$，对它用一次 Cauchy–Schwarz，直接得到 $$B_k^*\propto\sqrt{\kappa(T-1-k)}$$。博客那条 clipped power-law，在 NQM 里就是精确解，不是近似。

我取了一个幂律谱，量出来 $$s\approx0.49$$、$$\beta\approx1.96$$，落在 hard regime（$$s\le 1-1/\beta$$），正好对应 LLM 这一档。固定同样的 token 预算、同样的常数 learning rate，只换 batch schedule，比较下来是这样：

{% include figure.liquid
  path='assets/img/post-06-16/batch_schedule_experiment.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='同一 token 预算、同一常数 lr 下的对比。左：loss 的 log-log 总览；中：横轴换成线性 token 数，各 schedule 的差异与末尾暴跌一目了然；右：对应的 batch schedule，最优与解析式 $(T-t+1)^{1/2\beta-1}$、翻倍阶梯吻合。'
  zoomable=true
  alt='loss curves and batch schedules for the NQM experiment'
%}

前两张都是 loss 随 token 的下降：左边 log-log 总览，中间把横轴换成线性 token 数，schedule 之间的差异、还有末尾那一下暴跌就全铺开了。右边是对应的 batch schedule。几条观察：

- 最优 schedule（红）在大部分训练里把 batch 压在 $$B_{\min}$$ 上攒 optimizer step，loss 沿着小 batch 的噪声地板走；到末尾才幂律式把 batch 拉起来，loss 在那一下骤降到常数 batch（灰）一个数量级以下。这条「末尾暴跌到基线以下」的形状，和 WSD、最优 lr schedule 的图几乎一样。
- 右图里最优 schedule 的形状和解析式 $$(T-t+1)^{1/2\beta-1}$$（黑虚线）严丝合缝，Cauchy–Schwarz 的闭式解被直接验证。
- 多段翻倍（蓝色阶梯）几乎追平最优，两段式（绿）用一次切换也拿到了大部分收益。同预算下的最终 loss：常数 1.0×、两段式 3.8×、翻倍 9.9×、最优 10.2×。
- 更狠的是，最优的最终 loss 已经低于那个最优常数 batch 的噪声地板 $$L_\infty(B)$$。换句话说，那个常数 batch 再加多少 token 都到不了最优 schedule 的终点。

两个支撑事实也单独验证了：噪声地板确实 $$\propto 1/B$$（第 1.2 节，batch 翻倍地板减半），核函数 $$\kappa(j)$$ 确实是幂律（第 3 节）。

{% include figure.liquid
  path='assets/img/post-06-16/batch_schedule_diagnostics.png'
  class='img-fluid rounded z-depth-1'
  width='100%'
  caption='左：噪声 loss floor 精确 $\propto 1/B$。右：噪声核 $\kappa(j)$ 在中段是一条幂律，斜率给出 $\beta\approx1.96$。'
  zoomable=true
  alt='diagnostics: noise floor and kernel power law'
%}

代码在 `experiments/batch_schedule_nqm.py`，纯 NumPy，几秒钟跑完。改谱的指数 `S_SRC`、`C_NOISE`，就能在 easy / hard regime 之间切换，看到最优 schedule 从「全程缓增」变成「长平台 + 末尾翻倍」。

## 小结

后期翻 batch，本质是噪声项 $$\frac{\eta^2}{2B}\operatorname{tr}(HC)$$ 随 $$1/B$$ 缩，而 $$B_{\text{crit}}\sim \eta\operatorname{tr}(HC)/\|\mu\|^2$$ 随训练变大。learning rate 不动、batch 翻倍，效果接近一次 learning-rate decay，还顺带多吃机器。

它不是个例，而是 GPT-3、Llama 3、OLMo 都用过的 batch ramp / warmup，只是很少被画到主图上。

最优 schedule 能用泛函/变分解出来，形态是 clipped power-law；在 LLM 这种 hard regime 加上离散硬件约束下，它退化成「长时间小 batch + 后期一两次翻倍」。Apertus 那条 Double GBS，就是这个形态最省事的工程近似。

## 参考资料

- Apertus 技术报告（70B loss 曲线与 Double GBS 的出处）
- McCandlish, Kaplan, Amodei et al.：[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)（gradient noise scale）
- Smith, Kindermans, Le et al.：[Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489)
- Brown et al.：[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)（GPT-3 的 batch ramp）
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- OLMo / OLMo 2 技术报告，以及 OLMo 关于 critical batch size 的研究
- Wang, Li, Zhou, Wu et al.：Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws（FSL 的 batch size 版本）

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
