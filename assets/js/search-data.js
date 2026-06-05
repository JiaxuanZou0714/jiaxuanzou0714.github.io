// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "Blog",
          description: "Notes on mechanistic interpretability, deep learning theory, optimization, and scaling laws by Jiaxuan Zou.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "Publications",
          description: "Publications by Jiaxuan Zou, including research papers and preprints on mechanistic interpretability, deep learning theory, optimization, and scaling laws.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-projects",
          title: "Projects",
          description: "A collection of my research works, open-source tools, and other projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "Academic CV for Jiaxuan Zou.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-不要只学习-19-世纪的西方-文明中心论-世界主义与青年领袖的公共责任",
          title: "不要只学习 19 世纪的西方：文明中心论、世界主义与青年领袖的公共责任 <span class='ninja-badge ninja-category'>thoughts</span> <span class='ninja-badge ninja-tag'>#political-philosophy</span> <span class='ninja-badge ninja-tag'>#china</span> <span class='ninja-badge ninja-tag'>#technology</span> <span class='ninja-badge ninja-tag'>#leadership</span>",
        
        description: "读刘擎关于天下与新世界主义的文章后，我对文明中心论、19 世纪式强权想象，以及青年技术人公共责任的一些想法。",aliases: "thoughts, political-philosophy, china, technology, leadership",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/civilizational-centralism-and-young-technical-leaders/";
          
        },
      },{id: "post-重听杨植麟-bet-on-scaling-第一性原理和长期主义",
          title: "重听杨植麟：Bet on Scaling、第一性原理和长期主义 <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#agi</span> <span class='ninja-badge ninja-tag'>#llm</span> <span class='ninja-badge ninja-tag'>#scaling-law</span> <span class='ninja-badge ninja-tag'>#agent</span> <span class='ninja-badge ninja-tag'>#long-context</span>",
        
        description: "午饭时重听杨植麟和张小珺在 2024 年 1 月的对谈，记下一些关于 long context、scaling law、agent、AGI 组织形式和长期主义的想法。",aliases: "deep-learning, agi, llm, scaling-law, agent, long-context",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/bet-on-scaling-first-principles/";
          
        },
      },{id: "post-μp-map",
          title: "μP Map <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#deep-learning</span> <span class='ninja-badge ninja-tag'>#tensor-programs</span> <span class='ninja-badge ninja-tag'>#muP</span> <span class='ninja-badge ninja-tag'>#feature-learning</span>",
        
        description: "μP 相关博客的阅读导航与脉络梳理。",aliases: "deep-learning, deep-learning, tensor-programs, muP, feature-learning",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/mup-map/";
          
        },
      },{id: "post-在-llm-语境下-梯度里的噪声会如何影响-training-dynamics",
          title: "在 LLM 语境下，梯度里的噪声会如何影响 training dynamics？ <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#optimization</span> <span class='ninja-badge ninja-tag'>#deep-learning</span> <span class='ninja-badge ninja-tag'>#llm</span> <span class='ninja-badge ninja-tag'>#scaling-law</span>",
        
        description: "讨论 LLM 预训练后期的梯度噪声，以及块归一化更新为什么更像是在限制更新幅值，而不是在修正梯度方向。",aliases: "deep-learning, optimization, deep-learning, llm, scaling-law",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/llm-gradient-noise-training-dynamics/";
          
        },
      },{id: "post-并行性与表达能力的权衡-从-ac-0-tc-0-到-linear-attention-的理论边界",
          title: "并行性与表达能力的权衡：从 $AC^0$/$TC^0$ 到 Linear Attention 的理论边界 <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#llm</span> <span class='ninja-badge ninja-tag'>#reasoning</span> <span class='ninja-badge ninja-tag'>#transformer</span> <span class='ninja-badge ninja-tag'>#linear-attention</span> <span class='ninja-badge ninja-tag'>#complexity-theory</span>",
        
        description: "从电路复杂度的视角，统一解释为什么常数深度 Transformer 无法精确完成任意长度整数乘法，以及为什么更强的 linear attention 变体往往无法保持完全 token 并行性。",aliases: "deep-learning, llm, reasoning, transformer, linear-attention, complexity-theory",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/parallel-expressiveness-tradeoff-linear-attention/";
          
        },
      },{id: "post-有限宽度下随机高斯矩阵谱范数的偏置与涨落",
          title: "有限宽度下随机高斯矩阵谱范数的偏置与涨落 <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#random-matrix</span> <span class='ninja-badge ninja-tag'>#spectral-norm</span> <span class='ninja-badge ninja-tag'>#wishart</span> <span class='ninja-badge ninja-tag'>#tracy-widom</span> <span class='ninja-badge ninja-tag'>#finite-width</span>",
        
        description: "本文从 Wishart 随机矩阵理论出发，推导元素方差为 1/n 的高斯矩阵谱范数在有限宽度下的展开式，说明其不仅收敛到宏观极限 2，还带有 $n^{-2/3}$ 级别的偏置和 Tracy-Widom 型随机涨落。",aliases: "deep-learning, random-matrix, spectral-norm, wishart, tracy-widom, finite-width",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/finite-width-spectral-norm/";
          
        },
      },{id: "post-adam-与-muon-优化器更新矩阵的-frobenius-范数估计",
          title: "Adam 与 Muon 优化器更新矩阵的 Frobenius 范数估计 <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#optimizer</span> <span class='ninja-badge ninja-tag'>#adam</span> <span class='ninja-badge ninja-tag'>#muon</span> <span class='ninja-badge ninja-tag'>#frobenius-norm</span>",
        
        description: "本文严密推导并估计了 Adam 与 Muon 优化器在单步迭代中更新矩阵的 Frobenius 范数，并探讨了矩阵形状对范数量级的影响。",aliases: "deep-learning, optimizer, adam, muon, frobenius-norm",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/optimizer-update-matrix-norm/";
          
        },
      },{id: "post-球面之上-带有-hyperball-机制的优化器的-μp-缩放",
          title: "球面之上：带有 Hyperball 机制的优化器的 μP 缩放 <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#deep-learning</span> <span class='ninja-badge ninja-tag'>#spherical-dynamics</span> <span class='ninja-badge ninja-tag'>#muP</span> <span class='ninja-badge ninja-tag'>#optimizer</span>",
        
        description: "从连续时间球面动力学视角的第一性原理出发，探讨权重范数的内生依赖对超参数对齐的破坏，并严格推导各类 Hyperball 变体优化器实现特征空间对齐的底层数学机制。",aliases: "deep-learning, deep-learning, spherical-dynamics, muP, optimizer",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/spherical-hyperball/";
          
        },
      },{id: "post-球面之上-从球面动力学到-μp",
          title: "球面之上：从球面动力学到 μP <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#deep-learning</span> <span class='ninja-badge ninja-tag'>#spherical-dynamics</span> <span class='ninja-badge ninja-tag'>#muP</span> <span class='ninja-badge ninja-tag'>#rmsnorm</span>",
        
        description: "本文脱离 Tensor Programs 的概率论框架，从连续时间的球面动力学视角，严格推导在应用 RMSNorm 的网络架构中，如何通过对齐超球面上的动力学来实现大小网络的对齐。",aliases: "deep-learning, deep-learning, spherical-dynamics, muP, rmsnorm",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/spherical-dynamics-mup/";
          
        },
      },{id: "post-论当前-ai-界内-流形-概念使用的泛化与方法论边界",
          title: "论当前 AI 界内“流形”概念使用的泛化与方法论边界 <span class='ninja-badge ninja-category'>artificial-intelligence</span> <span class='ninja-badge ninja-tag'>#ai-theory</span> <span class='ninja-badge ninja-tag'>#mathematics</span> <span class='ninja-badge ninja-tag'>#manifold</span> <span class='ninja-badge ninja-tag'>#methodology</span>",
        
        description: "本文讨论 AI 理论研究中“流形”概念的泛化使用，并区分工程命名、几何直觉与严格数学论证之间的边界。",aliases: "artificial-intelligence, ai-theory, mathematics, manifold, methodology",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/manifold/";
          
        },
      },{id: "post-tensor-programs-二-从tensor-programs到-μp",
          title: "Tensor Programs (二)：从Tensor Programs到 μP <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#deep-learning</span> <span class='ninja-badge ninja-tag'>#tensor-programs</span> <span class='ninja-badge ninja-tag'>#muP</span> <span class='ninja-badge ninja-tag'>#feature-learning</span>",
        
        description: "本文对 Tensor Programs 导出的极大更新参数化（μP）的核心理论推导进行系统性梳理。Tensor Programs 理论在推导神经网络缩放法则时，其最基础且最核心的洞察在于：必须根据权重张量生成机制的不同，严格区分并应用大数定律（LLN）与中心极限定理（CLT）。",aliases: "deep-learning, deep-learning, tensor-programs, muP, feature-learning",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/tensor-programs-mup-intuition/";
          
        },
      },{id: "post-tensor-programs-一-从feature-learning-的谱条件到-μp",
          title: "Tensor Programs (一)：从Feature Learning 的谱条件到 μP <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#deep-learning</span> <span class='ninja-badge ninja-tag'>#tensor-programs</span> <span class='ninja-badge ninja-tag'>#muP</span> <span class='ninja-badge ninja-tag'>#feature-learning</span>",
        
        description: "本文介绍 Greg Yang 的 Tensor Programs 系列的入门论文——A Spectral Condition for Feature Learning，从谱范数的视角推导出 feature learning 所需的 scaling 条件，并由此重新推导 maximal update parametrization（μP）。",aliases: "deep-learning, deep-learning, tensor-programs, muP, feature-learning",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/spectral-condition-feature-learning/";
          
        },
      },{id: "post-从-gated-deltanet-到-kaczmarz",
          title: "从 Gated DeltaNet 到 Kaczmarz <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#deep-learning</span> <span class='ninja-badge ninja-tag'>#optimization</span> <span class='ninja-badge ninja-tag'>#linear-attention</span>",
        
        description: "本文从 Gated DeltaNet 的在线学习形式出发，并引入 Kaczmarz 算法作为 SGD 的替代方案，分析了其几何意义及与 Longhorn 的联系。",aliases: "deep-learning, deep-learning, optimization, linear-attention",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/kaczmarz/";
          
        },
      },{id: "post-如何对齐不同初始化大小下的-data-scaling-曲线",
          title: "如何对齐不同初始化大小下的 Data scaling 曲线 <span class='ninja-badge ninja-category'>deep-learning</span> <span class='ninja-badge ninja-tag'>#scaling-law</span>",
        
        description: "研究了 data scaling 的 empirical slope 关于初始化 std 的关系，并提出一种简单方法来对齐不同初始化大小下的 data scaling 曲线",aliases: "deep-learning, scaling-law",section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/data-scaling-and-std/";
          
        },
      },{id: "post-can-we-derive-scaling-law-from-first-principles",
          title: "Can We Derive Scaling Law From First Principles? <span class='ninja-badge ninja-category'>publications</span> <span class='ninja-badge ninja-tag'>#research</span> <span class='ninja-badge ninja-tag'>#scaling-law</span> <span class='ninja-badge ninja-tag'>#deep-learning</span> <span class='ninja-badge ninja-tag'>#pdf</span>",
        
        description: "New research available. Click to read the full PDF.",aliases: "publications, research, scaling-law, deep-learning, pdf",section: "Posts",
        handler: () => {
          
            window.location.href = "/assets/pdf/Can_We_Derive_Scaling_Law_From_First_Principles.pdf";
          
        },
      },{id: "news-i-joined-the-scalingopt-project-as-a-co-maintainer-working-on-optimizer-design-for-large-language-model-training",
          title: 'I joined the ScalingOpt project as a co-maintainer, working on optimizer design for...',
          description: "",
          section: "News",},{id: "news-our-nora-optimizer-has-been-included-in-the-scalingopt-optimizer-library",
          title: 'Our Nora optimizer has been included in the ScalingOpt optimizer library.',
          description: "",
          section: "News",},{id: "projects-scalingopt",
          title: 'ScalingOpt',
          description: "Exploring the core role of optimizers in the era of large language models, focusing on the synergy between optimizer design, model architecture, and training configurations under the Scaling Law paradigm. A systematic discussion platform and scientific benchmark for LLM training.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/scalingopt/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%33%31%34%30%31%34%33%34%39%37@%71%71.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/SmartPig_Joe", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/%E5%98%89%E8%BD%A9-%E9%82%B9-bb3439341", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=YE6iiwUAAAAJ", "_blank");
        },
      },{
        id: 'social-bilibili_social',
        title: 'Bilibili_social',
        section: 'Socials',
        handler: () => {
          window.open("https://space.bilibili.com/282545566?spm_id_from=333.1007.0.0", "_blank");
        },
      },{
        id: 'social-xiaohongshu_social',
        title: 'Xiaohongshu_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.xiaohongshu.com/user/profile/65fc3374000000000b00d833", "_blank");
        },
      },{
        id: 'social-wechat_social',
        title: 'Wechat_social',
        section: 'Socials',
        handler: () => {
          window.open("#", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
