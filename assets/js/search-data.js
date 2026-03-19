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
        },{id: "nav-μp-map",
          title: "μP Map",
          description: "μP 相关博客的阅读导航与脉络梳理。",
          section: "Navigation",
          handler: () => {
            window.location.href = "/mup-map/";
          },
        },{id: "nav-publications",
          title: "Publications",
          description: "Publications by Jiaxuan Zou, including research papers and preprints on mechanistic interpretability, deep learning theory, optimization, and scaling laws.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-q-amp-a",
          title: "Q&amp;A",
          description: "Some questions and answers about my research, background, and interests.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/autobiography/";
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
      },{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/assets/pdf/example_pdf.pdf", "_blank");
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%33%31%34%30%31%34%33%34%39%37@%71%71.%63%6F%6D", "_blank");
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
