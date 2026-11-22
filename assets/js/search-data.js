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
          description: "Research projects, benchmarks, and open-source work.",
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
        },{id: "en_posts-how-to-align-data-scaling-curves-under-different-initialization-scales",
          title: 'How to align data scaling curves under different initialization scales',
          description: "We study the relationship between the empirical slope of data scaling and the initialization std, and propose a simple method to align data scaling curves under different initialization scales.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/data-scaling-and-std/";
            },},{id: "en_posts-from-gated-deltanet-to-kaczmarz",
          title: 'From Gated DeltaNet to Kaczmarz',
          description: "This article starts from the online learning formulation of Gated DeltaNet, introduces the Kaczmarz algorithm as an alternative to SGD, and analyzes its geometric meaning and its connection to Longhorn.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/kaczmarz/";
            },},{id: "en_posts-tensor-programs-part-1-from-the-spectral-condition-of-feature-learning-to-μp",
          title: 'Tensor Programs (Part 1): From the Spectral Condition of Feature Learning to μP...',
          description: "This article introduces the introductory paper of Greg Yang&#39;s Tensor Programs series—A Spectral Condition for Feature Learning—which derives the scaling conditions required for feature learning from the perspective of the spectral norm, and then re-derives the Maximal Update Parametrization (μP) from them.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/spectral-condition-feature-learning/";
            },},{id: "en_posts-tensor-programs-part-2-from-tensor-programs-to-μp",
          title: 'Tensor Programs (Part 2): From Tensor Programs to μP',
          description: "This article systematically reviews the core theoretical derivations of the maximal update parameterization (μP) obtained from Tensor Programs. The most fundamental and central insight of Tensor Programs theory in deriving neural network scaling laws is that one must strictly distinguish and apply the law of large numbers (LLN) and the central limit theorem (CLT) according to the different generation mechanisms of weight tensors.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/tensor-programs-mup-intuition/";
            },},{id: "en_posts-on-the-loose-use-and-methodological-boundaries-of-the-quot-manifold-quot-concept-in-current-ai",
          title: 'On the Loose Use and Methodological Boundaries of the &amp;quot;Manifold&amp;quot; Concept in Current...',
          description: "This article discusses the loose use of the &quot;manifold&quot; concept in AI theory research and draws the boundaries between engineering terminology, geometric intuition, and rigorous mathematical argument.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/manifold/";
            },},{id: "en_posts-on-the-hypersphere-from-spherical-dynamics-to-μp",
          title: 'On the Hypersphere: From Spherical Dynamics to μP',
          description: "This article departs from the probabilistic framework of Tensor Programs and, from the perspective of continuous-time spherical dynamics, rigorously derives how to achieve alignment of networks of different sizes by aligning the dynamics on the hypersphere in network architectures that apply RMSNorm.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/spherical-dynamics-mup/";
            },},{id: "en_posts-on-the-hypersphere-μp-scaling-of-optimizers-with-the-hyperball-mechanism",
          title: 'On the Hypersphere: μP Scaling of Optimizers with the Hyperball Mechanism',
          description: "Starting from the first principles of continuous-time spherical dynamics, this article explores how the intrinsic dependence of weight norms disrupts hyperparameter alignment, and rigorously derives the underlying mathematical mechanisms by which various Hyperball optimizer variants achieve feature-space alignment.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/spherical-hyperball/";
            },},{id: "en_posts-frobenius-norm-estimation-of-the-update-matrices-for-adam-and-muon-optimizers",
          title: 'Frobenius Norm Estimation of the Update Matrices for Adam and Muon Optimizers',
          description: "This article rigorously derives and estimates the Frobenius norm of the update matrices of the Adam and Muon optimizers in a single iteration step, and explores the influence of matrix shape on the order of magnitude of the norm.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/optimizer-update-matrix-norm/";
            },},{id: "en_posts-bias-and-fluctuations-of-the-spectral-norm-of-random-gaussian-matrices-at-finite-width",
          title: 'Bias and Fluctuations of the Spectral Norm of Random Gaussian Matrices at Finite...',
          description: "Starting from Wishart random matrix theory, this article derives the expansion of the spectral norm of a Gaussian matrix with element variance 1/n at finite width, showing that it not only converges to the macroscopic limit 2, but also carries a bias of order $n^{-2/3}$ and Tracy-Widom type random fluctuations.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/finite-width-spectral-norm/";
            },},{id: "en_posts-the-trade-off-between-parallelism-and-expressiveness-theoretical-boundaries-from-ac-0-tc-0-to-linear-attention",
          title: 'The Trade-off Between Parallelism and Expressiveness: Theoretical Boundaries from $AC^0$/$TC^0$ to Linear Attention...',
          description: "From the perspective of circuit complexity, this post gives a unified explanation of why constant-depth Transformers cannot exactly perform integer multiplication of arbitrary length, and why stronger linear attention variants often fail to maintain full token parallelism.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/parallel-expressiveness-tradeoff-linear-attention/";
            },},{id: "en_posts-in-the-llm-context-how-does-noise-in-the-gradient-affect-training-dynamics",
          title: 'In the LLM context, how does noise in the gradient affect training dynamics?...',
          description: "A discussion of gradient noise in the late stages of LLM pretraining, and why blockwise normalized updates look more like constraining the update magnitude than correcting the gradient direction.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/llm-gradient-noise-training-dynamics/";
            },},{id: "en_posts-μp-map",
          title: 'μP Map',
          description: "A reading guide and thread overview for the μP-related blog posts.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/mup-map/";
            },},{id: "en_posts-re-listening-to-yang-zhilin-bet-on-scaling-first-principles-and-long-termism",
          title: 'Re-listening to Yang Zhilin: Bet on Scaling, First Principles, and Long-termism',
          description: "At lunch, I re-listened to the conversation between Yang Zhilin and Zhang Xiaojun from January 2024, jotting down some thoughts on long context, scaling laws, agents, AGI organizational forms, and long-termism.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/bet-on-scaling-first-principles/";
            },},{id: "en_posts-don-39-t-just-learn-from-the-19th-century-west-civilizational-centrism-cosmopolitanism-and-the-public-responsibility-of-young-leaders",
          title: 'Don&amp;#39;t Just Learn from the 19th-Century West: Civilizational Centrism, Cosmopolitanism, and the Public...',
          description: "After reading Liu Qing&#39;s article on tianxia and new cosmopolitanism, here are some of my thoughts on civilizational centrism, 19th-century-style imaginings of great power, and the public responsibility of young technologists.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/civilizational-centralism-and-young-technical-leaders/";
            },},{id: "en_posts-why-does-the-batch-size-need-to-be-doubled-midway-through-llm-pretraining",
          title: 'Why does the batch size need to be doubled midway through LLM pretraining?...',
          description: "Starting from the Double GBS phenomenon of Apertus 70B, we use the gradient noise scale, critical batch size, and calculus of variations to derive the optimal schedule for increasing the batch size midway through LLM pretraining, and validate it on the noisy quadratic model.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/why-double-batch-size-llm-pretraining/";
            },},{id: "en_posts-dasf-a-closed-loop-schedule-free-method-for-batch-size",
          title: 'DASF: A Closed-Loop Schedule-Free Method for Batch Size',
          description: "This post proposes DASF (Drift-Aware Schedule-Free): based on the duality of Schedule-Free (iterate averaging ↔ learning rate schedule, gradient averaging ↔ batch size schedule), it uses on-the-fly measured gradient statistics to set the effective batch size online, with no schedule and no tuning, eliminating the cost of training proxy models and fitting scaling laws for batch calibration. On real transformers it matches or exceeds tuned baselines, and provides a falsifiable negative result: under compute-optimal conditions, the optimal effective batch is approximately constant, not growing as √t.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/schedule-free-effective-batch-size/";
            },},{id: "en_posts-two-years-of-undergrad-my-deepest-takeaway-is-cognitive-compounding",
          title: 'Two years of undergrad: my deepest takeaway is cognitive compounding',
          description: "Many people tend to make long-term linear plans based on limited present cognition, easily falling into anxiety when short-term efforts yield no immediate return. Drawing from my two years of undergraduate experience, this essay reflects on cognitive compounding: consistent public writing, immersing in high-density environments, and frequently iterating cognition—and how this yields nonlinear growth over the long run.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/cognitive-compound-interest/";
            },},{id: "en_posts-hyperball-effective-lr-and-the-shape-of-peak-then-decay",
          title: 'Hyperball, effective lr, and the shape of peak-then-decay',
          description: "Addressing two core questions in pretraining learning rate scheduling: the effective learning rate directly governing optimization is dynamically determined by weight norms, and Hyperball eliminates this implicit schedule by constraining weight norms; peak-then-decay corresponds to the optimal solution of the bias–variance trade-off, where shapes satisfying this balance form a set not limited to a specific analytic form.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/hyperball-implicit-lr-schedule/";
            },},{id: "en_posts-pretraining-and-scaling-a-universal-methodology",
          title: 'Pretraining and scaling: a universal methodology',
          description: "Pretraining and scaling as a methodology across domains: learn world knowledge from data, test improvements across scales, and use efficiency, stability, and prediction to guide training.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/pretrain-scaling-methodology-scientific-perspective/";
            },},{id: "en_posts-do-contemporary-intellectuals-still-have-a-responsibility-to-participate-in-public-governance",
          title: 'Do Contemporary Intellectuals Still Have a Responsibility to Participate in Public Governance?',
          description: "From ancient Greek poleis and the Chinese literati tradition to the postmodern dismantling of grand narratives, questioning where intellectuals&#39; mandate to intervene in public governance truly comes from. This essay records how my initial assumptions collapsed, proposing an engaged posture on the ruins of postmodernism that acknowledges contingency yet still chooses commitment.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/intellectuals-public-governance-responsibility/";
            },},{id: "en_posts-how-to-build-a-scientific-scaling-ladder",
          title: 'How to Build a Scientific Scaling Ladder',
          description: "A practical guide to designing and building LLM pretraining scaling ladders: parameter and compute accounting, dense and MoE experimental design, hyperparameter search and schedules, data mixtures and multi-epoch repetition, loss scaling law fitting, downstream prediction, and extrapolation validation.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/how-to-build-scientific-scaling-ladder/";
            },},{id: "en_posts-pretraining-in-the-agentic-era-cannot-rely-on-nll-alone-notes-on-mai-beam-ml4-and-nemotron",
          title: 'Pretraining in the Agentic Era Cannot Rely on NLL Alone: Notes on MAI,...',
          description: "Reading the recent technical reports on MAI, Reflection Beam, Mistral Large 4, and Nemotron 3.5: legacy GDM training habits, and why PT and CPT should go beyond LR annealing to inject agentic interaction patterns and raise the RL starting point.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/agentic-native-pretraining-beyond-nll/";
            },},{id: "en_posts-on-the-day-the-riemann-hypothesis-was-proved-i-thought-of-classical-music",
          title: 'On the Day the Riemann Hypothesis Was Proved, I Thought of Classical Music...',
          description: "Written on the day the Lean and Mathlib communities verified Levent Alpöge and Claude&#39;s proof of the Riemann Hypothesis. A few reflections on mathematics, existential crisis, and music.",
          section: "En_posts",handler: () => {
              window.location.href = "/en/blog/2026/riemann-hypothesis-and-classical-music/";
            },},{id: "news-i-joined-the-scalingopt-project-as-a-co-maintainer-working-on-optimizer-design-for-large-language-model-training",
          title: 'I joined the ScalingOpt project as a co-maintainer, working on optimizer design for...',
          description: "",
          section: "News",},{id: "news-our-nora-optimizer-has-been-included-in-the-scalingopt-optimizer-library",
          title: 'Our Nora optimizer has been included in the ScalingOpt optimizer library.',
          description: "",
          section: "News",},{id: "news-i-joined-the-bytedance-seed-pre-training-team-as-a-research-intern",
          title: 'I joined the ByteDance Seed Pre-training team as a research intern.',
          description: "",
          section: "News",},{id: "news-i-wrapped-up-my-research-internship-with-the-bytedance-seed-pre-training-team-thank-you-to-everyone-at-seed-for-this-wonderful-journey",
          title: 'I wrapped up my research internship with the ByteDance Seed Pre-training team. Thank...',
          description: "",
          section: "News",},{id: "projects-scalingopt",
          title: 'ScalingOpt',
          description: "Exploring the core role of optimizers in the era of large language models, focusing on the synergy between optimizer design, model architecture, and training configurations under the Scaling Law paradigm. A systematic discussion platform and scientific benchmark for LLM training.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/scalingopt/";
            },},];
