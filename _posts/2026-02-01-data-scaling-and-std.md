---
layout: post
title: "如何对齐不同初始化大小下的 Data scaling 曲线"
date: 2026-02-01 10:00:00
description: "这是一个新的文章示例"
tags: [scaling-law]
categories: [deep-learning]
featured: true
giscus_comments: true

---

我们在 [上一篇 blog]({% post_url 2025-12-30-scaling-law %}) 中讨论了 scaling law 到底是如何产生的。我们在其预印本中增加了实验章节，其中有一系列图展示了不同$\alpha$下的 data scaling 曲线，比如其中一张图

{% include figure.liquid 
    path="assets/img/post-02-01/data-scaling-0p01.png" 
    class="img-fluid rounded z-depth-1 mx-auto d-block" 
    width="50%" 
    zoomable=true   
    alt="替代文本" 
%}

为了确保发生 feature learning，在这张图里，我们设定初始化的 std=0.01。但是如果我们把 std 增大一些，设置成 0.05，会发生什么呢？结果如下

{% include figure.liquid 
    path="assets/img/post-02-01/data-scaling-0p05.png" 
    class="img-fluid rounded z-depth-1 mx-auto d-block" 
    width="50%" 
    zoomable=true   
    alt="替代文本" 
%}

我们可以看见，std=0.05 时，data scaling 的曲线发生了偏移，那如果再增大一些呢？设置成 0.1，结果如下

{% include figure.liquid 
    path="assets/img/post-02-01/data-scaling-0p1.png" 
    class="img-fluid rounded z-depth-1 mx-auto d-block" 
    width="50%" 
    zoomable=true   
    alt="替代文本" 
%}
这个结果就很有趣了，随着初始化 std 增大，empirical 的直线逐渐偏离理论预测的直线。那如果我们把 empirical slope 关于 std 的曲线画出来会是什么样子？结果如下
{% include figure.liquid 
    path="assets/img/post-02-01/fix_lr.png" 
    class="img-fluid rounded z-depth-1 mx-auto d-block" 
    width="50%" 
    caption="初始化标准偏差与斜率的关系" 
    zoomable=true 
    alt="替代文本" 
%}
