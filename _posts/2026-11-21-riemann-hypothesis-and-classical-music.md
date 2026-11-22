---
layout: post
title: "黎曼猜想被证明的这一天，我想起了古典音乐"
date: 2026-11-21 12:00:00
description: "写在 Levent Alpöge 与 Claude 的黎曼猜想证明获得 Lean 与 Mathlib 社区验证的这一天。一点关于数学、存在主义危机与音乐的随感。"
tags: [ai, mathematics, music, classical-music, philosophy]
categories: [thoughts]
featured: false
giscus_comments: true
lang: zh-CN
_styles: |
  .essay-body p {
    margin-bottom: 1.4rem;
    line-height: 1.9;
  }
  .essay-media {
    max-width: 707px;
    margin: 2rem 0 2.2rem 0;
  }
  .essay-media-top {
    max-width: 707px;
    margin: 0.4rem 0 2.2rem 0;
  }
  #markdown-content .essay-media video,
  #markdown-content .essay-media-top video {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 10px;
    border: 1px solid var(--global-divider-color);
    background: #000;
  }
  .essay-photos {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  @media (max-width: 576px) {
    .essay-photos {
      grid-template-columns: 1fr;
    }
  }
  #markdown-content .essay-photos img {
    width: 100%;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    border-radius: 10px;
    border: 1px solid var(--global-divider-color);
    display: block;
  }
  .essay-caption {
    margin-top: 8px;
    font-size: 0.8rem;
    color: var(--global-text-color-light);
    text-align: center;
  }
  .essay-divider {
    border: none;
    border-top: 1px solid var(--global-divider-color);
    margin: 2.4rem 0 1.8rem 0;
  }
---

<figure class="essay-media-top">
  <video controls playsinline preload="metadata" poster="{{ '/assets/img/post-11-21/yundi-chopin-poster.jpg' | relative_url }}">
    <source src="{{ '/assets/video/yundi-li-chopin-2000.mp4' | relative_url }}" type="video/mp4">
  </video>
  <figcaption class="essay-caption">2000 年第十四届肖邦国际钢琴比赛 · 李云迪《第一钢琴协奏曲》第二乐章 Romance（<a href="https://www.bilibili.com/video/BV1Vt411h7nn" target="_blank" rel="noopener">Bilibili 全场</a>）</figcaption>
</figure>

<div class="essay-body" markdown="1">

这两天，Anthropic 的数学家 [Levent Alpöge](https://alpo.ge/) 放出了他和 Claude 合作完成的[黎曼猜想证明](https://alpo.ge/remark.pdf)——没有写成通常意义上的论文，而是一套 1700 万行代码的 Lean 形式化库。今天，Lean 与 Mathlib 社区（经 Lean FRO 与 Mathlib Initiative 背书）发布[联合声明](https://leanprover-community.github.io/blog/posts/riemann-hypothesis/)，经过官方内核和多个独立校验器复核，确认命题与证明本身均无误。一百六十多年过去，这道题确实被证明了。

在很长一段时间里，黎曼猜想都是人类数学最核心的象征之一。我们习惯把数学和哲学放在同等的位置上看待，一个在自然科学，一个在人文思辨，几乎并列在人类智识的最高处。数学家们也一直相信，好的数学理应是简洁而优美的。可面对这 1700 万行人类几乎无法通读的代码，我第一次有些动摇：数学本身或许并没有那么在意简洁和优美，只是人类的理解力有限，过去恰好只能沿着那些足够简洁、契合直觉的路径往前走，久而久之便把这一小块当成了全貌。AI 展示了数学发展的另一种可能，也让数学一点点不再属于人类所独有。这冲击的不只是数学家，对整个人类而言，大概都是一场无法回避的存在主义危机。

心里有些空落落的，不知怎么就想到了音乐，尤其是古典音乐。如果一定要找一块或许还属于人类的阵地，我脑海里最先浮现的便是它。音乐和数学一样，既讲智识，也讲浪漫，但它终究落在语言和符号秩序之外——乐谱只能记下音高与节拍，真正让人屏住呼吸的东西，从来不在纸面的规则里。而且，音乐里原本就没有“解决”这一说。古典乐理中固然有和声“解决”（[resolution](https://en.wikipedia.org/wiki/Resolution_(music))）这个术语，指不协和的张力落回协和，可从巴洛克一路走到古典主义、浪漫主义，作曲家们写下这些作品，从来不是为了给哪道难题交出一份答案。

<figure class="essay-media">
  <video controls playsinline preload="metadata" poster="{{ '/assets/img/post-11-21/chloe-chaconne-poster.jpg' | relative_url }}">
    <source src="{{ '/assets/video/chloe-chua-vitali-chaconne.mp4' | relative_url }}" type="video/mp4">
  </video>
  <figcaption class="essay-caption">维塔利《G 小调恰空》高潮片段 · 蔡珂宜（<a href="https://www.bilibili.com/video/BV1sHqNY9EXC" target="_blank" rel="noopener">Bilibili 完整版</a>）</figcaption>
</figure>

往大了看，古典、摇滚、爵士等其实都一样。AI 或许很快就能通过极快的排列组合，拼出各种风格、甚至相当好听的曲子，可那并不是人类发明音乐的初衷。音乐本来就不是为了解决什么问题而存在的，它是人心里积攒的情感、智识与浪漫，在某个时刻自然而然的迸发。当越来越多需要求解的事都可以交给机器，人或许也该从那种凡事都要寻找答案、追求最优解的惯性里退出来一点：听完一首协奏曲，去现场看一场演出，不为证明自己比机器更聪明，也不必把什么都折算成结果，只是让感受、思考和喜欢本身成立。

<figure class="essay-media">
  <div class="essay-photos">
    <img src="{{ '/assets/img/post-11-21/chloe-chua-concert.jpg' | relative_url }}" alt="蔡珂宜音乐会现场" loading="lazy">
    <img src="{{ '/assets/img/post-11-21/ray-chen-selfie.jpg' | relative_url }}" alt="与陈锐音乐会后合影" loading="lazy">
  </div>
  <figcaption class="essay-caption">之前去听蔡珂宜与陈锐的音乐会。</figcaption>
</figure>

</div>

<script>
(function () {
  var videos = document.querySelectorAll(".essay-media video, .essay-media-top video");
  videos.forEach(function (v) {
    v.addEventListener("play", function () {
      videos.forEach(function (other) {
        if (other !== v) other.pause();
      });
    });
  });
})();
</script>

<hr class="essay-divider">

## 参考

1. Lean and Mathlib communities (endorsed by the Lean FRO and the Mathlib Initiative), *[Lean verification of the Riemann Hypothesis](https://leanprover-community.github.io/blog/posts/riemann-hypothesis/)* (2026-11-21).
2. Levent Alpöge, *[A Claimed Proof of the Riemann Hypothesis](https://alpo.ge/remark.pdf)* (2026-11-20), 及 [X 发布帖](https://x.com/__alpoge__/status/2079028340955197566).
3. Joseph Howlett, *[Anthropic Mathematician Claims to Prove Riemann Hypothesis, Most Famous Problem in All of Math](https://www.scientificamerican.com/article/anthropic-mathematician-claims-to-prove-riemann-hypothesis-most-famous-problem-in-all-of-math/)*, Scientific American (2026-11-20).
4. State of Proof, *[A Claimed Proof of the Riemann Hypothesis — Paper Watch](https://stateofproof.org/paper-watch/paper-2026-11-20-a-claimed-proof-of-the-riemann-hypothesis)* (2026-11-20).

## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026riemannclassicalmusic,
  title={黎曼猜想被证明的这一天，我想起了古典音乐},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/riemann-hypothesis-and-classical-music/}
}
```
