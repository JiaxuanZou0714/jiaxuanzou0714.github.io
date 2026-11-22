---
source_sha: 15991109895eda65
layout: post
title: "On the Day the Riemann Hypothesis Was Proved, I Thought of Classical Music"
date: 2026-11-21 12:00:00
description: "Written on the day the Lean and Mathlib communities verified Levent Alpöge and Claude's proof of the Riemann Hypothesis. A few reflections on mathematics, existential crisis, and music."
tags: [ai, mathematics, music, classical-music, philosophy]
categories: [thoughts]
featured: false
giscus_comments: true
lang: en
permalink: /en/blog/2026/riemann-hypothesis-and-classical-music/
ref: riemann-hypothesis-and-classical-music
related_posts: false
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
  <figcaption class="essay-caption">14th International Chopin Piano Competition (2000) · Yundi Li, Piano Concerto No. 1, II. Romance (<a href="https://www.bilibili.com/video/BV1Vt411h7nn" target="_blank" rel="noopener">Full concert on Bilibili</a>)</figcaption>
</figure>

<div class="essay-body" markdown="1">

Over the past two days, Anthropic mathematician [Levent Alpöge](https://alpo.ge/) released a [proof of the Riemann Hypothesis](https://alpo.ge/remark.pdf) developed with Claude: not a conventional research paper, but a 17-million-line Lean formalization library. Today, the Lean and Mathlib communities (endorsed by the Lean FRO and the Mathlib Initiative) issued a [joint statement](https://leanprover-community.github.io/blog/posts/riemann-hypothesis/) confirming that, after checks by Lean's official kernel and several independent proof checkers, both the theorem statement and the proof itself are valid. After more than 160 years, the conjecture really has been proved.

For a long time, the Riemann Hypothesis stood as one of the central symbols of human mathematics. People tend to place mathematics and philosophy on equal footing, one in the natural sciences and the other in the humanities and pure thought, standing together near the summit of human intellect. Mathematicians have also long held the belief that good mathematics ought to be concise and elegant. Yet looking at 17 million lines of code that scarcely any human could read from start to finish, I felt that assumption waver for the first time: mathematics itself may not care much about conciseness or elegance. Perhaps human understanding is simply limited, so we happened to move along the paths clean and intuitive enough for us to follow, and eventually mistook that small slice for the whole. AI has revealed another way for mathematics to unfold, and it is steadily making mathematics less of a uniquely human capability. That unsettles more than mathematicians alone; for humanity as a whole, it feels like an unavoidable existential crisis.

Feeling a quiet sense of emptiness, I found myself thinking of music, and classical music in particular. If there is still ground that feels unmistakably human, it is the first thing that comes to mind. Like mathematics, music carries both intellect and romance, yet it lives outside language and symbolic order; a score can record pitches and rhythms, while the moments that actually make you hold your breath never reside in rules on a page. And in music, there is no real notion of "solving." Classical music theory does have the term [resolution](https://en.wikipedia.org/wiki/Resolution_(music)), where harmonic tension settles back into consonance, yet from the Baroque through the Classical and Romantic periods, composers never wrote these works to close the book on an open problem.

<figure class="essay-media">
  <video controls playsinline preload="metadata" poster="{{ '/assets/img/post-11-21/chloe-chaconne-poster.jpg' | relative_url }}">
    <source src="{{ '/assets/video/chloe-chua-vitali-chaconne.mp4' | relative_url }}" type="video/mp4">
  </video>
  <figcaption class="essay-caption">Vitali, Chaconne in G minor (climax excerpt) · Chloe Chua (<a href="https://www.bilibili.com/video/BV1sHqNY9EXC" target="_blank" rel="noopener">Full performance on Bilibili</a>)</figcaption>
</figure>

Stepping back, classical music, rock, jazz, and other genres all share the same root. AI will likely soon be able to recombine patterns at high speed and produce polished, even genuinely moving pieces in any of those styles, yet that was never why humans invented music. Music was never meant to solve anything; it is the natural outpouring of emotion, intellect, and romance that builds up inside us. As machines take over more and more of the things that need to be solved, perhaps we ought to step back a little from the habit of looking for an answer or an optimal path in everything: sit through a concerto, go hear a live concert, not to prove we are smarter than a machine, nor to convert every experience into an output, but simply to let feeling, thought, and delight stand on their own.

<figure class="essay-media">
  <div class="essay-photos">
    <img src="{{ '/assets/img/post-11-21/chloe-chua-concert.jpg' | relative_url }}" alt="Chloe Chua concert" loading="lazy">
    <img src="{{ '/assets/img/post-11-21/ray-chen-selfie.jpg' | relative_url }}" alt="Photo with Ray Chen after his concert" loading="lazy">
  </div>
  <figcaption class="essay-caption">From concerts by Chloe Chua and Ray Chen I attended earlier.</figcaption>
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

## References

1. Lean and Mathlib communities (endorsed by the Lean FRO and the Mathlib Initiative), *[Lean verification of the Riemann Hypothesis](https://leanprover-community.github.io/blog/posts/riemann-hypothesis/)* (2026-11-21).
2. Levent Alpöge, *[A Claimed Proof of the Riemann Hypothesis](https://alpo.ge/remark.pdf)* (2026-11-20), and [X post](https://x.com/__alpoge__/status/2079028340955197566).
3. Joseph Howlett, *[Anthropic Mathematician Claims to Prove Riemann Hypothesis, Most Famous Problem in All of Math](https://www.scientificamerican.com/article/anthropic-mathematician-claims-to-prove-riemann-hypothesis-most-famous-problem-in-all-of-math/)*, Scientific American (2026-11-20).
4. State of Proof, *[A Claimed Proof of the Riemann Hypothesis: Paper Watch](https://stateofproof.org/paper-watch/paper-2026-11-20-a-claimed-proof-of-the-riemann-hypothesis)* (2026-11-20).

## Citation

If you need to cite this article, please refer to:

```bibtex
@article{zou2026riemannclassicalmusic,
  title={黎曼猜想被证明的这一天，我想起了古典音乐},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/riemann-hypothesis-and-classical-music/}
}
```
