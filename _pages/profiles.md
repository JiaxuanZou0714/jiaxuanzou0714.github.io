---
layout: profiles
permalink: /autobiography/
title: Background
description: Notes on my research background and interests.
nav: false
nav_order: 7
giscus_comments: true

profiles:
  - align: right
    image: autobiography/teaching2.png
    image_circular: false
    more_info: >
---

<p class="text-muted">Language: English | <a href="{{ '/autobiography/zh/' | relative_url }}">中文</a></p>

## Why Mathematics

1. I have loved mathematics since childhood. To me, it represents a distilled form of human intelligence and one of the highest forms of abstraction through which we understand the world. For the same reason, I am also drawn to philosophy.

2. I care deeply about research **methodology**: how to do research and how to think about problems scientifically. At the undergraduate stage, my choice of major was mainly about systematic training in ways of thinking, methodology, and worldview. The ability to analyze, decompose, plan, and solve real problems requires sustained training, and mathematics provides an excellent environment for this. This training now shapes how I approach deep learning: when I encounter a phenomenon or problem, I first try to model it mathematically and then choose suitable tools, such as optimization, statistics, and dynamical systems, to analyze it. I believe bringing this scientific and systematic way of thinking into artificial intelligence is essential if AI is to become a rigorous science rather than remain only an empirical or engineering practice.

## Why Artificial Intelligence

1. Turing's question, "Can machines think?", raises a fundamental problem: can intelligence be reproduced through computation? Artificial intelligence is also a deeply interdisciplinary field, drawing from computer science, mathematics, statistics, cognitive science, and related areas. Since I am interested in all of these disciplines, AI gives me a natural platform for combining them in research.

2. AI is one of the most promising fields of our time. It is evolving quickly and has the potential to reshape many parts of the world. I also believe that intellectual work should contribute to society, so I hope to work in this field, help advance the underlying science and technology, and create positive social impact.

## Why Mathematics Instead of an AI Major

1. Most problems in AI, especially in machine learning and deep learning, are mathematical in nature. Although the field currently contains many empirical tricks, understanding its foundations and turning AI into a rigorous science requires analysis from mathematics and systematic scientific **methodology**. For this reason, I see building a strong mathematical foundation before studying AI in depth as the more natural path.

2. Some AI undergraduate programs in China are still developing their curricula and training structures, and they may not always keep pace with the field's rapid iteration. By contrast, mathematics programs are more mature and systematic. They provide stronger theoretical foundations and a wider set of mathematical tools. Many undergraduate AI courses can be learned independently, while advanced mathematics courses benefit more from structured instruction and close guidance.

3. As mentioned above, mathematics offers systematic and deep training in research thinking and **methodology**. Some AI programs are more oriented toward engineering techniques. Choosing mathematics helps me build a more solid scientific mindset, which I consider important for future systematic research in AI and beyond.

4. At present, parts of the AI field are highly commercialized and somewhat impatient. Some work overemphasizes short-term technical iteration and commercial application while paying too little attention to foundational scientific questions. This can lead students to chase performance metrics, such as SOTA scores and leaderboard rankings, or to assemble methods without addressing the underlying problem. I think the central question is not the metric itself, but what substantive problem has been solved. Research should be problem-driven and grounded in a scientific view of the field. In comparison, the academic atmosphere in mathematics is more focused on foundational problems and less driven by short-term returns, which helps cultivate patience and research taste.

> _Note: The above is only my personal assessment of program structures. It is not meant as a judgment of students in any major. Both mathematics and AI depend heavily on individual initiative, and both fields have outstanding people. I only mean that, for me, mathematics is the better-fitting path._

## How I Started Research

I formally began doing research in September 2024, when I had just entered university. At that time, my college was running a long-term program that allowed undergraduates to initiate and organize seminars, with support such as venues and publicity. I began to wonder: why not start a deep learning seminar myself? Teaching would help train my own understanding and communication, while the seminar could also become a platform for sustained exchange, connect people with similar interests, and contribute in some small way to the undergraduate research and deep learning community at Xi'an Jiaotong University.

That is how the deep learning seminar at Xi'an Jiaotong University began. It later attracted more than a thousand participants from across China and naturally led to many collaborations. This was close to what I had hoped for at the beginning, though I did not expect it to grow to that scale.

Because of this seminar, my first formal research opportunity came from faculty members at Xi'an Jiaotong University. They found me through the seminar and invited me to join their research group. Since then, I have become increasingly convinced of one thing: it is better to work on things that are long-term and right, not merely things that are easy.

## Questions I Currently Care About (as of April 2026)

1. **Analytical theory for nonlinear dynamics.** Most solvable results are still limited to linear networks or infinite-width limits. The training dynamics of real finite-width nonlinear networks remain largely a black box, and I want to understand what happens inside it.

2. **The origin and breaking points of scaling laws.** Why do power-law relationships emerge? Under what conditions do they fail? Some recent work suggests that scaling laws may undergo phase transitions at extremely large scales, which I find very interesting.

3. **The full phase diagram between the lazy regime and the rich regime.** We know that both the lazy regime and the rich regime exist, but what does the transition region between them look like? Is there a third regime? If so, what model behavior and training mechanism would it correspond to?

4. **A "standard model" of hyperparameters.** Can we build a unified framework that includes parametrization schemes such as μP and mean-field parametrization, and gives more systematic and interpretable principles for choosing hyperparameters? I hope future hyperparameter choices will not be only empirical tuning, but will instead rest on a theory that can be derived, compared, and transferred.

## My Research Taste

My current understanding of research can be summarized as "concrete-abstract-concrete."

I believe deep learning is first an empirical discipline, so research should not be detached from concrete phenomena. Whether we study scaling laws, the frequency principle, or empirical regularities in training dynamics, these phenomena should first be carefully observed, reproduced, and characterized. A good theoretical question is usually not invented from nowhere; it grows out of real phenomena.

But staying at the level of phenomena is not enough. I also believe that, at a deeper level, deep learning is ultimately a mathematical problem. We need to abstract empirical phenomena into analyzable objects, build self-consistent theoretical frameworks, and explain why these phenomena appear, when they appear, and under what conditions they fail.

Finally, theory must return to concrete practice. A valuable theory should not only be formally elegant. It should also explain the behavior of real models, and ideally guide training, architecture design, and hyperparameter selection. This is what I mean by "concrete-abstract-concrete": starting from phenomena, passing through theoretical abstraction, and returning to real problems.

## How I Learn a New Field

When I learn a new field, I usually begin with extensive reading to build an overall understanding of its core questions, representative methods, and historical development. Rather than chasing details from the beginning, I care more about what the field is trying to answer, which problems have already been solved, and where structural gaps remain.

At the same time, I have many conversations with AI. I use it to organize concepts, check my understanding, and find connections among different papers. For me, AI is more like a high-frequency feedback partner that helps expose places where my thinking is not yet clear.

I also value learning from more experienced researchers. Many judgments about research, problem taste, and technical intuition are not fully written in papers; they come from long-term observation of the field. Conversations with experienced researchers often help me distinguish more quickly between problems that merely look active and problems that are genuinely worth long-term investment.

## How I See the Relation Between Theory and Engineering

I still use "concrete-abstract-concrete" to think about the relation between theory and engineering.

Engineering provides real problems, empirical phenomena, and testable feedback. Without engineering practice, theory can easily become a formal game detached from reality. But with engineering experience alone, deep learning can remain a collection of tricks and trial-and-error tuning, making it difficult to build transferable, interpretable, and cumulative scientific knowledge.

Therefore, theory and engineering are not opposed. Good engineering problems can give rise to good theoretical questions, and good theory should in turn serve engineering practice: it should help us understand which empirical rules are stable and which are only accidentally effective; which designs can transfer to larger models and different tasks, and which fail when scale, data, or optimization conditions change.

## Books, Films, and Music That Have Influenced Me

Michel Foucault's _Discipline and Punish_ and Edward Yang's film _Yi Yi_.

For music, I like both classical music and rock. Classical music gives me a sense of structure, order, and patience, while rock reminds me to keep a desire for expression, criticality, and vitality. I am also learning the violin. It is difficult to get quick feedback from the violin, but precisely because of that, it trains long-termism and sensitivity to detail.

## What I Want to Do in the Next Few Years

In the next few years, I hope to continue working on things that are genuinely important. To me, this is not only about publishing papers or completing short-term milestones, but about continually searching for problems that may change how humans understand the world, what technology can do, and how society is structured.

In research, I hope to work on problems that connect empirical phenomena, mathematical mechanisms, and real model behavior. I want to start from concrete problems in deep learning, build clearer theoretical explanations, and let those explanations guide practice in return. In the longer term, I hope to help move AI from empirical engineering toward a more mature scientific system.

## To Be Continued

> Feel free to ask more questions in the comments; I will answer when I can.
