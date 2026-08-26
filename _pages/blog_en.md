---
layout: default
permalink: /en/blog/
title: Blog (English)
description: English translations of Jiaxuan Zou's notes on mechanistic interpretability, deep learning theory, optimization, and scaling laws.
keywords: mechanistic interpretability, deep learning theory, optimization, scaling laws, muP, tensor programs
nav: false
lang: en
---

<div class="post">

  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    <h2>{{ site.blog_description }}</h2>
  </div>

  <p>
    These are English translations of posts originally written in Chinese. Each one links back to
    its source. The originals live on the <a href="{{ '/blog/' | relative_url }}">main blog page</a>.
  </p>

  <ul class="post-list">
    {% assign en_posts = site.en_posts | sort: 'date' | reverse %}
    {% for post in en_posts %}
      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
      {% assign year = post.date | date: "%Y" %}
      <li>
        <h3>
          <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        </h3>
        <p>{{ post.description }}</p>
        <p class="post-meta">
          {{ read_time }} min read &nbsp; &middot; &nbsp;
          {{ post.date | date: '%B %d, %Y' }}
        </p>
        <p class="post-tags">
          <i class="fa-solid fa-calendar fa-sm"></i> {{ year }}
          {% assign tags = post.tags | join: "" %}
          {% if tags != "" %}
            &nbsp; &middot; &nbsp;
            {% for tag in post.tags %}
              <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}
              {% unless forloop.last %}&nbsp;{% endunless %}
            {% endfor %}
          {% endif %}
        </p>
      </li>
    {% endfor %}
  </ul>

</div>
