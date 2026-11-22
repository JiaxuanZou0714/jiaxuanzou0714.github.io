---
layout: default
permalink: /en/blog/
title: Blog (English)
description: English translations of Jiaxuan Zou's notes on mechanistic interpretability, deep learning theory, optimization, and scaling laws.
keywords: mechanistic interpretability, deep learning theory, optimization, scaling laws, muP, tensor programs
nav: false
lang: en
---

<div class="post blog-index">
  <header class="post-header">
    <h1 class="post-title">Blog (English)</h1>
    <p class="post-description">
      English translations of posts originally written in Chinese. Originals are on the
      <a href="{{ '/blog/' | relative_url }}">Chinese page</a>.
    </p>
  </header>

  <ul class="post-list">
    {% assign en_posts = site.en_posts | sort: 'date' | reverse %}
    {% for post in en_posts %}
      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
      <li>
        <div class="post-list-head">
          <span class="post-list-date">{{ post.date | date: '%Y-%m-%d' }}</span>
          <span class="post-list-title">
            <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </span>
        </div>
        {% if post.description and post.description != '' %}
          <p class="post-list-desc">{{ post.description }}</p>
        {% endif %}
        <p class="post-meta">
          {{ read_time }} min read
          {% if post.tags.size > 0 %}
            &middot;
            {% for tag in post.tags %}
              #{{ tag }}{% unless forloop.last %}, {% endunless %}
            {% endfor %}
          {% endif %}
        </p>
      </li>
    {% endfor %}
  </ul>
</div>
