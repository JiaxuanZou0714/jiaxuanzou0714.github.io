---
layout: default
permalink: /blog/
title: Blog
description: Notes on mechanistic interpretability, deep learning theory, optimization, and scaling laws by Jiaxuan Zou.
keywords: mechanistic interpretability, deep learning theory, optimization, scaling laws, muP, tensor programs
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 50
  sort_field: date
  sort_reverse: true
  trail:
    before: 1
    after: 3
---

<div class="post blog-index">
  <header class="post-header">
    <h1 class="post-title">Blog</h1>
    <p class="post-description">
      Notes on machine learning, optimization, and scaling. Most posts are in Chinese; English versions are on the
      <a href="{{ '/en/blog/' | relative_url }}">English page</a>.
    </p>
    {% if site.display_tags and site.display_tags.size > 0 %}
      <p class="tag-category-inline">
        Topics:
        {% for tag in site.display_tags %}
          <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">#{{ tag }}</a>{% unless forloop.last %}, {% endunless %}
        {% endfor %}
      </p>
    {% endif %}
  </header>

  {% assign featured_posts = site.posts | where: "featured", "true" %}
  {% if featured_posts.size > 0 %}
    <section class="blog-section">
      <h2>Selected Notes</h2>
      <ul class="selected-post-list">
        {% for post in featured_posts %}
          <li>
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
            <span class="post-meta">({{ post.date | date: '%b %Y' }})</span>
          </li>
        {% endfor %}
      </ul>
    </section>
  {% endif %}

  <section class="blog-section">
    <h2>All Posts</h2>
    <ul class="post-list">
      {% if page.pagination.enabled %}
        {% assign postlist = paginator.posts %}
      {% else %}
        {% assign postlist = site.posts %}
      {% endif %}

      {% for post in postlist %}
        {% if post.external_source == blank %}
          {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
        {% else %}
          {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
        {% endif %}
        {% assign year = post.date | date: "%Y" %}

        <li>
          <div class="post-list-head">
            <span class="post-list-date">{{ post.date | date: '%Y-%m-%d' }}</span>
            <span class="post-list-title">
              {% if post.redirect == blank %}
                <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
              {% elsif post.redirect contains '://' %}
                <a class="post-title" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
              {% else %}
                <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
              {% endif %}
            </span>
          </div>
          {% if post.description and post.description != '' %}
            <p class="post-list-desc">{{ post.description }}</p>
          {% endif %}
          <p class="post-meta">
            {{ read_time }} min read
            {% if post.external_source %} &middot; {{ post.external_source }}{% endif %}
            {% if site.enable_goatcounter_analytics and post.redirect == blank %}
              <span class="post-views" {% include goatcounter-paths.liquid document=post %}>
                &middot; <span class="goatcounter-count">--</span> views
              </span>
            {% endif %}
            {% if post.tags.size > 0 %}
              &middot;
              {% for tag in post.tags %}
                <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">#{{ tag }}</a>{% unless forloop.last %}, {% endunless %}
              {% endfor %}
            {% endif %}
          </p>
        </li>
      {% endfor %}
    </ul>
  </section>

  {% if page.pagination.enabled %}
    {% include pagination.liquid %}
  {% endif %}
</div>
