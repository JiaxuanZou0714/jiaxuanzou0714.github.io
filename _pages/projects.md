---
layout: page
title: Projects
permalink: /projects/
description: Research projects, benchmarks, and open-source work.
nav: true
nav_order: 4
display_categories: [research]
horizontal: false
---

<div class="projects">
{% if site.enable_project_categories and page.display_categories %}
  {% for category in page.display_categories %}
    {% assign categorized_projects = site.projects | where: "category", category %}
    {% assign sorted_projects = categorized_projects | sort: "importance" %}
    {% if sorted_projects.size > 0 %}
      <ul class="project-list">
        {% for project in sorted_projects %}
          {% include projects.liquid %}
        {% endfor %}
      </ul>
    {% endif %}
  {% endfor %}
{% else %}
  {% assign sorted_projects = site.projects | sort: "importance" %}
  <ul class="project-list">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </ul>
{% endif %}
</div>
