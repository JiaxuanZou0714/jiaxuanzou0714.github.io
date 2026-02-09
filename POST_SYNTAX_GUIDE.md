# al-folio 网站内容编辑与语法完全指南

本文档全面梳理了 al-folio 主题中用于撰写文章（Posts）和页面（Pages）的各类语法规则。文档基于现有的代码库配置、模板文件及示例文本进行整理，涵盖了从基础 Markdown 到高级交互式图表的全部功能。

## 目录

1. [Front Matter 配置](#1-front-matter-配置)
2. [基础文本格式](#2-基础文本格式)
3. [多媒体嵌入](#3-多媒体嵌入)
4. [代码与技术文档](#4-代码与技术文档)
5. [数学公式](#5-数学公式)
6. [图表与数据可视化](#6-图表与数据可视化)
7. [引用与参考文献](#7-引用与参考文献)
8. [特殊布局与组件](#8-特殊布局与组件)

---

## 1. Front Matter 配置

每篇文章的头部 YAML 区域（Front Matter）控制了该文章启用的功能。

### 常用配置项

```yaml
---
layout: post              # 布局：post, distill, page 等
title: "文章标题"
date: 2024-01-01 12:00:00 # 发布时间
description: "文章简短描述"
tags: [tag1, tag2]        # 标签
categories: [cat1]        # 分类
featured: true            # 是否设为精选文章
giscus_comments: true     # 是否启用评论
toc:                      # 目录配置
  sidebar: left           # 侧边栏目录 (left/right)
redirect: /assets/pdf/example.pdf # 重定向到其他页面或文件
---
```

### 功能开关

以下功能需要在 Front Matter 中显式开启才能生效：

| 功能 | 配置键 | 说明 |
| :--- | :--- | :--- |
| 数学公式 | `math: true` | 启用 MathJax |
| Mermaid 图表 | `mermaid: { enabled: true, zoomable: true }` | 启用流程图/类图等 |
| Chart.js | `chart: { chartjs: true }` | 启用 Chart.js 图表 |
| ECharts | `chart: { echarts: true }` | 启用 ECharts 图表 |
| Vega-Lite | `chart: { vega_lite: true }` | 启用 Vega-Lite 图表 |
| Plotly | `chart: { plotly: true }` | 启用 Plotly 交互式图表 |
| Map (Leaflet) | `map: true` | 启用地图 |
| TikZ | `tikzjax: true` | 启用 LaTeX TikZ 绘图 |
| Typograms | `typograms: true` | 启用 ASCII 绘图 |
| 代码差异 | `code_diff: true` | 启用 Diff2Html |
| 伪代码 | `pseudocode: true` | 启用算法伪代码渲染 |
| 文章引用 | `citation: true` | 在文末显示本文的引用格式 |
| 标签页 | `tabs: true` | 启用内容标签页 |
| 高级图片 | `images: { compare: true, slider: true }` | 启用图片滑块和对比 |

---

## 2. 基础文本格式

### Markdown 标准语法

支持 GitHub Flavored Markdown (GFM)。

```markdown
**加粗**
*斜体*
[链接文本](URL)
# 一级标题
## 二级标题
- 无序列表
1. 有序列表
```

### 自定义提示框 (Alerts/Blockquotes)

基于 Kramdown 的类属性语法，支持 Tip, Warning, Danger 三种样式。

**语法：**
```markdown
> ##### TIP
> 这是一个提示信息。
{: .block-tip }

> ##### WARNING
> 这是一个警告信息。
{: .block-warning }

> ##### DANGER
> 这是一个危险警告。
{: .block-danger }
```

### 表格

**基础 Markdown 表格：**

```markdown
| 列 1 | 列 2 |
| :--- | :--- |
| 内容 | 内容 |
```

**Bootstrap Table (高级表格)：**
需要在 Front Matter 中设置 `pretty_table: true`。

```html
<table data-toggle="table" data-url="{{ '/assets/json/table_data.json' | relative_url }}">
  <thead>
    <tr>
      <th data-field="id">ID</th>
      <th data-field="name">Name</th>
    </tr>
  </thead>
</table>
```

---

## 3. 多媒体嵌入

### 图片 (Figure)

使用 `figure.liquid` 标签插入图片，支持响应式、标题、类名和缩放。

**语法：**
```liquid
{% include figure.liquid 
    path="assets/img/image.jpg" 
    class="img-fluid rounded z-depth-1" 
    width="50%" 
    caption="这是一个示例图片" 
    zoomable=true 
    alt="替代文本" 
%}
```

**参数说明：**
*   `path`: 图片路径（必需）
*   `class`: CSS 类名（如 `img-fluid` 响应式, `rounded` 圆角, `z-depth-1` 阴影）
*   `width`/`height`: 尺寸
*   `caption`: 图片下方标题
*   `zoomable`: 是否允许点击放大
*   `loading`: `eager` 或 `lazy` (默认 lazy)

### 图片滑块与对比 (Advanced Images)

需在 Front Matter 启用 `images: { compare: true, slider: true }`。

**图片滑块 (Swiper)：**
```html
<swiper-container keyboard="true" navigation="true" pagination="true">
  <swiper-slide>{% include figure.liquid path="assets/img/1.jpg" class="img-fluid" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid path="assets/img/2.jpg" class="img-fluid" %}</swiper-slide>
</swiper-container>
```

**图片对比 (Comparison)：**
```html
<img-comparison-slider>
  {% include figure.liquid path="assets/img/before.jpg" slot="first" %}
  {% include figure.liquid path="assets/img/after.jpg" slot="second" %}
</img-comparison-slider>
```

### 音频 (Audio)

**语法：**
```liquid
{% include audio.liquid 
    path="assets/audio/sound.mp3" 
    controls=true 
    caption="音频标题" 
%}
```

### 视频 (Video)

支持本地文件或嵌入 iframe。

**本地文件：**
```liquid
{% include video.liquid 
    path="assets/video/movie.mp4" 
    class="img-fluid rounded z-depth-1" 
    controls=true 
    autoplay=true
%}
```

**嵌入视频 (如 YouTube)：**
```liquid
{% include video.liquid 
    path="https://www.youtube.com/embed/VIDEO_ID" 
    class="img-fluid rounded z-depth-1" 
%}
```

### Twitter (X) 嵌入

**语法：**
```liquid
{% twitter https://twitter.com/username/status/123456789 %}
```
或者带参数：
```liquid
{% twitter https://twitter.com/username maxwidth=500 limit=3 %}
```

### 图片画廊 (Gallery)

支持多种画廊库 (Lightbox2, PhotoSwipe, Spotlight, Venobox)。需要在 Front Matter 中开启对应库。

**示例 (Lightbox2)：**
```html
<a href="image-large.jpg" data-lightbox="gallery-name" data-title="标题">
    <img src="image-thumb.jpg" alt="Image">
</a>
```

---

## 4. 代码与技术文档

### 代码块

**标准语法：**
````markdown
```python
def hello():
    print("Hello World")
```
````

**Liquid 高亮标签 (Jekyll 原生)：**
```liquid
{% highlight python %}
def hello():
    print("Hello World")
{% endhighlight %}
```

### 代码差异对比 (Diff2Html)

需配置 `code_diff: true`。

````markdown
```diff2html
diff --git a/file.js b/file.js
index 123..456 100644
--- a/file.js
+++ b/file.js
@@ -1,3 +1,3 @@
- old code
+ new code
```
````

### 伪代码 (Pseudocode)

需配置 `pseudocode: true`。使用 LaTeX 风格语法。

````markdown
```pseudocode
\begin{algorithm}
\caption{Quicksort}
\begin{algorithmic}
\PROCEDURE{Quicksort}{$$A, p, r$$}
    \IF{$$p < r$$}
        \STATE ...
    \ENDIF
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
````

### Jupyter Notebook

需安装插件并在 Front Matter 启用相关配置（通常自动支持）。

```liquid
{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/notebook.ipynb' | relative_url %}
{% jupyter_notebook jupyter_path %}
{:/nomarkdown}
```
*注意：需使用 `{::nomarkdown}` 包裹以防止 Markdown 解析器干扰。*

---

## 5. 数学公式

使用 MathJax 渲染。

**行内公式：**
使用 `$$ ... $$` 或 `$ ... $`。
例如：`$$ E = mc^2 $$`

**块级公式：**
双美元符号包裹，并独占一行。
```latex
$$
\sum_{i=1}^n i = \frac{n(n+1)}{2}
$$
```

---

## 6. 图表与数据可视化

所有图表需在 Front Matter 中开启对应模块。

### Mermaid (流程图/时序图)

````markdown
```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```
````

### Chart.js (常用统计图)

使用 JSON 配置。

````markdown
```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["Red", "Blue", "Yellow"],
    "datasets": [{
      "label": "Votes",
      "data": [12, 19, 3]
    }]
  }
}
```
````

### ECharts (交互式图表)

````markdown
```echarts
{
  "xAxis": {
    "type": "category",
    "data": ["Mon", "Tue", "Wed"]
  },
  "yAxis": {
    "type": "value"
  },
  "series": [{
    "data": [150, 230, 224],
    "type": "line"
  }]
}
```
````

### Plotly (交互式图表)

````markdown
```plotly
{
  "data": [
    {
      "x": [1, 2, 3, 4],
      "y": [10, 15, 13, 17],
      "type": "scatter"
    }
  ]
}
```
````

### Typograms (ASCII 风格图表)

````markdown
```typograms
+----+
|    |---> Diagram
+----+
```
````

### Leaflet 地图 (GeoJSON)

````markdown
```geojson
{
  "type": "FeatureCollection",
  "features": [ ... ]
}
```
````

### TikZ (LaTeX 绘图)

````html
<script type="text/tikz">
  \begin{tikzpicture}
    \draw (0,0) circle (1cm);
  \end{tikzpicture}
</script>
````

---

## 7. 引用与参考文献

### 插入引用

**Distill 风格：**
使用 `<d-cite>` 标签。
```html
<d-cite key="author2023paper"></d-cite>
```
*需要配合 `bibliography: file.bib` 在 Front Matter 中指定 bib 文件。*

**脚注：**
```html
<d-footnote>这里是脚注内容。</d-footnote>
```

---

## 8. 特殊布局与组件

### 标签页 (Tabs)

需在 Front Matter 启用 `tabs: true`。

```liquid
{% tabs group-name %}

{% tab group-name tab-1 %}
Content for tab 1
{% endtab %}

{% tab group-name tab-2 %}
Content for tab 2
{% endtab %}

{% endtabs %}
```

### 折叠详情块 (Details)

使用 Liquid 标签创建可折叠内容。

```liquid
{% details 点击展开详情 %}
这里是隐藏的内容，支持 **Markdown** 和 $$ Math $$。
{% enddetails %}
```

### 目录 (TOC)

**自动生成：**
在 Front Matter 中添加：
```yaml
toc:
  sidebar: left # 或 right
```

**手动指定 (Distill 布局)：**
```yaml
toc:
  - name: 章节名称
    subsections:
      - name: 子章节
```

### GitHub 仓库卡片

**单个仓库：**
```liquid
{% include repository/repo.liquid repository="username/repo-name" %}
```

**用户资料：**
```liquid
{% include repository/repo_user.liquid username="username" %}
```
