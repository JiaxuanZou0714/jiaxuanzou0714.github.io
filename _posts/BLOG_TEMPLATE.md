---
layout: post
title: "在这里填写标题"
date: 2026-03-08 12:00:00
description: "用 1-2 句话概括本文的问题、方法和结论。"
tags: [tag1, tag2, tag3]
categories: [deep-learning]
featured: false
giscus_comments: true
toc:
  sidebar: left
---

复制本文时，建议先把文件名改成 YYYY-MM-DD-slug.md，再同步修改 title、date、description、tags 和文末引用里的 slug。

本文先用一段话说明写作动机、核心问题，以及这篇文章准备解决什么。

## 1.

## 2.

## 3.

……

### 3.1 ……

### 3.2

### 3.3

## 4.

## 5.

……


## 参考文献

[1]

[2]

……


## 引用

如果您需要引用本文，请参考：

```bibtex
@article{zou2026slug,
  title={在这里填写标题},
  author={Zou, Jiaxuan},
  journal={Jiaxuan's Blog},
  year={2026},
  url={https://jiaxuanzou0714.github.io/blog/2026/slug/}
}
```


注意：

表格用法：

```markdown
| 列1 | 列2 | 列3 |
| :--- | :---: | ---: |
| 左对齐 | 居中对齐 | 右对齐 |
{: .table .table-striped .table-sm style="font-size: 0.5em;"}
```

公式用法：
行内行间的公式都直接用 $$ 包裹，行间公式会自动居中

注意行间公式与上下文之间要空一行，否则可能会出现渲染问题：

```markdown
这是行内公式 $$E=mc^2$$，这是行间公式：

$$
E=mc^2
$$

下文继续……
```

公式竖线规范（推荐）：

```markdown
绝对值/模长请写成 \lvert x \rvert，例如：\lvert v_i \rvert
范数请写成 \lVert x \rVert，例如：\lVert g_t \rVert_{p+1}
```

这样比直接写 `|x|` 或 `\|x\|` 更稳定，能减少不同渲染环境下的歧义。