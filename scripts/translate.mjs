#!/usr/bin/env node
//
// Pre-translate the Chinese posts in _posts/ into English documents in _en_posts/.
//
// Translation happens here rather than during `jekyll build` so that the deploy
// workflow never needs DEEPSEEK_API_KEY and never depends on the API being up.
// Both the generated markdown and the block cache are committed.
//
// Usage:
//   node scripts/translate.mjs                  translate everything that changed
//   node scripts/translate.mjs --dry-run        report work to be done, call no API
//   node scripts/translate.mjs --only <slug>    restrict to one post
//   node scripts/translate.mjs --force          ignore the cache and retranslate
//
// Requires DEEPSEEK_API_KEY, read from the environment or from a local .env.

import { readFile, writeFile, mkdir, readdir } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
import process from "node:process";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const SOURCE_DIR = path.join(ROOT, "_posts");
const OUTPUT_DIR = path.join(ROOT, "_en_posts");
const CACHE_PATH = path.join(ROOT, "_data", "i18n", "zh-en.json");

const SOURCE_LANG = "zh-CN";
const TARGET_LANG = "en";

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

function loadEnvFile(filePath) {
  if (!existsSync(filePath)) return;
  for (const line of readFileSync(filePath, "utf8").split("\n")) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (!match) continue;
    let value = match[2];
    if (/^(".*"|'.*')$/s.test(value)) value = value.slice(1, -1);
    if (!(match[1] in process.env)) process.env[match[1]] = value;
  }
}

loadEnvFile(path.join(ROOT, ".env"));

const API_KEY = process.env.DEEPSEEK_API_KEY;
const API_URL = process.env.DEEPSEEK_API_URL || "https://api.deepseek.com/chat/completions";
const MODEL = process.env.DEEPSEEK_MODEL || "deepseek-v4-flash";
const CONCURRENCY = Number(process.env.TRANSLATE_CONCURRENCY || 6);
const MAX_TOKENS = Number(process.env.TRANSLATE_MAX_TOKENS || 8192);

const args = process.argv.slice(2);
const DRY_RUN = args.includes("--dry-run");
const FORCE = args.includes("--force");
const ONLY = (() => {
  const i = args.indexOf("--only");
  return i !== -1 ? args[i + 1] : null;
})();

// ---------------------------------------------------------------------------
// Sentinel-based protection
//
// Anything matched here is lifted out before the text reaches the model and put
// back verbatim afterwards. Order matters: the outermost constructs go first so
// that a `$$` or a backtick inside a fenced block is not matched on its own.
// ---------------------------------------------------------------------------

const SENTINEL = (n) => `@@KEEP${n}@@`;
const SENTINEL_RE = /@@KEEP(\d+)@@/g;

const PROTECT_RULES = [
  { name: "fence", re: /^(```|~~~)[^\n]*\n[\s\S]*?^\1[^\n]*$/gm },
  { name: "liquid_raw", re: /\{%\s*raw\s*%\}[\s\S]*?\{%\s*endraw\s*%\}/g },
  { name: "liquid_tag", re: /\{%[\s\S]*?%\}/g },
  { name: "liquid_var", re: /\{\{[\s\S]*?\}\}/g },
  { name: "math", re: /\$\$[\s\S]*?\$\$/g },
  { name: "inline_code", re: /`[^`\n]+`/g },
  { name: "html_tag", re: /<\/?[a-zA-Z][^>\n]*>/g },
  { name: "link_dest", re: /\]\([^)\s]*(?:\s+"[^"]*")?\)/g },
];

function protect(text) {
  if (SENTINEL_RE.test(text)) {
    SENTINEL_RE.lastIndex = 0;
    throw new Error("Source already contains an @@KEEP<n>@@ token; pick a different sentinel.");
  }
  SENTINEL_RE.lastIndex = 0;

  const items = [];
  let out = text;
  for (const rule of PROTECT_RULES) {
    out = out.replace(rule.re, (match) => {
      const token = SENTINEL(items.length);
      items.push({ name: rule.name, text: match });
      return token;
    });
  }
  return { text: out, items };
}

// Later rules can wrap earlier sentinels: `[label]({% post_url x %})` protects
// the Liquid tag first, then the link destination swallows that sentinel. One
// replace pass would leave the inner token unexpanded, so iterate to a fixpoint.
function restore(text, items) {
  let out = text;
  for (let pass = 0; pass < 10; pass += 1) {
    const next = out.replace(SENTINEL_RE, (match, n) => {
      const item = items[Number(n)];
      return item ? item.text : match;
    });
    if (next === out) return out;
    out = next;
  }
  throw new Error("Sentinel restore did not converge; a placeholder is self-referential.");
}

function sentinelIds(text) {
  return [...text.matchAll(SENTINEL_RE)].map((m) => m[1]).sort();
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

function cacheKey(text) {
  return createHash("sha256")
    .update(`${MODEL}\u0000${SOURCE_LANG}\u0000${TARGET_LANG}\u0000${text}`)
    .digest("hex")
    .slice(0, 32);
}

async function loadCache() {
  if (!existsSync(CACHE_PATH)) return {};
  try {
    return JSON.parse(await readFile(CACHE_PATH, "utf8"));
  } catch (error) {
    throw new Error(`Cache at ${CACHE_PATH} is not valid JSON: ${error.message}`);
  }
}

async function saveCache(cache) {
  await mkdir(path.dirname(CACHE_PATH), { recursive: true });
  // Sort keys so the committed diff only shows genuinely new entries.
  const sorted = Object.fromEntries(Object.entries(cache).sort(([a], [b]) => a.localeCompare(b)));
  await writeFile(CACHE_PATH, `${JSON.stringify(sorted, null, 2)}\n`, "utf8");
}

// ---------------------------------------------------------------------------
// DeepSeek
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = [
  "You translate Chinese technical writing into English for an academic machine learning blog.",
  "The author researches mechanistic interpretability, deep learning theory, optimization, and scaling laws.",
  "",
  "Rules:",
  "1. Output only the translation. No preamble, no commentary, no code fences around the result.",
  "2. Tokens of the form @@KEEP<number>@@ are placeholders for math, code, and markup.",
  "   Reproduce every one of them exactly as written, the same number of times, in the same order.",
  "   Never translate, renumber, reformat, or drop them.",
  "3. Preserve the markdown structure of the input exactly: heading levels, list markers,",
  "   blockquote markers, indentation, table pipes and alignment rows, and line breaks.",
  "4. Use standard English terminology from the machine learning literature.",
  "   Examples: 谱范数 -> spectral norm, 学习率 -> learning rate, 预训练 -> pre-training,",
  "   缩放律 -> scaling law, 梯度噪声 -> gradient noise, 宽度极限 -> width limit,",
  "   特征学习 -> feature learning, 参数化 -> parametrization, 动量 -> momentum.",
  "5. Keep proper nouns, paper titles, author names, and library names in their original form.",
  "6. Text that is already English stays as it is.",
  "7. Write in the same register as the source: direct technical prose, no added hedging or filler.",
].join("\n");

async function callDeepSeek(text, { attempt = 1 } = {}) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: MODEL,
      temperature: 0,
      max_tokens: MAX_TOKENS,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: text },
      ],
    }),
  });

  if (!response.ok) {
    const retryable = response.status === 429 || response.status >= 500;
    const body = await response.text().catch(() => "");
    if (retryable && attempt < 5) {
      const delay = Math.min(2 ** attempt * 1000, 30000);
      await new Promise((r) => setTimeout(r, delay));
      return callDeepSeek(text, { attempt: attempt + 1 });
    }
    throw new Error(`DeepSeek ${response.status}: ${body.slice(0, 400)}`);
  }

  const data = await response.json();
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content !== "string" || content.trim() === "") {
    throw new Error(`DeepSeek returned no content: ${JSON.stringify(data).slice(0, 400)}`);
  }
  return { content: content.trim(), usage: data.usage || {} };
}

const stats = { hits: 0, calls: 0, promptTokens: 0, completionTokens: 0, wouldCall: 0 };

// Run `worker` over `items` with a bounded number of in-flight requests.
async function mapPool(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (true) {
      const i = next++;
      if (i >= items.length) return;
      results[i] = await worker(items[i], i);
    }
  });
  await Promise.all(runners);
  return results;
}

// Identical blocks requested concurrently must not each be paid for.
const inFlight = new Map();

async function translateUnit(source, cache) {
  const key = cacheKey(source);
  if (!FORCE && cache[key] !== undefined) {
    stats.hits += 1;
    return cache[key];
  }
  if (DRY_RUN) {
    stats.wouldCall += 1;
    return source;
  }
  if (!API_KEY) throw new Error("DEEPSEEK_API_KEY is not set. Put it in .env or the environment.");
  if (inFlight.has(key)) return inFlight.get(key);

  const pending = translateUncached(source, key, cache);
  inFlight.set(key, pending);
  try {
    return await pending;
  } finally {
    inFlight.delete(key);
  }
}

async function translateUncached(source, key, cache) {
  const expected = sentinelIds(source);
  let result = null;

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const { content, usage } = await callDeepSeek(
      attempt === 1
        ? source
        : `${source}\n\n[The previous attempt lost or altered a @@KEEP<number>@@ placeholder. Reproduce all of them exactly.]`,
    );
    stats.calls += 1;
    stats.promptTokens += usage.prompt_tokens || 0;
    stats.completionTokens += usage.completion_tokens || 0;

    const got = sentinelIds(content);
    if (got.join(",") === expected.join(",")) {
      result = content;
      break;
    }
    process.stderr.write(
      `  placeholder mismatch on attempt ${attempt} (expected ${expected.length}, got ${got.length})\n`,
    );
  }

  if (result === null) {
    throw new Error(
      "Model kept dropping @@KEEP@@ placeholders after 3 attempts. Source block:\n" +
        `${source.slice(0, 300)}\n`,
    );
  }

  cache[key] = result;
  return result;
}

// ---------------------------------------------------------------------------
// Markdown handling
// ---------------------------------------------------------------------------

const CJK_RE = /[\u4e00-\u9fff]/;

function splitFrontMatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!match) return { frontMatter: null, body: raw };
  return { frontMatter: match[1], body: raw.slice(match[0].length) };
}

// Replace a scalar front matter value, preserving the original quoting style.
async function translateFrontMatterField(frontMatter, key, cache) {
  const re = new RegExp(`^(${key}:[ \\t]*)(.+)$`, "m");
  const match = frontMatter.match(re);
  if (!match) return frontMatter;

  const raw = match[2].trim();
  let quote = "";
  let value = raw;
  if (/^".*"$/s.test(raw) || /^'.*'$/s.test(raw)) {
    quote = raw[0];
    value = raw.slice(1, -1);
  }
  if (!CJK_RE.test(value)) return frontMatter;

  const translated = await translateUnit(value, cache);
  // Always emit double quotes so that apostrophes introduced by the translation
  // cannot terminate the scalar.
  const escaped = translated.replace(/"/g, '\\"');
  return frontMatter.replace(re, `$1"${escaped}"`);
}

function setFrontMatterField(frontMatter, key, value) {
  const re = new RegExp(`^${key}:[ \\t]*.*$`, "m");
  if (re.test(frontMatter)) return frontMatter.replace(re, `${key}: ${value}`);
  return `${frontMatter}\n${key}: ${value}`;
}

// Liquid includes carry human-readable caption/alt/title attributes. The tag as
// a whole is protected, so these are translated separately inside the payload.
async function translateLiquidAttributes(tagText, cache) {
  const attrRe = /\b(caption|alt|title)(\s*=\s*)(['"])([\s\S]*?)\3/g;
  const pieces = [];
  let lastIndex = 0;
  let match;
  while ((match = attrRe.exec(tagText)) !== null) {
    const [full, name, eq, quote, value] = match;
    pieces.push(tagText.slice(lastIndex, match.index));
    if (CJK_RE.test(value)) {
      const translated = await translateUnit(value, cache);
      pieces.push(`${name}${eq}${quote}${translated.replace(new RegExp(quote, "g"), "")}${quote}`);
    } else {
      pieces.push(full);
    }
    lastIndex = match.index + full.length;
  }
  pieces.push(tagText.slice(lastIndex));
  return pieces.join("");
}

async function translateBody(body, cache) {
  const { text, items } = protect(body);

  // Blank lines separate translation units. Splitting with a capturing group
  // keeps the exact separators so the document reassembles byte-for-byte.
  const parts = text.split(/(\n[ \t]*\n)/);

  const jobs = [];
  for (let i = 0; i < parts.length; i += 2) {
    const block = parts[i];
    if (!CJK_RE.test(block)) continue;
    const leading = block.match(/^\s*/)[0];
    const trailing = block.match(/\s*$/)[0];
    const core = block.slice(leading.length, block.length - trailing.length);
    if (core === "") continue;
    jobs.push({ index: i, leading, core, trailing });
  }

  await mapPool(jobs, CONCURRENCY, async (job) => {
    const translated = await translateUnit(job.core, cache);
    parts[job.index] = job.leading + translated + job.trailing;
  });

  const tags = items.filter((item) => item.name === "liquid_tag" && CJK_RE.test(item.text));
  await mapPool(tags, CONCURRENCY, async (item) => {
    item.text = await translateLiquidAttributes(item.text, cache);
  });

  return restore(parts.join(""), items);
}

async function translateFile(fileName, cache) {
  const raw = await readFile(path.join(SOURCE_DIR, fileName), "utf8");
  const { frontMatter, body } = splitFrontMatter(raw);
  if (frontMatter === null) return { fileName, skipped: "no front matter" };
  if (!/^lang:[ \t]*zh-CN[ \t]*$/m.test(frontMatter)) {
    return { fileName, skipped: `lang is not ${SOURCE_LANG}` };
  }

  const slug = fileName.replace(/\.md$/, "");
  const year = slug.slice(0, 4);
  const urlSlug = slug.replace(/^\d{4}-\d{2}-\d{2}-/, "");

  let fm = frontMatter;
  fm = await translateFrontMatterField(fm, "title", cache);
  fm = await translateFrontMatterField(fm, "description", cache);
  fm = setFrontMatterField(fm, "lang", TARGET_LANG);
  // An explicit permalink avoids relying on collection permalink placeholders,
  // and `ref` is what the language switcher pairs the two documents on.
  fm = setFrontMatterField(fm, "permalink", `/en/blog/${year}/${urlSlug}/`);
  fm = setFrontMatterField(fm, "ref", urlSlug);
  // site.related_posts is populated from the Chinese posts, so the widget would
  // recommend articles this reader cannot read.
  fm = setFrontMatterField(fm, "related_posts", "false");

  const translatedBody = await translateBody(body, cache);

  if (DRY_RUN) {
    // Translation is the identity in a dry run, so the reassembled body must
    // match the source exactly. Any drift is a protect/split/restore bug.
    if (translatedBody !== body) {
      const at = [...body].findIndex((ch, i) => ch !== translatedBody[i]);
      return {
        fileName,
        roundTripFailed: `bodies diverge at offset ${at}: ` +
          `${JSON.stringify(body.slice(at, at + 60))} vs ` +
          `${JSON.stringify(translatedBody.slice(at, at + 60))}`,
      };
    }
    return { fileName, planned: true };
  }

  const output = `---\n${fm}\n---\n${translatedBody}`;
  await mkdir(OUTPUT_DIR, { recursive: true });
  await writeFile(path.join(OUTPUT_DIR, fileName), output, "utf8");
  return { fileName, written: true };
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main() {
  const cache = await loadCache();
  const before = Object.keys(cache).length;

  let files = (await readdir(SOURCE_DIR)).filter((f) => f.endsWith(".md")).sort();
  if (ONLY) files = files.filter((f) => f.includes(ONLY));
  if (files.length === 0) {
    process.stderr.write(ONLY ? `No post matches --only ${ONLY}\n` : "No posts found\n");
    process.exit(1);
  }

  if (DRY_RUN) process.stdout.write("Dry run: no API calls will be made.\n\n");

  // Files run one at a time so a failure points at a single post; the blocks
  // within each file are what get parallelised.
  const results = [];
  for (const file of files) {
    process.stdout.write(`${file}\n`);
    try {
      results.push(await translateFile(file, cache));
    } catch (error) {
      await saveCache(cache);
      process.stderr.write(`\nFailed on ${file}: ${error.message}\n`);
      process.stderr.write("Cache was saved, so a rerun resumes from here.\n");
      process.exit(1);
    }
  }

  if (!DRY_RUN) await saveCache(cache);

  const written = results.filter((r) => r.written || r.planned).length;
  const skipped = results.filter((r) => r.skipped);
  const broken = results.filter((r) => r.roundTripFailed);

  if (broken.length > 0) {
    process.stderr.write("\nRound-trip check failed; refusing to translate.\n");
    for (const b of broken) process.stderr.write(`  ${b.fileName}: ${b.roundTripFailed}\n`);
    process.exit(1);
  }

  process.stdout.write("\n");
  process.stdout.write(
    DRY_RUN
      ? `would write    ${written} file(s) to _en_posts/\n`
      : `written        ${written} file(s) to _en_posts/\n`,
  );
  for (const s of skipped) process.stdout.write(`skipped        ${s.fileName} (${s.skipped})\n`);
  process.stdout.write(`cache hits     ${stats.hits}\n`);
  if (DRY_RUN) {
    process.stdout.write(`would call     ${stats.wouldCall} unit(s)\n`);
  } else {
    process.stdout.write(`api calls      ${stats.calls}\n`);
    process.stdout.write(`prompt tokens  ${stats.promptTokens}\n`);
    process.stdout.write(`output tokens  ${stats.completionTokens}\n`);
    process.stdout.write(`cache entries  ${before} -> ${Object.keys(cache).length}\n`);
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
});
