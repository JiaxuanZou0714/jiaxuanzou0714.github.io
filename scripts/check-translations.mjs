#!/usr/bin/env node
//
// Compare each generated English post against its Chinese source and report
// anything that looks like translation damage rather than translation.
//
// Errors are things a translation must never do: lose a formula, drop an image,
// restructure the document, leak a placeholder, or leave a paragraph in Chinese.
// Warnings need a human to look, because a legitimate translation can trip them:
// English word order moves a formula within a sentence, and Chinese numeric
// idioms like "两万多" are properly spelled out rather than kept as digits.
//
// Usage: node scripts/check-translations.mjs [--verbose]

import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const SOURCE_DIR = path.join(ROOT, "_posts");
const OUTPUT_DIR = path.join(ROOT, "_en_posts");
const VERBOSE = process.argv.includes("--verbose");

const REQUIRED_KEYS = ["lang", "permalink", "ref", "related_posts"];

function splitFrontMatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!match) return { frontMatter: null, body: raw };
  return { frontMatter: match[1], body: raw.slice(match[0].length) };
}

// Strip the regions a translator is required to leave alone, so the remaining
// text is the prose that actually got translated.
function proseOnly(text) {
  return text
    .replace(/^(```|~~~)[\s\S]*?^\1/gm, "")
    .replace(/\$\$[\s\S]*?\$\$/g, "")
    .replace(/`[^`\n]+`/g, "")
    .replace(/\{%[\s\S]*?%\}/g, "")
    .replace(/\]\([^)]*\)/g, "](L)")
    .replace(/https?:\/\/\S+/g, "U")
    .replace(/\/en\/blog\/\d{4}\/[a-z0-9-]*\//g, "L");
}

const count = (text, re) => (text.match(re) || []).length;
const multiset = (list) => [...list].sort().join("\u0000");

function diffCounts(a, b) {
  const tally = (list) => list.reduce((m, x) => m.set(x, (m.get(x) || 0) + 1), new Map());
  const ta = tally(a);
  const tb = tally(b);
  const lost = [];
  const gained = [];
  for (const [k, n] of ta) {
    const d = n - (tb.get(k) || 0);
    if (d > 0) lost.push(d > 1 ? `${k}×${d}` : k);
  }
  for (const [k, n] of tb) {
    const d = n - (ta.get(k) || 0);
    if (d > 0) gained.push(d > 1 ? `${k}×${d}` : k);
  }
  return { lost, gained };
}

async function checkOne(fileName) {
  const errors = [];
  const warnings = [];

  const rawSrc = await readFile(path.join(SOURCE_DIR, fileName), "utf8");
  const rawOut = await readFile(path.join(OUTPUT_DIR, fileName), "utf8");
  const src = splitFrontMatter(rawSrc);
  const out = splitFrontMatter(rawOut);

  if (out.frontMatter === null) {
    return { errors: ["front matter delimiters are missing or malformed"], warnings };
  }
  for (const key of REQUIRED_KEYS) {
    if (!new RegExp(`^${key}:`, "m").test(out.frontMatter)) {
      errors.push(`front matter is missing \`${key}\``);
    }
  }
  for (const key of ["title", "description"]) {
    const value = out.frontMatter.match(new RegExp(`^${key}:[ \\t]*(.+)$`, "m"))?.[1]?.trim();
    if (value && /^"/.test(value) && !/^".*(?<!\\)"$/.test(value)) {
      errors.push(`front matter \`${key}\` has an unbalanced double quote`);
    }
  }

  const a = src.body;
  const b = out.body;

  if (/@@KEEP\d+@@/.test(b)) errors.push("a @@KEEP@@ placeholder leaked into the output");

  // Formulas may move within a sentence, and a \text{} annotation inside one is
  // prose that should be translated. Everything else about the mathematics has
  // to be identical, so compare with annotation contents blanked out.
  const mathA = a.match(/\$\$[\s\S]*?\$\$/g) || [];
  const mathB = b.match(/\$\$[\s\S]*?\$\$/g) || [];
  const blankAnnotations = (s) => s.replace(/\\(text|mathrm|textbf|mbox)\{[^{}]*\}/g, "\\$1{}");
  const structureA = mathA.map(blankAnnotations);
  const structureB = mathB.map(blankAnnotations);
  if (multiset(structureA) !== multiset(structureB)) {
    errors.push(`math structure differs (${mathA.length} source, ${mathB.length} output)`);
  } else if (structureA.join("\u0000") !== structureB.join("\u0000")) {
    warnings.push("a formula moved within its sentence (English word order)");
  }

  // Math is protected wholesale, so a \text{} annotation inside a formula never
  // reaches the translator and renders as Chinese on an English page.
  const mathText = [];
  for (const span of mathB) {
    for (const m of span.matchAll(/\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}/g)) {
      if (/[\u4e00-\u9fff]/.test(m[1])) mathText.push(m[1]);
    }
  }
  if (mathText.length > 0) {
    errors.push(`Chinese inside math text macros: ${mathText.map((t) => `\\text{${t}}`).join(", ")}`);
  }

  const pathsA = [...a.matchAll(/path='([^']*)'/g)].map((m) => m[1]);
  const pathsB = [...b.matchAll(/path='([^']*)'/g)].map((m) => m[1]);
  if (pathsA.join("|") !== pathsB.join("|")) errors.push("image paths changed");

  const liquidA = count(a, /\{%[\s\S]*?%\}/g);
  const liquidB = count(b, /\{%[\s\S]*?%\}/g);
  // post_url tags become plain /en/blog/ links, so the output may legitimately
  // have fewer. It must never have more.
  if (liquidB > liquidA) errors.push(`output gained Liquid tags (${liquidA} -> ${liquidB})`);

  // Link structure must survive. Site-relative links are exempt because
  // post_url cross-references are rewritten to /en/blog/ paths on purpose.
  for (const [label, re] of [
    ["markdown links", /\[[^\]]*\]\(/g],
    ["external URLs", /\]\(https?:\/\//g],
  ]) {
    const ca = count(a, re);
    const cb = count(b, re);
    if (ca !== cb) errors.push(`${label} ${ca} -> ${cb}`);
  }

  const blocksA = a.split(/\n[ \t]*\n/).length;
  const blocksB = b.split(/\n[ \t]*\n/).length;
  if (blocksA !== blocksB) errors.push(`paragraph count changed (${blocksA} -> ${blocksB})`);

  const headsA = [...a.matchAll(/^(#+) /gm)].map((m) => m[1].length);
  const headsB = [...b.matchAll(/^(#+) /gm)].map((m) => m[1].length);
  if (headsA.join(",") !== headsB.join(",")) {
    errors.push(`heading structure changed (${headsA.length} -> ${headsB.length})`);
  }

  const stripped = proseOnly(b);
  const chineseBlocks = stripped
    .split(/\n[ \t]*\n/)
    .filter((block) => (block.match(/[\u4e00-\u9fff]/g) || []).length > 10);
  if (chineseBlocks.length > 0) {
    errors.push(`${chineseBlocks.length} paragraph(s) still in Chinese`);
    if (VERBOSE) for (const c of chineseBlocks) errors.push(`    ${c.trim().slice(0, 90)}`);
  }

  // Kramdown block attributes like {: .table .table-striped} carry styling, and
  // dropping one silently renders the table unstyled.
  const attrA = count(a, /^\{:.*\}$/gm);
  const attrB = count(b, /^\{:.*\}$/gm);
  if (attrA !== attrB) errors.push(`kramdown block attributes ${attrA} -> ${attrB}`);

  for (const [label, re] of [
    ["list items", /^\s*[-*+] /gm],
    ["numbered items", /^\s*\d+\. /gm],
    ["blockquote lines", /^\s*> /gm],
    ["table rows", /^\|/gm],
  ]) {
    const ca = count(a, re);
    const cb = count(b, re);
    if (ca !== cb) warnings.push(`${label} ${ca} -> ${cb}`);
  }

  const numsA = proseOnly(a).match(/\d+(?:[.,]\d+)*/g) || [];
  const numsB = stripped.match(/\d+(?:[.,]\d+)*/g) || [];
  const { lost, gained } = diffCounts(numsA, numsB);
  if (lost.length || gained.length) {
    const parts = [];
    if (lost.length) parts.push(`dropped ${lost.slice(0, 8).join(", ")}`);
    if (gained.length) parts.push(`added ${gained.slice(0, 8).join(", ")}`);
    warnings.push(`numbers in prose: ${parts.join("; ")}`);
  }

  return { errors, warnings };
}

async function main() {
  const files = (await readdir(OUTPUT_DIR)).filter((f) => f.endsWith(".md")).sort();
  if (files.length === 0) {
    process.stderr.write("No translations found in _en_posts/\n");
    process.exit(1);
  }

  let errorFiles = 0;
  let warnFiles = 0;

  for (const file of files) {
    const { errors, warnings } = await checkOne(file);
    if (errors.length === 0 && warnings.length === 0) continue;
    process.stdout.write(`${file}\n`);
    for (const e of errors) process.stdout.write(`  ERROR    ${e}\n`);
    for (const w of warnings) process.stdout.write(`  warning  ${w}\n`);
    if (errors.length) errorFiles += 1;
    else warnFiles += 1;
  }

  process.stdout.write(
    `\n${files.length} translation(s) checked: ` +
      `${errorFiles} with errors, ${warnFiles} with warnings only\n`,
  );
  if (errorFiles > 0) process.exit(1);
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
});
