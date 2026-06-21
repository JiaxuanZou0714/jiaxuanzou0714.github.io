window.MathJax = {
  loader: { load: ["[tex]/boldsymbol"] },
  startup: {
    typeset: false,
    ready() {
      MathJax.startup.defaultReady();

      const mathRoots = Array.from(
        document.querySelectorAll(
          ".post, d-title, d-article, d-appendix, #toc-sidebar",
        ),
      );
      const elements = mathRoots.length > 0 ? mathRoots : [document.body];

      MathJax.startup.promise = MathJax.startup.promise
        .then(() => MathJax.typesetPromise(elements))
        .then(() => {
          // Store the original LaTeX source on every rendered container so that
          // it can be copied (click-to-copy + selection-to-markdown below).
          const doc = MathJax.startup.document;
          if (!doc || !doc.math) return;
          for (const item of doc.math) {
            const root = item.typesetRoot;
            if (!root || !root.setAttribute) continue;
            root.setAttribute("data-tex", item.math);
            if (item.display) root.setAttribute("data-tex-display", "true");
            root.setAttribute("title", "点击复制 LaTeX");
          }
        });
    },
  },
  tex: {
    tags: "ams",
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    packages: { "[+]": ["boldsymbol"] },
  },
  options: {
    renderActions: {
      addCss: [
        200,
        function (doc) {
          const style = document.createElement("style");
          style.innerHTML = `
          .mjx-container {
            color: inherit;
          }
          mjx-container[display="true"] {
            display: block;
            overflow-x: auto;
            overflow-y: hidden;
          }
        `;
          document.head.appendChild(style);
        },
        "",
      ],
    },
  },
};

// ---------------------------------------------------------------------------
// Copy LaTeX from rendered math
//   1. Click a formula  -> copies its raw LaTeX source.
//   2. Select text + copy -> formulas are restored to their original markdown
//      delimiters ($...$ for inline, $$...$$ for display) instead of being
//      dropped/garbled by the browser's default copy.
// The original LaTeX is read from the data-tex attribute set after typesetting.
// ---------------------------------------------------------------------------
(function () {
  "use strict";

  // Set to true to also include $...$ / $$...$$ delimiters when clicking a formula.
  const CLICK_COPY_WITH_DELIMITERS = false;

  function getTex(container) {
    return container.getAttribute("data-tex") || "";
  }

  function isDisplay(container) {
    return (
      container.getAttribute("display") === "true" ||
      container.getAttribute("data-tex-display") === "true"
    );
  }

  // Wrap a formula's LaTeX in its markdown delimiters.
  function wrapTex(container, blockNewlines) {
    const tex = getTex(container);
    if (isDisplay(container)) {
      return blockNewlines ? "\n\n$$" + tex + "$$\n\n" : "$$" + tex + "$$";
    }
    return "$" + tex + "$";
  }

  // Serialize a selection Range to text, substituting rendered math with its
  // original markdown and keeping rough block structure (paragraphs, lists...).
  const BLOCK_TAGS = new Set([
    "P",
    "DIV",
    "SECTION",
    "ARTICLE",
    "LI",
    "UL",
    "OL",
    "DL",
    "DT",
    "DD",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "BLOCKQUOTE",
    "PRE",
    "FIGURE",
    "FIGCAPTION",
    "TABLE",
    "TR",
    "HR",
  ]);

  function serializeFragmentWithLatex(frag) {
    // Replace each math container with its markdown form.
    frag
      .querySelectorAll("mjx-container[data-tex]")
      .forEach(function (container) {
        container.replaceWith(
          document.createTextNode(wrapTex(container, true)),
        );
      });

    let out = "";
    function walk(node) {
      if (node.nodeType === Node.TEXT_NODE) {
        out += node.nodeValue;
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      const tag = node.tagName;
      if (tag === "BR") {
        out += "\n";
        return;
      }
      const block = BLOCK_TAGS.has(tag);
      if (block && out && !out.endsWith("\n")) out += "\n";
      node.childNodes.forEach(walk);
      if (block && out && !out.endsWith("\n")) out += "\n";
    }
    // Start from the fragment's children: a DocumentFragment is neither a text
    // nor an element node, so walking it directly would bail immediately.
    frag.childNodes.forEach(walk);

    return out.replace(/\n{3,}/g, "\n\n").trim();
  }

  function setClipboard(event, text) {
    if (!text) return false;
    event.clipboardData.setData("text/plain", text);
    event.preventDefault();
    return true;
  }

  // --- Selection copy: restore original markdown for any selected formulas ---
  document.addEventListener("copy", function (event) {
    try {
      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0 || selection.isCollapsed)
        return;

      const range = selection.getRangeAt(0);
      const ancestor = range.commonAncestorContainer;
      const ancestorEl =
        ancestor.nodeType === Node.ELEMENT_NODE
          ? ancestor
          : ancestor.parentElement;

      // Selection sits entirely inside a single formula -> copy that formula.
      const enclosing =
        ancestorEl && ancestorEl.closest("mjx-container[data-tex]");
      if (enclosing) {
        setClipboard(event, wrapTex(enclosing, false));
        return;
      }

      // Clone the selection and rebuild it only if it actually contains math.
      // Detecting math via the cloned fragment is far more reliable than
      // Range.intersectsNode (which is missing/buggy in some engines).
      const frag = range.cloneContents();
      if (!frag.querySelector("mjx-container[data-tex]")) return; // no math -> native copy

      setClipboard(event, serializeFragmentWithLatex(frag));
    } catch (e) {
      // On any unexpected error, leave the browser's native copy untouched.
    }
  });

  // --- Click a formula to copy its LaTeX ---
  document.addEventListener("click", function (event) {
    if (!event.target.closest) return;
    const container = event.target.closest("mjx-container[data-tex]");
    if (!container) return;

    // Don't hijack the click if the user is in the middle of selecting text.
    const selection = window.getSelection();
    if (
      selection &&
      !selection.isCollapsed &&
      selection.toString().trim().length > 0
    )
      return;

    const text = CLICK_COPY_WITH_DELIMITERS
      ? wrapTex(container, false)
      : getTex(container);
    if (!text) return;

    const onDone = function () {
      showToast(container, "已复制 LaTeX");
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard
        .writeText(text)
        .then(onDone)
        .catch(function () {
          legacyCopy(text) && onDone();
        });
    } else if (legacyCopy(text)) {
      onDone();
    }
  });

  // Fallback for non-secure contexts where navigator.clipboard is unavailable.
  function legacyCopy(text) {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch (e) {
      return false;
    }
  }

  // --- Small "copied" toast shown near the clicked formula ---
  let toastEl = null;
  let toastTimer = null;
  function showToast(target, message) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.className = "math-copy-toast";
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = message;
    const rect = target.getBoundingClientRect();
    toastEl.style.left = window.scrollX + rect.left + rect.width / 2 + "px";
    toastEl.style.top = window.scrollY + rect.top - 8 + "px";
    // Force reflow so re-triggering the animation works on rapid clicks.
    void toastEl.offsetWidth;
    toastEl.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toastEl.classList.remove("visible");
    }, 1200);
  }

  // --- Styles: clickable cursor, subtle hover, toast ---
  const style = document.createElement("style");
  style.innerHTML = `
    mjx-container[data-tex] {
      cursor: pointer;
      border-radius: 3px;
      transition: background-color 0.15s ease;
    }
    mjx-container[data-tex]:hover {
      background-color: rgba(128, 128, 128, 0.15);
    }
    .math-copy-toast {
      position: absolute;
      transform: translate(-50%, -100%);
      background: rgba(0, 0, 0, 0.82);
      color: #fff;
      font-size: 0.75rem;
      line-height: 1;
      padding: 6px 10px;
      border-radius: 6px;
      white-space: nowrap;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.15s ease;
      z-index: 9999;
    }
    .math-copy-toast.visible {
      opacity: 1;
    }
  `;
  (document.head || document.documentElement).appendChild(style);
})();
