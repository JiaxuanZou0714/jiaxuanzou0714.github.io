window.MathJax = {
  loader: { load: ["[tex]/boldsymbol"] },
  startup: {
    typeset: false,
    ready() {
      MathJax.startup.defaultReady();

      const mathRoots = Array.from(document.querySelectorAll(".post, d-title, d-article, d-appendix, #toc-sidebar"));
      const elements = mathRoots.length > 0 ? mathRoots : [document.body];

      MathJax.startup.promise = MathJax.startup.promise.then(() => MathJax.typesetPromise(elements));
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
