window.MathJax = {
  tex: {
    tags: "ams",
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
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
