function determineGiscusTheme() {
  
    let theme = document.documentElement.getAttribute("data-theme") || "system";

    if (theme === "dark") return "dark";
    if (theme === "light") return "light";

    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    return prefersDark ? "dark" : "light";
  
}

(function setupGiscus() {
  const container = document.getElementById("giscus_thread");
  if (!container) return;
  let loaded = false;

  function loadComments() {
    if (loaded) return;
    loaded = true;
    // Read the theme when comments load, including changes made while reading.
    let giscusTheme = determineGiscusTheme();

    let giscusAttributes = {
      src: "https://giscus.app/client.js",
      "data-repo": "JiaxuanZou0714/jiaxuanzou0714.github.io",
      "data-repo-id": "R_kgDOPvg-0Q",
      "data-category": "Announcements",
      "data-category-id": "DIC_kwDOPvg-0c4C1wDF",
      "data-mapping": "title",
      "data-strict": "1",
      "data-reactions-enabled": "1",
      "data-emit-metadata": "0",
      "data-input-position": "top",
      "data-theme": giscusTheme,
      "data-lang": "zh-CN",
      crossorigin: "anonymous",
      async: true,
    };

    let giscusScript = document.createElement("script");
    Object.entries(giscusAttributes).forEach(([key, value]) =>
      giscusScript.setAttribute(key, value)
    );
    giscusScript.addEventListener("error", () => {
      loaded = false;
      giscusScript.remove();
      const retry = document.createElement("button");
      retry.className = "btn btn-sm btn-outline-secondary";
      retry.textContent = "zh-CN".startsWith("zh") ? "重新加载评论" : "Retry loading comments";
      retry.addEventListener("click", () => {
        retry.remove();
        loadComments();
      }, { once: true });
      container.appendChild(retry);
    }, { once: true });
    container.appendChild(giscusScript);
  }

  if (!("IntersectionObserver" in window)) {
    loadComments();
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) {
      observer.disconnect();
      loadComments();
    }
  }, { rootMargin: "600px" });
  observer.observe(container);
})();
