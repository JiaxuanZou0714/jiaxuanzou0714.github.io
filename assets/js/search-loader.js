const searchAssetState = {
  moduleLoaded: false,
  dataLoaded: false,
};

let searchLoadPromise;

const searchAssetUrls = {
  module: "/assets/js/search/ninja-keys.min.js",
  data: "/assets/js/search-data.js",
};

const isMacPlatform = () => navigator.platform.toUpperCase().indexOf("MAC") >= 0;

const getAssetBaseUrl = () => {
  const currentScript = document.currentScript || document.querySelector('script[src*="/assets/js/search-loader.js"]');

  if (!currentScript) {
    return "";
  }

  const scriptUrl = new URL(currentScript.src, window.location.origin);
  return scriptUrl.pathname.replace(/\/assets\/js\/search-loader\.js$/, "");
};

const toRelativeUrl = (path) => {
  const baseUrl = getAssetBaseUrl();
  return `${baseUrl}${path}`;
};

const updateSearchShortcutLabel = () => {
  const shortcutKeyElement = document.querySelector("#search-toggle .nav-link");
  if (!shortcutKeyElement || !isMacPlatform()) {
    return;
  }

  shortcutKeyElement.innerHTML = '&#x2318; k <i class="fa-solid fa-magnifying-glass"></i>';
};

const ensureSearchElement = () => {
  let ninjaKeys = document.querySelector("ninja-keys");

  if (!ninjaKeys) {
    ninjaKeys = document.createElement("ninja-keys");
    ninjaKeys.setAttribute("hideBreadcrumbs", "");
    ninjaKeys.setAttribute("noAutoLoadMdIcons", "");
    ninjaKeys.setAttribute("placeholder", "Type to start searching");
    document.body.appendChild(ninjaKeys);
  }

  if (typeof determineComputedTheme === "function" && determineComputedTheme() === "dark") {
    ninjaKeys.classList.add("dark");
  } else {
    ninjaKeys.classList.remove("dark");
  }

  return ninjaKeys;
};

const collapseNavbarIfExpanded = () => {
  const navbarNav = document.getElementById("navbarNav");
  if (!navbarNav || !navbarNav.classList.contains("show")) {
    return;
  }

  if (window.jQuery) {
    window.jQuery(navbarNav).collapse("hide");
    return;
  }

  navbarNav.classList.remove("show");
};

const loadScriptOnce = ({ src, type = "text/javascript" }) => {
  const absoluteSrc = toRelativeUrl(src);
  const existingScript = document.querySelector(`script[data-search-src="${absoluteSrc}"]`);

  if (existingScript) {
    if (existingScript.dataset.loaded === "true") {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      existingScript.addEventListener("load", () => resolve(), { once: true });
      existingScript.addEventListener("error", () => reject(new Error(`Failed to load ${absoluteSrc}`)), { once: true });
    });
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = absoluteSrc;
    script.type = type;
    script.dataset.searchSrc = absoluteSrc;

    script.addEventListener(
      "load",
      () => {
        script.dataset.loaded = "true";
        resolve();
      },
      { once: true }
    );
    script.addEventListener("error", () => reject(new Error(`Failed to load ${absoluteSrc}`)), { once: true });

    document.body.appendChild(script);
  });
};

const ensureSearchLoaded = async () => {
  if (!searchLoadPromise) {
    searchLoadPromise = (async () => {
      const ninjaKeys = ensureSearchElement();

      if (!searchAssetState.moduleLoaded) {
        await loadScriptOnce({ src: searchAssetUrls.module, type: "module" });
        await customElements.whenDefined("ninja-keys");
        searchAssetState.moduleLoaded = true;
      }

      if (!searchAssetState.dataLoaded) {
        await loadScriptOnce({ src: searchAssetUrls.data });
        searchAssetState.dataLoaded = true;
      }

      return ninjaKeys;
    })().catch((error) => {
      searchLoadPromise = undefined;
      throw error;
    });
  }

  return searchLoadPromise;
};

window.openSearchModal = async () => {
  collapseNavbarIfExpanded();
  const ninjaKeys = await ensureSearchLoaded();
  ninjaKeys.open();
};

document.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  const usesSearchShortcut = key === "k" && (isMacPlatform() ? event.metaKey : event.ctrlKey);
  const searchAlreadyLoaded = !!customElements.get("ninja-keys") && !!document.querySelector("ninja-keys");

  if (!usesSearchShortcut || searchAlreadyLoaded) {
    return;
  }

  event.preventDefault();
  window.openSearchModal();
});

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", updateSearchShortcutLabel, { once: true });
} else {
  updateSearchShortcutLabel();
}