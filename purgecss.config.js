module.exports = {
  content: ["_site/**/*.html", "_site/**/*.js"],
  css: ["_site/assets/css/*.css"],
  output: "_site/assets/css/",
  skippedContentGlobs: ["_site/assets/**/*.html"],
  // medium-zoom adds these classes at runtime (the library is loaded from a
  // CDN, so its class names appear in neither the built HTML nor any local JS).
  // Without safelisting, PurgeCSS strips their rules — including the z-index
  // that lifts a zoomed image/overlay above the TOC sidebar.
  // MDB (now served locally) creates waves-ripple elements at runtime when
  // elements opt into the waves effect, so keep those rules as well.
  safelist: [":root", /data-theme/, /^medium-zoom/, /^waves/],
};
