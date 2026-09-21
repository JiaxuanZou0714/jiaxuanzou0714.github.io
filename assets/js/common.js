$(document).ready(function () {
  // add toggle functionality to abstract, award and bibtex buttons
  $("a.abstract").click(function () {
    $(this).parent().parent().find(".abstract.hidden").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.award").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.bibtex").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden").toggleClass("open");
  });
  // bootstrap-toc
  if ($("#toc-sidebar").length) {
    // remove related publications years from the TOC
    $(".publications h2").each(function () {
      $(this).attr("data-toc-skip", "");
    });

    // Fix heading IDs that start with digits — CSS selectors used by
    // Bootstrap ScrollSpy (querySelector) don't allow IDs starting with
    // a digit, which silently breaks scroll tracking.
    var renamedIds = {};
    $("h1, h2, h3, h4, h5, h6").each(function () {
      if (this.id && /^\d/.test(this.id)) {
        renamedIds[this.id] = "sec-" + this.id;
        this.id = "sec-" + this.id;
      }
    });

    // Markdown-authored links still point at the pre-rename IDs, so they
    // would update the URL without scrolling anywhere.
    $('a[href^="#"]').each(function () {
      var renamed = renamedIds[decodeURIComponent(this.getAttribute("href").slice(1))];
      if (renamed) {
        this.setAttribute("href", "#" + renamed);
      }
    });

    // Same problem for a fragment that was already in the URL on load:
    // the browser resolved it against the old ID before this ran.
    var landed = renamedIds[decodeURIComponent(window.location.hash.slice(1))];
    if (landed) {
      window.history.replaceState(null, "", "#" + landed);
      document.getElementById(landed).scrollIntoView();
    }

    var navSelector = "#toc-sidebar";
    var $myNav = $(navSelector);
    Toc.init($myNav);
    $("body").scrollspy({
      target: navSelector,
      offset: 100,
    });

    // Refresh scrollspy after MathJax finishes rendering,
    // since equations change page height and heading positions.
    // The MathJax configuration can exist before startup.promise is created.
    document.addEventListener("mathjax:typeset", function () {
      $("body").scrollspy("refresh");
    });
  }

  // add css to jupyter notebooks
  const cssLink = document.createElement("link");
  cssLink.href = "../css/jupyter.css";
  cssLink.rel = "stylesheet";
  cssLink.type = "text/css";

  let jupyterTheme = determineComputedTheme();

  $(".jupyter-notebook-iframe-container iframe").each(function () {
    $(this).contents().find("head").append(cssLink);

    if (jupyterTheme == "dark") {
      $(this).bind("load", function () {
        $(this).contents().find("body").attr({
          "data-jp-theme-light": "false",
          "data-jp-theme-name": "JupyterLab Dark",
        });
      });
    }
  });

  // trigger popovers
  $('[data-toggle="popover"]').popover({
    trigger: "hover",
  });
});
