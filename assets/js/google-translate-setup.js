// Google Translate lazy loader and toggle helpers.
// Externalized from scripts.liquid so every page's HTML is ~5KB smaller and
// this file can be cached across navigations.

function applyNoTranslateToElements() {
  // Protect code and structured math elements from being translated
  var protectedSelectors = [
    'code',
    'pre',
    '.highlight',
    '.MathJax',
    '.MathJax_Display',
    'mjx-container',
    'math',
    '.equation',
    'script[type^="math/tex"]'
  ];
  document.querySelectorAll(protectedSelectors.join(', ')).forEach(function(el) {
    el.classList.add('notranslate');
  });
}

document.addEventListener('DOMContentLoaded', applyNoTranslateToElements);

// Catch dynamic math elements that load later (like MathJax rendering)
window.addEventListener('load', function() {
  setTimeout(applyNoTranslateToElements, 1000);
});

window.googleTranslateConfig = {
  pageLanguage: "zh-CN",
  includedLanguages: "en,zh-CN"
};

window.getGoogleTranslateLanguage = function () {
  var sourceLanguage = window.googleTranslateConfig.pageLanguage;
  var match = document.cookie.match(/(?:^|; )googtrans=([^;]+)/);

  if (!match) {
    return sourceLanguage;
  }

  var parts = decodeURIComponent(match[1]).split('/');
  var targetLanguage = parts[parts.length - 1];

  if (!targetLanguage || targetLanguage === parts[parts.length - 2]) {
    return sourceLanguage;
  }

  return targetLanguage;
};

window.updateGoogleTranslateToggleState = function () {
  var currentLanguage = window.getGoogleTranslateLanguage();
  var buttons = document.querySelectorAll('.translate-toggle-button');

  buttons.forEach(function (button) {
    var isActive = button.dataset.lang === currentLanguage;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
  });
};

window.setGoogleTranslateLanguage = function (language) {
  var sourceLanguage = window.googleTranslateConfig.pageLanguage;
  var cookieValue = '/' + sourceLanguage + '/' + language;

  document.cookie = 'googtrans=' + encodeURIComponent(cookieValue) + ';path=/;max-age=31536000';
  window.updateGoogleTranslateToggleState();
  window.location.reload();
};

window.loadGoogleTranslate = function () {
  if (document.querySelector('script[data-google-translate="true"]')) {
    return;
  }

  var script = document.createElement('script');
  script.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
  script.async = true;
  script.defer = true;
  script.dataset.googleTranslate = 'true';
  document.body.appendChild(script);
};

document.addEventListener('DOMContentLoaded', function () {
  window.updateGoogleTranslateToggleState();

  if (window.getGoogleTranslateLanguage() !== window.googleTranslateConfig.pageLanguage) {
    window.loadGoogleTranslate();
  }
});

window.googleTranslateElementInit = function () {
  new google.translate.TranslateElement(
    {
      pageLanguage: window.googleTranslateConfig.pageLanguage,
      includedLanguages: window.googleTranslateConfig.includedLanguages,
      layout: google.translate.TranslateElement.InlineLayout.SIMPLE,
      autoDisplay: false
    },
    "google_translate_element"
  );

  window.updateGoogleTranslateToggleState();
};
