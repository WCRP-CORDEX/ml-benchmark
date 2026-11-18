/*
 * MathJax configuration: renders the LaTeX math of the pages (`$...$` inline,
 * `$$...$$` display), which the `pymdownx.arithmatex` extension (mkdocs.yml)
 * wraps in elements of class `arithmatex`. Only those elements are processed.
 * MathJax itself is loaded after this file (see `extra_javascript`).
 */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// typeset again on every page load, including instant navigation
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
