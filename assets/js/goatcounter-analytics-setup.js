document.addEventListener("DOMContentLoaded", () => {
  const counters = Array.from(
    document.querySelectorAll("[data-goatcounter-path]"),
  );
  const endpoint = window.goatcounterEndpoint;
  if (!endpoint || counters.length === 0) {
    return;
  }

  const siteRoot = endpoint.replace(/\/count\/?$/, "/");
  const requests = new Map();
  const groupedCounters = new Map();

  // Keep language-specific analytics and include both historical counters.
  counters.forEach((element) => {
    const paths = [
      ...new Set(
        [
          element.getAttribute("data-goatcounter-path"),
          element.getAttribute("data-goatcounter-alternate-path"),
        ].filter(Boolean),
      ),
    ].sort();
    if (paths.length === 0) return;
    const key = JSON.stringify(paths);
    const group = groupedCounters.get(key) || { paths, elements: [] };
    group.elements.push(element);
    groupedCounters.set(key, group);
  });

  function loadCount(path) {
    if (!requests.has(path)) {
      requests.set(
        path,
        fetch(`${siteRoot}counter/${encodeURIComponent(path)}.json`)
          .then((response) => {
            // GoatCounter returns 404 for a path without recorded visits.
            if (response.status === 404) return { count: "0" };
            if (!response.ok)
              throw new Error(`Failed to load counter for ${path}`);
            return response.json();
          })
          .then((data) => {
            // The API formats counts with thousands separators.
            const raw = String(data?.count ?? "").replace(
              /[,\s\u00a0\u202f]/g,
              "",
            );
            const count = Number(raw);
            if (!/^\d+$/.test(raw) || !Number.isSafeInteger(count)) {
              throw new Error(`Invalid counter for ${path}`);
            }
            return count;
          }),
      );
    }
    return requests.get(path);
  }

  groupedCounters.forEach(({ paths, elements }) => {
    const display = (value) =>
      elements.forEach((element) => {
        const node = element.querySelector(".goatcounter-count");
        if (node) node.textContent = value;
      });
    Promise.all(paths.map(loadCount))
      .then((counts) =>
        display(
          counts
            .reduce((total, count) => total + count, 0)
            .toLocaleString("en-US"),
        ),
      )
      // Do not display an incomplete total when one language fails to load.
      .catch(() => display("--"));
  });
});
