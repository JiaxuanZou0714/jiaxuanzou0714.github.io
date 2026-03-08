document.addEventListener('DOMContentLoaded', () => {
  const analyticsScript = document.querySelector('script[data-goatcounter]');
  const counters = Array.from(document.querySelectorAll('[data-goatcounter-path]'));

  if (!analyticsScript || counters.length === 0) {
    return;
  }

  const endpoint = analyticsScript.getAttribute('data-goatcounter');
  if (!endpoint) {
    return;
  }

  const siteRoot = endpoint.replace(/\/count\/?$/, '/');
  const groupedCounters = new Map();

  counters.forEach((element) => {
    const path = element.getAttribute('data-goatcounter-path');
    if (!path) {
      return;
    }

    const existingGroup = groupedCounters.get(path) || [];
    existingGroup.push(element);
    groupedCounters.set(path, existingGroup);
  });

  groupedCounters.forEach((elements, path) => {
    const counterUrl = `${siteRoot}counter/${encodeURIComponent(path)}.json`;

    fetch(counterUrl)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load counter for ${path}`);
        }
        return response.json();
      })
      .then((data) => {
        const count = data && typeof data.count === 'string' ? data.count : '0';
        elements.forEach((element) => {
          const valueNode = element.querySelector('.goatcounter-count');
          if (valueNode) {
            valueNode.textContent = count;
          }
        });
      })
      .catch(() => {
        elements.forEach((element) => {
          const valueNode = element.querySelector('.goatcounter-count');
          if (valueNode) {
            valueNode.textContent = '--';
          }
        });
      });
  });
});