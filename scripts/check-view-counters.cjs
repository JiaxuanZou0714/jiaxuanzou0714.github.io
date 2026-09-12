const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const code = fs.readFileSync(
  "assets/js/goatcounter-analytics-setup.js",
  "utf8",
);
async function run(pairs, responses) {
  const calls = [];
  const elements = pairs.map(([path, alternate]) => ({
    value: { textContent: "--" },
    getAttribute: (name) =>
      name === "data-goatcounter-path" ? path : alternate,
    querySelector() {
      return this.value;
    },
  }));
  vm.runInNewContext(code, {
    window: { goatcounterEndpoint: "https://example.goatcounter.com/count" },
    document: {
      addEventListener: (_, f) => f(),
      querySelectorAll: () => elements,
    },
    fetch: async (url) => {
      const path = decodeURIComponent(url.split("/counter/")[1].slice(0, -5));
      calls.push(path);
      const value = responses[path];
      if (value instanceof Error) throw value;
      return {
        ok: typeof value !== "number",
        status: typeof value === "number" ? value : 200,
        json: async () => ({ count: value }),
      };
    },
  });
  await new Promise(setImmediate);
  return { values: elements.map((e) => e.value.textContent), calls };
}
(async () => {
  const result = await run(
    [
      ["/zh/", "/en/"],
      ["/en/", "/zh/"],
      ["/zh/", "/en/"],
    ],
    { "/zh/": "1,026", "/en/": "74" },
  );
  assert.deepEqual(result.values, ["1,100", "1,100", "1,100"]);
  assert.equal(result.calls.length, 2);
  assert.deepEqual(
    (await run([["/zh/", "/en/"]], { "/zh/": "26", "/en/": 404 })).values,
    ["26"],
  );
  assert.deepEqual(
    (await run([["/zh/", "/en/"]], { "/zh/": "26", "/en/": 500 })).values,
    ["--"],
  );
  assert.deepEqual(
    (
      await run([["/zh/", "/en/"]], {
        "/zh/": "26",
        "/en/": new Error("offline"),
      })
    ).values,
    ["--"],
  );
  assert.deepEqual((await run([["/zh/", null]], { "/zh/": "26" })).values, [
    "26",
  ]);
  assert.deepEqual((await run([["/zh/", "/zh/"]], { "/zh/": "26" })).values, [
    "26",
  ]);
  assert.deepEqual(
    (await run([["/zh/", "/en/"]], { "/zh/": "26", "/en/": "bad" })).values,
    ["--"],
  );
  console.log("View counters: 7 scenarios passed.");
})();
