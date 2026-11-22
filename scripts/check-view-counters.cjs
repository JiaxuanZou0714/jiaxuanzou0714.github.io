const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const code = fs.readFileSync(
  "assets/js/goatcounter-analytics-setup.js",
  "utf8",
);
async function run(pairs, responses, storage = new Map()) {
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
    sessionStorage: {
      getItem(key) {
        return storage.get(key) ?? null;
      },
      setItem(key, value) {
        storage.set(key, value);
      },
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
  const storage = new Map();
  const pairs = [["/zh/", "/en/"]];
  await run(pairs, { "/zh/": "26", "/en/": "74" }, storage);
  const cached = await run(pairs, {}, storage);
  assert.deepEqual(cached.values, ["100"]);
  assert.equal(
    cached.calls.length,
    0,
    "Fresh counts should be reused across pages",
  );

  for (const [key, value] of storage) {
    const entry = JSON.parse(value);
    entry.savedAt -= 6 * 60 * 1000;
    storage.set(key, JSON.stringify(entry));
  }
  const expired = await run(pairs, { "/zh/": "27", "/en/": "75" }, storage);
  assert.deepEqual(expired.values, ["102"]);
  assert.equal(expired.calls.length, 2);

  storage.clear();
  await run(pairs, { "/zh/": "26", "/en/": 500 }, storage);
  const retry = await run(pairs, { "/en/": "74" }, storage);
  assert.deepEqual(retry.values, ["100"]);
  assert.deepEqual(retry.calls, ["/en/"], "Failed requests must not be cached");

  for (const invalid of [
    "broken JSON",
    '{"count":-1,"savedAt":0}',
    JSON.stringify({ count: 42, savedAt: Date.now() + 60000 }),
  ]) {
    const badStorage = new Map([
      ["goatcounter:https://example.goatcounter.com//zh/", invalid],
    ]);
    assert.deepEqual(
      (await run([["/zh/", null]], { "/zh/": "26" }, badStorage)).values,
      ["26"],
    );
  }

  const blockedStorage = {
    get() {
      throw new Error("Storage blocked");
    },
    set() {
      throw new Error("Storage blocked");
    },
  };
  assert.deepEqual(
    (await run(pairs, { "/zh/": "26", "/en/": "74" }, blockedStorage)).values,
    ["100"],
  );
  console.log("View counters: 14 scenarios passed.");
})();
