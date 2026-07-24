/**
 * Recursive cellular Seed — Claude SparkCellularHero mechanics.
 * Only Seed mask + per-bar colors differ from the original.
 */

const BACKGROUND = "#F7F9FC";
const LIGHT_DOT = "#C7D5E6";
const DARK_DOT = "#5F7188";

// Four bars, left → right (deep navy → blue → cyan).
const BAR_PALETTES = [
  ["#115FB3", "#0A4F9C", "#06356E", "#052B5C"],
  ["#2B7AE8", "#1A6FE0", "#0E5FD4", "#3C8CFF"],
  ["#5BA0FF", "#3C8CFF", "#0095FD", "#4A94FF"],
  ["#7AF0E6", "#48C6EF", "#00CBD4", "#6FE8DF"],
];
const BAR_SOLIDS = ["#0A3A7A", "#1A6FE0", "#3C8CFF", "#00CBD4"];
const BAR_STROKES = ["#115FB3", "#1A6FE0", "#3C8CFF", "#48C6EF"];

const MASK_ROWS = [
  "0000000000000000000",
  "0000000000000000000",
  "0000000000000000000",
  "0110000000000001110",
  "0111000000000001110",
  "0111000000000001110",
  "0111000000000001110",
  "0111000000011001110",
  "0111000000111001110",
  "0111001110111001110",
  "0111001110111001110",
  "0111001110111001110",
  "0111001110111001110",
  "0111001110111001110",
  "0111001110000001110",
  "0110001110000001110",
  "0000000000000000000",
  "0000000000000000000",
  "0000000000000000000",
];

const START_X = 9;
const START_Y = 10;
const SCALE_STEP = Math.log(19 / 0.78);
const SWATCH_SIZE = 64;
const PHASE_TILE_SIZE = 96;
const SOLID_TILE_SIZE = 96;
const FULL_SOURCE_SIZE = 304;
const GLYPH_SOURCE_SIZE = 304;
const MIPMAP_MIN_SIZE = 19;
const MIPMAP_STOP_SIZE = 38;
const SWATCH_REFERENCE_SIZE = 128;
const SWATCH_REFERENCE_NOISE_COUNT = 220;
const SWATCH_NOISE_COUNT = Math.round(
  SWATCH_REFERENCE_NOISE_COUNT *
    (SWATCH_SIZE / SWATCH_REFERENCE_SIZE) *
    (SWATCH_SIZE / SWATCH_REFERENCE_SIZE),
);
const INITIALIZATION_IDLE_TIMEOUT = 1000;

const MASK = [];
const MASK_SET = new Set();
for (let y = 0; y < 19; y += 1) {
  for (let x = 0; x < 19; x += 1) {
    if (MASK_ROWS[y][x] !== "1") continue;
    const index = y * 19 + x;
    MASK.push({ index, x, y, distance: Math.hypot(x - START_X, y - START_Y) });
    MASK_SET.add(index);
  }
}
MASK.sort((a, b) => a.distance - b.distance || a.index - b.index);

function buildNeighbors(indexes) {
  const neighbors = new Map();
  for (const index of indexes) {
    const x = index % 19;
    const y = Math.floor(index / 19);
    const adjacent = [];
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const candidate = (y + dy) * 19 + x + dx;
      if (indexes.has(candidate)) adjacent.push(candidate);
    }
    neighbors.set(index, adjacent);
  }
  return neighbors;
}

const VISIBLE_NEIGHBORS = buildNeighbors(MASK_SET);

/** Assign every visible cell to a bar, left → right. */
function computeBars() {
  const visited = new Set();
  const groups = [];

  for (const { x: sx, y: sy, index: start } of MASK) {
    if (visited.has(start)) continue;
    const cells = [];
    const queue = [[sx, sy, start]];
    visited.add(start);
    while (queue.length) {
      const [x, y, index] = queue.pop();
      cells.push({ x, y, index });
      for (const neighbor of VISIBLE_NEIGHBORS.get(index) || []) {
        if (visited.has(neighbor)) continue;
        visited.add(neighbor);
        queue.push([neighbor % 19, (neighbor / 19) | 0, neighbor]);
      }
    }
    const cx = cells.reduce((sum, cell) => sum + cell.x, 0) / cells.length;
    groups.push({ cx, cells });
  }

  groups.sort((a, b) => a.cx - b.cx);
  const barOf = new Map();
  groups.forEach((group, bar) => {
    for (const cell of group.cells) barOf.set(cell.index, bar);
  });
  return barOf;
}

const BAR_OF = computeBars();

// Seed's four bars are disconnected. These invisible row-9 cells connect their
// simulation graph, allowing Claude's single center agent (index 180) to drive
// the whole icon without adding visible bridge pixels.
const CENTER_INDEX = 9 * 19 + 9;
const VIRTUAL_CONNECTORS = new Set([4, 5, 9, 13, 14].map((x) => 9 * 19 + x));
const GRAPH_SET = new Set([...MASK_SET, ...VIRTUAL_CONNECTORS]);
const NEIGHBORS = buildNeighbors(GRAPH_SET);
const GRAPH_SIZE = GRAPH_SET.size;

function mulberry32(seed) {
  return function random() {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let value = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function makeCanvas(size) {
  const element = document.createElement("canvas");
  element.width = size;
  element.height = size;
  return element;
}

function contextFor(canvas) {
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas 2D is unavailable");
  return context;
}

function makeMipmaps(source) {
  const mipmaps = [source];
  let current = source;
  while (current.width > MIPMAP_STOP_SIZE) {
    const size = Math.max(MIPMAP_MIN_SIZE, current.width >> 1);
    const next = makeCanvas(size);
    const context = contextFor(next);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(current, 0, 0, size, size);
    mipmaps.push(next);
    current = next;
  }
  return mipmaps;
}

function mipmapFor(mipmaps, targetSize) {
  for (let i = mipmaps.length - 1; i >= 0; i -= 1) {
    if (mipmaps[i].width >= targetSize) return mipmaps[i];
  }
  return mipmaps[0];
}

function roundedRect(context, x, y, width, height, radius) {
  context.beginPath();
  if (typeof context.roundRect === "function") {
    context.roundRect(x, y, width, height, radius);
    return;
  }

  const corner = Math.max(
    0,
    Math.min(Number(radius) || 0, Math.abs(width) / 2, Math.abs(height) / 2),
  );
  const right = x + width;
  const bottom = y + height;
  context.moveTo(x + corner, y);
  context.lineTo(right - corner, y);
  context.quadraticCurveTo(right, y, right, y + corner);
  context.lineTo(right, bottom - corner);
  context.quadraticCurveTo(right, bottom, right - corner, bottom);
  context.lineTo(x + corner, bottom);
  context.quadraticCurveTo(x, bottom, x, bottom - corner);
  context.lineTo(x, y + corner);
  context.quadraticCurveTo(x, y, x + corner, y);
  context.closePath();
}

function makeSwatch(seed, palette, stroke) {
  const random = mulberry32(seed * 9301 + 49297);
  const swatch = makeCanvas(SWATCH_SIZE);
  const context = contextFor(swatch);
  const pick = () => palette[Math.floor(random() * palette.length)];

  context.globalAlpha = 0.88;
  context.fillStyle = pick();
  context.fillRect(0, 0, SWATCH_SIZE, SWATCH_SIZE);

  for (let i = 0; i < 5; i += 1) {
    const x = SWATCH_SIZE * random();
    const y = SWATCH_SIZE * random();
    const radius = SWATCH_SIZE * (0.25 + 0.35 * random());
    const gradient = context.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, pick());
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    context.globalAlpha = 0.14 + 0.14 * random();
    context.fillStyle = gradient;
    context.fillRect(0, 0, SWATCH_SIZE, SWATCH_SIZE);
  }

  context.globalAlpha = 0.06;
  for (let i = 0; i < SWATCH_NOISE_COUNT; i += 1) {
    const size = SWATCH_SIZE * ((1 + 2 * random()) / SWATCH_REFERENCE_SIZE);
    context.fillStyle = pick();
    context.fillRect(
      SWATCH_SIZE * random(),
      SWATCH_SIZE * random(),
      size,
      size,
    );
  }

  context.globalAlpha = 0.14;
  context.strokeStyle = stroke;
  context.lineWidth = SWATCH_SIZE * 0.05;
  context.shadowColor = stroke;
  context.shadowBlur = SWATCH_SIZE * 0.1;
  roundedRect(
    context,
    SWATCH_SIZE * 0.07,
    SWATCH_SIZE * 0.07,
    SWATCH_SIZE * 0.86,
    SWATCH_SIZE * 0.86,
    SWATCH_SIZE * 0.2,
  );
  context.stroke();

  context.shadowBlur = 0;
  context.globalAlpha = 1;
  context.globalCompositeOperation = "destination-in";
  context.fillStyle = "#000";
  const a = random() * Math.PI * 2;
  const b = random() * Math.PI * 2;
  const c = random() * Math.PI * 2;
  context.beginPath();
  for (let i = 0; i <= 56; i += 1) {
    const t = (i / 56) * Math.PI * 2;
    const cos = Math.cos(t);
    const sin = Math.sin(t);
    const radius =
      (1 / Math.pow(cos ** 4 + sin ** 4, 0.25)) *
      (SWATCH_SIZE * 0.46) *
      (1 +
        0.03 * Math.sin(3 * t + a) +
        0.018 * Math.sin(7 * t + b) +
        0.01 * Math.sin(11 * t + c));
    const x = SWATCH_SIZE * 0.5 + cos * radius;
    const y = SWATCH_SIZE * 0.5 + sin * radius;
    if (i === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.closePath();
  context.fill();
  context.globalCompositeOperation = "source-over";
  return swatch;
}

function growthOrder(seed) {
  const random = mulberry32(seed * 0x9e3779b1);
  const occupied = new Set();
  const order = [];
  const frontier = [];

  const add = (x, y) => {
    const index = y * 19 + x;
    if (occupied.has(index)) return false;
    occupied.add(index);
    order.push({ x, y });
    return true;
  };

  const addFrontier = (x, y) => {
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const nx = x + dx;
      const ny = y + dy;
      if (
        nx >= 0 &&
        ny >= 0 &&
        nx < 19 &&
        ny < 19 &&
        !occupied.has(ny * 19 + nx)
      ) {
        frontier.push({ x: nx, y: ny });
      }
    }
  };

  add(9, 9);
  const walkers = 5 + Math.floor(random() * 3);
  for (let walker = 0; walker < walkers; walker += 1) {
    let x = 9;
    let y = 9;
    for (let step = 0; step < 38; step += 1) {
      const options = [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ].filter(([dx, dy]) => {
        const nx = x + dx;
        const ny = y + dy;
        return (
          nx >= 0 &&
          ny >= 0 &&
          nx < 19 &&
          ny < 19 &&
          !occupied.has(ny * 19 + nx)
        );
      });
      if (!options.length) break;
      let choice = options[0];
      let best = -Infinity;
      for (const option of options) {
        const score =
          Math.hypot(x + option[0] - 9, y + option[1] - 9) + 1.2 * random();
        if (score > best) {
          best = score;
          choice = option;
        }
      }
      x += choice[0];
      y += choice[1];
      add(x, y);
      if (x === 0 || y === 0 || x === 18 || y === 18) break;
    }
  }

  for (const cell of order) addFrontier(cell.x, cell.y);
  while (order.length < 361 && frontier.length) {
    const pick = Math.floor(random() * frontier.length);
    const cell = frontier[pick];
    frontier[pick] = frontier[frontier.length - 1];
    frontier.pop();
    if (!add(cell.x, cell.y)) continue;
    addFrontier(cell.x, cell.y);
  }
  return order;
}

function clipRounded(context, size) {
  context.globalCompositeOperation = "destination-in";
  context.fillStyle = "#000";
  roundedRect(context, 0, 0, size, size, size * 0.18);
  context.fill();
  context.globalCompositeOperation = "source-over";
}

function makeSolidTile(seed, solids) {
  const random = mulberry32(seed * 0x165667b1);
  const tile = makeCanvas(SOLID_TILE_SIZE);
  const context = contextFor(tile);
  roundedRect(
    context,
    0,
    0,
    SOLID_TILE_SIZE,
    SOLID_TILE_SIZE,
    SOLID_TILE_SIZE * 0.18,
  );
  context.fillStyle = solids[seed % solids.length];
  context.fill();

  context.save();
  roundedRect(
    context,
    0,
    0,
    SOLID_TILE_SIZE,
    SOLID_TILE_SIZE,
    SOLID_TILE_SIZE * 0.18,
  );
  context.clip();
  for (let i = 0; i < 4; i += 1) {
    const x = SOLID_TILE_SIZE * random();
    const y = SOLID_TILE_SIZE * random();
    const radius = SOLID_TILE_SIZE * (0.35 + 0.35 * random());
    const gradient = context.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, solids[Math.floor(random() * solids.length)]);
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    context.globalAlpha = 0.22;
    context.fillStyle = gradient;
    context.fillRect(0, 0, SOLID_TILE_SIZE, SOLID_TILE_SIZE);
  }
  context.restore();
  return tile;
}

function makeTileSet(seed, swatches, solids) {
  const order = growthOrder(seed);

  const paintSubcells = (context, size, count) => {
    const pitch = size / 19;
    const square = pitch * 0.78;
    const inset = (pitch - square) / 2;
    for (let i = 0; i < count; i += 1) {
      const { x, y } = order[i];
      const hash =
        ((x * 0x466f45d) ^ (y * 0x127409f) ^ (seed * 0x4f9ffb7)) >>> 0;
      const swatch = swatches[hash % 12];
      context.save();
      context.translate(
        x * pitch + inset + square / 2,
        y * pitch + inset + square / 2,
      );
      context.scale((hash >> 4) & 1 ? -1 : 1, (hash >> 5) & 1 ? -1 : 1);
      context.drawImage(swatch, -square / 2, -square / 2, square, square);
      context.restore();
    }
  };

  const phases = Array.from({ length: 8 }, (_, phase) => {
    const tile = makeCanvas(PHASE_TILE_SIZE);
    const context = contextFor(tile);
    paintSubcells(context, PHASE_TILE_SIZE, Math.ceil(((phase + 1) / 8) * 361));
    clipRounded(context, PHASE_TILE_SIZE);
    return tile;
  });

  const fullCanvas = makeCanvas(FULL_SOURCE_SIZE);
  const fullContext = contextFor(fullCanvas);
  paintSubcells(fullContext, FULL_SOURCE_SIZE, 361);
  clipRounded(fullContext, FULL_SOURCE_SIZE);

  return {
    phases,
    full: makeMipmaps(fullCanvas),
    solid: makeSolidTile(seed, solids),
  };
}

function makeTextureAtlas() {
  // One tile family per bar so L→R gradient reads clearly when filled.
  const barSwatches = BAR_PALETTES.map((palette, bar) =>
    Array.from({ length: 12 }, (_, i) =>
      makeSwatch(bar * 12 + i + 1, palette, BAR_STROKES[bar]),
    ),
  );
  const barTiles = BAR_PALETTES.map((palette, bar) =>
    Array.from({ length: 4 }, (_, i) =>
      makeTileSet(bar * 4 + i + 1, barSwatches[bar], [
        BAR_SOLIDS[bar],
        palette[0],
        palette[2],
        BAR_SOLIDS[bar],
      ]),
    ),
  );

  const tileFor = (index) => {
    const bar = BAR_OF.get(index) ?? 0;
    return barTiles[bar][index % barTiles[bar].length];
  };

  const glyph = makeCanvas(GLYPH_SOURCE_SIZE);
  const context = contextFor(glyph);
  const pitch = GLYPH_SOURCE_SIZE / 19;
  const square = pitch * 0.78;
  const inset = (pitch - square) / 2;
  for (const { index, x, y } of MASK) {
    context.drawImage(
      tileFor(index).solid,
      x * pitch + inset,
      y * pitch + inset,
      square,
      square,
    );
  }

  return { barTiles, agentGlyph: makeMipmaps(glyph) };
}

let textureAtlas = null;
let textureAtlasReferences = 0;

function getTextureAtlas() {
  if (textureAtlas) return textureAtlas;
  textureAtlas = makeTextureAtlas();
  return textureAtlas;
}

function retainTextureAtlas() {
  const atlas = getTextureAtlas();
  textureAtlasReferences += 1;
  return atlas;
}

function releaseTextureAtlas(atlas) {
  if (!atlas) return;
  textureAtlasReferences = Math.max(0, textureAtlasReferences - 1);
  if (textureAtlasReferences === 0 && textureAtlas === atlas) {
    textureAtlas = null;
  }
}

function tileFor(atlas, index) {
  const bar = BAR_OF.get(index) ?? 0;
  return atlas.barTiles[bar][index % atlas.barTiles[bar].length];
}

function freshState(now) {
  return {
    cells: new Map(),
    agents: [],
    lastTick: now,
    ticks: 0,
    doneAt: 0,
  };
}

function addAgent(simulation, now, instant, index) {
  simulation.agents.push({
    index,
    previous: index,
    since: now,
    bornAt: instant ? -Infinity : now,
    dieAt: 0,
  });
}

function seedAgent(simulation, now, instant = false) {
  addAgent(simulation, now, instant, CENTER_INDEX);
}

function frontierCells(simulation, reserved) {
  const result = [];
  const seen = new Set();
  for (const index of simulation.cells.keys()) {
    for (const neighbor of NEIGHBORS.get(index) || []) {
      if (
        !simulation.cells.has(neighbor) &&
        !reserved.has(neighbor) &&
        !seen.has(neighbor)
      ) {
        seen.add(neighbor);
        result.push(neighbor);
      }
    }
  }
  return result;
}

const NOOP_INSTANCE = Object.freeze({ destroy() {} });

export function mountSparkCellular(canvas) {
  if (!canvas || typeof canvas.getContext !== "function") return NOOP_INSTANCE;

  const root = document.documentElement;
  const container =
    typeof canvas.closest === "function"
      ? canvas.closest(".home-spark-cellular")
      : null;
  const toggle =
    container && typeof container.querySelector === "function"
      ? container.querySelector("[data-spark-cellular-toggle]")
      : null;
  const motionQuery =
    typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-reduced-motion: reduce)")
      : { matches: false };
  let reducedMotion = Boolean(motionQuery.matches);

  const setContainerState = (nextState) => {
    if (container?.dataset) container.dataset.sparkState = nextState;
  };
  const setToggleLabel = (paused) => {
    if (!toggle) return;
    const label = paused ? "Resume animation" : "Pause animation";
    toggle.setAttribute("aria-pressed", String(paused));
    toggle.setAttribute("aria-label", label);
    toggle.setAttribute("title", label);
  };

  if (container?.style) {
    container.style.setProperty(
      "--spark-band-width",
      `${document.documentElement.clientWidth}px`,
    );
  }
  setContainerState(reducedMotion ? "static" : "idle");
  if (toggle) {
    toggle.hidden = true;
    setToggleLabel(false);
  }

  if (typeof canvas.setAttribute === "function") {
    canvas.setAttribute("aria-hidden", "true");
    canvas.setAttribute("role", "presentation");
  }
  if (canvas.style) canvas.style.pointerEvents = "none";
  canvas.hidden = reducedMotion;

  let context;
  try {
    context = canvas.getContext("2d");
  } catch {
    setContainerState("fallback");
    container?.style?.removeProperty("--spark-band-width");
    canvas.hidden = true;
    return NOOP_INSTANCE;
  }
  if (!context) {
    setContainerState("fallback");
    container?.style?.removeProperty("--spark-band-width");
    canvas.hidden = true;
    return NOOP_INSTANCE;
  }

  let atlas = null;
  let width = 0;
  let height = 0;
  let ratio = 1;
  let centerX = 0;
  let centerY = 0;
  let baseSize = 0;
  let background = BACKGROUND;
  let dot = LIGHT_DOT;
  let state;
  let level = 0;
  let focus = 0;
  let focusVelocity = 0;
  let lastFrame = performance.now();
  let tickRandom = mulberry32(0xc1a0de);
  let frameId = null;
  let initializationHandle = null;
  let initializationHandleType = null;
  let resizeFrameId = null;
  let running = false;
  let intersecting = typeof IntersectionObserver !== "function";
  let initializedOnce = false;
  let initializationFailed = false;
  let pausedByUser = false;
  let pausedAt = null;
  let destroyed = false;
  let resizeObserver = null;
  let visibilityObserver = null;
  let themeObserver = null;

  function syncBandWidth() {
    if (!container?.style) return;
    container.style.setProperty(
      "--spark-band-width",
      `${document.documentElement.clientWidth}px`,
    );
  }

  function updateToggle() {
    if (!toggle) return;
    toggle.hidden = !atlas || reducedMotion || initializationFailed;
    setToggleLabel(pausedByUser);
  }

  function updateUiState() {
    if (reducedMotion) setContainerState("static");
    else if (initializationFailed) setContainerState("fallback");
    else if (!atlas) setContainerState("idle");
    else setContainerState(running ? "running" : "paused");
    updateToggle();
  }

  function updateThemeColors() {
    const styles = window.getComputedStyle(root);
    background =
      styles.getPropertyValue("--global-bg-color").trim() || BACKGROUND;
    dot = root.getAttribute("data-theme") === "dark" ? DARK_DOT : LIGHT_DOT;
  }

  function clearCanvas() {
    width = 0;
    height = 0;
    centerX = 0;
    centerY = 0;
    baseSize = 0;
    canvas.width = 0;
    canvas.height = 0;
  }

  function resize() {
    if (destroyed || reducedMotion || !atlas) return;

    ratio = Math.min(2, window.devicePixelRatio || 1);
    width = Math.max(0, Number(canvas.clientWidth) || 0);
    height = Math.max(0, Number(canvas.clientHeight) || 0);
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";

    centerX = 0.5 * width;
    centerY = 0.5 * height;
    baseSize = 0.66 * Math.min(0.92 * width, 0.9 * height);
  }

  function simulationTick(simulation, now) {
    while (now - simulation.lastTick >= 450 && !simulation.doneAt) {
      simulation.lastTick += 450;
      simulation.ticks += 1;
      const tickAt = simulation.lastTick;
      const reserved = new Set(
        simulation.agents
          .filter((agent) => !agent.dieAt)
          .map((agent) => agent.index),
      );
      const nextAgents = [];

      for (const agent of simulation.agents) {
        if (agent.dieAt) {
          if (tickAt - agent.dieAt < 350) nextAgents.push(agent);
          continue;
        }

        if (!simulation.cells.has(agent.index)) {
          simulation.cells.set(agent.index, tickAt);
        }
        const available = (NEIGHBORS.get(agent.index) || []).filter(
          (index) => !simulation.cells.has(index) && !reserved.has(index),
        );

        if (!available.length) {
          agent.dieAt = tickAt;
          nextAgents.push(agent);
          continue;
        }

        const next = available[Math.floor(tickRandom() * available.length)];
        agent.previous = agent.index;
        agent.index = next;
        agent.since = tickAt;
        reserved.add(next);
        nextAgents.push(agent);
      }

      simulation.agents = nextAgents;
      const activeCount = simulation.agents.filter(
        (agent) => !agent.dieAt,
      ).length;
      const availableFrontier = frontierCells(simulation, reserved);
      const rate = 7 / (1 + 0.25 * level);
      const desiredAgents = Math.min(
        48,
        Math.max(1, Math.floor((simulation.ticks / rate) ** 2)),
      );

      let count = activeCount;
      while (count < desiredAgents && availableFrontier.length) {
        const pick = Math.floor(tickRandom() * availableFrontier.length);
        const index = availableFrontier[pick];
        availableFrontier[pick] =
          availableFrontier[availableFrontier.length - 1];
        availableFrontier.pop();
        addAgent(simulation, tickAt, false, index);
        reserved.add(index);
        count += 1;
      }

      if (!availableFrontier.length && count === 0) simulation.doneAt = tickAt;
    }
  }

  function drawBackground() {
    if (destroyed || width <= 0 || height <= 0) return;

    context.globalAlpha = 1;
    context.globalCompositeOperation = "source-over";
    context.fillStyle = BACKGROUND;
    context.fillRect(0, 0, width, height);
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);
  }

  function drawDots(x, y, spacing) {
    if (spacing < 2 || (width / spacing + 2) * (height / spacing + 2) > 12000)
      return;
    const visibility = Math.max(0, Math.min(1, (spacing - 2) / 7));
    const opacity = (0.45 + 0.45 * Math.min(1, spacing / 90)) * visibility;
    if (opacity < 0.01) return;

    const startX = (((x + spacing / 2) % spacing) + spacing) % spacing;
    const startY = (((y + spacing / 2) % spacing) + spacing) % spacing;
    context.fillStyle = dot;
    context.globalAlpha = opacity;
    for (let py = startY; py <= height; py += spacing) {
      for (let px = startX; px <= width; px += spacing) {
        context.fillRect(px - 1, py - 1, 2, 2);
      }
    }
    context.globalAlpha = 1;
  }

  function cellPosition(index, scale) {
    const unit = scale / 19;
    return [
      centerX + 2 * unit * ((index % 19) - 9),
      centerY + 2 * unit * (Math.floor(index / 19) - 9),
    ];
  }

  function drawCell(index, bornAt, now, scale) {
    const unit = scale / 19;
    if (unit < 0.4 || !MASK_SET.has(index)) return;

    const [x, y] = cellPosition(index, scale);
    const size = 2 * unit * 0.78;
    const tile = tileFor(atlas, index);
    const age = now - bornAt;

    if (age < 700) {
      const phase = Math.min(7, Math.floor((age / 700) * 8));
      context.drawImage(
        tile.phases[phase],
        x - size / 2,
        y - size / 2,
        size,
        size,
      );
      return;
    }

    const blend = Math.min(1, (age - 700) / 1600);
    if (blend < 1) {
      context.drawImage(
        mipmapFor(tile.full, size * ratio),
        x - size / 2,
        y - size / 2,
        size,
        size,
      );
    }
    if (blend > 0) {
      context.globalAlpha = blend;
      context.drawImage(tile.solid, x - size / 2, y - size / 2, size, size);
      context.globalAlpha = 1;
    }
  }

  function drawAgent(agent, now, scale) {
    const unit = scale / 19;
    if (unit < 0.4) return;

    const progress = Math.min(1, (now - agent.since) / 450);
    const eased = 1 - (1 - progress) ** 2;
    const [fromX, fromY] = cellPosition(agent.previous, scale);
    const [toX, toY] = cellPosition(agent.index, scale);
    const x = fromX + (toX - fromX) * eased;
    const y = fromY + (toY - fromY) * eased;

    let alpha;
    if (agent.dieAt) alpha = (1 - Math.min(1, (now - agent.dieAt) / 350)) ** 2;
    else alpha = 1 - (1 - Math.min(1, (now - agent.bornAt) / 350)) ** 3;
    if (alpha <= 0) return;

    const agentUnit = unit * alpha;
    if (agentUnit < 0.4) return;
    const glyphSize = 2 * agentUnit * 0.78;
    context.drawImage(
      mipmapFor(atlas.agentGlyph, glyphSize * ratio),
      x - glyphSize / 2,
      y - glyphSize / 2,
      glyphSize,
      glyphSize,
    );
  }

  function drawFrame(now, scale) {
    if (destroyed) return;

    drawBackground();
    const outerSpacing = (scale / 19) * 2;
    drawDots(centerX, centerY, outerSpacing);
    drawDots(centerX, centerY, (0.78 * outerSpacing) / 19);

    if (!state) return;
    for (const [index, bornAt] of state.cells)
      drawCell(index, bornAt, now, scale);
    for (const agent of state.agents) drawAgent(agent, now, scale);
  }

  function currentScale() {
    return baseSize * Math.exp(level * SCALE_STEP - focus);
  }

  function redraw(now = pausedAt ?? performance.now()) {
    if (destroyed || !atlas) return;
    drawFrame(now, currentScale());
  }

  function restart(now) {
    level = 0;
    focus = -0.04 * SCALE_STEP;
    focusVelocity = 0;
    tickRandom = mulberry32((now | 0) ^ 0xc1a0de);
    state = freshState(now);
    seedAgent(state, now);
    lastFrame = now;
  }

  function shiftSimulationTimestamps(duration) {
    if (!state || duration <= 0) return;

    state.lastTick += duration;
    if (state.doneAt) state.doneAt += duration;
    for (const [index, bornAt] of state.cells) {
      state.cells.set(index, bornAt + duration);
    }
    for (const agent of state.agents) {
      agent.since += duration;
      agent.bornAt += duration;
      if (agent.dieAt) agent.dieAt += duration;
    }
  }

  /**
   * Claude's exact recursive cut: the full, centered agent glyph in the next
   * generation occupies the same pixels as the completed Seed in the prior one.
   */
  function advanceGeneration(now) {
    state = freshState(now);
    state.lastTick = now;
    seedAgent(state, now, true);
    level += 1;
    if (level >= 3) {
      level -= 3;
      focus -= 3 * SCALE_STEP;
    }
  }

  function update(now) {
    frameId = null;
    if (!running || destroyed) return;

    const delta = Math.min(50, now - lastFrame);
    lastFrame = now;

    simulationTick(state, now);

    if (state.doneAt && now - state.doneAt >= 500) {
      advanceGeneration(now);
    }

    const fraction = state.cells.size / GRAPH_SIZE;
    const offset = state.doneAt ? 1 : -0.04 + 0.34 * fraction;
    const target = (level + offset) * SCALE_STEP;

    const steps = Math.max(1, Math.ceil(delta / 12));
    const stepDelta = delta / steps;
    for (let i = 0; i < steps; i += 1) {
      const acceleration =
        0.00000081 * (target - focus) - 0.00153 * focusVelocity;
      focusVelocity += acceleration * stepDelta;
      focus += focusVelocity * stepDelta;
    }

    drawFrame(now, currentScale());
    frameId = window.requestAnimationFrame(update);
  }

  function shouldRun() {
    return (
      !destroyed &&
      Boolean(atlas) &&
      !reducedMotion &&
      !pausedByUser &&
      intersecting &&
      !document.hidden &&
      width > 0 &&
      height > 0
    );
  }

  function start() {
    if (running || !shouldRun()) return;
    const now = performance.now();

    if (!initializedOnce) {
      restart(now);
      initializedOnce = true;
    } else if (pausedAt !== null) {
      shiftSimulationTimestamps(Math.max(0, now - pausedAt));
      pausedAt = null;
      lastFrame = now;
    }

    running = true;
    frameId = window.requestAnimationFrame(update);
    updateUiState();
  }

  function stop() {
    if (running && initializedOnce && pausedAt === null) pausedAt = lastFrame;
    running = false;
    if (frameId !== null) {
      window.cancelAnimationFrame(frameId);
      frameId = null;
    }
  }

  function releaseAtlasReference() {
    if (!atlas) return;
    releaseTextureAtlas(atlas);
    atlas = null;
  }

  function cancelPendingInitialization() {
    if (initializationHandle === null) return;

    if (
      initializationHandleType === "idle" &&
      typeof window.cancelIdleCallback === "function"
    ) {
      window.cancelIdleCallback(initializationHandle);
    } else {
      window.clearTimeout(initializationHandle);
    }
    initializationHandle = null;
    initializationHandleType = null;
  }

  function canInitialize() {
    return (
      !destroyed &&
      !reducedMotion &&
      !initializationFailed &&
      !atlas &&
      intersecting &&
      !document.hidden
    );
  }

  function markAtlasFailure() {
    cancelPendingInitialization();
    stop();
    releaseAtlasReference();
    initializationFailed = true;
    initializedOnce = false;
    pausedAt = null;
    state = undefined;
    clearCanvas();
    canvas.hidden = true;
    updateUiState();
  }

  function initializeAtlas() {
    if (!canInitialize()) return;

    try {
      atlas = retainTextureAtlas();
    } catch {
      markAtlasFailure();
      return;
    }

    if (destroyed || reducedMotion) {
      releaseAtlasReference();
      return;
    }

    canvas.hidden = false;
    updateThemeColors();
    updateUiState();
    resize();
    drawBackground();
    syncPlayback();
  }

  function scheduleInitialization() {
    if (!canInitialize() || initializationHandle !== null) return;

    const runInitialization = () => {
      initializationHandle = null;
      initializationHandleType = null;
      initializeAtlas();
    };

    if (
      typeof window.requestIdleCallback === "function" &&
      typeof window.cancelIdleCallback === "function"
    ) {
      initializationHandleType = "idle";
      initializationHandle = window.requestIdleCallback(runInitialization, {
        timeout: INITIALIZATION_IDLE_TIMEOUT,
      });
    } else {
      initializationHandleType = "timeout";
      initializationHandle = window.setTimeout(runInitialization, 0);
    }
  }

  function enterReducedMotion() {
    cancelPendingInitialization();
    stop();
    releaseAtlasReference();
    initializationFailed = false;
    initializedOnce = false;
    pausedAt = null;
    state = undefined;
    clearCanvas();
    canvas.hidden = true;
    updateUiState();
  }

  function syncPlayback() {
    if (!atlas) {
      stop();
      scheduleInitialization();
      updateUiState();
      return;
    }

    if (shouldRun()) start();
    else stop();
    updateUiState();
  }

  function handleResize() {
    if (destroyed) return;
    updateThemeColors();
    if (atlas) {
      resize();
      redraw();
    }
    syncPlayback();
  }

  function handleWindowResize() {
    if (destroyed || resizeFrameId !== null) return;
    resizeFrameId = window.requestAnimationFrame(() => {
      resizeFrameId = null;
      syncBandWidth();
      if (!resizeObserver) handleResize();
    });
  }

  function handleThemeChange() {
    if (destroyed) return;
    updateThemeColors();
    redraw();
  }

  function handleVisibilityChange() {
    if (destroyed) return;
    if (document.hidden) cancelPendingInitialization();
    syncPlayback();
  }

  function handleMotionChange(event) {
    if (destroyed) return;
    const nextReducedMotion = Boolean(event.matches);
    if (nextReducedMotion === reducedMotion) return;

    reducedMotion = nextReducedMotion;
    if (reducedMotion) {
      enterReducedMotion();
      return;
    }

    initializationFailed = false;
    canvas.hidden = false;
    updateUiState();
    if (intersecting) syncPlayback();
  }

  function handleToggleClick() {
    if (destroyed || reducedMotion || !atlas) return;
    pausedByUser = !pausedByUser;
    updateToggle();
    syncPlayback();
  }

  updateThemeColors();
  clearCanvas();
  syncBandWidth();
  updateUiState();

  if (typeof ResizeObserver === "function") {
    resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(canvas);
  }

  if (typeof IntersectionObserver === "function") {
    visibilityObserver = new IntersectionObserver(
      ([entry]) => {
        if (destroyed) return;
        intersecting = Boolean(entry?.isIntersecting);
        if (!intersecting) cancelPendingInitialization();
        syncPlayback();
      },
      { threshold: 0.1 },
    );
    visibilityObserver.observe(canvas);
  } else {
    syncPlayback();
  }

  if (typeof MutationObserver === "function") {
    themeObserver = new MutationObserver(handleThemeChange);
    themeObserver.observe(root, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  }

  window.addEventListener("resize", handleWindowResize, { passive: true });
  document.addEventListener("visibilitychange", handleVisibilityChange);
  if (typeof motionQuery.addEventListener === "function") {
    motionQuery.addEventListener("change", handleMotionChange);
  } else if (typeof motionQuery.addListener === "function") {
    motionQuery.addListener(handleMotionChange);
  }
  if (toggle && typeof toggle.addEventListener === "function") {
    toggle.addEventListener("click", handleToggleClick);
  }

  return {
    destroy() {
      if (destroyed) return;
      cancelPendingInitialization();
      stop();
      destroyed = true;
      if (resizeFrameId !== null) {
        window.cancelAnimationFrame(resizeFrameId);
        resizeFrameId = null;
      }
      resizeObserver?.disconnect();
      visibilityObserver?.disconnect();
      themeObserver?.disconnect();
      window.removeEventListener("resize", handleWindowResize);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (typeof motionQuery.removeEventListener === "function") {
        motionQuery.removeEventListener("change", handleMotionChange);
      } else if (typeof motionQuery.removeListener === "function") {
        motionQuery.removeListener(handleMotionChange);
      }
      if (toggle && typeof toggle.removeEventListener === "function") {
        toggle.removeEventListener("click", handleToggleClick);
        toggle.hidden = true;
      }
      releaseAtlasReference();
      state = undefined;
      clearCanvas();
      canvas.hidden = true;
      setContainerState(reducedMotion ? "static" : "idle");
    },
  };
}

const AUTO_INSTANCES =
  typeof document === "undefined"
    ? []
    : Array.from(
        document.querySelectorAll("[data-spark-cellular-canvas]"),
        (canvas) => mountSparkCellular(canvas),
      );

if (typeof window !== "undefined" && AUTO_INSTANCES.length) {
  window.addEventListener("pagehide", (event) => {
    if (event.persisted) return;
    for (const instance of AUTO_INSTANCES) instance.destroy();
  });
}
