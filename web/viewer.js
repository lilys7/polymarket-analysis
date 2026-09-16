import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvas = document.getElementById("c");
const statusEl = document.getElementById("status");
const pickEl = document.getElementById("pick");
const legendEl = document.getElementById("legend");
const clusterListEl = document.getElementById("clusterList");
const clusterStatsEl = document.getElementById("clusterStats");
const colorModeEl = document.getElementById("colorMode");
const sizeEl = document.getElementById("size");
const resetEl = document.getElementById("reset");
const showAllEl = document.getElementById("showAll");
const hideAllEl = document.getElementById("hideAll");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x111318, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(2.4, 1.8, 2.6);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.target.set(0, 0, 0);
scene.add(new THREE.AxesHelper(1.2));

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.03;
const pointer = new THREE.Vector2();

let meta;
let arrays;
let viewPoints = null;
let viewIndex = new Uint32Array(0);

// empty selected + hideAll=false => original (everything on)
const selected = new Set();
let hideAll = false;

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function parseBin(buffer, meta) {
  const n = meta.n;
  let offset = 0;
  const f32 = () => {
    const view = new Float32Array(buffer, offset, n);
    offset += n * 4;
    return view;
  };
  const zx = f32();
  const zy = f32();
  const zz = f32();
  const xRaw = f32();
  const yRaw = f32();
  const zRaw = f32();
  const pnl = f32();
  const ppv = f32();
  const hhi = f32();
  const outlier = f32();
  const cluster = new Int16Array(buffer, offset, n);
  offset += n * 2;
  const label = new Uint8Array(buffer, offset, n);
  return { zx, zy, zz, xRaw, yRaw, zRaw, pnl, ppv, hhi, outlier, cluster, label };
}

function clusterKey(cid) {
  if (cid === meta.noise_label) return "noise";
  if (meta.top_cluster_ids.includes(cid)) return String(cid);
  return "other";
}

function clusterColor(clusterId) {
  if (clusterId === meta.noise_label) return hexToRgb(meta.noise_color);
  const idx = meta.top_cluster_ids.indexOf(clusterId);
  if (idx >= 0) return hexToRgb(meta.top_colors[idx]);
  return hexToRgb(meta.other_dense_color);
}

function lerpColor(a, b, t) {
  const u = Math.min(1, Math.max(0, t));
  return [
    a[0] + (b[0] - a[0]) * u,
    a[1] + (b[1] - a[1]) * u,
    a[2] + (b[2] - a[2]) * u,
  ];
}

function colorFor(i, mode) {
  if (mode === "cluster") return clusterColor(arrays.cluster[i]);
  if (mode === "label") {
    const name = meta.label_names[String(arrays.label[i])] || "unlabeled";
    return hexToRgb(meta.label_colors[name]);
  }
  if (mode === "hhi") return lerpColor([0.22, 0.28, 0.55], [0.95, 0.78, 0.22], arrays.hhi[i]);
  return lerpColor([0.85, 0.25, 0.25], [0.25, 0.75, 0.35], (arrays.ppv[i] + 0.4) / 0.8);
}

function isShowingAll() {
  return !hideAll && selected.size === 0;
}

function pointVisible(i) {
  if (isShowingAll()) return true;
  if (hideAll && selected.size === 0) return false;
  return selected.has(clusterKey(arrays.cluster[i]));
}

function collectVisible() {
  const ids = [];
  for (let i = 0; i < meta.n; i++) {
    if (pointVisible(i)) ids.push(i);
  }
  return ids;
}

function disposeView() {
  if (!viewPoints) return;
  scene.remove(viewPoints);
  viewPoints.geometry.dispose();
  viewPoints.material.dispose();
  viewPoints = null;
}

function rebuildView() {
  const ids = collectVisible();
  viewIndex = Uint32Array.from(ids);
  disposeView();
  const m = ids.length;
  const positions = new Float32Array(m * 3);
  const colors = new Float32Array(m * 3);
  const mode = colorModeEl.value;
  for (let k = 0; k < m; k++) {
    const i = ids[k];
    positions[k * 3] = arrays.zx[i];
    positions[k * 3 + 1] = arrays.zy[i];
    positions[k * 3 + 2] = arrays.zz[i];
    const rgb = colorFor(i, mode);
    colors[k * 3] = rgb[0];
    colors[k * 3 + 1] = rgb[1];
    colors[k * 3 + 2] = rgb[2];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.PointsMaterial({
    size: Number(sizeEl.value),
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.82,
    depthWrite: false,
  });
  viewPoints = new THREE.Points(geometry, mat);
  scene.add(viewPoints);
  updateStats(ids);
  updateClusterList();
  const n = ids.length;
  if (isShowingAll()) {
    statusEl.textContent = `${meta.n.toLocaleString()} traders · showing all`;
  } else {
    statusEl.textContent = `${n.toLocaleString()} traders visible`;
  }
}

function fillColors(mode) {
  if (!viewPoints) return;
  const colors = viewPoints.geometry.getAttribute("color");
  for (let k = 0; k < viewIndex.length; k++) {
    const rgb = colorFor(viewIndex[k], mode);
    colors.setXYZ(k, rgb[0], rgb[1], rgb[2]);
  }
  colors.needsUpdate = true;
}

function clusterItems() {
  return [
    { key: "noise", color: meta.noise_color, text: "noise (−1)" },
    { key: "other", color: meta.other_dense_color, text: "other dense" },
    ...meta.top_cluster_ids.map((id, i) => ({
      key: String(id),
      color: meta.top_colors[i],
      text: `cluster ${id}`,
    })),
  ];
}

function updateClusterList() {
  if (!clusterListEl.children.length) {
    clusterItems().forEach((item) => {
      const row = document.createElement("div");
      row.className = "swatch selectable";
      row.dataset.key = item.key;
      row.innerHTML = `<span class="dot" style="background:${item.color}"></span><span>${item.text}</span>`;
      row.addEventListener("click", () => toggleCluster(item.key));
      clusterListEl.appendChild(row);
    });
  }
  const showingAll = isShowingAll();
  clusterListEl.querySelectorAll(".swatch").forEach((row) => {
    const on = selected.has(row.dataset.key);
    row.classList.toggle("selected", on);
    row.classList.toggle("dim", !showingAll && !on);
  });
}

function toggleCluster(key) {
  hideAll = false;
  if (selected.has(key)) selected.delete(key);
  else selected.add(key);
  rebuildView();
}

function renderLegend(mode) {
  legendEl.innerHTML = "";
  const add = (color, text) => {
    const row = document.createElement("div");
    row.className = "swatch";
    row.innerHTML = `<span class="dot" style="background:${color}"></span><span>${text}</span>`;
    legendEl.appendChild(row);
  };
  if (mode === "cluster") return;
  if (mode === "label") {
    Object.entries(meta.label_colors).forEach(([k, v]) => add(v, k));
  } else if (mode === "hhi") {
    add("#38468c", "low HHI (generalist)");
    add("#f2c738", "high HHI (specialist)");
  } else {
    add("#d94040", "low / negative PPV");
    add("#40bf59", "high PPV");
  }
}

function fmt(x, d = 3) {
  if (!Number.isFinite(x)) return "—";
  return x.toLocaleString(undefined, { maximumFractionDigits: d });
}

function pct(part, n) {
  return n ? ((100 * part) / n).toFixed(1) : "0.0";
}

function updateStats(ids) {
  const n = ids.length;
  if (!n) {
    clusterStatsEl.innerHTML = "<b>Visible set</b><br/>no traders";
    return;
  }
  let pnl = 0;
  let ppv = 0;
  let hhi = 0;
  let sharp = 0;
  let awful = 0;
  for (let k = 0; k < n; k++) {
    const i = ids[k];
    pnl += arrays.pnl[i];
    ppv += arrays.ppv[i];
    hhi += arrays.hhi[i];
    if (arrays.label[i] === 3) sharp += 1;
    if (arrays.label[i] === 0) awful += 1;
  }
  clusterStatsEl.innerHTML = `
    <b>Visible set</b><br/>
    n: ${n.toLocaleString()}<br/>
    mean pnl: ${fmt(pnl / n, 2)}<br/>
    mean ppv: ${fmt(ppv / n, 4)}<br/>
    mean hhi: ${fmt(hhi / n, 3)}<br/>
    sharp: ${pct(sharp, n)}% · awful: ${pct(awful, n)}%
  `;
}

function showPick(i) {
  const label = meta.label_names[String(arrays.label[i])] || "unlabeled";
  const cid = arrays.cluster[i];
  const clusterName =
    cid === meta.noise_label
      ? "noise (−1)"
      : meta.top_cluster_ids.includes(cid)
        ? `manifold ${cid}`
        : `other dense (${cid})`;
  pickEl.innerHTML = `
    <b>Trader ${i.toLocaleString()}</b><br/>
    cluster: ${clusterName}<br/>
    label: ${label}<br/>
    pnl: ${fmt(arrays.pnl[i], 2)}<br/>
    ppv: ${fmt(arrays.ppv[i], 4)}<br/>
    hhi: ${fmt(arrays.hhi[i], 3)}<br/>
    book impact: ${fmt(arrays.xRaw[i], 3)}<br/>
    timing var: ${fmt(arrays.yRaw[i], 1)}<br/>
    log size: ${fmt(arrays.zRaw[i], 3)}
  `;
}

function onPointer(event) {
  if (!viewPoints || !viewIndex.length) return;
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(viewPoints);
  if (!hits.length) return;
  showPick(viewIndex[hits[0].index]);
}

async function load() {
  const metaRes = await fetch("./data/meta.json");
  if (!metaRes.ok) {
    statusEl.textContent = "meta.json missing. From the repo root run: python3 main.py --web";
    return;
  }
  meta = await metaRes.json();
  const binRes = await fetch("./data/points.bin");
  if (!binRes.ok) {
    statusEl.textContent = "points.bin missing. Run python3 main.py --web";
    return;
  }
  arrays = parseBin(await binRes.arrayBuffer(), meta);
  rebuildView();
  renderLegend(colorModeEl.value);
}

colorModeEl.addEventListener("change", () => {
  fillColors(colorModeEl.value);
  renderLegend(colorModeEl.value);
});
sizeEl.addEventListener("input", () => {
  if (viewPoints) viewPoints.material.size = Number(sizeEl.value);
});
resetEl.addEventListener("click", () => {
  camera.position.set(2.4, 1.8, 2.6);
  controls.target.set(0, 0, 0);
  controls.update();
});
showAllEl.addEventListener("click", () => {
  hideAll = false;
  selected.clear();
  rebuildView();
});
hideAllEl.addEventListener("click", () => {
  hideAll = true;
  selected.clear();
  rebuildView();
});
canvas.addEventListener("click", onPointer);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function tick() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

tick();
load().catch((err) => {
  statusEl.textContent = String(err);
});
