import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvas = document.getElementById("c");
const statusEl = document.getElementById("status");
const pickEl = document.getElementById("pick");
const legendEl = document.getElementById("legend");
const colorModeEl = document.getElementById("colorMode");
const hideNoiseEl = document.getElementById("hideNoise");
const sizeEl = document.getElementById("size");
const resetEl = document.getElementById("reset");

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
let densePoints;
let noisePoints;
let denseIndex;
let noiseIndex;
let denseColors;
let noiseColors;

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

function paint(attr, indexMap, mode) {
  for (let k = 0; k < indexMap.length; k++) {
    const rgb = colorFor(indexMap[k], mode);
    attr.setXYZ(k, rgb[0], rgb[1], rgb[2]);
  }
  attr.needsUpdate = true;
}

function fillColors(mode) {
  paint(denseColors, denseIndex, mode);
  paint(noiseColors, noiseIndex, mode);
}

function makeCloud(indexMap, colorAttr) {
  const m = indexMap.length;
  const positions = new Float32Array(m * 3);
  for (let k = 0; k < m; k++) {
    const i = indexMap[k];
    positions[k * 3] = arrays.zx[i];
    positions[k * 3 + 1] = arrays.zy[i];
    positions[k * 3 + 2] = arrays.zz[i];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", colorAttr);
  const mat = new THREE.PointsMaterial({
    size: Number(sizeEl.value),
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.82,
    depthWrite: false,
  });
  const cloud = new THREE.Points(geometry, mat);
  cloud.userData.indexMap = indexMap;
  scene.add(cloud);
  return cloud;
}

function renderLegend(mode) {
  legendEl.innerHTML = "";
  const add = (color, text) => {
    const row = document.createElement("div");
    row.className = "swatch";
    row.innerHTML = `<span class="dot" style="background:${color}"></span><span>${text}</span>`;
    legendEl.appendChild(row);
  };
  if (mode === "cluster") {
    add(meta.noise_color, "noise (−1)");
    add(meta.other_dense_color, "other dense");
    meta.top_cluster_ids.forEach((id, i) => add(meta.top_colors[i], `cluster ${id}`));
  } else if (mode === "label") {
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
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects([densePoints, noisePoints].filter(Boolean));
  if (!hits.length) return;
  const map = hits[0].object.userData.indexMap;
  showPick(map[hits[0].index]);
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

  const dense = [];
  const noise = [];
  for (let i = 0; i < meta.n; i++) {
    if (arrays.cluster[i] === meta.noise_label) noise.push(i);
    else dense.push(i);
  }
  denseIndex = Uint32Array.from(dense);
  noiseIndex = Uint32Array.from(noise);
  denseColors = new THREE.BufferAttribute(new Float32Array(denseIndex.length * 3), 3);
  noiseColors = new THREE.BufferAttribute(new Float32Array(noiseIndex.length * 3), 3);
  densePoints = makeCloud(denseIndex, denseColors);
  noisePoints = makeCloud(noiseIndex, noiseColors);

  fillColors(colorModeEl.value);
  noisePoints.visible = !hideNoiseEl.checked;
  renderLegend(colorModeEl.value);
  statusEl.textContent = `${meta.n.toLocaleString()} traders · drag to orbit`;
}

colorModeEl.addEventListener("change", () => {
  fillColors(colorModeEl.value);
  renderLegend(colorModeEl.value);
});
hideNoiseEl.addEventListener("change", () => {
  if (noisePoints) noisePoints.visible = !hideNoiseEl.checked;
});
sizeEl.addEventListener("input", () => {
  const s = Number(sizeEl.value);
  if (densePoints) densePoints.material.size = s;
  if (noisePoints) noisePoints.material.size = s;
});
resetEl.addEventListener("click", () => {
  camera.position.set(2.4, 1.8, 2.6);
  controls.target.set(0, 0, 0);
  controls.update();
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
