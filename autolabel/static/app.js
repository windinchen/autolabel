"use strict";

const state = {
  ds: null, list: [], idx: -1, item: null,
  selObj: -1, selKpt: -1,
  view: { scale: 1, ox: 0, oy: 0 },
  img: null, dirty: false, saveTimer: null, addMode: false,
  drag: null, spaceDown: false,
  batchFilter: null,
  focus: false, todo: [], selTodo: -1,
};
const batchState = { list: [], selected: new Set(), loading: false };
// 本页只服务姿态库：未分组虚拟行的 ID 带库名，提示词库的批次不在这页
const POSE_LIB = "pose";
const POSE_UNGROUPED = "__ungrouped__:pose";
// 低于这个置信度的点进「待办清单」，和 QC 的 low_conf 口径一致
const LOWCONF = 0.4;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const imgCache = new Map();
// 连续翻页时后发的请求可能先回来：只有最新一次的加载结果才允许写 state.item
let loadSeq = 0;

// --------------------------------------------------------------------------- //
// 初始化
// --------------------------------------------------------------------------- //
async function boot() {
  let ds;
  try {
    ds = await getJson("/api/dataset");
  } catch (err) {
    $("hint").textContent = "数据集读不出来：" + err.message;
    toast("数据集读不出来：" + err.message, true);
    return;
  }
  state.ds = ds;
  if (state.ds.batches) batchState.list = poseBatches(state.ds.batches);
  renderModelOptions(state.ds.models);
  setFlagPolicy(state.ds);
  setHeaderCount(state.ds);
  document.title = "姿态标注 · " + state.ds.name;
  rebuildList();
  if (state.list.length) selectIndex(0);
  $("filter").onchange = () => { rebuildList(); };
  $("sort").onchange = () => { rebuildList(); };
  $("show-hidden").onchange = draw;
  $("show-box").onchange = draw;
  $("btn-approve").onclick = () => setStatus("approved");
  $("btn-reject").onclick = () => setStatus("rejected");
  $("btn-next").onclick = () => selectIndex(state.idx + 1);
  $("btn-prev").onclick = () => selectIndex(state.idx - 1);
  $("btn-flagged").onclick = () => jumpFlagged(1);
  $("btn-undo").onclick = undo;
  $("btn-add").onclick = toggleAdd;
  window.addEventListener("keydown", onKeyDown);
  initNav();
  initShell();
  initUpload();
  initBatches();
  initExports();
  window.addEventListener("keyup", (e) => { if (e.code === "Space") state.spaceDown = false; });
  window.addEventListener("resize", () => { resizeCanvas(); draw(); });
  resizeCanvas();
  draw();
  // 门户页的「去导入素材」会带 ?upload=1 过来，直接把上传弹层摊开
  if (new URLSearchParams(location.search).get("upload") === "1") {
    openUpload(true);
    history.replaceState(null, "", "/pose");
  }
}

/** 顶栏只报姿态库张数，顺带说明素材池里还有多少能直接「从素材池添加」。 */
function setHeaderCount(ds) {
  renderExportNote("export-note", ds, "pose");
  const mine = (ds.pose_items || ds.items || []).length;
  const pool = ds.pool || {};
  const spare = Math.max(0, (pool.files || 0) - (pool.pose || 0));
  const need = (ds.items || []).filter(needsAttention).length;
  $("ds-name").textContent = " · " + ds.name;
  $("capsule-set").textContent = `${ds.name} · 姿态库 ${mine} 帧`;
  $("capsule-count").textContent = need ? `待确认 ${need}` : "全部干净";
  $("queue-capsule").title = "焦点队列：模型自己觉得没把握的帧"
    + (spare ? `；素材池另有 ${spare} 张可直接「从素材池添加」` : "");
}

/** 模型自己标了警告、或整体置信度偏低，就算「需要人看一眼」。
 * 「两个模型结论一致」是加分项，不算问题——它以前也被算进红点和「待确认 N 处」。 */
function needsAttention(it) {
  return issueFlags(it).length > 0
    || (it.score !== null && it.score !== undefined && it.score < flagPolicy.lowScore);
}

function rebuildList() {
  const filter = $("filter").value, sort = $("sort").value;
  const current = state.item ? state.item.image : null;
  let items = state.ds.items.slice();
  if (state.batchFilter) items = items.filter((i) => (i.batch_id || POSE_UNGROUPED) === state.batchFilter);
  const statusOf = (i) => i.status || "unreviewed";
  if (filter === "flagged") items = items.filter(needsAttention);
  else if (filter === "all") { /* 全部：不筛 */ }
  else if (filter === "pending" || filter === "unreviewed") {
    // 「待复核」= 还没定论的，包含改过但没接受的，否则改完的帧会从队列里消失
    items = items.filter((i) => !["approved", "rejected"].includes(statusOf(i)));
  } else items = items.filter((i) => statusOf(i) === filter);
  items.sort(sort === "name"
    ? (a, b) => a.image.localeCompare(b.image)
    : (a, b) => (b.priority - a.priority) || a.image.localeCompare(b.image));
  state.list = items;
  const emptyBox = $("empty-state");
  emptyBox.textContent = state.ds.items.length
    ? "当前筛选条件下没有图片，试试切换右上角筛选或取消批次分组。"
    : "这个数据集还没有图片：把文件拖进本页、点「导入素材」，或用命令行 autolabel import 导入后即可开始复核。";
  emptyBox.classList.toggle("hidden", items.length > 0);
  if (!items.length) {
    state.item = null;
    state.idx = -1;
    renderProgress();
    renderFilmstrip();
    draw();
    return;
  }
  const pos = items.findIndex((i) => i.image === current);
  selectIndex(pos >= 0 ? pos : 0, true);
  renderProgress();
  renderFilmstrip();
}

function renderProgress() {
  const counts = { approved: 0, rejected: 0, edited: 0, unreviewed: 0 };
  state.ds.items.forEach((i) => { counts[i.status || "unreviewed"] = (counts[i.status || "unreviewed"] || 0) + 1; });
  const done = counts.approved + counts.rejected + counts.edited;
  const pct = state.ds.items.length ? (done / state.ds.items.length) * 100 : 0;
  $("progress-bar").style.width = pct.toFixed(1) + "%";
  $("progress-track").classList.toggle("on", pct > 0);  // 一张都没处理时别在顶栏挂条灰线
  $("progress-text").textContent =
    `已处理 ${done} / ${state.ds.items.length} · 接受 ${counts.approved} · 驳回 ${counts.rejected}`
    + ` · 修改 ${counts.edited} · 待确认 ${state.ds.items.filter(needsAttention).length}`;
  $("queue-count").textContent = state.list.length + " 帧";
  renderHudPos();
}

function renderHudPos() {
  const host = $("hud-pos");
  if (!host) return;
  host.textContent = "";
  host.append("第 ");
  const b = document.createElement("b");
  b.textContent = String(state.idx >= 0 ? state.idx + 1 : 0);
  host.append(b, ` / ${state.list.length}`);
}

function renderFilmstrip() {
  const strip = $("filmstrip");
  strip.innerHTML = "";
  const from = Math.max(0, state.idx - 60), to = Math.min(state.list.length, state.idx + 60);
  for (let i = from; i < to; i++) {
    const it = state.list[i];
    const div = document.createElement("div");
    const issues = issueFlags(it);
    const pluses = positiveFlags(it);
    div.className = "thumb" + (i === state.idx ? " active" : "")
      + (issues.length ? " flagged" : "") + (it.status === "approved" ? " approved" : "")
      + (it.status === "rejected" ? " rejected" : "");
    div.title = it.image + "\n优先级 " + it.priority + "\n"
      + (issues.length ? issues.map(flagLabel).join("、") : "没有要处理的标记")
      + (pluses.length ? "\n加分项：" + pluses.map(flagLabel).join("、") : "");
    const img = document.createElement("img");
    img.src = "/thumb/" + encodeURIComponent(it.image);
    img.loading = "lazy";
    div.appendChild(img);
    const meta = document.createElement("span");
    meta.className = "thumb-meta";
    const fn = document.createElement("span");
    fn.className = "fn";
    fn.textContent = it.image.split("/").pop();
    meta.appendChild(fn);
    const st = document.createElement("span");
    st.className = "st";
    st.textContent = statusLabel(it.status);
    if (issueFlags(it).length) {
      const em = document.createElement("em");
      em.textContent = issueFlags(it).length + " 项标记";
      st.appendChild(document.createTextNode(" · "));
      st.appendChild(em);
    } else if (it.objects) {
      st.textContent += " · " + it.objects + " 人";
    }
    meta.appendChild(st);
    div.appendChild(meta);
    div.onclick = () => selectIndex(i);
    strip.appendChild(div);
  }
  const active = strip.querySelector(".thumb.active");
  if (active) active.scrollIntoView({ inline: "center", block: "nearest" });
}

// --------------------------------------------------------------------------- //
// 载入与绘制
// --------------------------------------------------------------------------- //
async function selectIndex(i, keep = false) {
  if (!state.list.length) { state.item = null; renderPanel2(); return; }
  if (state.dirty) await save(true);
  i = clamp(i, 0, state.list.length - 1);
  const seq = ++loadSeq;
  state.idx = i;
  const meta = state.list[i];
  let item;
  try {
    item = await getJson("/api/item?image=" + encodeURIComponent(meta.image));
  } catch (err) {
    if (seq !== loadSeq) return;
    // 最常见的是图在另一个标签页被删了：说清楚，别留一张白画布让人以为软件卡死
    state.item = null;
    state.selObj = -1;
    state.selKpt = -1;
    $("hint").textContent = "读不到这张图的标注：" + err.message;
    toast("读不到 " + meta.image + "：" + err.message, true);
    renderPanel();
    renderFilmstrip();
    draw();
    return;
  }
  if (seq !== loadSeq) return;
  state.item = item;
  state.selObj = state.item.objects.length ? 0 : -1;
  state.selKpt = -1;
  await loadImage(meta.image);
  if (seq !== loadSeq) return;
  fitView();
  draw();
  renderPanel();
  renderFilmstrip();
  renderProgress();
  $("hint").textContent = meta.image + (state.item.time_sec != null ? ` · t=${state.item.time_sec}s` : "");
}

function loadImage(rel) {
  return new Promise((resolve) => {
    const url = "/image/" + encodeURIComponent(rel);
    if (imgCache.has(url)) { state.img = imgCache.get(url); resolve(); return; }
    const img = new Image();
    img.onload = () => {
      if (imgCache.size > 12) imgCache.delete(imgCache.keys().next().value);
      imgCache.set(url, img);
      state.img = img;
      resolve();
    };
    img.onerror = () => { state.img = null; resolve(); };
    img.src = url;
  });
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  state.cw = rect.width;
  state.ch = rect.height;
}

function fitView() {
  if (!state.img) return;
  const pad = 24;
  const scale = Math.min((state.cw - pad) / state.img.width, (state.ch - pad) / state.img.height);
  state.view.scale = scale;
  state.view.ox = (state.cw - state.img.width * scale) / 2;
  state.view.oy = (state.ch - state.img.height * scale) / 2;
}

const toScreen = (x, y) => [x * state.view.scale + state.view.ox, y * state.view.scale + state.view.oy];
const toWorld = (x, y) => [(x - state.view.ox) / state.view.scale, (y - state.view.oy) / state.view.scale];

function draw() {
  if (!state.cw) resizeCanvas();
  ctx.clearRect(0, 0, state.cw, state.ch);
  if (!state.img || !state.item) return;
  const { scale, ox, oy } = state.view;
  ctx.imageSmoothingEnabled = scale < 3;
  ctx.drawImage(state.img, ox, oy, state.img.width * scale, state.img.height * scale);
  const showHidden = $("show-hidden").checked, showBox = $("show-box").checked;
  const focusOn = state.focus && state.selObj >= 0;
  state.item.objects.forEach((obj, oi) => {
    const selected = oi === state.selObj;
    const color = objColor(obj, selected);
    ctx.globalAlpha = focusOn && !selected ? 0.16 : 1;
    const w = state.item.width || state.img.width, h = state.item.height || state.img.height;
    if (showBox) {
      const x1 = (obj.cx - obj.w / 2) * w, y1 = (obj.cy - obj.h / 2) * h;
      const bw = obj.w * w, bh = obj.h * h;
      const [sx, sy] = toScreen(x1, y1);
      ctx.lineWidth = selected ? 2 : 1.2;
      ctx.strokeStyle = color;
      if (state.focus && selected) drawCorners(sx, sy, bw * scale, bh * scale);
      else ctx.strokeRect(sx, sy, bw * scale, bh * scale);
      ctx.font = cfont(11);
      const vis = obj.keypoints.filter((k) => k.v !== 0).length;
      const label = `#${oi} k${vis}/${obj.keypoints.length}` +
        (obj.score ? ` ${(obj.score || 0).toFixed(2)}` : "");
      const tw = ctx.measureText(label).width + 6;
      const labelH = Math.round(14 * canvasPalette().scale);
      ctx.fillStyle = canvasPalette().veil;
      ctx.fillRect(sx, sy - labelH, tw, labelH);
      ctx.fillStyle = photoLabelInk(color);
      ctx.fillText(label, sx + 3, sy - 4);
    }
    ctx.lineWidth = selected ? 2.4 : 1.6;
    ctx.strokeStyle = color;
    (state.ds.skeleton || []).forEach(([a, b]) => {
      const ka = obj.keypoints[a], kb = obj.keypoints[b];
      if (!ka || !kb || ka.v === 0 || kb.v === 0) return;
      const [x1, y1] = toScreen(ka.x * w, ka.y * h);
      const [x2, y2] = toScreen(kb.x * w, kb.y * h);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    });
    obj.keypoints.forEach((k, ki) => {
      if (k.v === 0 && !showHidden) return;
      const [sx, sy] = toScreen(k.x * w, k.y * h);
      const sel = selected && ki === state.selKpt;
      const r = sel ? 7 : k.v === 2 ? 4.5 : 3.5;
      ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = k.v === 2 ? color : k.v === 1 ? canvasPalette().amber : canvasPalette().keyHidden;
      ctx.fill();
      ctx.lineWidth = sel ? 2.5 : 1;
      ctx.strokeStyle = sel ? canvasPalette().ink : canvasPalette().ringIdle;
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  });
  drawTodoBadges();
  if (state.drag && state.drag.type === "box" && state.drag.rect) {
    const [x1, y1] = state.drag.rect, [x2, y2] = state.drag.end;
    ctx.setLineDash([5, 4]); ctx.strokeStyle = rgbaOf(canvasPalette().amber, .9); ctx.lineWidth = 1.5;
    ctx.strokeRect(...toScreen(Math.min(x1, x2), Math.min(y1, y2)), (x2 - x1) * scale, (y2 - y1) * scale);
    ctx.setLineDash([]);
  }
}

function objColor(obj, selected) {
  const p = canvasPalette();
  const base = (obj.flags && obj.flags.length) ? p.coral : p.amber;
  return selected ? p.sky : base;
}

/** 标签压在深色底板上，字要用压得住的亮色：描边跟着主题，底板上的字两种主题都偏亮。 */
function photoLabelInk(color) {
  const p = canvasPalette();
  if (color === p.sky) return p.photoSky;
  if (color === p.coral) return p.photoCoral;
  return p.photoAmber;
}

// --------------------------------------------------------------------------- //
// 侧栏
// --------------------------------------------------------------------------- //
function renderPanel2() {
  $("meta").textContent = "";
  $("flags").innerHTML = "";
  $("objects").innerHTML = "";
  $("obj-count").textContent = "";
  state.todo = []; state.selTodo = -1;
  if ($("todo")) { $("todo").innerHTML = ""; $("todo-count").textContent = ""; }
  if ($("hud-segs")) { $("hud-segs").innerHTML = ""; $("hud-flag").textContent = "0 处待确认"; }
}

function renderPanel() {
  const it = state.item;
  if (!it) { renderPanel2(); return; }
  const meta = $("meta");
  meta.textContent = "";
  const lines = [["文件", it.image], ["尺寸", `${it.width} × ${it.height}`], ["状态", statusLabel(it.status)]];
  if (it.time_sec !== null && it.time_sec !== undefined) lines.push(["时间点", it.time_sec + " s"]);
  for (const [key, value] of lines) {
    const row = document.createElement("div");
    row.append(key + " ");
    const b = document.createElement("b");
    b.textContent = value;
    row.appendChild(b);
    meta.appendChild(row);
  }
  renderTodo();
  const flags = $("flags");
  flags.innerHTML = "";
  // 要处理的排前面，加分项排后面并染成绿色，免得把"两个模型一致"当成待办
  const flagRows = issueFlags(it).map((f) => [f, false])
    .concat(positiveFlags(it).map((f) => [f, true]));
  if (!flagRows.length) {
    const div = document.createElement("div");
    div.className = "flag ok";
    div.innerHTML = '<b>没有要处理的标记</b><span class="hint">这一帧可以直接 Q 接受</span>';
    flags.appendChild(div);
  }
  flagRows.forEach(([f, plus]) => {
    const div = document.createElement("div");
    div.className = "flag" + (plus ? " ok" : "");
    div.innerHTML = `<b>${flagLabel(f)}</b><span class="hint">${(state.ds.flag_hints || {})[f] || "人工确认"}</span>`;
    div.title = f + (plus ? "（加分项，不占待办）" : "");
    flags.appendChild(div);
  });
  $("obj-count").textContent = ` · ${it.objects.length} 人`;
  const box = $("objects");
  box.innerHTML = "";
  it.objects.forEach((obj, oi) => {
    const wrap = document.createElement("div");
    const selected = oi === state.selObj;
    wrap.className = "obj" + (selected ? " selected" : "");
    const visible = obj.keypoints.filter((k) => k.v !== 0).length;
    const head = document.createElement("div");
    head.className = "obj-head";
    head.innerHTML = `<span class="name">#${oi} ${(state.ds.class_names || {})["" + obj.cls] || "person"}</span>
      <span>${obj.score ? `<span class="chip">${obj.score.toFixed(2)}</span> ` : ""}
      <span class="chip ${visible < 5 ? "warn" : ""}">${visible}/${obj.keypoints.length}</span></span>`;
    head.onclick = () => { state.selObj = oi; state.selKpt = -1; renderPanel(); draw(); };
    wrap.appendChild(head);
    if (!selected) {
      // 七个人 × 十七个点全摊开就没法看了：未选中的只留一行摘要，点一下才展开
      const sub = document.createElement("div");
      sub.className = "obj-sub";
      const scored = obj.keypoints.filter((k) => k.conf != null);
      const avg = scored.length
        ? (scored.reduce((sum, k) => sum + k.conf, 0) / scored.length).toFixed(2) : "—";
      const hits = (state.todo || []).filter((row) => row.oi === oi).length;
      sub.innerHTML = `均置信 ${avg} · 点开展开 ${obj.keypoints.length} 个关键点`
        + (hits ? ` · <em>${hits} 处待办</em>` : "");
      wrap.appendChild(sub);
    } else {
      const kpts = document.createElement("div");
      kpts.className = "kpts";
      obj.keypoints.forEach((k, ki) => {
        const row = document.createElement("div");
        row.className = "kpt-row" + (ki === state.selKpt ? " selected" : "");
        row.innerHTML = `<span>${state.ds.keypoint_names[ki] || ki}</span>
          <span class="muted">${k.conf != null ? k.conf.toFixed(2) : "-"}</span>
          <span class="vbtn v${k.v}">${k.v}</span>`;
        row.onclick = (e) => {
          if (e.target.classList.contains("vbtn")) {
            k.v = (k.v + 1) % 3;
            markDirty();
          }
          state.selObj = oi; state.selKpt = ki;
          renderPanel(); draw();
        };
        kpts.appendChild(row);
      });
      wrap.appendChild(kpts);
    }
    box.appendChild(wrap);
  });
}

// --------------------------------------------------------------------------- //
// 交互
// --------------------------------------------------------------------------- //
function hitTest(mx, my) {
  const it = state.item;
  if (!it) return null;
  const w = it.width || (state.img && state.img.width) || 1;
  const h = it.height || (state.img && state.img.height) || 1;
  const tol = 9 / state.view.scale;
  let best = null, bestD = tol;
  it.objects.forEach((obj, oi) => {
    obj.keypoints.forEach((k, ki) => {
      if (k.v === 0 && !$("show-hidden").checked) return;
      const d = Math.hypot(k.x * w - mx, k.y * h - my);
      if (d < bestD) { bestD = d; best = { oi, ki }; }
    });
  });
  return best;
}

canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const [wx, wy] = toWorld(mx, my);
  if (e.button === 1 || state.spaceDown) { state.drag = { type: "pan", mx, my }; return; }
  if (state.addMode) { state.drag = { type: "box", rect: [wx, wy], end: [wx, wy] }; return; }
  const hit = hitTest(wx, wy);
  if (hit) {
    state.selObj = hit.oi; state.selKpt = hit.ki;
    state.drag = { type: "kpt", oi: hit.oi, ki: hit.ki };
    renderPanel(); draw(); return;
  }
  const it = state.item;
  if (it) {
    const w = it.width, h = it.height;
    for (let oi = it.objects.length - 1; oi >= 0; oi--) {
      const o = it.objects[oi];
      const x1 = (o.cx - o.w / 2) * w, y1 = (o.cy - o.h / 2) * h;
      if (wx >= x1 && wx <= x1 + o.w * w && wy >= y1 && wy <= y1 + o.h * h) {
        state.selObj = oi; state.selKpt = -1; renderPanel(); draw(); return;
      }
    }
  }
  state.drag = { type: "pan", mx, my };
});

window.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (!state.drag) return;
  const d = state.drag;
  if (d.type === "pan") {
    state.view.ox += mx - d.mx; state.view.oy += my - d.my;
    d.mx = mx; d.my = my; draw();
  } else if (d.type === "kpt") {
    const [wx, wy] = toWorld(mx, my);
    const it = state.item, o = it.objects[d.oi];
    if (!o) return;
    const k = o.keypoints[d.ki];
    k.x = clamp(wx / (it.width || 1), 0, 1);
    k.y = clamp(wy / (it.height || 1), 0, 1);
    if (k.v === 0) k.v = 2;
    draw();
    markDirty();
  } else if (d.type === "box") {
    d.end = toWorld(mx, my); draw();
  }
});

window.addEventListener("mouseup", () => {
  const d = state.drag;
  state.drag = null;
  if (!d || d.type !== "box") return;
  const [x1, y1] = d.rect, [x2, y2] = d.end;
  const w = state.item.width, h = state.item.height;
  const bw = Math.abs(x2 - x1), bh = Math.abs(y2 - y1);
  if (bw < 4 || bh < 4) { draw(); return; }
  const n = state.ds.num_keypoints;
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  state.item.objects.push({
    cls: 0, cx: cx / w, cy: cy / h, w: bw / w, h: bh / h, score: 1, flags: ["manual"],
    keypoints: Array.from({ length: n }, () => ({ x: cx / w, y: cy / h, v: 0, conf: 0 })),
  });
  state.selObj = state.item.objects.length - 1;
  state.selKpt = -1;
  state.addMode = false;
  $("btn-add").classList.remove("active");
  markDirty(); renderPanel(); draw();
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const [wx, wy] = toWorld(mx, my);
  state.view.scale = clamp(state.view.scale * factor, 0.05, 40);
  state.view.ox = mx - wx * state.view.scale;
  state.view.oy = my - wy * state.view.scale;
  draw();
}, { passive: false });

function onKeyDown(e) {
  const confirm = $("confirm-modal");
  if (confirm && !confirm.classList.contains("hidden")) {
    if (e.key === "Escape") { e.preventDefault(); closeConfirm(false); }
    else if (e.key === "Enter") { e.preventDefault(); closeConfirm(true); }
    return;
  }
  const pooled = $("pool-modal");
  if (pooled && !pooled.classList.contains("hidden")) {
    if (e.key === "Escape") openPool(false);
    return;
  }
  if (shellKeys(e)) return;
  const modal = $("upload-modal");
  if (modal && !modal.classList.contains("hidden")) {
    if (e.key === "Escape") openUpload(false);
    return;
  }
  const batchModal = $("batch-modal");
  if (batchModal && !batchModal.classList.contains("hidden")) {
    if (e.key === "Escape") openBatches(false);
    return;
  }
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  if (e.metaKey || e.ctrlKey) {
    // ⌘/Ctrl 加字母是浏览器与系统的（全选、收藏、页内查找、下载、书签栏、切标签页…），
    // 以前这些组合会顺手翻页、改关键点可见性、甚至跳去模型页。只按直觉接住撤销。
    if ((e.key === "z" || e.key === "Z") && !e.shiftKey && !e.altKey) {
      e.preventDefault();
      undo();;
    }
    return;
  }
  const step = e.shiftKey ? 5 : 1;
  if (e.altKey && state.selObj >= 0 && state.selKpt >= 0 && e.key.startsWith("Arrow")) {
    e.preventDefault();
    const k = state.item.objects[state.selObj].keypoints[state.selKpt];
    const w = state.item.width, h = state.item.height;
    if (e.key === "ArrowLeft") k.x -= step / w;
    if (e.key === "ArrowRight") k.x += step / w;
    if (e.key === "ArrowUp") k.y -= step / h;
    if (e.key === "ArrowDown") k.y += step / h;
    k.x = clamp(k.x, 0, 1); k.y = clamp(k.y, 0, 1);
    if (k.v === 0) k.v = 2;
    markDirty(); draw();
    return;
  }
  switch (e.key) {
    case " ": state.spaceDown = true; e.preventDefault(); break;
    case "ArrowRight": case "ArrowDown": case "d": selectIndex(state.idx + 1); break;
    case "ArrowLeft": case "ArrowUp": case "a": selectIndex(state.idx - 1); break;
    case "v": case "V": setFocus(); break;
    case "j": case "J": cycleTodo(1); break;
    case "k": case "K": cycleTodo(-1); break;
    case "?": openHelp(true); break;
    case "q": case "Q": setStatus("approved"); break;
    case "e": case "E": setStatus("rejected"); break;
    case "f": case "F": jumpFlagged(1); break;
    case "u": case "U": if (!runPendingUndo()) undo(); break;
    case "x": case "X": deleteObject(); break;
    case "z": case "Z": fitView(); draw(); break;
    case "[": cycleObject(-1); break;
    case "]": cycleObject(1); break;
    case "b": case "B": openBatches(true); break;
    case "w": case "W": openUpload(true); break;
    case "m": case "M": location.href = "/models"; break;
    case "o": case "O": location.href = "/objects"; break;
    case "0": case "1": case "2": setVisibility(parseInt(e.key, 10)); break;
    default: return;
  }
}

function cycleObject(dir) {
  const n = state.item ? state.item.objects.length : 0;
  if (!n) return;
  state.selObj = (state.selObj + dir + n) % n;
  state.selKpt = -1;
  renderPanel(); draw();
}

function setVisibility(v) {
  const o = state.item && state.item.objects[state.selObj];
  if (!o) { toast("先选中一个目标"); return; }
  if (state.selKpt >= 0 && o.keypoints[state.selKpt]) o.keypoints[state.selKpt].v = v;
  else o.keypoints.forEach((k) => { if (k.v !== 0) k.v = v; });
  markDirty(); renderPanel(); draw();
}

function deleteObject() {
  if (!state.item || state.selObj < 0) return;
  state.item.objects.splice(state.selObj, 1);
  state.selObj = Math.min(state.selObj, state.item.objects.length - 1);
  state.selKpt = -1;
  markDirty(); renderPanel(); draw();
}

// --------------------------------------------------------------------------- //
// 保存 / 状态
// --------------------------------------------------------------------------- //
function markDirty() {
  state.dirty = true;
  $("save-state").textContent = "未保存…";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => save(false), 600);
}

async function save(silent) {
  if (!state.item) return;
  const payload = { image: state.item.image, objects: state.item.objects };
  try {
    const res = await fetch("/api/item", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    state.dirty = false;
    $("save-state").textContent = `已保存 ${new Date().toLocaleTimeString("zh-CN")} · ${data.objects.length} 人`;
    if (data.flags) {
      state.item.flags = data.flags;
      const meta = state.list[state.idx];
      if (meta) {
        meta.flags = [...(data.flags || []), ...(data.object_flags || [])].sort();
        meta.objects = data.objects.length;
        meta.status = data.status;
        const dsItem = state.ds.items.find((i) => i.image === meta.image);
        if (dsItem) Object.assign(dsItem, meta);
      }
    }
    renderProgress(); renderFilmstrip();
  } catch (err) {
    $("save-state").textContent = "保存失败: " + err.message;
    toast("保存失败: " + err.message, true);
  }
}

async function setStatus(status) {
  if (!state.item) return;
  if (state.dirty) await save(true);
  const target = state.item.image;
  const previous = state.item.status || "unreviewed";
  const res = await fetch("/api/status", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: state.item.image, status }),
  });
  const data = await res.json();
  if (!res.ok) { toast("失败: " + data.error, true); return; }
  const meta = state.list[state.idx];
  if (meta) meta.status = status;
  const dsItem = state.ds.items.find((i) => i.image === state.item.image);
  if (dsItem) dsItem.status = status;
  const said = { approved: "已接受", rejected: "已驳回", edited: "已保存修改" }[status] || ("已更新 " + status);
  toast(said);
  showUndoBar(`${said} · ${target.split("/").pop()} · 点撤销退回「${statusLabel(previous)}」`,
    () => revertStatus(target, previous));
  renderProgress(); renderFilmstrip();
  if (status !== "edited") selectIndex(state.idx + 1);
}

/** 把某一帧的状态改回去（撤销条用）：只回退状态，不动已经改好的坐标。 */
async function revertStatus(image, previous) {
  const res = await fetch("/api/status", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image, status: previous }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) { toast("撤销失败：" + (data.error || "后端没答应"), true); return; }
  const row = state.ds.items.find((i) => i.image === image);
  if (row) row.status = previous;
  toast("已撤销，这一帧回到" + statusLabel(previous));
  // 回到刚才那一帧，用户才看得见撤销生效了（筛选停在"待复核"时它会重新进队列）
  await refreshDataset({ select: image });
}

async function undo() {
  if (!state.item) return;
  const res = await fetch("/api/undo", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: state.item.image }),
  });
  const data = await res.json();
  if (!res.ok) { toast(data.error || "没有可撤销的历史", true); return; }
  state.dirty = false;
  await selectIndex(state.idx);
  hideUndoBar();
  toast("已撤销一次修改");
}

function jumpFlagged(dir) {
  // 与速查表写的一致：带 QC 标记**或整体置信度偏低**都算，正向标记不算
  for (let i = state.idx + dir; i >= 0 && i < state.list.length; i += dir) {
    if (needsAttention(state.list[i])) { selectIndex(i); return; }
  }
  toast("没有更多待确认的图片了");
}

window.addEventListener("beforeunload", (e) => { if (state.dirty) { e.preventDefault(); e.returnValue = ""; } });
boot();

// --------------------------------------------------------------------------- //
// 上传素材（图片 / 图片目录 / 视频）+ 后台标注任务进度
// --------------------------------------------------------------------------- //
const up = { files: [], poll: null, seenStages: {}, busy: false, warm: false, hadActive: false };

function upLog(msg) {
  const el = $("up-log");
  el.textContent += `[${new Date().toLocaleTimeString("zh-CN")}] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}

function renderUpFiles() {
  const files = up.files;
  const el = $("up-files");
  if (!files.length) { el.textContent = "尚未选择文件"; $("up-start").disabled = true; return; }
  const videos = files.filter(isVideo), images = files.filter(isImage);
  const total = files.reduce((a, f) => a + f.size, 0);
  const names = files.slice(0, 12).map((f) => f.name).join("、");
  el.textContent = `已选 ${files.length} 个文件（视频 ${videos.length} · 图片 ${images.length} · 共 ${kb(total)}）：`
    + names + (files.length > 12 ? ` …等 ${files.length} 个` : "");
  el.title = files.map((f) => f.name).join("\n");
  $("up-start").disabled = false;
}

function addUpFiles(list) {
  const seen = new Set(up.files.map((f) => f.name + ":" + f.size));
  let added = 0, skipped = 0;
  for (const f of Array.from(list || [])) {
    if (!isImage(f) && !isVideo(f)) { skipped++; continue; }
    const key = f.name + ":" + f.size;
    if (seen.has(key)) continue;
    seen.add(key); up.files.push(f); added++;
  }
  renderUpFiles();
  if (skipped) upLog(`忽略 ${skipped} 个不支持的文件（只支持图片与视频）`);
  return added;
}

function openUpload(open) {
  $("upload-modal").classList.toggle("hidden", !open);
}

/** 清除正在执行/排队的后台任务（带确认弹窗） */
async function clearJobs() {
  if (confirmBoxState.resolve) return;
  const ok = await confirmBox({
    title: "清除正在执行的任务",
    body: "排队中的任务立即删除（上传的临时文件一并清理）；正在标注的任务会在当前图片标完后停止，"
      + "已写入的标注和坐标文件保留。此操作无法撤销。",
    okLabel: "确定清除",
  });
  if (!ok) return;
  let data;
  try {
    const res = await fetch("/api/jobs/clear", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
  } catch (err) {
    toast("清除任务失败：" + err.message, true);
    return;
  }
  const parts = [];
  if (data.cancelling) parts.push(`${data.cancelling} 个任务正在取消`);
  if (data.cancelled) parts.push(`${data.cancelled} 个排队任务已删除`);
  if (data.dropped) parts.push(`清理 ${data.dropped} 条历史`);
  const msg = parts.length ? parts.join(" · ") : "当前没有正在执行的任务";
  upLog("清除任务：" + msg);
  toast(msg);
  if (data.cancelling) {
    up.hadActive = true;  // 等运行中的任务真正停下后再刷新列表
    startJobPolling();
    return;
  }
  stopJobPolling();
  setBusy(false, null);
  $("up-start").disabled = up.files.length === 0;
  await refreshDataset();
}

async function uploadOne(file) {
  const res = await fetch("/api/upload?name=" + encodeURIComponent(file.name), { method: "POST", body: file });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function startUpload() {
  const files = up.files.slice();
  if (!files.length) { toast("先选择文件或目录", true); return; }
  const chosen = $("up-model").selectedOptions[0];
  const modelValue = $("up-model").value;
  const vlmProfile = modelValue.indexOf("vlm:") === 0 ? modelValue.slice(4) : "";
  if (!vlmProfile && chosen && chosen.dataset.ready === "0") {
    upLog(`提示：${chosen.value} 的权重还没下载完，首个任务会先下载权重，期间进度条不走，属正常`);
  }
  if (vlmProfile && chosen && chosen.dataset.ready === "0") {
    upLog(`提示：这条多模态配置最近一次连通性测试没通过，任务可能报连不上，先到「模型管理」点「测试」看看原因`);
  }
  if (state.dirty) await save(true);
  $("up-start").disabled = true;
  const params = {
    model: $("up-model").value,
    batch_name: $("up-batch-name") ? $("up-batch-name").value.trim() : "",
    vlm_profile: vlmProfile,
    fps: parseFloat($("up-fps").value),
    max_frames: parseInt($("up-max-frames").value, 10) || 300,
    label: $("up-label").checked,
    video_dedupe: $("up-keep-similar").checked ? "none" : "perceptual",
  };
  upLog(`开始上传 ${files.length} 个文件…`);
  const stored = { image: [], video: [] };
  let sent = 0;
  for (let i = 0; i < files.length; i += 3) {
    const batch = files.slice(i, i + 3);
    const results = await Promise.all(batch.map(async (f) => {
      try {
        const info = await uploadOne(f);
        return { ok: true, name: f.name, info };
      } catch (err) {
        return { ok: false, name: f.name, error: err.message };
      }
    }));
    for (const r of results) {
      if (r.ok) { stored[r.info.kind].push(r.info.stored); upLog(`已上传 ${r.name}（${kb(r.info.bytes)}）`); }
      else upLog(`上传失败 ${r.name}: ${r.error}`);
    }
    sent += batch.length;
    $("up-bar").style.width = Math.round((sent / files.length) * 60) + "%";
  }
  for (const kind of ["video", "image"]) {
    if (!stored[kind].length) continue;
    const res = await fetch("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, files: stored[kind], params }),
    });
    const data = await res.json();
    if (!res.ok) { upLog(`任务启动失败: ${data.error}`); continue; }
    upLog(kind === "video" ? `视频任务 ${data.id} 已启动（抽帧 → 标注）` : `图片任务 ${data.id} 已启动`);
  }
  up.files = [];
  renderUpFiles();
  startJobPolling();
}

function startJobPolling() {
  stopJobPolling();
  up.poll = setInterval(pollJobs, 1200);
  pollJobs();
}
function stopJobPolling() {
  if (up.poll) { clearInterval(up.poll); up.poll = null; }
}

async function pollJobs() {
  let jobs = [];
  try {
    const res = await fetch("/api/jobs");
    jobs = (await res.json()).jobs || [];
  } catch (err) {
    return;
  }
  const active = jobs.filter((j) => !j.finished);
  for (const job of jobs) {
    const key = job.id + ":" + job.stage;
    if (up.seenStages[job.id] === key) continue;
    up.seenStages[job.id] = key;
    // 页面刚打开时别把历史已完成任务当成新消息刷屏
    if (up.warm || !job.finished) upLog(`任务 ${job.id} · ${job.message}`);
  }
  up.warm = true;
  setBusy(active.length > 0, active.length ? active[active.length - 1] : null);
  $("up-bar").style.width = jobPercent(active) >= 0 ? (60 + jobPercent(active) * 0.4) + "%" : "100%";
  if (!active.length) {
    stopJobPolling();
    $("up-start").disabled = up.files.length === 0;
    if (!up.hadActive) return;
    up.hadActive = false;
    const last = jobs[0];
    if (last && last.result && last.result.coords) upLog("坐标文件目录：" + last.result.coords);
    await refreshDataset();
  } else {
    up.hadActive = true;
  }
}

function jobPercent(active) {
  if (!active.length) return -1;
  const j = active[active.length - 1];
  if (j.total > 0) return Math.round((j.done / j.total) * 100);
  return { queued: 5, importing: 20, labeling: 55, qc: 85, coords: 95 }[j.stage] ?? 10;
}

function setBusy(busy, job) {
  up.busy = busy;
  const strip = $("job-strip");
  strip.classList.toggle("hidden", !busy);
  strip.classList.toggle("err", Boolean(busy && job && job.stage === "failed"));
  const cancelling = Boolean(busy && job && job.cancel_requested);
  if (busy && job) {
    const extra = job.total > 0 ? ` ${job.done}/${job.total}` : "";
    $("job-strip-text").textContent = (cancelling ? "取消中 · " : "")
      + `后台标注进行中 · 任务 ${job.id}${extra} · ${job.message}（期间请先别改标注，完成后列表会自动刷新）`;
  }
  for (const id of ["strip-clear-jobs", "up-clear-jobs"]) {
    const el = $(id);
    if (!el) continue;
    el.disabled = !busy || cancelling;
    if (id === "strip-clear-jobs") el.textContent = cancelling ? "取消中…" : "清除任务";
  }
  for (const id of ["btn-approve", "btn-reject", "btn-add", "btn-undo"]) {
    const el = $(id);
    if (el) el.disabled = busy;
  }
}

function toVlmChoice(profile) {
  const tested = profile.last_test || null;
  return {
    id: profile.id, name: profile.name, kind: profile.kind, provider: profile.provider,
    model: profile.model, ok: !!(tested && tested.ok), tested: !!tested,
    checked_at: tested ? (tested.checked_at || "") : "",
  };
}

const CAPABILITY_GROUP_LABEL = {
  pose: "姿态估计（YOLO-pose · 快且稳）",
  vlm: "多模态大模型（逐图询问 · 慢但灵活）",
  custom_task: "自定义任务权重",
};

/** 统一模型清单：优先用 /api/dataset 的 model_rows（带中文说明与兼容性），退化成权重行也要能选。 */
function registryRows() {
  const rows = (state.ds && Array.isArray(state.ds.model_rows) && state.ds.model_rows.length)
    ? state.ds.model_rows : [];
  if (Array.isArray(rows) && rows.length) return rows;
  return ((state.ds && state.ds.models) || []).map((w) => ({
    id: String(w.name).replace(/\.pt$/, ""), display_name: w.name, filename: w.name,
    capability: "pose", ready: Boolean(w.ready), size_mb: w.size_mb || 0, expected_mb: 0,
    status: w.ready ? "available" : (w.size_mb ? "incomplete" : "missing"),
    status_label: w.ready ? "可用" : (w.size_mb ? "文件不完整" : "未下载"),
    compatible: true, compatibility_reasons: [], note: "", description: "",
  }));
}

function modelOption(row) {
  const capability = row.capability || "pose";
  const option = document.createElement("option");
  option.value = capability === "vlm" ? "vlm:" + row.id : (row.filename || row.id + ".pt");
  const bits = [];
  if (capability === "vlm") {
    bits.push(row.kind === "local" ? "本地" : "云端", row.model || "未填模型名",
      row.tested ? (row.ready ? "已连通" : "测试失败") : "未测试");
  } else {
    if (row.params_m) bits.push(row.params_m + "M 参数");
    bits.push(row.ready ? `已就绪 ${mb(row.size_mb || 0)} MB` : `预计 ${mb(row.expected_mb || 0)} MB`);
  }
  const blocked = capability !== "vlm" && !row.ready;
  // 关键点数量对不上当前数据集的模型选了也白选，直接置灰，原因写在「模型管理」的说明里
  const mismatch = capability !== "vlm" && Boolean(row.ready) && row.compatible === false;
  const tail = blocked ? " · 未下载，先到「模型管理」下载）"
    : mismatch ? " · 与当前数据集关键点数量不匹配，先看「模型管理」）" : "）";
  option.textContent = `${row.display_name || row.id}（${bits.join(" · ")}` + tail;
  option.disabled = blocked || mismatch;
  option.dataset.ready = row.ready ? "1" : "0";
  option.dataset.modelId = row.id;
  return option;
}

/** 选中模型的中文说明直接摊在下拉框下面，省得用户凭文件名猜哪个能用。 */
function renderModelDesc() {
  const box = $("up-model-desc");
  const select = $("up-model");
  if (!box || !select) return;
  const chosen = select.selectedOptions[0];
  const row = registryRows().find((r) => String(r.id) === String(chosen && chosen.dataset.modelId));
  box.innerHTML = "";
  if (!row) { box.textContent = "（这个条目没有说明信息）"; return; }
  const facts = [];
  if (row.capability === "vlm") {
    const tested = row.tested ? (row.ready ? "最近测试通过" : "最近测试没通过") : "未测过连通性";
    facts.push(`${row.kind === "local" ? "本地服务" : "云端 API"} ${row.base_url || "未填地址"}`,
      `模型名 ${row.model || "未填"}`, `Key ${row.api_key_masked || "未设置"}`, tested);
  } else {
    if (row.params_m) facts.push(`参数约 ${row.params_m}M`);
    if (row.map_pose) facts.push(`mAP50-95 约 ${row.map_pose}`);
    facts.push(row.ready ? `本地已有 ${mb(row.size_mb || 0)} MB` : `需下载约 ${mb(row.expected_mb || 0)} MB`);
    facts.push(`${row.num_keypoints || 17} 关键点`);
    if (row.speed) facts.push(`速度 ${row.speed}`);
    if (row.recommended_for) facts.push(`适合 ${row.recommended_for}`);
  }
  if (row.note) facts.push(`备注：${row.note}`);
  if (!row.compatible) facts.push(`不兼容：${(row.compatibility_reasons || []).join("；")}`);
  const desc = document.createElement("div");
  desc.textContent = row.description || "（目录里没写说明）";
  const meta = document.createElement("div");
  meta.textContent = facts.filter(Boolean).join(" ｜ ");
  box.append(desc, meta);
}

function renderModelOptions(models, vlm) {
  const select = $("up-model");
  if (!select) return;
  const previous = select.value;
  const rows = registryRows();
  if (!rows.length) return;
  select.innerHTML = "";
  for (const capability of ["pose", "vlm", "custom_task"]) {
    const group = rows.filter((r) => (r.capability || "pose") === capability)
      .slice().sort((a, b) => Number(b.ready === true) - Number(a.ready === true));
    if (!group.length) continue;
    // 一条都能没用（全没下载或全不兼容）时至少留一条能选的，行为退回「首次标注自动经镜像下载」
    const usable = group.some((r) => r.ready && r.compatible !== false);
    const firstFallback = capability === "vlm" || usable ? -1 : 0;
    const optgroup = document.createElement("optgroup");
    optgroup.label = CAPABILITY_GROUP_LABEL[capability] || capability;
    group.forEach((row, index) => {
      const option = modelOption(row);
      if (index === firstFallback) {
        option.disabled = false;
        option.textContent = option.textContent.replace(" · 未下载，先到「模型管理」下载）",
          " · 未下载，选它会在首个任务时自动下载）");
      }
      optgroup.appendChild(option);
    });
    select.appendChild(optgroup);
  }
  if (!select.selectedOptions.length || select.selectedOptions[0].disabled) {
    const first = Array.from(select.options).find((o) => !o.disabled);
    if (first) select.value = first.value;
  }
  const restore = Array.from(select.options).find((o) => o.value === previous && !o.disabled);
  if (restore) select.value = previous;
  renderModelDesc();
}

async function refreshDataset(opts) {
  const current = (opts && opts.select) || (state.item ? state.item.image : null);
  const res = await fetch("/api/dataset");
  state.ds = await res.json();
  if (state.ds.batches) batchState.list = poseBatches(state.ds.batches);
  renderModelOptions(state.ds.models);
  setFlagPolicy(state.ds);
  setHeaderCount(state.ds);
  rebuildList();
  if (!state.list.length) { state.item = null; draw(); return; }
  const found = current ? state.list.findIndex((i) => i.image === current) : -1;
  await selectIndex(found >= 0 ? found : 0);
  // 各操作自己都有更具体的提示（导出路径、删除张数、任务结果），这里默认不再补一条
  // 泛化 toast，否则会把刚弹出来的关键提示顶掉。
  if (opts && opts.announce) toast("列表已更新");
}

/** 顶栏导出按钮：坐标是直接下载的文件，训练集是目录，只能让后端生成后回报路径。 */
function initExports() {
  const button = $("btn-export-yolo");
  if (button) button.onclick = () => exportYoloDataset(button);
  renderExportNote("export-note", state.ds, "pose");
}

async function exportYoloDataset(button) {
  if (button.disabled) return;
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "导出中…";
  try {
    const data = await postJson("/api/export", { val_ratio: 0.2 });
    const info = data.export || {};
    const opened = await revealExport(info.out, { quiet: true });
    const verb = info.reused ? "标签没有变化，已复用上次导出" : "YOLO 训练集已导出";
    toast(`${verb}（${scopeLabel(info.scope || "reviewed")}）：${exportSummary(info, "pose")} `
      + `→ ${info.out}` + openedSuffix(opened));
  } catch (err) {
    toast("导出失败：" + err.message, true);
  }
  button.disabled = false;
  button.textContent = label;
  await refreshDataset();
}

function initUpload() {
  $("btn-upload").onclick = () => openUpload(true);
  $("up-close").onclick = () => openUpload(false);
  $("up-clear").onclick = () => { up.files = []; $("up-log").textContent = ""; $("up-bar").style.width = "0"; renderUpFiles(); };
  $("up-start").onclick = startUpload;
  $("up-clear-jobs").onclick = clearJobs;
  $("strip-clear-jobs").onclick = clearJobs;
  // 素材池里已有的图（多为提示词库先传的）直接引用同一份文件，不必重传
  $("up-pool").onclick = () => openPool(true, POSE_LIB, () => refreshDataset());
  $("up-model").onchange = renderModelDesc;
  for (const id of ["file-images", "file-dir", "file-video"]) {
    $(id).onchange = (e) => { addUpFiles(e.target.files); e.target.value = ""; };
  }
  const drop = $("up-drop");
  for (const evt of ["dragenter", "dragover"]) {
    drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.add("over"); });
  }
  for (const evt of ["dragleave", "drop"]) {
    drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.remove("over"); });
  }
  drop.addEventListener("drop", (e) => {
    const files = e.dataTransfer ? e.dataTransfer.files : null;
    if (files && addUpFiles(files)) toast(`已加入 ${files.length} 个文件`);
  });
  window.addEventListener("dragover", (e) => { e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    const files = e.dataTransfer ? e.dataTransfer.files : null;
    if (files && files.length) { openUpload(true); addUpFiles(files); e.preventDefault(); }
  });
  $("up-start").disabled = true;
  renderUpFiles();
  pollJobs().then(() => { if (up.busy) startJobPolling(); });
}

// --------------------------------------------------------------------------- //
// 批次管理：按每次上传/导入任务分组查看，可整批删除已标注的数据与记录
// 两个标注库各自记账，这页只列姿态库的批次
// --------------------------------------------------------------------------- //
const BATCH_SOURCE = { upload: "网页上传", cli: "命令行", unknown: "历史", "-": "历史" };
const BATCH_STATUS = {
  running: "标注中", done: "已完成", cancelled: "已取消", failed: "失败",
  interrupted: "已中断", "-": "历史数据", unknown: "未知",
};
const BATCH_KIND = { video: "视频抽帧", image: "图片", prompt: "提示词标注",
                     prompt_upload: "提示词上传", pool: "素材池添加", mixed: "混合" };
const BATCH_MODULE = { pose: "姿态标注", objects: "提示词标注", unknown: "未识别" };

/** 只留姿态库的批次：提示词库的批次在这一页看，两边进度互不掺水。 */
function poseBatches(rows) {
  return (rows || []).filter((b) => (b.lib || b.module || "pose") === POSE_LIB);
}

function openBatches(open) {
  $("batch-modal").classList.toggle("hidden", !open);
  if (open) refreshBatches();
}

async function refreshBatches() {
  try {
    const res = await fetch("/api/batches");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    batchState.list = poseBatches(data.batches);
  } catch (err) {
    toast("批次读取失败：" + err.message, true);
    batchState.list = [];
  }
  for (const id of Array.from(batchState.selected)) {
    if (!batchState.list.some((b) => b.id === id)) batchState.selected.delete(id);
  }
  renderBatches();
}

function renderBatches() {
  const box = $("batch-list");
  box.innerHTML = "";
  if (!batchState.list.length) {
    box.innerHTML = '<div class="batch-empty muted small">还没有批次记录：网页上传素材或命令行 import / video / auto 都会自动登记。</div>';
  }
  for (const batch of batchState.list) {
    const row = document.createElement("div");
    row.className = "batch-row" + (batchState.selected.has(batch.id) ? " selected" : "");
    const checked = batchState.selected.has(batch.id) ? "checked" : "";
    const files = (batch.files || []).slice(0, 3).join("、")
      + (batch.file_count > 3 ? ` 等 ${batch.file_count} 个` : "");
    row.innerHTML = `
      <input type="checkbox" ${checked}>
      <div class="batch-main">
        <div class="batch-name"></div>
        <div class="muted small batch-meta"></div>
      </div>
      <div class="batch-nums">
        <span>图片 <b>${batch.items}</b></span>
        <span>已标 <b>${batch.labeled}</b></span>
        <span>人体 <b>${batch.people}</b></span>
        <span>待复核 <b>${batch.flagged}</b></span>${batch.prompt_boxes ? ` <span>提示词框 <b>${batch.prompt_boxes}</b></span>` : ""}
      </div>
      <span class="badge ${batch.status}">${BATCH_STATUS[batch.status] || batch.status}</span>
      <button class="peek">只看这批</button>`;
    const nameCell = row.querySelector(".batch-name");
    const module = batch.module || "unknown";
    const badge = document.createElement("span");
    badge.className = "badge module " + module;
    badge.textContent = BATCH_MODULE[module] || module;
    nameCell.append(badge, document.createTextNode(" " + batch.name));
    row.querySelector(".batch-meta").textContent =
      `${batch.created_at || "时间未知"} · ${BATCH_SOURCE[batch.source] || batch.source} · `
      + `${BATCH_KIND[batch.kind] || batch.kind || "混合"}`
      + (files ? ` · ${files}` : "") + (batch.message ? ` · ${batch.message}` : "");
    row.querySelector("input").onchange = (e) => {
      if (e.target.checked) batchState.selected.add(batch.id);
      else batchState.selected.delete(batch.id);
      renderBatches();
    };
    row.querySelector(".peek").onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setBatchFilter(batch.id);
      openBatches(false);
      toast(`只看「${batch.name}」的 ${batch.items} 张`);
    };
    box.appendChild(row);
  }
  const chosen = batchState.list.filter((b) => batchState.selected.has(b.id));
  const images = chosen.reduce((a, b) => a + b.items, 0);
  $("batch-note").textContent = chosen.length ? `已选 ${chosen.length} 个批次 · 共 ${images} 张图片` : "";
  $("batch-delete").disabled = !chosen.length;
  $("batch-all").checked = batchState.list.length > 0 && chosen.length === batchState.list.length;
}

function setBatchFilter(id) {
  state.batchFilter = id || null;
  const chip = $("batch-chip");
  if (state.batchFilter) {
    const row = batchState.list.find((b) => b.id === state.batchFilter);
    chip.textContent = `只看批次：${row ? row.name : state.batchFilter}（${row ? row.items : 0} 张）×`;
    chip.classList.remove("hidden");
  } else {
    chip.textContent = "";
    chip.classList.add("hidden");
  }
  rebuildList();
}

async function deleteSelectedBatches() {
  const ids = Array.from(batchState.selected);
  if (!ids.length) { toast("先勾选要删除的批次", true); return; }
  const chosen = batchState.list.filter((b) => ids.includes(b.id));
  const images = chosen.reduce((a, b) => a + b.items, 0);
  const ok = await confirmBox({
    title: "删除姿态库的这些批次",
    body: `将删除「姿态标注」的 ${chosen.length} 个批次、共 ${images} 张图，连同骨骼标签、复核记录、QC 叠加图与批次记录一起清掉。`
      + "素材文件只有在提示词库不再引用时才会删，否则只摘姿态库引用（原图和检测框留给提示词库）。"
      + "删除前会在 .autolabel/deleted/ 写一份快照（含标签原文），界面里不能一键恢复。",
    okLabel: "确定删除",
  });
  if (!ok) return;
  let data;
  try {
    const res = await fetch("/api/batches/delete", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
  } catch (err) {
    toast("删除失败：" + err.message, true);
    return;
  }
  batchState.selected.clear();
  if (state.batchFilter && data.batches.includes(state.batchFilter)) setBatchFilter(null);
  upLog(`删除批次 ${data.batches.join("、")}：图片 ${data.items} 张 · 文件 ${data.files_removed} 个`);
  toast(`已从姿态库移除 ${data.items} 张 · 清理 ${data.files_removed} 个文件`
    + (data.snapshot ? "（快照已保存）" : ""));
  await refreshBatches();
  await refreshDataset();
}

function initBatches() {
  $("btn-batches").onclick = () => openBatches(true);
  $("batch-close").onclick = () => openBatches(false);
  $("batch-refresh").onclick = refreshBatches;
  $("batch-delete").onclick = deleteSelectedBatches;
  $("batch-all").onchange = (e) => {
    batchState.selected = e.target.checked ? new Set(batchState.list.map((b) => b.id)) : new Set();
    renderBatches();
  };
  $("batch-chip").onclick = () => { setBatchFilter(null); toast("已取消批次筛选"); };
}

// --------------------------------------------------------------------------- //
// 壳层：待办清单（编号 ↔ 画布编号）、聚焦模式、撤销条、命令面板、快捷键速查
// --------------------------------------------------------------------------- //
function toggleAdd() {
  state.addMode = !state.addMode;
  $("btn-add").classList.toggle("active", state.addMode);
  toast(state.addMode ? "拖动画框新增目标" : "退出新增模式");
}

/**
 * 这一帧哪些关键点值得动手：贴边/偏出框、被标成遮挡、或置信度偏低。
 * 按「先修错的、再看可疑的」排序，最多取 9 条——画布编号超过一位数就开始挡人了。
 */
function attentionList(item) {
  if (!item) return [];
  const names = (state.ds && state.ds.keypoint_names) || {};
  const rows = [];
  (item.objects || []).forEach((obj, oi) => {
    (obj.keypoints || []).forEach((k, ki) => {
      if (k.v === 0) return;  // 忽略的点本来就不产生监督，不用再看
      const low = k.conf !== null && k.conf !== undefined && k.conf < LOWCONF;
      const edge = k.x <= 0.004 || k.x >= 0.996 || k.y <= 0.004 || k.y >= 0.996;
      if (!low && !edge && k.v !== 1) return;
      const why = [];
      if (edge) why.push("贴边，可能偏出框");
      if (k.v === 1) why.push("标为遮挡");
      if (low) why.push(`置信度 ${Number(k.conf).toFixed(2)} 偏低`);
      rows.push({
        oi, ki, label: names[ki] || ("关键点 " + ki), reason: why.join(" · "),
        sev: edge || k.v === 1 ? 1 : 2, v: k.v,
      });
    });
  });
  rows.sort((a, b) => a.sev - b.sev || a.oi - b.oi || a.ki - b.ki);
  return rows.slice(0, 9);
}

function renderTodo() {
  const box = $("todo");
  if (!box) return;
  box.innerHTML = "";
  state.todo = attentionList(state.item);
  if (state.selTodo >= state.todo.length) state.selTodo = state.todo.length ? 0 : -1;
  if (state.item && !state.todo.length) {
    const tip = document.createElement("div");
    tip.className = "todo-empty";
    tip.textContent = "没有要点的关键点：置信度都够，也没有贴边或被遮挡的点。按 Q 接受就进下一帧。";
    box.appendChild(tip);
  }
  state.todo.forEach((row, i) => {
    const line = document.createElement("button");
    line.type = "button";
    line.className = "todo-row" + (row.sev > 1 ? " mut" : "") + (i === state.selTodo ? " on" : "");
    const no = document.createElement("span");
    no.className = "todo-no";
    no.textContent = String(i + 1);
    const main = document.createElement("span");
    main.className = "todo-main";
    const title = document.createElement("b");
    title.textContent = `${row.label} · 第 ${row.oi + 1} 人`;
    const why = document.createElement("p");
    why.textContent = row.reason;
    main.append(title, why);
    const v = document.createElement("span");
    v.className = "chip " + (row.v === 2 ? "ok" : "warn");
    v.textContent = "v=" + row.v;
    line.append(no, main, v);
    line.onclick = () => selectTodo(i);
    box.appendChild(line);
  });
  $("todo-count").textContent = state.todo.length ? state.todo.length + " 处" : "干净";
  const flag = $("hud-flag");
  flag.textContent = state.todo.length ? state.todo.length + " 处待确认" : "无需修改";
  flag.className = state.todo.length ? "chip bad" : "chip ok";
  renderHudSegs();
}

function selectTodo(i) {
  const row = state.todo[i];
  if (!row) return;
  state.selTodo = i;
  state.selObj = row.oi;
  state.selKpt = row.ki;
  centerOn(row.oi, row.ki);
  renderPanel();
  draw();
}

function cycleTodo(dir) {
  if (!state.todo.length) { toast("这一帧没有待办的关键点"); return; }
  const next = state.selTodo < 0
    ? (dir > 0 ? 0 : state.todo.length - 1)
    : (state.selTodo + dir + state.todo.length) % state.todo.length;
  selectTodo(next);
}

/** 选中的点快跑出视野时把它挪回中间，省得手动拖画布。 */
function centerOn(oi, ki) {
  const obj = state.item && state.item.objects[oi];
  const k = obj && obj.keypoints[ki];
  if (!k) return;
  const [sx, sy] = toScreen(k.x * state.item.width, k.y * state.item.height);
  const cx = state.cw / 2, cy = state.ch / 2;
  if (Math.abs(sx - cx) > state.cw * 0.22) state.view.ox += cx - sx;
  if (Math.abs(sy - cy) > state.ch * 0.22) state.view.oy += cy - sy;
}

/** 待办点在画布上的编号：右侧清单第几条，画布上就是几号。 */
function drawTodoBadges() {
  const it = state.item;
  if (!it || !state.todo.length) return;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  state.todo.forEach((row, i) => {
    const obj = it.objects[row.oi];
    const k = obj && obj.keypoints[row.ki];
    if (!k) return;
    const [sx, sy] = toScreen(k.x * it.width, k.y * it.height);
    const hot = row.sev === 1;
    ctx.beginPath();
    ctx.arc(sx, sy, i === state.selTodo ? 13 : 10, 0, Math.PI * 2);
    ctx.fillStyle = rgbaOf(canvasPalette()[hot ? "coral" : "amber"], hot ? .2 : .16);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(sx, sy, 8, 0, Math.PI * 2);
    ctx.fillStyle = hot ? canvasPalette().coral : canvasPalette().amber;
    ctx.fill();
    ctx.fillStyle = canvasPalette().onAccent;
    ctx.font = cfont(10, "600");
    ctx.fillText(String(i + 1), sx, sy + 0.5);
  });
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

function drawCorners(x, y, w, h) {
  const l = clamp(Math.min(w, h) * 0.22, 8, 22);
  ctx.beginPath();
  ctx.moveTo(x, y + l); ctx.lineTo(x, y); ctx.lineTo(x + l, y);
  ctx.moveTo(x + w - l, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + l);
  ctx.moveTo(x + w, y + h - l); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - l, y + h);
  ctx.moveTo(x + l, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - l);
  ctx.stroke();
}

/** HUD 的分段进度：一目了然这帧的点有多少是「模型真看清了」。 */
function renderHudSegs() {
  const host = $("hud-segs");
  if (!host) return;
  host.innerHTML = "";
  const kps = [];
  ((state.item && state.item.objects) || []).forEach((o) => (o.keypoints || []).forEach((k) => kps.push(k)));
  const n = 10;
  const vis = kps.filter((k) => k.v === 2).length;
  const occ = kps.filter((k) => k.v === 1).length;
  const solid = kps.length ? Math.round(vis / kps.length * n) : 0;
  const soft = kps.length ? Math.round(occ / kps.length * n) : 0;
  for (let i = 0; i < n; i++) {
    const seg = document.createElement("i");
    if (i < solid) seg.className = "ok";
    else if (i < solid + soft) seg.className = "next";
    host.appendChild(seg);
  }
  host.title = `可见 ${vis} · 遮挡 ${occ} · 忽略 ${kps.length - vis - occ}`;
}

function setFocus(on) {
  state.focus = on === undefined ? !state.focus : Boolean(on);
  $("stage").classList.toggle("focus", state.focus);
  $("btn-focus").classList.toggle("active", state.focus);
  $("rail-queue").classList.toggle("on", true);
  draw();
  toast(state.focus ? "聚焦模式：只显示当前目标" : "已显示全部目标");
}







/** 命令面板：动作 + 筛选切换 + 按文件名直接跳帧。 */




/** 命令面板里点了文件名：当前位置找不到就把筛选放开再找。 */
async function gotoImage(image) {
  let pos = state.list.findIndex((i) => i.image === image);
  if (pos < 0) {
    $("filter").value = "all";
    rebuildList();
    pos = state.list.findIndex((i) => i.image === image);
  }
  if (pos < 0) { toast("当前队列里没有这张", true); return; }
  await selectIndex(pos);
}

function initShell() {
  registerShellCommands(poseCommands(), poseFrameMatches);
  initShellChrome({
    onQueue: () => {
      const col = $("queue-col");
      col.classList.toggle("hidden");
      $("rail-queue").classList.toggle("on", !col.classList.contains("hidden"));
      resizeCanvas();
      draw();
    },
    onUpload: () => openUpload(true),
    onBatches: () => openBatches(true),
  });
  $("btn-batches").addEventListener("click", () => openMenu(false));
  $("btn-focus").onclick = () => setFocus();
  $("btn-undo-bar").onclick = () => { if (!runPendingUndo()) undo(); };
  renderHudSegs();
}

/** 命令面板里的姿态动作；筛选切换也做成命令，省得先开菜单再找下拉框。 */
function poseCommands() {
  const cmds = [
    { label: "接受当前帧", key: "Q", run: () => setStatus("approved") },
    { label: "驳回当前帧", key: "E", run: () => setStatus("rejected") },
    { label: "撤销上一次修改", key: "U", run: undo },
    { label: "跳到下一张待确认", key: "F", run: () => jumpFlagged(1) },
    { label: "聚焦模式", key: "V", run: () => setFocus() },
    { label: "新增目标（拖框）", run: toggleAdd },
    { label: "适应窗口", key: "Z", run: () => { fitView(); draw(); } },
    { label: "导入素材", key: "W", run: () => openUpload(true) },
    { label: "从素材池添加", run: () => openPool(true, POSE_LIB) },
    { label: "批次管理", key: "B", run: () => openBatches(true) },
    { label: "导出坐标 CSV", run: () => downloadHref("/api/coords?format=csv") },
    { label: "导出坐标 JSONL", run: () => downloadHref("/api/coords?format=jsonl") },
    { label: "导出 YOLO 训练集", run: () => $("btn-export-yolo").click() },
    { label: "快捷键速查", key: "?", run: () => openHelp(true) },
    { label: "打开提示词标注", key: "O", run: () => { location.href = "/objects"; } },
    { label: "打开模型管理", key: "M", run: () => { location.href = "/models"; } },
  ];
  for (const [value, label] of [["pending", "筛选：待复核"], ["flagged", "筛选：仅带标记"],
    ["edited", "筛选：已修改"], ["approved", "筛选：已接受"], ["rejected", "筛选：已驳回"], ["all", "筛选：全部"]]) {
    cmds.push({ label, tag: "筛选", run: () => { $("filter").value = value; rebuildList(); } });
  }
  return cmds;
}

/** 命令面板里直接敲文件名跳帧；当前筛不到就放开筛选再找。 */
function poseFrameMatches(query) {
  const hits = [];
  for (const it of state.list) {
    if (String(it.image).toLowerCase().includes(query)) {
      hits.push({
        label: String(it.image).split("/").pop(), tag: statusLabel(it.status),
        run: () => gotoImage(it.image),
      });
    }
    if (hits.length >= 8) { break; }
  }
  return hits;
}

/* 换白天/夜晚或调字号后重画：颜色和字号都是从外观里取的，不重画会留在旧配色 */
window.addEventListener("appearancechange", () => { if (state.img) { draw(); } });
