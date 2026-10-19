"use strict";

// 二级模块「提示词标注」与「目标检测」共用这一份脚本，绝不再复制一份：
// 两个模块都是「检测框 + 人工复核」，界面骨架一模一样，差别只在**谁出框**——
//   提示词标注：多模态大模型按你写的字段描述在画面里找物品（可联网、每张一次请求、类别随便定）；
//   目标检测：本地 YOLO 权重按它自带的类别表出框（不联网、单张几十毫秒、类别由权重决定）。
// 所以差异全部收在下面这张表里，由 <body data-page data-lib> 选一行；
// 正文里再出现写死的库名，等于把另一个页面悄悄写坏。

const PAGE = document.body.dataset.page || "";
const LIB = document.body.dataset.lib || "";

const LIB_CONFIG = {
  // local=false：大模型按描述找物品；local=true：本机权重按自带类别出框
  objects: {
    lib: "objects", prefix: "prompt", local: false, page: "/objects", peer: "detect",
    name: "提示词标注", fieldWord: "字段", classTable: "字段登记表",
    items: "object_items", classes: "fields", labelDir: "prompt_labels/",
    runUrl: "/api/prompt/run", uploadUrl: "/api/prompt/upload",
    runLabel: "批量提示词标注", manageLabel: "字段管理",
    exportKind: "detect", emptyImage: "提示词库还没有图片",
    note: "提示词标注 = 多模态大模型按你写的字段描述找物品：类别随你定、可联网、慢；"
      + "要快就换「目标检测」——本机 YOLO 权重按它自带的类别表出框，不联网、单张几十毫秒。",
  },
  detect: {
    lib: "detect", prefix: "detect", local: true, page: "/detect", peer: "objects",
    name: "目标检测", fieldWord: "类别", classTable: "本库类别表",
    items: "detect_items", classes: "detect_classes", labelDir: "detect_labels/",
    runUrl: "/api/detect/run", uploadUrl: "/api/detect/upload",
    runLabel: "批量目标检测", manageLabel: "类别管理",
    exportKind: "yolo_detect", emptyImage: "目标检测库还没有图片",
    note: "目标检测 = 本机 YOLO 权重按它自带的类别表（COCO-80 等）出框：不联网、单张几十毫秒，"
      + "类别由权重决定；要标权重不认识的东西，去「提示词标注」让多模态大模型按你写的描述找。",
  },
};
const CFG = LIB_CONFIG[LIB] || LIB_CONFIG.objects;
const IS_LOCAL = Boolean(CFG.local);
// 另一个出框的库：两个库共用 images/，界面上要能一键过去，也要说清两者分工
const OTHER = LIB_CONFIG[CFG.peer] || CFG;
// ---- 库的差异到此为止，下面的正文对两个库必须一模一样 ----

/** 本库在索引里的字段名：提示词库是 prompt_*，检测库是 detect_*。 */
const pk = (name) => `${CFG.prefix}_${name}`;

const state = {
  ds: null, list: [], idx: -1, item: null,
  boxes: [], sel: -1, dirty: false, saveTimer: null,
  view: { scale: 1, ox: 0, oy: 0 },
  img: null, drag: null, spaceDown: false,
  drawField: "",
};
const run = { poll: null, busy: false, logEl: "run-log", seen: {}, warm: false, hadActive: false };
// 本模块独立上传：文件先落到 uploads/，再由 <lib>_upload 任务入库并直接出检测框
const up = { files: [] };
const fieldState = { editing: null };
// 目标检测页的权重状态：下拉框选项、权重自带类别、上次选的那一档（后端 prefs 记的）
const modelState = {
  loading: false, models: [], unusable: [], device: "auto",
  model: "", conf: 0.25, prefsClasses: [],
  list: [], listModel: "", listLoading: false, note: "", result: "",
  // 开放词表权重（YOLOE / YOLO-World）：类别不是权重自带的，而是这次现填的英文词
  openVocab: "", vocab: [],
};
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const imgCache = new Map();
const HANDLE_PX = 9;

const SAVE_STATE = { idle: "就绪", dirty: "有改动未保存", saving: "保存中…", saved: "已保存" };

/** 本库的类别表：提示词库读 `fields`，检测库读 `detect_classes`。 */
function fieldRows() {
  const table = (state.ds && state.ds[CFG.classes]) || {};
  return table.fields || [];
}

function fieldByClass(classId) {
  return fieldRows().find((f) => Number(f.class_id) === Number(classId)) || null;
}

function fieldName(classId) {
  const field = fieldByClass(classId);
  return field ? `${field.name}（${field.label || field.name}）` : `class_${classId}`;
}

function fieldColor(classId) {
  const field = fieldByClass(classId);
  return (field && field.color) || canvasPalette().fieldFallback;
}

function flagHint(flag) {
  const overview = (state.ds && state.ds[CFG.prefix]) || {};
  const hints = overview.flag_hints || {};
  return hints[flag] || "";
}

/** 页面顶部一句话：把「谁出框、快不快、类别谁说了算」讲明白，并给对面模块留个入口。 */
function renderLibNote() {
  let host = $("lib-note");
  if (!host) {
    host = document.createElement("div");
    host.id = "lib-note";
    host.className = "export-note bar muted small";
    const anchor = $("export-note");
    if (anchor && anchor.parentNode) { anchor.parentNode.insertBefore(host, anchor.nextSibling); }
    else { document.body.append(host); }
  }
  host.textContent = CFG.note;
  const link = document.createElement("a");
  link.href = OTHER.page;
  link.textContent = "去「" + OTHER.name + "」";
  host.append(link);
}

// --------------------------------------------------------------------------- //
// 初始化与列表
// --------------------------------------------------------------------------- //
async function bootObjects() {
  initNav();
  $("filter").onchange = rebuildList;
  $("sort").onchange = rebuildList;
  $("show-labels").onchange = draw;
  $("btn-approve").onclick = () => setStatus("approved");
  $("btn-reject").onclick = () => setStatus("rejected");
  $("btn-delete").onclick = deleteBox;
  $("btn-undo").onclick = undoItem;
  $("btn-next").onclick = () => selectIndex(state.idx + 1);
  $("btn-prev").onclick = () => selectIndex(state.idx - 1);
  $("btn-flagged").onclick = () => jumpFlagged(1);
  $("btn-fields").onclick = () => openFields(true);
  renderLibNote();
  $("btn-run").onclick = () => openRun(true);
  initUploadPanel();
  $("draw-field").onchange = (e) => { state.drawField = e.target.value; };
  $("strip-clear-jobs").onclick = clearJobs;
  initFieldPanel();
  initRunPanel();
  initModelPanel();
  initProbePanel();
  bindCanvas();
  initShell();
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", (e) => { if (e.code === "Space") state.spaceDown = false; });
  window.addEventListener("resize", () => { resizeCanvas(); draw(); });
  window.addEventListener("beforeunload", (e) => { if (state.dirty) { e.preventDefault(); e.returnValue = ""; } });
  resizeCanvas();
  await refreshDataset(true);
  // 门户卡片的「去导入素材」/「从素材池添加」带参数跳过来，直接摊开对应弹层
  const params = new URLSearchParams(location.search);
  if (params.get("pool") === "1") openPool(true, CFG.lib, () => refreshDataset());
  else if (params.get("upload") === "1") openUpload(true);
  // 权重清单不加载模型、只列本机文件，开页就取一次：下拉框与「用不了的权重」都得等着人
  if (IS_LOCAL) loadModelChoices();
  pollJobs().then(() => { if (run.busy) startJobPolling(); });
}

/** 本库的条目：这本库自己一本账，素材文件与其它库共用。 */
function libItems() {
  return (state.ds && state.ds[CFG.items]) || [];
}

/** 顶栏只报本库张数，顺带说明素材池里还有多少能直接「从素材池添加」。 */
function setHeaderCount(ds) {
  renderExportNote("export-note", ds, CFG.exportKind);
  const mine = libItems().length;
  const pool = ds.pool || {};
  const spare = Math.max(0, (pool.files || 0) - (pool[CFG.lib] || 0));
  const need = libItems().filter(needsAttention).length;
  $("ds-name").textContent = " · " + ds.name;
  $("capsule-set").textContent = `${ds.name} · ${CFG.name} ${mine} 帧`;
  $("capsule-count").textContent = need ? `待确认 ${need}` : "全部干净";
  $("queue-capsule").title = "焦点队列：带标记或还没标过检测框的帧"
    + (spare ? `；素材池另有 ${spare} 张可直接「从素材池添加」` : "");
}

/** 带标记，或压根还没标过检测框，都算「需要人看一眼」（正向标记不算问题）。 */
function needsAttention(it) {
  return issueFlags(it, pk("flags")).length > 0
    || it[pk("objects")] === null || it[pk("objects")] === undefined;
}

async function refreshDataset(keepImage) {
  // 调用方有时直接指定要继续停在哪个文件，别总按当前选中项猜
  const current = typeof keepImage === "string" ? keepImage
    : (state.item ? state.item.image : null);
  try {
    state.ds = await getJson("/api/dataset");
  } catch (err) {
    toast("数据集读取失败：" + err.message, true);
    return;
  }
  setFlagPolicy(state.ds);
  setHeaderCount(state.ds);
  document.title = CFG.name + " · " + state.ds.name;
  renderFieldChoices();
  rebuildList(current);
}

function currentFilter() { return $("filter") ? $("filter").value : "pending"; }

function matchesFilter(item, filter) {
  const status = item[pk("status")] || "unreviewed";
  const flagged = (item[pk("flags")] || []).length > 0;
  const labeled = item[pk("objects")] !== null && item[pk("objects")] !== undefined;
  if (filter === "all") return true;
  if (filter === "flagged") return flagged;
  if (filter === "unlabeled") return !labeled;
  if (filter === "pending") return !["approved", "rejected"].includes(status);
  return status === filter;
}

function rebuildList(keepImage) {
  if (!state.ds) return;
  const filter = currentFilter(), sort = $("sort") ? $("sort").value : "priority";
  const current = keepImage === undefined ? (state.item ? state.item.image : null) : keepImage;
  let items = libItems().filter((i) => matchesFilter(i, filter));
  items.sort(sort === "name"
    ? (a, b) => a.image.localeCompare(b.image)
    : (a, b) => ((b[pk("priority")] || 0) - (a[pk("priority")] || 0)) || a.image.localeCompare(b.image));
  state.list = items;
  const emptyBox = $("empty-state");
  emptyBox.textContent = libItems().length
    ? "当前筛选下没有图片，试试切右上角筛选（比如「全部」）。"
    : `${CFG.emptyImage}：点右上角「导入素材」上传图片和视频，或在上传弹层里「从素材池添加」现成素材。`;
  emptyBox.classList.toggle("hidden", items.length > 0);
  renderProgress();
  renderFilmstrip();
  if (!items.length) {
    state.item = null; state.idx = -1; state.boxes = []; state.sel = -1;
    renderPanel(); draw();
    return;
  }
  const pos = items.findIndex((i) => i.image === current);
  selectIndex(pos >= 0 ? pos : 0);
}

function renderProgress() {
  const items = libItems();
  const counts = { approved: 0, rejected: 0, edited: 0, unreviewed: 0 };
  let boxes = 0;
  items.forEach((i) => {
    const status = i[pk("status")] || "unreviewed";
    counts[status] = (counts[status] || 0) + 1;
    boxes += Number(i[pk("objects")] || 0);
  });
  const done = counts.approved + counts.rejected + counts.edited;
  const pct = items.length ? (done / items.length) * 100 : 0;
  $("progress-bar").style.width = pct.toFixed(1) + "%";
  $("progress-track").classList.toggle("on", pct > 0);  // 一张都没处理时别在顶栏挂条灰线
  $("progress-text").textContent =
    `已复核 ${done}/${items.length} (${pct.toFixed(1)}%) · 接受 ${counts.approved} · 驳回 ${counts.rejected}`
    + ` · 修改 ${counts.edited} · 框 ${boxes}`;
  renderQueueCounts();
}

function renderFilmstrip() {
  const strip = $("filmstrip");
  strip.innerHTML = "";
  const from = Math.max(0, state.idx - 60), to = Math.min(state.list.length, state.idx + 60);
  for (let i = from; i < to; i++) {
    const it = state.list[i];
    const flagged = issueFlags(it, pk("flags")).length > 0;
    const status = it[pk("status")];
    const div = document.createElement("div");
    div.className = "thumb" + (i === state.idx ? " active" : "")
      + (flagged ? " flagged" : "")
      + (status === "approved" ? " approved" : "")
      + (status === "rejected" ? " rejected" : "");
    div.title = it.image + "\n" + (STATUS_LABEL[status] || status)
      + "\n框 " + (it[pk("objects")] === null || it[pk("objects")] === undefined
        ? "未标" : it[pk("objects")])
      + "\n" + (it[pk("flags")] || []).map(flagLabel).join("、");
    const img = document.createElement("img");
    img.src = "/thumb/" + encodeURIComponent(it.image);
    img.loading = "lazy";
    const meta = document.createElement("span");
    meta.className = "thumb-meta";
    const fn = document.createElement("span");
    fn.className = "fn";
    fn.textContent = String(it.image).split("/").pop();
    meta.append(fn);
    const line = document.createElement("span");
    line.className = "st";
    line.textContent = statusLabel(status);
    const boxes = it[pk("objects")];
    if (issueFlags(it, pk("flags")).length) {
      const em = document.createElement("em");
      em.textContent = issueFlags(it, pk("flags")).length + " 项标记";
      line.append(document.createTextNode(" · "), em);
    } else {
      line.textContent += " · " + (boxes === null || boxes === undefined ? "未标" : boxes + " 框");
    }
    meta.append(line);
    div.append(img, meta);
    div.onclick = () => selectIndex(i);
    strip.append(div);
  }
  const active = strip.querySelector(".thumb.active");
  if (active) active.scrollIntoView({ inline: "center", block: "nearest" });
}

// --------------------------------------------------------------------------- //
// 单张图片的载入、绘制与编辑
// --------------------------------------------------------------------------- //
async function selectIndex(i, keep) {
  if (!state.list.length) { state.item = null; state.boxes = []; renderPanel(); draw(); return; }
  if (state.dirty) await save(true);
  state.idx = clamp(i, 0, state.list.length - 1);
  const meta = state.list[state.idx];
  $("loading").classList.remove("hidden");
  let item;
  try {
    item = await getJson("/api/prompt/item?image=" + encodeURIComponent(meta.image)
      + "&lib=" + encodeURIComponent(CFG.lib));
  } catch (err) {
    $("loading").classList.add("hidden");
    toast("读取标注失败：" + err.message, true);
    return;
  }
  state.item = item;
  state.boxes = (item.objects || []).map((o) => ({ ...o, flags: o.flags || [] }));
  state.sel = state.boxes.length ? 0 : -1;
  await loadImage(meta.image);
  fitView();
  draw();
  renderPanel();
  renderFilmstrip();
  renderProgress();
  $("loading").classList.add("hidden");
  $("hint").textContent = meta.image
    + (item.time_sec !== null && item.time_sec !== undefined ? ` · t=${item.time_sec}s` : "")
    + (item.error ? ` · ${item.error}` : "");
  if (!keep) setSaveState(SAVE_STATE.idle);
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
  const stage = $("stage");
  canvas.width = Math.max(320, stage.clientWidth);
  canvas.height = Math.max(240, stage.clientHeight);
}

function fitView() {
  if (!state.img) { state.view = { scale: 1, ox: 0, oy: 0 }; return; }
  const scale = Math.min(canvas.width / state.img.width, canvas.height / state.img.height);
  state.view.scale = scale;
  state.view.ox = (canvas.width - state.img.width * scale) / 2;
  state.view.oy = (canvas.height - state.img.height * scale) / 2;
}

const toScreen = (x, y) => [x * state.view.scale + state.view.ox, y * state.view.scale + state.view.oy];

window.addEventListener("appearancechange", () => { draw(); });

/** 框上的置信度：没拿到分数就显示 —，别拿 0.00 冒充「模型很有把握地给了 0 分」。
 *
 *  detection 标签只有 5 列，不落这一列；分数靠索引里按几何签名存的那张表认回来，
 *  认不到就是真没有（老数据、或框被挪过位置）。
 */
const boxScore = (v) => (v === null || v === undefined || v === "" ? "—" : Number(v).toFixed(2));

/** 分割轮廓（像素点列）：少于 3 个点就不画，残缺轮廓画出来是幽灵多边形。 */
const polyPoints = (box) => (Array.isArray(box.poly) && box.poly.length > 2 ? box.poly : []);

/** 有轮廓时方框改成虚线细描边：轮廓才是主角，但框还得看得见、拖得动。 */
function drawPolygon(points, color, selected) {
  ctx.beginPath();
  points.forEach(([px, py], i) => {
    const [sx, sy] = toScreen(px, py);
    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
  });
  ctx.closePath();
  ctx.fillStyle = color + (selected ? "3a" : "1a");
  ctx.fill();
  ctx.lineWidth = selected ? 2.4 : 1.6;
  ctx.strokeStyle = color;
  ctx.stroke();
}

async function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.img) return;
  const v = state.view;
  ctx.drawImage(state.img, state.view.ox, state.view.oy,
                state.img.width * v.scale, state.img.height * v.scale);
  const showLabels = $("show-labels") ? $("show-labels").checked : true;
  state.boxes.forEach((box, index) => {
    const [x1, y1] = toScreen(box.xyxy[0], box.xyxy[1]);
    const [x2, y2] = toScreen(box.xyxy[2], box.xyxy[3]);
    const color = fieldColor(box.cls);
    const selected = index === state.sel;
    const poly = polyPoints(box);
    ctx.lineWidth = selected ? 2.5 : 1.6;
    ctx.strokeStyle = color;
    if (poly.length) {
      drawPolygon(poly, color, selected);
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    } else {
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = color + (selected ? "44" : "1f");
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
    if (selected) {
      ctx.fillStyle = color;
      for (const [hx, hy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) {
        ctx.fillRect(hx - 3.5, hy - 3.5, 7, 7);
      }
      if (poly.length) drawPolyHandles(poly, color);
    }
    if (!showLabels) return;
    const small = (box.flags || []).includes("small_box") ? " ⚠" : "";
    const edge = (box.flags || []).includes("edge_box") ? " ✂" : "";
    const shape = poly.length ? " ◇" : "";
    const label = `${fieldName(box.cls)} · ${boxScore(box.score)}${small}${edge}${shape}`;
    ctx.font = cfont(12);
    const w = ctx.measureText(label).width + 8;
    const top = Math.max(0, y1 - 16);
    ctx.fillStyle = canvasPalette().veil;
    ctx.fillRect(x1, top, w, 15);
    ctx.fillStyle = color;
    ctx.fillText(label, x1 + 4, top + 11);
  });
}

function boxAt(mx, my) {
  for (let i = state.boxes.length - 1; i >= 0; i--) {
    const b = state.boxes[i];
    const [x1, y1] = toScreen(b.xyxy[0], b.xyxy[1]);
    const [x2, y2] = toScreen(b.xyxy[2], b.xyxy[3]);
    if (mx >= x1 - 4 && mx <= x2 + 4 && my >= y1 - 4 && my <= y2 + 4) return i;
  }
  return -1;
}

/** 轮廓顶点（屏幕坐标）与边中点：前者能拖、后者拖一下就插一个新点。 */
function polyHandles(box) {
  const points = polyPoints(box);
  const vertices = points.map(([px, py]) => toScreen(px, py));
  const middles = vertices.map((point, index) => {
    const next = vertices[(index + 1) % vertices.length];
    return [(point[0] + next[0]) / 2, (point[1] + next[1]) / 2];
  });
  return { vertices, middles };
}

function drawPolyHandles(poly, color) {
  const { vertices, middles } = polyHandles({ poly });
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = color;
  ctx.fillStyle = canvasPalette().veil;
  for (const [hx, hy] of middles) {
    ctx.beginPath();
    ctx.arc(hx, hy, 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
  ctx.fillStyle = color;
  for (const [hx, hy] of vertices) {
    ctx.beginPath();
    ctx.arc(hx, hy, 4.2, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** 轮廓顶点的命中测试（按屏幕像素给容差，缩得再小也点得着）。 */
function vertexAt(mx, my) {
  if (state.sel < 0) { return null; }
  const box = state.boxes[state.sel];
  if (!box || !polyPoints(box).length) { return null; }
  const { vertices, middles } = polyHandles(box);
  for (let index = 0; index < vertices.length; index += 1) {
    if (Math.abs(mx - vertices[index][0]) <= HANDLE_PX
      && Math.abs(my - vertices[index][1]) <= HANDLE_PX) return { kind: "vertex", index };
  }
  for (let index = 0; index < middles.length; index += 1) {
    if (Math.abs(mx - middles[index][0]) <= HANDLE_PX
      && Math.abs(my - middles[index][1]) <= HANDLE_PX) return { kind: "middle", index };
  }
  return null;
}

/** 像素点列（画布口径）写回框，并把方框收成轮廓的外接框——YOLO 那 4 列必须包住轮廓。 */
function setPoly(box, points) {
  const clean = points.map(([px, py]) => [
    Number(clamp(px, 0, state.img.width).toFixed(2)),
    Number(clamp(py, 0, state.img.height).toFixed(2)),
  ]);
  box.poly = clean;
  const xs = clean.map((p) => p[0]);
  const ys = clean.map((p) => p[1]);
  const [x1, x2] = [Math.min(...xs), Math.max(...xs)];
  const [y1, y2] = [Math.min(...ys), Math.max(...ys)];
  const w = Math.max(0.001, (x2 - x1) / state.img.width);
  const h = Math.max(0.001, (y2 - y1) / state.img.height);
  box.cx = Number((((x1 + x2) / 2) / state.img.width).toFixed(6));
  box.cy = Number((((y1 + y2) / 2) / state.img.height).toFixed(6));
  box.w = Number(w.toFixed(6));
  box.h = Number(h.toFixed(6));
  box.xyxy = [Number(x1.toFixed(2)), Number(y1.toFixed(2)),
              Number(x2.toFixed(2)), Number(y2.toFixed(2))];
  return box;
}

/** 方框被动过时轮廓跟着走：按新老方框做仿射，和后端 carry_polygons 同一套算法。 */
function rescalePoly(points, from, to) {
  if (!points || !points.length || from[2] <= from[0] || from[3] <= from[1]) return points || [];
  const sx = (to[2] - to[0]) / (from[2] - from[0]);
  const sy = (to[3] - to[1]) / (from[3] - from[1]);
  return points.map(([px, py]) => [
    Number((to[0] + (px - from[0]) * sx).toFixed(2)),
    Number((to[1] + (py - from[1]) * sy).toFixed(2)),
  ]);
}

function handleAt(mx, my) {
  if (state.sel < 0) return null;
  const b = state.boxes[state.sel];
  const [x1, y1] = toScreen(b.xyxy[0], b.xyxy[1]);
  const [x2, y2] = toScreen(b.xyxy[2], b.xyxy[3]);
  const corners = { nw: [x1, y1], ne: [x2, y1], sw: [x1, y2], se: [x2, y2] };
  for (const [key, [hx, hy]] of Object.entries(corners)) {
    if (Math.abs(mx - hx) <= HANDLE_PX && Math.abs(my - hy) <= HANDLE_PX) return key;
  }
  return null;
}

function eventPoint(e) {
  const rect = canvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

function toWorldNorm(mx, my) {
  if (!state.img) return [0, 0];
  return [clamp((mx - state.view.ox) / (state.view.scale * state.img.width), 0, 1),
          clamp((my - state.view.oy) / (state.view.scale * state.img.height), 0, 1)];
}

function bindCanvas() {
  canvas.addEventListener("wheel", (e) => {
    if (!state.img) return;
    e.preventDefault();
    const [mx, my] = eventPoint(e);
    const [wx, wy] = toWorldNorm(mx, my);
    const next = clamp(state.view.scale * (e.deltaY < 0 ? 1.12 : 1 / 1.12), 0.05, 24);
    state.view.scale = next;
    const [sx, sy] = toScreen(wx * state.img.width, wy * state.img.height);
    state.view.ox += mx - sx;
    state.view.oy += my - sy;
    draw();
  }, { passive: false });

  canvas.addEventListener("mousedown", (e) => {
    if (!state.img) return;
    const [mx, my] = eventPoint(e);
    if (e.button === 1 || state.spaceDown) {
      state.drag = { mode: "pan", mx, my, ox: state.view.ox, oy: state.view.oy };
      return;
    }
    const spot = vertexAt(mx, my);
    if (spot) {
      const box = state.boxes[state.sel];
      const points = polyPoints(box).map((p) => p.slice());
      if (e.altKey && spot.kind === "vertex") {
        if (points.length <= 3) { toast("轮廓最少留 3 个点", true); return; }
        points.splice(spot.index, 1);
        setPoly(box, points);
        markDirty(); renderBoxes(); draw();
        toast(`删掉一个顶点，还剩 ${points.length} 个`);
        return;
      }
      if (spot.kind === "middle") {
        const a = points[spot.index];
        const b = points[(spot.index + 1) % points.length];
        points.splice(spot.index + 1, 0, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
      }
      state.drag = { mode: "vertex", index: spot.kind === "vertex" ? spot.index : spot.index + 1, points };
      return;
    }
    const handle = handleAt(mx, my);
    if (handle) {
      const box = state.boxes[state.sel];
      state.drag = { mode: "resize", handle, from: box.xyxy.slice(), poly: polyPoints(box).map((p) => p.slice()) };
      return;
    }
    const hit = boxAt(mx, my);
    if (hit >= 0) {
      state.sel = hit;
      state.drag = { mode: "move", mx, my, box: { ...state.boxes[hit] },
                     from: state.boxes[hit].xyxy.slice(),
                     poly: polyPoints(state.boxes[hit]).map((p) => p.slice()) };
      renderBoxes(); draw();
      return;
    }
    if (!state.drawField) { toast(`先在右侧选「绘制${CFG.fieldWord}」，再拖框补画目标`, true); return; }
    const field = fieldByClass(state.drawField);
    const [nx, ny] = toWorldNorm(mx, my);
    state.drag = { mode: "new", nx, ny };
    draw();
  });

  window.addEventListener("mousemove", (e) => {
    if (!state.img) return;
    const [mx, my] = eventPoint(e);
    if (!state.drag) { hoverCursor(mx, my); return; }
    const drag = state.drag;
    if (drag.mode === "pan") {
      state.view.ox = drag.ox + (mx - drag.mx);
      state.view.oy = drag.oy + (my - drag.my);
      draw();
      return;
    }
    const [nx, ny] = toWorldNorm(mx, my);
    if (drag.mode === "new") {
      const box = normBox(state.drawField, drag.nx, drag.ny, nx, ny);
      state.preview = box;
      drawPreview(box);
      return;
    }
    const box = state.boxes[state.sel];
    if (!box) return;
    if (drag.mode === "vertex") {
      const points = drag.points.map((p) => p.slice());
      points[drag.index] = [nx * state.img.width, ny * state.img.height];
      setPoly(box, points);
      markDirty(); draw();
      return;
    }
    const [bx1, by1, bx2, by2] = toWorldBox(box);
    let next;
    if (drag.mode === "move") {
      const dx = (mx - drag.mx) / (state.view.scale * state.img.width);
      const dy = (my - drag.my) / (state.view.scale * state.img.height);
      const w = bx2 - bx1, h = by2 - by1;
      next = normBox(box.cls, clamp(bx1 + dx, 0, 1 - w), clamp(by1 + dy, 0, 1 - h),
                     clamp(bx1 + dx, 0, 1 - w) + w, clamp(by1 + dy, 0, 1 - h) + h);
    } else {
      let [x1, y1, x2, y2] = [bx1, by1, bx2, by2];
      if (drag.handle.includes("w")) x1 = nx;
      if (drag.handle.includes("e")) x2 = nx;
      if (drag.handle.includes("n")) y1 = ny;
      if (drag.handle.includes("s")) y2 = ny;
      next = normBox(box.cls, x1, y1, x2, y2);
    }
    Object.assign(box, next);
    // 拖的是方框也不能把轮廓丢在原地：按新老方框仿射过去（后端同口径，两边不会打架）
    if (drag.poly && drag.poly.length) box.poly = rescalePoly(drag.poly, drag.from, box.xyxy);
    markDirty(); draw();
  });

  window.addEventListener("mouseup", async () => {
    const drag = state.drag;
    state.drag = null;
    if (!drag) return;
    if (drag.mode === "new") {
      const box = state.preview;
      state.preview = null;
      draw();
      if (!box) return;
      if (box.w < 0.004 || box.h < 0.004) { toast("框太小，已忽略（按 X 删掉多余的框）"); return; }
      state.boxes.push(box);
      state.sel = state.boxes.length - 1;
      markDirty(); renderBoxes(); draw();
      toast(`已补画一个 ${fieldName(box.cls)}`);
    } else if (drag.mode === "move" || drag.mode === "resize" || drag.mode === "vertex") {
      markDirty(); renderBoxes();
    }
  });
}

/** 光标形状就是可拖性提示：轮廓点、方框角点、平移，各给一种。 */
function hoverCursor(mx, my) {
  const want = vertexAt(mx, my) ? "grab"
    : (handleAt(mx, my) ? "nwse-resize" : (boxAt(mx, my) >= 0 ? "move" : ""));
  if (canvas.style.cursor !== want) { canvas.style.cursor = want; }
}

function toWorldBox(box) {
  const [x1, y1, x2, y2] = box.xyxy;
  const w = state.img.width, h = state.img.height;
  return [x1 / w, y1 / h, x2 / w, y2 / h];
}

/** 任意两点 → 归一化检测框（带像素 xyxy，前端画与提交都用它）。 */
function normBox(cls, x1, y1, x2, y2) {
  const cx1 = Math.min(x1, x2), cx2 = Math.max(x1, x2);
  const cy1 = Math.min(y1, y2), cy2 = Math.max(y1, y2);
  const w = Math.max(0.001, cx2 - cx1), h = Math.max(0.001, cy2 - cy1);
  const width = state.item ? state.item.width : 1;
  const height = state.item ? state.item.height : 1;
  return {
    cls: Number(cls), cx: Number(((cx1 + cx2) / 2).toFixed(6)), cy: Number(((cy1 + cy2) / 2).toFixed(6)),
    w: Number(w.toFixed(6)), h: Number(h.toFixed(6)), score: 1, flags: ["manual"],
    xyxy: [cx1 * width, cy1 * height, cx2 * width, cy2 * height].map((v) => Number(v.toFixed(2))),
  };
}

function drawPreview(box) {
  draw();
  if (!box) return;
  const [x1, y1] = toScreen(box.xyxy[0], box.xyxy[1]);
  const [x2, y2] = toScreen(box.xyxy[2], box.xyxy[3]);
  ctx.setLineDash([5, 4]);
  ctx.strokeStyle = fieldColor(box.cls);
  ctx.lineWidth = 1.8;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
}

function markDirty() {
  state.dirty = true;
  setSaveState(SAVE_STATE.dirty);
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => save(false), 900);
}

function setSaveState(text) { $("save-state").textContent = text; }

async function save(silent) {
  if (!state.item || !state.dirty) return;
  state.dirty = false;
  setSaveState(SAVE_STATE.saving);
  clearTimeout(state.saveTimer);
  const objects = state.boxes.map((b) => ({
    cls: b.cls, cx: b.cx, cy: b.cy, w: b.w, h: b.h, score: b.score, flags: b.flags,
    // 拖过顶点的轮廓原样交回（像素点）；没轮廓就是空表，后端照旧按 IoU 从磁盘认领
    poly: polyPoints(b),
  }));
  try {
    const data = await postJson("/api/prompt/item",
      { image: state.item.image, objects, lib: CFG.lib });
    applyItemUpdate(data);
    setSaveState(SAVE_STATE.saved);
    if (!silent) toast("已保存");
  } catch (err) {
    state.dirty = true;
    setSaveState(SAVE_STATE.dirty);
    toast("保存失败：" + err.message, true);
  }
}

/** 后端会重算标记与状态，这里把返回值同步回本地与列表。 */
function applyItemUpdate(data) {
  const meta = libItems().find((i) => i.image === state.item.image);
  if (meta) {
    meta[pk("objects")] = data[pk("objects")];
    meta[pk("status")] = data[pk("status")];
    meta[pk("flags")] = data[pk("flags")] || [];
    meta[pk("score")] = data[pk("score")];
  }
  state.item.status = data[pk("status")];
  state.item.flags = data[pk("flags")] || [];
  state.item[pk("status")] = data[pk("status")];
  if (data.objects) {
    state.boxes = data.objects.map((o) => ({ ...o, flags: o.flags || [] }));
    if (state.sel >= state.boxes.length) state.sel = state.boxes.length - 1;
  }
  renderBoxes(); renderFlags(); renderFilmstrip(); renderProgress(); draw();
}

async function setStatus(status) {
  if (!state.item) return;
  if (state.dirty) await save(true);
  try {
    await postJson("/api/prompt/status", { image: state.item.image, status, lib: CFG.lib });
  } catch (err) {
    toast("改状态失败：" + err.message, true);
    return;
  }
  const target = state.item.image;
  const previous = state.item[pk("status")] || state.item.status || "unreviewed";
  state.item.status = status;
  const meta = libItems().find((i) => i.image === target);
  if (meta) meta[pk("status")] = status;
  const said = status === "approved" ? "已接受" : "已驳回";
  toast(said);
  // 撤销必须认准刚才那一帧：这一页接受后会重建队列并翻页，
  // 直接调"撤销坐标修改"会把下一帧的框回滚掉，而接受 / 驳回照旧留着。
  showUndoBar(`${said} · ${String(target).split("/").pop()} · 点撤销退回「${statusLabel(previous)}」`,
    () => revertStatus(target, previous));
  if (currentFilter() !== "all") {
    const nextImage = state.list[state.idx + (status === "rejected" ? 0 : 1)];
    rebuildList(nextImage ? nextImage.image : null);
  } else {
    renderFilmstrip(); renderProgress(); renderFlags();
  }
}

/** 撤销条：把那一帧的复核状态改回去，已经画好的框不动。 */
async function revertStatus(image, previous) {
  try {
    await postJson("/api/prompt/status", { image, status: previous, lib: CFG.lib });
  } catch (err) {
    toast("撤销失败：" + err.message, true);
    return;
  }
  const row = libItems().find((i) => i.image === image);
  if (row) row[pk("status")] = previous;
  toast("已撤销，这一帧回到" + statusLabel(previous));
  await refreshDataset(image);
}

async function undoItem() {
  if (!state.item) return;
  const ok = await confirmBox({
    title: "撤销上一次保存",
    body: "把检测框回滚到上一次保存前的版本：删多了的框能救回来，刚补画的框会被撤掉。"
      + "没有备份时（从没改过这张图）会提示没有可回滚的版本。",
    okLabel: "撤销",
  });
  if (!ok) return;
  state.dirty = false;
  clearTimeout(state.saveTimer);
  try {
    await postJson("/api/prompt/undo", { image: state.item.image, lib: CFG.lib });
  } catch (err) {
    toast("撤销失败：" + err.message, true);
    return;
  }
  hideUndoBar();
  toast("已撤销");
  await selectIndex(state.idx);
  await refreshDataset(state.item.image);
}

function deleteBox() {
  if (state.sel < 0 || !state.boxes.length) { toast("先点选一个框再删除", true); return; }
  const removed = state.boxes.splice(state.sel, 1)[0];
  state.sel = state.boxes.length ? clamp(state.sel, 0, state.boxes.length - 1) : -1;
  markDirty(); renderBoxes(); draw();
  toast(`已删除 ${fieldName(removed.cls)} 的框`);
}

function cycleBox(dir) {
  if (!state.boxes.length) return;
  state.sel = (state.sel + dir + state.boxes.length) % state.boxes.length;
  renderBoxes(); draw();
}

function jumpFlagged(dir) {
  // 与顶栏「待确认 N」同口径：带标记或还没标过框都算，正向标记不算
  for (let i = state.idx + dir; i >= 0 && i < state.list.length; i += dir) {
    if (needsAttention(state.list[i])) { selectIndex(i); return; }
  }
  toast("没有更多待确认的图片了");
}

function onKeyDown(e) {
  if (shellKeys(e)) return;
  const confirm = $("confirm-modal");
  if (confirm && !confirm.classList.contains("hidden")) {
    if (e.key === "Escape") { e.preventDefault(); closeConfirm(false); }
    else if (e.key === "Enter") { e.preventDefault(); closeConfirm(true); }
    return;
  }
  for (const id of ["pool-modal", "field-modal", "run-modal", "up-modal"]) {
    const modal = $(id);
    if (modal && !modal.classList.contains("hidden")) {
      if (e.key === "Escape") openModal(id, false);
      return;
    }
  }
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  if (e.metaKey || e.ctrlKey) {
    // ⌘/Ctrl 加字母是浏览器与系统的（全选、收藏、页内查找、下载、书签栏、切标签页…），
    // 以前这些组合会顺手翻页、改关键点可见性、甚至跳去模型页。只按直觉接住撤销。
    if ((e.key === "z" || e.key === "Z") && !e.shiftKey && !e.altKey) {
      e.preventDefault();
      undoItem();;
    }
    return;
  }
  switch (e.key) {
    case " ": state.spaceDown = true; e.preventDefault(); break;
    case "ArrowRight": case "ArrowDown": case "d": selectIndex(state.idx + 1); break;
    case "ArrowLeft": case "ArrowUp": case "a": selectIndex(state.idx - 1); break;
    case "q": case "Q": setStatus("approved"); break;
    case "e": case "E": setStatus("rejected"); break;
    case "f": case "F": jumpFlagged(1); break;
    case "u": case "U": if (!runPendingUndo()) undoItem(); break;
    case "x": case "X": case "Delete": case "0": deleteBox(); break;
    case "z": case "Z": fitView(); draw(); break;
    case "[": cycleBox(-1); break;
    case "]": cycleBox(1); break;
    case "g": case "G": openRun(true); break;
    case "w": case "W": openUpload(true); break;
    case "m": case "M": location.href = "/models"; break;
    case "p": case "P": location.href = "/pose"; break;
    case "o": case "O": location.href = OTHER.page; break;
    case "1": openFields(true); break;
    default: return;
  }
}

// --------------------------------------------------------------------------- //
// 右侧面板
// --------------------------------------------------------------------------- //
function renderPanel() {
  renderMeta(); renderFlags(); renderBoxes(); renderFieldChoices();
}

function renderMeta() {
  const box = $("meta");
  if (!state.item) { box.textContent = "（没有选中图片）"; return; }
  const it = state.item;
  box.textContent = [
    `文件 ${it.image}`,
    `尺寸 ${it.width}×${it.height}`,
    `状态 ${STATUS_LABEL[it.status] || it.status}`,
    `框数 ${state.boxes.length}`,
    `平均置信度 ${dash(it.score === null || it.score === undefined ? "—" : Number(it.score).toFixed(3))}`,
    // 目标检测库把出框的权重记在索引里，复核时要能说清这批框是谁画的
    it.model ? `出框权重 ${it.model}` : "",
    it.stale ? "本库类别表 / 权重已变更，建议重标" : "",
    it.video ? `来源视频 ${it.video}` : "来源 图片",
    it.has_backup ? "有可撤销的备份" : "暂无备份",
  ].filter(Boolean).join("\n");
  box.style.whiteSpace = "pre-wrap";
}

function renderFlags() {
  const box = $("flags");
  box.innerHTML = "";
  for (const flag of state.item ? state.item.flags || [] : []) {
    const el = document.createElement("div");
    el.className = "flag";
    el.textContent = flagLabel(flag);
    el.title = flag;
    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = flagHint(flag) || "（未知标记）";
    el.append(hint);
    box.append(el);
  }
}

function renderBoxes() {
  const box = $("boxes");
  if (!box) return;
  box.innerHTML = "";
  $("box-count").textContent = state.boxes.length ? ` ${state.boxes.length} 个` : "（0 个）";
  if (!state.boxes.length) {
    box.innerHTML = `<div class="muted small">还没有框：空白处拖动画一个，`
      + `或点「${CFG.runLabel}」让${IS_LOCAL ? "本机权重" : "大模型"}出框。</div>`;
    return;
  }
  state.boxes.forEach((boxItem, index) => {
    const row = document.createElement("div");
    row.className = "obj" + (index === state.sel ? " selected" : "");
    const head = document.createElement("div");
    head.className = "obj-head";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = `#${index + 1} `;
    const dot = document.createElement("span");
    dot.style.color = fieldColor(boxItem.cls);
    dot.textContent = fieldName(boxItem.cls);
    name.append(dot);
    const chips = document.createElement("span");
    chips.textContent = ` ${boxScore(boxItem.score)} `;
    for (const flag of boxItem.flags || []) {
      const chip = document.createElement("span");
      // 正向标记（人工画的框、轮廓已落盘）不染红：它们是说明，不是问题
      chip.className = "chip" + (flagPolicy.positive.has(flag) || flag === "manual" ? "" : " warn");
      chip.textContent = flagLabel(flag);
      chip.title = flag + " · " + (flagHint(flag) || "人工确认");
      chips.append(chip);
    }
    head.append(name, chips);
    head.onclick = () => { state.sel = index; renderBoxes(); draw(); };
    const controls = document.createElement("div");
    controls.className = "kpt-row";
    const select = document.createElement("select");
    select.title = "改成别的" + CFG.fieldWord;
    for (const option of fieldOptions()) {
      const el = document.createElement("option");
      el.value = option.class_id;
      el.textContent = `${option.name}（${option.label || option.name}）`;
      el.selected = Number(option.class_id) === Number(boxItem.cls);
      select.append(el);
    }
    select.onchange = (e) => {
      boxItem.cls = Number(e.target.value);
      boxItem.flags = (boxItem.flags || []).filter((f) => f !== "unknown_class");
      markDirty(); renderBoxes(); draw();
      toast(`#${index + 1} 改成 ${fieldName(boxItem.cls)}`);
    };
    const del = document.createElement("button");
    del.className = "vbtn v0";
    del.textContent = "删除";
    del.onclick = (e) => { e.stopPropagation(); state.sel = index; deleteBox(); };
    controls.append(select, del);
    row.append(head, controls);
    box.append(row);
  });
}

function fieldOptions() {
  return fieldRows().slice();
}

function renderFieldChoices() {
  const options = fieldOptions();
  for (const id of ["draw-field"]) {
    const select = $(id);
    if (!select) continue;
    const previous = select.value;
    select.innerHTML = "";
    if (!options.length) {
      const el = document.createElement("option");
      el.value = "";
      el.textContent = `（还没有${CFG.fieldWord}，先点「${CFG.manageLabel}」新建）`;
      select.append(el);
      select.disabled = true;
      continue;
    }
    select.disabled = false;
    for (const option of options) {
      const el = document.createElement("option");
      el.value = option.class_id;
      el.textContent = `${option.name}（${option.label || option.name}）· 已标 ${option.boxes} 框`;
      select.append(el);
    }
    const restore = options.some((f) => String(f.class_id) === String(previous));
    state.drawField = restore ? previous : String(options[0].class_id);
    select.value = state.drawField;
  }
}

// --------------------------------------------------------------------------- //
// 字段管理
// --------------------------------------------------------------------------- //
function openModal(id, open) {
  const el = $(id);
  if (el) el.classList.toggle("hidden", !open);
}

function openFields(open) {
  openModal("field-modal", open);
  if (open) {
    renderFieldPanel();
    // 类别管理面板第一件事是选权重：清单没取过就取一次，取完自动列出它自带的类别
    if (IS_LOCAL) loadModelChoices();
  }
}

function initFieldPanel() {
  $("field-close").onclick = () => openFields(false);
  $("field-reset").onclick = () => fillFieldForm(null);
  $("field-save").onclick = saveField;
  $("field-name").onkeydown = (e) => { if (e.key === "Enter") saveField(); };
}

function renderFieldPanel() {
  const box = $("field-list");
  const rows = fieldOptions();
  const info = (state.ds && state.ds[CFG.classes]) || {};
  box.innerHTML = "";
  if (!rows.length) {
    box.innerHTML = `<div class="batch-empty muted small">还没有${CFG.fieldWord}：`
      + (IS_LOCAL ? "上面选一个本机权重，把它自带的类别勾几个「导入所选」。"
        : "在下面填英文字段名 + 中文显示名 + 描述提示词，点「保存字段」。")
      + "</div>";
  }
  for (const row of rows) {
    const el = document.createElement("div");
    el.className = "model-row";
    el.innerHTML = '<span class="dot"></span><div class="model-main"><div class="model-name"></div>'
      + '<div class="muted small model-meta"></div></div><span class="chip"></span>'
      + '<button class="edit">编辑</button><button class="danger del">删除</button>';
    el.querySelector(".dot").style.background = row.color;
    el.querySelector(".model-name").textContent = `${row.name} · ${row.label || row.name}`;
    el.querySelector(".model-meta").textContent = `class_id ${row.class_id} · `
      + (row.source ? `来自 ${row.source} · ` : "")
      + (row.prompt_hint ? row.prompt_hint
        : (IS_LOCAL ? "（模型自带类别，出框只看类别号）" : "（还没写描述提示词）"));
    el.querySelector(".chip").textContent = `${row.boxes} 框`;
    el.querySelector(".edit").onclick = (e) => { e.preventDefault(); fillFieldForm(row); };
    el.querySelector(".del").onclick = (e) => { e.preventDefault(); deleteField(row); };
    box.append(el);
  }
  if ((info.orphans || []).length) {
    const warn = document.createElement("div");
    warn.className = "flag";
    warn.textContent = `标签里还有孤儿类别号：${info.orphans.join("、")}`;
    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = `这些类别号在${CFG.classTable}里已经没有了（${CFG.fieldWord}被删过或手工改过盘），导出时会被跳过。`;
    warn.append(hint);
    box.append(warn);
  }
}

function fillFieldForm(row) {
  fieldState.editing = row ? row.id : "";
  $("field-id").value = row ? row.id : "";
  $("field-name").value = row ? row.name : "";
  $("field-label").value = row ? (row.label || "") : "";
  $("field-hint").value = row ? (row.prompt_hint || "") : "";
  $("field-log").textContent = row ? `正在编辑「${row.name}」，class_id 固定为 ${row.class_id}` : "";
  if ($("field-hint")) $("field-hint").disabled = false;
  $("field-name").focus();
}

async function saveField() {
  const payload = {
    name: $("field-name").value.trim(),
    label: $("field-label").value.trim(),
    prompt_hint: $("field-hint").value.trim(),
  };
  if ($("field-id").value) payload.id = $("field-id").value;
  if (!payload.name) { fieldLog(`先填${CFG.fieldWord}名（英文）`, true); return; }
  let data;
  try {
    data = await postJson("/api/fields/save", { field: payload, lib: CFG.lib });
  } catch (err) {
    fieldLog("保存失败：" + err.message, true);
    return;
  }
  fieldLog(`已保存「${data.field.name}」 · class_id ${data.field.class_id}`
    + ` · 本库共 ${data.fields.length} 个${CFG.fieldWord}`);
  await refreshDataset(state.item ? state.item.image : null);
  renderFieldPanel();
  renderRunFields();
  if (IS_LOCAL) renderClassImportList();
  toast(`${CFG.fieldWord} ${data.field.name} 已保存`);
}

async function deleteField(row) {
  const hasBoxes = Number(row.boxes || 0) > 0;
  const body = hasBoxes
    ? `「${row.name}」还有 ${row.boxes} 个已标注的框。删除${CFG.fieldWord}会连同这些框一起从 ${CFG.labelDir} 里清掉，`
      + "类别号 " + row.class_id + " 之后不会被复用。此操作无法在界面里撤销。"
    : `将删除${CFG.fieldWord}「${row.name}」（class_id ${row.class_id}）。已标注的数据不受影响。`;
  const ok = await confirmBox({ title: `删除${CFG.fieldWord}`, body,
    okLabel: hasBoxes ? "连框一起删" : "确定删除" });
  if (!ok) return;
  try {
    await postJson("/api/fields/delete", { id: row.id, drop_boxes: hasBoxes, lib: CFG.lib });
  } catch (err) {
    toast(`删除${CFG.fieldWord}失败：` + err.message, true);
    return;
  }
  toast(`已删除${CFG.fieldWord} ${row.name}`);
  if (String(state.drawField) === String(row.class_id)) state.drawField = "";
  await refreshDataset(state.item ? state.item.image : null);
  renderFieldPanel();
  renderRunFields();
  if (IS_LOCAL) renderClassImportList();
}

function fieldLog(msg, isErr) {
  const box = $("field-log");
  box.textContent = msg || "";
  box.classList.toggle("err", Boolean(isErr));
}

// --------------------------------------------------------------------------- //
// 目标检测专属：本机权重 → 它自带的类别表 → 导进本库类别表
// 界面顺序就是这里的调用顺序：先挑权重，再勾它认识的类别，「批量目标检测」只吃本库这张表。
// 这一节在提示词标注页是彻底的空操作（页面上没有 #cls-* / #*-model 这些元素）。
// --------------------------------------------------------------------------- //

function initModelPanel() {
  const select = $("cls-model");
  if (select) {
    select.onchange = () => {
      modelState.model = select.value;
      renderModelChoices();
      loadModelClasses(select.value);
    };
  }
  const search = $("cls-search");
  if (search) search.oninput = () => renderClassImportList();
  const vocabInput = $("cls-vocab");
  if (vocabInput) vocabInput.oninput = () => { renderClassImportList(); };
  const reload = $("cls-reload");
  if (reload) reload.onclick = () => { modelState.listModel = ""; loadModelChoices(); };
  if ($("cls-vocab")) { $("cls-vocab").value = ""; }
  const all = $("cls-all");
  if (all) all.onclick = () => toggleAllClassRows(true);
  const clear = $("cls-clear");
  if (clear) clear.onclick = () => toggleAllClassRows(false);
  const invert = $("cls-invert");
  if (invert) invert.onclick = invertClassRows;
  const add = $("cls-import");
  if (add) add.onclick = importModelClasses;
  const probeSelect = $("probe-model");
  if (probeSelect) {
    probeSelect.onchange = () => {
      modelState.model = probeSelect.value;
      renderModelChoices();
    };
  }
}

// --------------------------------------------------------------------------- //
// 试词：这个词权重到底认不认、最高给了多少分。
//
// 「一个框都没有」以前有三种完全不同的原因却只有一句话：权重压根没这一类、
// 分被阈值挡了、图里确实没这个目标。下一步动作互不相同，所以这里一次分开报，
// 并且只跑一张图、只读不写——试错不该把已标好的框覆盖掉。
// --------------------------------------------------------------------------- //
const probe = { busy: false, data: null };

function initProbePanel() {
  if (!$("probe-modal")) { return; }   // 提示词页没这块：那边是问大模型，不问权重类别表
  $("btn-probe").onclick = () => openProbe(true);
  $("btn-probe-menu").onclick = () => { openMenu(false); openProbe(true); };
  $("probe-close").onclick = () => openProbe(false);
  $("probe-run").onclick = () => runProbe();
  $("probe-import").onclick = () => importProbedWords();
  $("probe-apply").onclick = applyProbeSuggestion;
}

function openProbe(open) {
  if (!$("probe-modal")) { return; }
  openModal("probe-modal", open);
  if (!open) { return; }
  renderModelChoices();
  const box = $("probe-words");
  // 每次都按当前选择重算默认词：类别管理里刚改过词表，这里再留旧词就是误导
  if (box) { box.value = prefillProbeWords(); }
  renderProbeTarget();
  renderProbeResult(probe.data);
}

/** 默认拿哪些词试：词表输入框 → 勾中的类别 → 本库类别表，一路兜底。 */
function prefillProbeWords() {
  const words = splitWords(($("cls-vocab") || {}).value || "");
  if (words.length) { return words.slice(0, 12).join(", "); }
  const picked = checkedFields("cls-list");
  if (picked.length) { return picked.slice(0, 12).join(", "); }
  return fieldOptions().map((row) => row.name).slice(0, 12).join(", ");
}

/** 用户往词框里打的分隔符不止一种：中文逗号、分号、换行都得认。 */
function splitWords(text) {
  return String(text || "").split(/[,，;；、\n\r]+/).map((v) => v.trim()).filter(Boolean);
}

function probeTargetImage() {
  if (state.item && state.item.image) { return String(state.item.image); }
  const items = libItems();
  return items.length ? String(items[0].image) : "";
}

function renderProbeTarget() {
  const rel = probeTargetImage();
  const el = $("probe-image");
  if (el) {
    el.textContent = rel
      ? `试的这张图：${rel}${state.item ? "" : "（当前没打开图，取队列里第一张）"}`
      : `${CFG.name}里还没有图：先导入图片再试词`;
  }
  const run = $("probe-run");
  if (run) { run.disabled = !rel || probe.busy; }
}

async function runProbe() {
  if (probe.busy) { return; }
  const words = String(($("probe-words") || {}).value || "").trim();
  if (!words) { toast("先说要试哪些词，比如 t-shirt, shirt, top", true); return; }
  const image = probeTargetImage();
  if (!image) { toast(`${CFG.name}里还没有图：先导入图片再试词`, true); return; }
  const payload = { image, words, model: ($("probe-model") || {}).value || modelState.model };
  const imgsz = ($("probe-imgsz") || {}).value;
  if (imgsz) { payload.imgsz = Number(imgsz); }
  probe.busy = true;
  renderProbeBusy(true);
  renderProbeTarget();
  try {
    probe.data = await postJson("/api/detect/probe", payload);
  } catch (err) {
    probe.data = null;
    toast("试检失败：" + err.message, true);
  }
  probe.busy = false;
  renderProbeBusy(false);
  renderProbeTarget();
  renderProbeResult(probe.data);
}

function renderProbeBusy(on) {
  const run = $("probe-run");
  if (!run) { return; }
  run.disabled = on;
  run.textContent = on ? "试检中…" : "开始试检";
}

function renderProbeResult(data) {
  const box = $("probe-result");
  const actions = $("probe-actions");
  if (!box) { return; }
  box.innerHTML = "";
  if (!data) {
    box.textContent = "填好词点「开始试检」：一个词一行，告诉你是权重没这一类、分被阈值挡了，"
      + "还是图里确实没这个目标。";
    if (actions) { actions.hidden = true; }
    return;
  }
  const head = document.createElement("div");
  head.className = "muted small";
  const vocab = data.open_vocab ? `开放词表（${data.open_vocab}）` : `权重自带 ${data.class_count} 类`;
  head.textContent = `${data.model} · ${data.task_name} · ${vocab} · 算力 ${data.device} · `
    + `分辨率 ${data.imgsz} · 阈值 ${data.conf}`;
  box.append(head);
  for (const row of data.words || []) {
    const line = document.createElement("div");
    line.className = "probe-row " + (row.over ? "ok" : (row.best > 0 ? "low" : "none"));
    const name = document.createElement("b");
    name.textContent = row.word;
    const bar = document.createElement("span");
    bar.className = "probe-bar";
    const fill = document.createElement("i");
    fill.style.width = Math.max(2, Math.min(100, Number(row.best) * 100)) + "%";
    bar.append(fill);
    const score = document.createElement("span");
    score.className = "mono";
    score.textContent = Number(row.best).toFixed(3);
    const boxes = document.createElement("span");
    boxes.textContent = row.over ? `${row.boxes} 框` : "无框";
    const advice = document.createElement("span");
    advice.className = "muted small probe-advice";
    advice.textContent = row.advice;
    line.append(name, bar, score, boxes, advice);
    box.append(line);
  }
  const usable = probedUsable();
  const foot = document.createElement("div");
  foot.className = "muted small";
  if (usable.length === 0) {
    foot.textContent = "这些词一个分都没给：先确认图里真有这个目标，"
      + "再把词换成更常见的英文写法（普通权重还得换开放词表权重）。";
  } else if (data.suggested_conf) {
    foot.textContent = `已经有词能出框；检不出框的词最高 ${data.missed_best}：把置信度降到 `
      + data.suggested_conf + " 就一起标得出来，下面两个按钮照顺序点。";
  } else {
    foot.textContent = `${usable.length} 个词都已经能出框：导入类别表就能直接开跑。`;
  }
  box.append(foot);
  if (actions) { actions.hidden = usable.length === 0; }
}

const probedUsable = () => ((probe.data || {}).words || []).filter((row) => Number(row.best) > 0);

/** 把试出来有分的词写进本库类别表（与「类别管理」的导入同一条路）。 */
async function importProbedWords() {
  const usable = probedUsable().map((row) => row.word);
  if (!usable.length) { return; }
  const button = $("probe-import");
  if (button) { button.disabled = true; }
  try {
    const data = await postJson("/api/detect/classes/add", {
      model: ($("probe-model") || {}).value || modelState.model, names: usable,
    });
    const lost = data.not_in_model || [];
    toast(`类别表已更新：新增 ${data.added || 0} · 复用 ${data.updated || 0}`
      + (lost.length ? ` · 这个权重认不出 ${lost.length} 个` : ""));
    await refreshDataset(state.item ? state.item.image : null);
    renderFieldPanel();
    renderRunFields();
    renderClassImportList();
  } catch (err) {
    toast("导入失败：" + err.message, true);
  }
  if (button) { button.disabled = false; }
}

/** 按建议阈值去标注：把算好的阈值填进批量面板并摊开它，省得用户回去手动改。 */
function applyProbeSuggestion() {
  const data = probe.data || {};
  openProbe(false);
  const conf = $("run-conf");
  if (conf && data.suggested_conf) { conf.value = Math.max(0.01, Number(data.suggested_conf)); }
  openRun(true);
  if (data.suggested_conf) { toast(`置信度已调到 ${data.suggested_conf}，勾好类别就能开跑`); }
}

/** 取本机权重清单（不加载模型，几百毫秒），顺带把上次用的权重/类别/阈值回填回来。 */
async function loadModelChoices() {
  if (!IS_LOCAL || modelState.loading) { return; }
  modelState.loading = true;
  try {
    const data = await getJson("/api/detect/models");
    modelState.models = data.models || [];
    modelState.unusable = data.unusable || [];
    modelState.device = data.device || "auto";
    const prefs = data.prefs || {};
    const known = modelState.models.map((row) => row.name);
    const previous = modelState.model;
    // 上次选的权重如今用不了（被删了 / 换成姿态权重）就退回第一个能用的，别留个空下拉框
    modelState.model = known.includes(previous) ? previous
      : (known.includes(prefs.model) ? prefs.model : (known[0] || ""));
    if (typeof prefs.conf === "number") modelState.conf = prefs.conf;
    if (Array.isArray(prefs.classes)) modelState.prefsClasses = prefs.classes.slice();
    renderModelChoices();
    if ($("cls-list") && modelState.model && modelState.listModel !== modelState.model) {
      loadModelClasses(modelState.model);
    }
  } catch (err) {
    modelState.note = "本机权重清单读取失败：" + err.message;
    modelState.models = [];
    renderModelChoices();
  }
  modelState.loading = false;
}

/** 把「用不了的权重」按原因归并：同一种原因只说一遍，文件名收成一行。
 *
 *  以前逐个拼「名字：原因」，本机一摆就是十一个姿态权重，屏幕上刷出十一模
 *  一样的两句话，真正要看的「人体骨骼请回姿态标注」被埋在最底下。原因相同的
 *  归成一条，名字最多列三个，完整清单挂在这行的悬停提示里。
 */
function unusableWeightsNote(rows) {
  const groups = new Map();
  for (const row of rows || []) {
    const reason = row.note || "这个任务的产出不是检测框";
    if (!groups.has(reason)) { groups.set(reason, []); }
    groups.get(reason).push(String(row.name || "未命名权重"));
  }
  const parts = [];
  const titles = [];
  for (const [reason, names] of groups) {
    const cut = names.length > 3;
    const listed = cut ? `${names.slice(0, 3).join("、")} 等 ${names.length} 个（悬停看全部）`
      : names.join("、");
    parts.push(names.length === 1 ? `${names[0]}：${reason}` : `${reason} —— ${listed}`);
    titles.push(`${names.join("、")}：${reason}`);
  }
  return {
    text: parts.length ? "用不了的权重 · " + parts.join("；") : "",
    title: titles.join("\n"),
  };
}

/** 三个下拉框（类别管理 / 批量运行 / 导入素材）共用同一份权重清单与同一个选择。 */
function renderModelChoices() {
  for (const id of ["cls-model", "run-model", "up-model", "probe-model"]) {
    const select = $(id);
    if (!select) { continue; }
    const previous = select.value || modelState.model;
    select.innerHTML = "";
    if (!modelState.models.length) {
      const el = document.createElement("option");
      el.value = "";
      el.textContent = "（models/ 里还没有能用在检测库的权重）";
      select.append(el);
      select.disabled = true;
      continue;
    }
    select.disabled = false;
    for (const row of modelState.models) {
      const el = document.createElement("option");
      el.value = row.name;
      el.textContent = `${row.name}（${row.task_name || row.task} · ${kb(row.bytes || 0)}）`;
      el.title = row.note || "";
      select.append(el);
    }
    select.value = modelState.models.some((row) => row.name === previous)
      ? previous : (modelState.model || modelState.models[0].name);
  }
  const active = ($("cls-model") || {}).value || ($("run-model") || {}).value
    || ($("probe-model") || {}).value || "";
  if (active) { modelState.model = active; }
  // 用不了的权重不许从界面上消失：摆在下拉框下面，说清它为什么不能用在检测库
  const unusable = unusableWeightsNote(modelState.unusable);
  for (const id of ["cls-model-note", "run-model-note", "up-model-note"]) {
    const el = $(id);
    if (!el) { continue; }
    el.textContent = unusable.text;
    el.title = unusable.title;
  }
}

/** 加载权重、列出它自带的类别（第一次要一两秒：得真把权重读进来才知道有哪些类）。 */
async function loadModelClasses(model) {
  const box = $("cls-list");
  if (!box || !model || modelState.listLoading) { return; }
  modelState.listLoading = true;
  modelState.listModel = model;
  modelState.note = `正在加载 ${model}，读它自带的类别表…（第一次加载权重要一两秒）`;
  renderClassImportList();
  try {
    const data = await getJson("/api/detect/model/classes?model=" + encodeURIComponent(model)
      + "&device=" + encodeURIComponent(modelState.device || ""));
    modelState.list = data.classes || [];
    modelState.openVocab = String(data.open_vocab || "");
    modelState.vocab = Array.isArray(data.vocab) ? data.vocab.slice() : [];
    if (modelState.openVocab) {
      const box = $("cls-vocab");
      if (box && !box.value) {
        box.value = (modelState.vocab.length ? modelState.vocab : modelState.prefsClasses).join(", ");
      }
      modelState.note = `${data.model} 是「${data.task_name || data.task}」权重 · `
        + `开放词表（${modelState.openVocab}）· 类别不来自权重自带类别表，`
        + "而是你在下面填的英文词——想标什么填什么，不用管权重 trained 过哪些类";
    } else {
      modelState.note = `${data.model} 是「${data.task_name || data.task}」权重 · `
        + `自带 ${data.class_count} 个类别` + (data.device ? ` · 这次用 ${data.device} 加载` : "");
    }
  } catch (err) {
    modelState.list = [];
    modelState.openVocab = "";
    modelState.vocab = [];
    modelState.note = `读 ${model} 的类别表失败：${err.message}`;
  }
  modelState.listLoading = false;
  renderClassImportList();
}

/** 权重自带类别的勾选列表：中文名 + 英文类名 + class_id，支持搜索 / 全选 / 反选。 */
function renderClassImportList() {
  const box = $("cls-list");
  if (!box) { return; }
  if ($("cls-note")) { $("cls-note").textContent = modelState.note || ""; }
  if ($("cls-result")) { $("cls-result").textContent = modelState.result || ""; }
  const note = $("cls-result");
  if (note) { note.classList.toggle("err", String(modelState.result).startsWith("导入失败")); }
  const openVocab = Boolean(modelState.openVocab);
  if ($("cls-toolbar")) { $("cls-toolbar").hidden = openVocab; }
  if ($("cls-vocab-row")) { $("cls-vocab-row").hidden = !openVocab; }
  const checked = new Set(checkedFields("cls-list"));
  const query = String(($("cls-search") || {}).value || "").trim().toLowerCase();
  const known = new Set(fieldOptions().map((row) => String(row.name).toLowerCase()));
  box.innerHTML = "";
  if (openVocab) {
    // 开放词表权重没有"自带类别表"可勾：用户在输入框里写要找的英文词，
    // 这里只负责实时告诉他哪些是新增、哪些已经在类别表里（类别号不会再变）
    const words = vocabWords();
    if (!words.length) {
      box.textContent = modelState.listLoading ? "加载中…"
        : "还没填词。想标什么就填什么，比如 person, dog, traffic cone（只能英文，"
          + "填中文不会报错，但一个框都找不到）。";
    }
    for (const word of words) {
      const line = document.createElement("div");
      line.className = "check-row";
      const hit = known.has(word.toLowerCase());
      const text = document.createElement("span");
      text.textContent = word;
      const meta = document.createElement("span");
      meta.className = "muted small";
      meta.textContent = hit ? " — 已在本库类别表，重复加入只会复用原类别号" : " — 新增类别";
      line.append(text, meta);
      box.append(line);
    }
    syncClassButtons();
    return;
  }
  if (!modelState.list.length) {
    box.textContent = modelState.listLoading ? "加载中…"
      : (modelState.model ? "这个权重没回报任何类别。" : "先选一个本机权重，才能列出它自带的类别。");
    syncClassButtons();
    return;
  }
  let shown = 0;
  for (const row of modelState.list) {
    const name = String(row.name || "");
    const label = String(row.label || name);
    if (query && !name.toLowerCase().includes(query) && !label.includes(query)
      && String(row.class_id) !== query) { continue; }
    shown += 1;
    const line = document.createElement("label");
    line.className = "check-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = name;
    input.checked = checked.size ? checked.has(name) : known.has(name.toLowerCase());
    const text = document.createElement("span");
    text.textContent = `${label} · ${name}`;
    const meta = document.createElement("span");
    meta.className = "muted small";
    meta.textContent = ` — class_id ${row.class_id}${known.has(name.toLowerCase()) ? " · 已在本库类别表" : ""}`;
    line.append(input, text, meta);
    box.append(line);
  }
  if (!shown) {
    box.textContent = `没有匹配「${query}」的类别（共 ${modelState.list.length} 个，换个词或清空搜索）。`;
  }
  syncClassButtons();
}

/** 读输入框里的开放词表：逗号/换行/分号都当分隔符，顺手去重。 */
function vocabWords() {
  const raw = String(($("cls-vocab") || {}).value || "");
  const seen = new Set();
  const words = [];
  for (const piece of raw.split(/[,;\n\r]+/)) {
    const word = piece.trim().replace(/\s+/g, " ");
    if (!word || seen.has(word.toLowerCase())) { continue; }
    seen.add(word.toLowerCase());
    words.push(word);
  }
  return words;
}

function syncClassButtons() {
  const box = $("cls-list");
  if (!box) { return; }
  if (modelState.openVocab) {
    const words = vocabWords();
    const tooMany = words.length > 80;
    if ($("cls-count")) {
      $("cls-count").textContent = words.length ? `填了 ${words.length} 个词${tooMany ? " · 上限 80" : ""}` : "";
    }
    if ($("cls-import")) {
      $("cls-import").disabled = !words.length || tooMany || !modelState.model;
      $("cls-import").textContent = words.length ? `加入类别表（${words.length}）` : "加入类别表";
    }
    return;
  }
  const inputs = Array.from(box.querySelectorAll("input"));
  const picked = inputs.filter((input) => input.checked);
  if ($("cls-count")) {
    $("cls-count").textContent = inputs.length
      ? `选中 ${picked.length} / ${inputs.length} 个` : "";
  }
  if ($("cls-import")) {
    $("cls-import").disabled = !picked.length || !modelState.model;
    $("cls-import").textContent = picked.length ? `导入所选（${picked.length}）` : "导入所选";
  }
}

function toggleAllClassRows(on) {
  const box = $("cls-list");
  if (!box) { return; }
  for (const input of box.querySelectorAll("input")) input.checked = on;
  syncClassButtons();
}

function invertClassRows() {
  const box = $("cls-list");
  if (!box) { return; }
  for (const input of box.querySelectorAll("input")) input.checked = !input.checked;
  syncClassButtons();
}

/** 把勾中的权重自带类别写进本库类别表：同名永不换号，所以对老标签是安全的。 */
async function importModelClasses() {
  const names = modelState.openVocab ? vocabWords() : checkedFields("cls-list");
  if (!names.length) {
    modelState.result = modelState.openVocab ? "先在输入框里填要找的英文词" : "先勾选至少一个类别";
    renderClassImportList();
    return;
  }
  const button = $("cls-import");
  if (button) { button.disabled = true; }
  let data;
  try {
    data = await postJson("/api/detect/classes/add", {
      model: modelState.model, names, device: modelState.device || "",
    });
  } catch (err) {
    modelState.result = "导入失败：" + err.message;
    renderClassImportList();
    if (button) { button.disabled = false; }
    return;
  }
  const lost = data.not_in_model || [];
  modelState.result = `新增 ${data.added || 0} / 复用 ${data.updated || 0}`
    + (data.open_vocab ? ` · 词表已备好（${data.vocab.length} 个词），去「批量目标检测」开跑`
      : ` / 该权重认不出 ${lost.length}` + (lost.length ? `（${lost.join("、")}）` : ""))
    + ((data.skipped || []).length ? ` · 还有 ${data.skipped.length} 个没进表：${data.skipped.join("、")}` : "")
    + ` · 本库类别表现有 ${data.fields.length} 个，下一个类别号 ${data.next_class_id}`;
  await refreshDataset(state.item ? state.item.image : null);
  renderFieldPanel();
  renderRunFields();
  renderClassImportList();
  toast(`本库类别表已更新：新增 ${data.added || 0} · 复用 ${data.updated || 0}`);
  if (button) { button.disabled = false; }
}

// --------------------------------------------------------------------------- //
// 批量提示词标注 / 批量目标检测（走任务队列，可清除）
// --------------------------------------------------------------------------- //
function openRun(open) {
  openModal("run-modal", open);
  if (open) {
    renderRunFields();
    renderVlmChoices();
    if (IS_LOCAL) { renderModelChoices(); renderRunFields(); }
  }
}

function initRunPanel() {
  $("run-close").onclick = () => openRun(false);
  $("run-start").onclick = startRun;
  $("run-clear-jobs").onclick = clearJobs;
  $("strip-clear-jobs").onclick = clearJobs;
  $("run-export").onclick = () => exportDataset($("run-export"));
  $("btn-export-yolo").onclick = () => exportDataset($("btn-export-yolo"));
  $("run-all-fields").onchange = (e) => {
    for (const box of $("run-fields").querySelectorAll("input")) box.checked = e.target.checked;
  };
  // 换权重等于换一张类别表：勾选列表要跟着重画，不然留着上一版权重的类别
  const modelSelect = $("run-model");
  if (modelSelect) modelSelect.onchange = () => { modelState.model = modelSelect.value; };
}

function renderRunFields(boxId = "run-fields") {
  const box = $(boxId);
  if (!box) return;
  const previous = new Set(checkedFields(boxId));
  // 第一次摊开时用后端记的「上次勾的那几个」，用户没选过才全勾
  const preferred = IS_LOCAL && modelState.prefsClasses.length
    ? new Set(modelState.prefsClasses) : null;
  box.innerHTML = "";
  const rows = fieldOptions();
  if (!rows.length) {
    box.innerHTML = `<div class="batch-empty muted small">还没有${CFG.fieldWord}：先点「${CFG.manageLabel}」`
      + (IS_LOCAL ? "从权重自带的类别表里导入。" : "新建要标的物品。") + "</div>";
    return;
  }
  for (const row of rows) {
    const label = document.createElement("label");
    label.className = "check-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = row.name;
    input.checked = previous.size ? previous.has(row.name)
      : (preferred ? preferred.has(row.name) : true);
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = row.color;
    const text = document.createElement("span");
    text.textContent = `${row.name}（${row.label || row.name}）· 已标 ${row.boxes} 框`;
    const hint = document.createElement("span");
    hint.className = "muted small";
    hint.textContent = row.prompt_hint ? ` — ${row.prompt_hint}`
      : (IS_LOCAL ? ` — class_id ${row.class_id}` : " — 还没写描述提示词");
    label.append(input, dot, text, hint);
    box.append(label);
  }
}

function checkedFields(boxId = "run-fields") {
  const box = $(boxId);
  if (!box) return [];
  return Array.from(box.querySelectorAll("input:checked")).map((i) => i.value);
}

function renderVlmChoices(selectId = "run-vlm") {
  const select = $(selectId);
  if (!select) return;
  const vlm = (state.ds && state.ds.vlm) || { profiles: [], default: "" };
  const previous = select.value;
  select.innerHTML = "";
  if (!vlm.profiles.length) {
    const el = document.createElement("option");
    el.value = "";
    el.textContent = "（没有多模态配置，去「模型管理」新建）";
    select.append(el);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  for (const profile of vlm.profiles) {
    const el = document.createElement("option");
    el.value = profile.id;
    el.textContent = `${profile.name}（${profile.kind === "local" ? "本地" : "云端"} · ${profile.model}`
      + `${profile.tested ? (profile.ok ? " · 已连通" : " · 测试失败") : " · 未测试"}）`;
    select.append(el);
  }
  select.value = vlm.profiles.some((p) => p.id === previous) ? previous : (vlm.default || vlm.profiles[0].id);
}

async function startRun() {
  const picked = checkedFields();
  if (!picked.length) { runLog(`先勾选至少一个${CFG.fieldWord}`, true); return; }
  const scope = $("run-scope").value;
  let images = [];
  if (scope === "current") images = state.list.map((i) => i.image);
  else if (scope === "all") images = libItems().map((i) => i.image);
  const payload = {
    images,
    limit: parseInt($("run-limit").value, 10) || 0,
    overwrite: $("run-overwrite").checked,
  };
  if (IS_LOCAL) {
    payload.model = $("run-model") ? $("run-model").value : modelState.model;
    payload.classes = picked;
    payload.conf = clamp(readNumber("run-conf", 0.25), 0.01, 0.95);
    if (!payload.model) { runLog("先选一个能用在本库的本机权重", true); return; }
    // 点一下就跑掉一整批，确认文案必须把「用哪个权重、几类、多快」摆在眼前
    const go = await confirmBox({
      title: CFG.runLabel,
      body: `本机权重 ${payload.model} · ${picked.length} 个${CFG.fieldWord}`
        + `（${picked.slice(0, 6).join("、")}${picked.length > 6 ? "…" : ""}）· 置信度阈值 ${payload.conf}。`
        + `${images.length ? `本次 ${images.length} 张` : "本次按「只标还没标过的」挑图"}。`
        + "本地推理不联网，比问多模态大模型快得多：单张通常几十毫秒，几百张也就一两分钟。"
        + (payload.overwrite ? "已勾「覆盖已有框」：人工改过的框也会被替换。" : ""),
      okLabel: "开始出框",
    });
    if (!go) return;
  } else {
    payload.fields = picked;
    payload.prompt_extra = ($("run-extra") || {}).value
      ? $("run-extra").value.trim() : "";
    const vlm = ($("run-vlm") || {}).value;
    if (vlm) payload.vlm_profile = vlm;
    if (scope === "unlabeled" && !payload.limit) payload.limit = 0;
  }
  let job;
  try {
    job = await postJson(CFG.runUrl, payload);
  } catch (err) {
    runLog("启动失败：" + err.message, true);
    toast("启动失败：" + err.message, true);
    return;
  }
  runLog(`任务 ${job.id} 已启动 · ${job.file_count} 张 · ${CFG.fieldWord} ${picked.join("、")}`);
  toast(`${CFG.runLabel}已启动（${job.file_count} 张）`);
  run.warm = true;
  startJobPolling();
}

async function exportDataset(trigger) {
  const button = trigger || $("run-export");
  if (button.disabled) return;
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "导出中…";
  try {
    const data = await postJson("/api/prompt/export", { val_ratio: DEFAULT_VAL_RATIO, lib: CFG.lib });
    const info = data.export || {};
    const boxes = info.boxes || {};
    const boxesN = (boxes.train || 0) + (boxes.val || 0);
    const classes = (info.classes || []).map((c) => c.name).join("、") || "无";
    const opened = await revealExport(info.out, { quiet: true });
    runLog(`已导出 ${info.train} 训练 / ${info.val} 验证 · ${boxesN} 框 · 类别 ${classes} · ${info.out}`
      + (info.skipped_note ? ` · 没收进来的 ${info.skipped_note}` : ""));
    const verb = info.reused ? "标签没有变化，已复用上次导出" : "YOLO 训练集已导出";
    toast(`${verb}（${scopeLabel(info.scope || "reviewed")}）：${exportSummary(info, CFG.exportKind)} `
      + `→ ${info.out}` + openedSuffix(opened));
  } catch (err) {
    runLog("导出失败：" + err.message, true);
    toast("导出失败：" + err.message, true);
  }
  button.disabled = false;
  button.textContent = label;
  await refreshDataset();
}

function runLog(msg, isErr) {
  appendLog("run-log", msg, isErr);
}

function startJobPolling() {
  stopJobPolling();
  run.poll = setInterval(pollJobs, 1500);
  pollJobs();
}

function stopJobPolling() {
  if (run.poll) { clearInterval(run.poll); run.poll = null; }
}

async function pollJobs() {
  let jobs = [];
  try {
    jobs = (await getJson("/api/jobs")).jobs || [];
  } catch (err) {
    return;
  }
  const mine = jobs.filter((j) => String(j.kind || "").startsWith(CFG.prefix));
  const active = mine.filter((j) => !j.finished);
  for (const job of mine) {
    const key = job.id + ":" + job.stage;
    if (run.seen[job.id] === key) continue;
    run.seen[job.id] = key;
    if (run.warm || !job.finished) runLog(`任务 ${job.id} · ${job.message}`);
  }
  run.warm = true;
  setJobBusy(active.length > 0, active.length ? active[active.length - 1] : null);
  if (!active.length) {
    stopJobPolling();
    if (!run.hadActive) return;
    run.hadActive = false;
    const last = mine[0];
    if (last && last.result && last.result.coords) runLog("坐标文件目录：" + last.result.coords);
    await refreshDataset(state.item ? state.item.image : null);
    renderFieldPanel();
    renderRunFields();
    // 跑完一轮后端会记下这次用的权重与类别，取回来下次开页直接回填
    if (IS_LOCAL) loadModelChoices();
  } else {
    run.hadActive = true;
  }
}

function jobPercent(job) {
  if (!job) return 10;
  if (job.total > 0) return Math.round((job.done / job.total) * 100);
  return { queued: 5, importing: 25, labeling: 55, coords: 92 }[job.stage] ?? 12;
}

function setJobBusy(busy, job) {
  run.busy = busy;
  const strip = $("job-strip");
  strip.classList.toggle("hidden", !busy);
  strip.classList.toggle("err", Boolean(busy && job && job.stage === "failed"));
  if (busy && job) {
    const extra = job.total > 0 ? ` ${job.done}/${job.total}` : "";
    const cancelling = Boolean(job.cancel_requested);
    const verb = String(job.kind || "").endsWith("_upload") ? "上传并标注" : CFG.name;
    $("job-strip-text").textContent = (cancelling ? "取消中 · " : "")
      + `${verb}进行中 · 任务 ${job.id}${extra} · `
      + `${job.message}（期间请先别改标注，完成后列表会自动刷新）`;
    $("run-bar").style.width = jobPercent(job) + "%";
    $("up-bar").style.width = jobPercent(job) + "%";
  }
  for (const id of ["strip-clear-jobs", "run-clear-jobs", "up-clear-jobs"]) {
    const el = $(id);
    if (!el) continue;
    el.disabled = !busy || Boolean(job && job.cancel_requested);
    el.textContent = job && job.cancel_requested ? "取消中…" : "清除任务";
  }
  for (const id of ["btn-approve", "btn-reject", "btn-delete", "btn-undo", "run-start", "up-start"]) {
    const el = $(id);
    if (el) el.disabled = busy;
  }
}

async function clearJobs() {
  if (confirmBoxState.resolve) return;
  const ok = await confirmBox({
    title: "清除正在执行的任务",
    body: "排队中的任务立即删除；正在标注的任务会在当前图片标完后停止，已写入的检测框和坐标文件保留。此操作无法撤销。",
    okLabel: "确定清除",
  });
  if (!ok) return;
  let data;
  try {
    data = await postJson("/api/jobs/clear", {});
  } catch (err) {
    toast("清除任务失败：" + err.message, true);
    return;
  }
  const parts = [];
  if (data.cancelling) parts.push(`${data.cancelling} 个任务正在取消`);
  if (data.cancelled) parts.push(`${data.cancelled} 个排队任务已删除`);
  if (data.dropped) parts.push(`清理 ${data.dropped} 条历史`);
  const msg = parts.length ? parts.join(" · ") : "当前没有正在执行的任务";
  runLog("清除任务：" + msg);
  toast(msg);
  if (data.cancelling) {
    run.hadActive = true;
    startJobPolling();
    return;
  }
  stopJobPolling();
  setJobBusy(false, null);
  $("run-bar").style.width = "0";
}

// --------------------------------------------------------------------------- //
// 本模块独立上传：与其它库互不影响，入库后直接按本库的方式出框
// --------------------------------------------------------------------------- //
function openUpload(open) {
  openModal("up-modal", open);
  if (open) {
    renderRunFields("up-fields");
    renderVlmChoices("up-vlm");
    if (IS_LOCAL) { renderModelChoices(); renderRunFields("up-fields"); }
    renderUpFiles();
  }
}

function initUploadPanel() {
  $("btn-upload").onclick = () => openUpload(true);
  $("up-close").onclick = () => openUpload(false);
  $("up-start").onclick = startUpload;
  $("up-clear-jobs").onclick = clearJobs;
  // 素材池里已有的图（多为姿态库先传的）直接引用同一份文件，不必重传
  $("up-pool").onclick = () => openPool(true, CFG.lib, () => refreshDataset());
  $("up-clear").onclick = () => {
    up.files = [];
    $("up-log").textContent = "";
    $("up-bar").style.width = "0";
    renderUpFiles();
  };
  $("up-all-fields").onchange = (e) => {
    for (const box of $("up-fields").querySelectorAll("input")) box.checked = e.target.checked;
  };
  for (const id of ["up-file-images", "up-file-dir", "up-file-video"]) {
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
}

function renderUpFiles() {
  const files = up.files;
  const el = $("up-files");
  if (!files.length) {
    el.textContent = "尚未选择文件";
    $("up-start").disabled = true;
    return;
  }
  const videos = files.filter(isVideo), images = files.filter(isImage);
  const total = files.reduce((a, f) => a + f.size, 0);
  const names = files.slice(0, 12).map((f) => f.name).join("、");
  el.textContent = `已选 ${files.length} 个文件（视频 ${videos.length} · 图片 ${images.length} · 共 ${kb(total)}）：`
    + names + (files.length > 12 ? ` …等 ${files.length} 个` : "");
  el.title = files.map((f) => f.name).join("\n");
  $("up-start").disabled = run.busy;
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
  if (skipped) upLog(`忽略 ${skipped} 个不支持的文件（只支持图片与视频）`, true);
  return added;
}

function upLog(msg, isErr) {
  appendLog("up-log", msg, isErr);
}

async function uploadOne(file, onProgress) {
  return uploadFile(file, onProgress);
}

async function startUpload() {
  const files = up.files.slice();
  if (!files.length) { toast("先选择文件或目录", true); return; }
  const labeling = $("up-label").checked;
  const picked = checkedFields("up-fields");
  if (labeling && !picked.length) {
    upLog(`先点「${CFG.manageLabel}」备好${CFG.fieldWord}并在上面勾选，`
      + "或取消「入库后自动标注」改为只入库", true);
    toast(`还没有可标的${CFG.fieldWord}：先去「${CFG.manageLabel}」`, true);
    return;
  }
  $("up-start").disabled = true;
  upLog(`开始上传 ${files.length} 个文件…`);
  const stored = [];
  // 进度按字节走，不按"传完几个文件"跳：一条 500 MB 的录像能让人盯着静止的条看几分钟
  const totalBytes = files.reduce((sum, f) => sum + f.size, 0) || 1;
  const acked = new Map();
  const paintUpload = () => {
    let done = 0;
    acked.forEach((value) => { done += value; });
    $("up-bar").style.width = Math.round((done / totalBytes) * 60) + "%";
  };
  for (let i = 0; i < files.length; i += 3) {
    const chunk = files.slice(i, i + 3);
    const results = await Promise.all(chunk.map(async (f, offset) => {
      const slot = i + offset;
      try {
        const info = await uploadOne(f, (loaded) => { acked.set(slot, loaded); paintUpload(); });
        acked.set(slot, f.size);
        return { ok: true, name: f.name, info };
      } catch (err) {
        acked.set(slot, f.size);      // 失败的文件也要把进度条推走，否则永远停在 90%
        return { ok: false, name: f.name, error: err.message };
      }
    }));
    for (const r of results) {
      if (r.ok) { stored.push(r.info.stored); upLog(`已上传 ${r.name}（${kb(r.info.bytes)}）`); }
      else upLog(`上传失败 ${r.name}: ${r.error}`, true);
    }
    paintUpload();
  }
  if (!stored.length) { $("up-start").disabled = false; return; }
  let job;
  try {
    const payload = {
      files: stored,
      label: labeling,
      fps: readNumber("up-fps", 0.5),
      max_frames: Math.round(readNumber("up-max-frames", 600)),
      video_dedupe: $("up-keep-similar").checked ? "none" : "perceptual",
      batch_name: $("up-batch-name").value.trim(),
    };
    if (IS_LOCAL) {
      // 目标检测的上传任务吃本机权重：模型 / 类别 / 阈值，和大模型那套参数完全不同
      payload.model = ($("up-model") || {}).value || modelState.model;
      payload.classes = picked;
      payload.conf = clamp(readNumber("up-conf", 0.25), 0.01, 0.95);
    } else {
      payload.fields = picked;
      payload.vlm_profile = ($("up-vlm") || {}).value || "";
      payload.prompt_extra = ($("up-extra") || {}).value ? $("up-extra").value.trim() : "";
    }
    job = await postJson(CFG.uploadUrl, payload);
  } catch (err) {
    upLog("任务启动失败：" + err.message, true);
    toast("任务启动失败：" + err.message, true);
    $("up-start").disabled = false;
    return;
  }
  upLog(`任务 ${job.id} 已启动 · ${job.file_count} 个文件 · `
    + (labeling ? `${CFG.fieldWord} ${picked.join("、")}，入库后${IS_LOCAL ? "本机权重直接出框" : "逐图问大模型"}`
      : "只入库不标注"));
  toast("上传任务已启动");
  up.files = [];
  renderUpFiles();
  run.warm = true;
  run.hadActive = true;
  startJobPolling();
}

bootObjects();

// --------------------------------------------------------------------------- //
// 壳层接线：命令面板动作 + 队列计数 + HUD（面板、弹层与快捷键由 common.js 提供）
// --------------------------------------------------------------------------- //
function initShell() {
  registerShellCommands(objectsCommands(), objectsFrameMatches);
  initShellChrome({
    onQueue: () => {
      const col = $("queue-col");
      col.classList.toggle("hidden");
      $("rail-queue").classList.toggle("on", !col.classList.contains("hidden"));
      resizeCanvas();
      draw();
    },
    onUpload: () => openUpload(true),
    onFields: () => openFields(true),
    onRun: () => openRun(true),
  });
  $("btn-fields").addEventListener("click", () => openMenu(false));
  $("btn-undo-bar").onclick = () => { if (!runPendingUndo()) undoItem(); };
}

/** 命令面板里的动作：两个出框库共用，措辞跟着本库的口径走。 */
function objectsCommands() {
  const coords = (format) => "/api/prompt/coords?format=" + format + "&lib=" + encodeURIComponent(CFG.lib);
  const cmds = [
    { label: "接受当前图", key: "Q", run: () => setStatus("approved") },
    { label: "驳回当前图", key: "E", run: () => setStatus("rejected") },
    { label: "删除选中框", key: "X", run: deleteBox },
    { label: "撤销这张图的上一次保存", key: "U", run: undoItem },
    { label: "跳到下一张待复核", key: "F", run: () => jumpFlagged(1) },
    { label: "适应窗口", key: "Z", run: () => { fitView(); draw(); } },
    { label: CFG.runLabel, key: "G", run: () => openRun(true) },
    { label: CFG.manageLabel, key: "1", run: () => openFields(true) },
    { label: "导入素材", key: "W", run: () => openUpload(true) },
    { label: "从素材池添加", run: () => openPool(true, CFG.lib, () => refreshDataset()) },
    { label: "导出坐标 CSV", run: () => downloadHref(coords("csv")) },
    { label: "导出坐标 JSONL", run: () => downloadHref(coords("jsonl")) },
    { label: "导出数据集", run: () => $("run-export").click() },
    { label: "快捷键速查", key: "?", run: () => openHelp(true) },
    { label: "打开姿态标注", key: "P", run: () => { location.href = "/pose"; } },
    { label: "打开模型管理", key: "M", run: () => { location.href = "/models"; } },
    { label: "打开" + OTHER.name, tag: "换个出框方式", run: () => { location.href = OTHER.page; } },
  ];
  if (IS_LOCAL) {
    cmds.splice(8, 0, { label: "从权重导入类别", tag: "类别", run: () => openFields(true) });
    if ($("probe-modal")) {
      cmds.splice(9, 0, {
        label: "试词：这个词权重认不认", tag: "类别", run: () => openProbe(true),
      });
    }
  }
  for (const [value, label] of [["pending", "筛选：待复核"], ["flagged", "筛选：仅带标记"],
    ["edited", "筛选：已修改"], ["approved", "筛选：已接受"], ["rejected", "筛选：已驳回"],
    ["unlabeled", "筛选：未标注"], ["all", "筛选：全部"]]) {
    cmds.push({ label, tag: "筛选", run: () => { $("filter").value = value; rebuildList(); } });
  }
  return cmds;
}

function objectsFrameMatches(query) {
  const hits = [];
  for (const it of state.list) {
    if (String(it.image).toLowerCase().includes(query)) {
      hits.push({
        label: String(it.image).split("/").pop(), tag: statusLabel(it[pk("status")]),
        run: () => gotoFrame(it.image),
      });
    }
    if (hits.length >= 8) { break; }
  }
  return hits;
}

/** 命令面板点了文件名：当前筛不到就放开筛选再找。 */
async function gotoFrame(image) {
  let pos = state.list.findIndex((i) => i.image === image);
  if (pos < 0) {
    $("filter").value = "all";
    rebuildList(image);
    pos = state.list.findIndex((i) => i.image === image);
  }
  if (pos < 0) { toast("当前队列里没有这张", true); return; }
  await selectIndex(pos);
}

/** 队列头与画布 HUD 的计数：看队列多长、这一张还剩几处可疑。 */
function renderQueueCounts() {
  const items = libItems();
  $("queue-count").textContent = state.list.length + " 帧";
  $("capsule-count").textContent = items.filter(needsAttention).length
    ? `待确认 ${items.filter(needsAttention).length}` : "全部干净";
  const flags = (state.item && state.item.flags) || [];
  const hud = $("hud-flag");
  hud.textContent = flags.length ? flags.length + " 处待确认" : "无需修改";
  hud.className = flags.length ? "chip bad" : "chip ok";
  const pos = $("hud-pos");
  pos.textContent = "";
  pos.append("第 ");
  const b = document.createElement("b");
  b.textContent = String(state.idx >= 0 ? state.idx + 1 : 0);
  pos.append(b, ` / ${state.list.length}`);
}
