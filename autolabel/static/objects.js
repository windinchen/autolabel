"use strict";

// 二级模块「提示词标注」：自定义字段（英文类名 + 中文显示名 + 描述提示词），
// 用多模态大模型按描述在画面里找物品出检测框，复核时删掉标错的、补画漏掉的。

const state = {
  ds: null, list: [], idx: -1, item: null,
  boxes: [], sel: -1, dirty: false, saveTimer: null,
  view: { scale: 1, ox: 0, oy: 0 },
  img: null, drag: null, spaceDown: false,
  drawField: "",
};
const run = { poll: null, busy: false, logEl: "run-log", seen: {}, warm: false, hadActive: false };
// 本模块独立上传：文件先落到 uploads/，再由 prompt_upload 任务入库并直接出检测框
const up = { files: [] };
const fieldState = { editing: null };
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const imgCache = new Map();
const HANDLE_PX = 9;

const SAVE_STATE = { idle: "就绪", dirty: "有改动未保存", saving: "保存中…", saved: "已保存" };

function fieldByClass(classId) {
  const fields = (state.ds && state.ds.fields && state.ds.fields.fields) || [];
  return fields.find((f) => Number(f.class_id) === Number(classId)) || null;
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
  const hints = (state.ds && state.ds.prompt && state.ds.prompt.flag_hints) || {};
  return hints[flag] || "";
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
  $("btn-run").onclick = () => openRun(true);
  initUploadPanel();
  $("draw-field").onchange = (e) => { state.drawField = e.target.value; };
  $("strip-clear-jobs").onclick = clearJobs;
  initFieldPanel();
  initRunPanel();
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
  if (params.get("pool") === "1") openPool(true, "objects", () => refreshDataset());
  else if (params.get("upload") === "1") openUpload(true);
  pollJobs().then(() => { if (run.busy) startJobPolling(); });
}

/** 提示词库的条目：这本库自己一本账，素材文件与姿态库共用。 */
function libItems() {
  return (state.ds && state.ds.object_items) || [];
}

/** 顶栏只报提示词库张数，顺带说明素材池里还有多少能直接「从素材池添加」。 */
function setHeaderCount(ds) {
  renderExportNote("export-note", ds, "detect");
  const mine = libItems().length;
  const pool = ds.pool || {};
  const spare = Math.max(0, (pool.files || 0) - (pool.objects || 0));
  const need = libItems().filter(needsAttention).length;
  $("ds-name").textContent = " · " + ds.name;
  $("capsule-set").textContent = `${ds.name} · 提示词库 ${mine} 帧`;
  $("capsule-count").textContent = need ? `待确认 ${need}` : "全部干净";
  $("queue-capsule").title = "焦点队列：带标记或还没标过检测框的帧"
    + (spare ? `；素材池另有 ${spare} 张可直接「从素材池添加」` : "");
}

/** 带标记，或压根还没标过检测框，都算「需要人看一眼」。 */
function needsAttention(it) {
  return (it.prompt_flags || []).length > 0
    || it.prompt_objects === null || it.prompt_objects === undefined;
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
  setHeaderCount(state.ds);
  document.title = "提示词标注 · " + state.ds.name;
  renderFieldChoices();
  rebuildList(current);
}

function currentFilter() { return $("filter") ? $("filter").value : "pending"; }

function matchesFilter(item, filter) {
  const status = item.prompt_status || "unreviewed";
  const flagged = (item.prompt_flags || []).length > 0;
  const labeled = item.prompt_objects !== null && item.prompt_objects !== undefined;
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
    : (a, b) => ((b.prompt_priority || 0) - (a.prompt_priority || 0)) || a.image.localeCompare(b.image));
  state.list = items;
  const emptyBox = $("empty-state");
  emptyBox.textContent = libItems().length
    ? "当前筛选下没有图片，试试切右上角筛选（比如「全部」）。"
    : "提示词库还没有图片：点右上角「导入素材」上传图片和视频，或在上传弹层里「从素材池添加」现成素材。";
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
    counts[i.prompt_status || "unreviewed"] = (counts[i.prompt_status || "unreviewed"] || 0) + 1;
    boxes += Number(i.prompt_objects || 0);
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
    const flagged = (it.prompt_flags || []).length > 0;
    const div = document.createElement("div");
    div.className = "thumb" + (i === state.idx ? " active" : "")
      + (flagged ? " flagged" : "")
      + (it.prompt_status === "approved" ? " approved" : "")
      + (it.prompt_status === "rejected" ? " rejected" : "");
    div.title = it.image + "\n" + (STATUS_LABEL[it.prompt_status] || it.prompt_status)
      + "\n框 " + (it.prompt_objects === null || it.prompt_objects === undefined ? "未标" : it.prompt_objects)
      + "\n" + (it.prompt_flags || []).map(flagLabel).join("、");
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
    line.textContent = statusLabel(it.prompt_status);
    const boxes = it.prompt_objects;
    if ((it.prompt_flags || []).length) {
      const em = document.createElement("em");
      em.textContent = (it.prompt_flags || []).length + " 项标记";
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
    item = await getJson("/api/prompt/item?image=" + encodeURIComponent(meta.image));
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
    ctx.lineWidth = selected ? 2.5 : 1.6;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = color + (selected ? "44" : "1f");
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    if (selected) {
      ctx.fillStyle = color;
      for (const [hx, hy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) {
        ctx.fillRect(hx - 3.5, hy - 3.5, 7, 7);
      }
    }
    if (!showLabels) return;
    const small = (box.flags || []).includes("small_box") ? " ⚠" : "";
    const edge = (box.flags || []).includes("edge_box") ? " ✂" : "";
    const label = `${fieldName(box.cls)} · ${Number(box.score || 0).toFixed(2)}${small}${edge}`;
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
    const handle = handleAt(mx, my);
    if (handle) { state.drag = { mode: "resize", handle }; return; }
    const hit = boxAt(mx, my);
    if (hit >= 0) {
      state.sel = hit;
      state.drag = { mode: "move", mx, my, box: { ...state.boxes[hit] } };
      renderBoxes(); draw();
      return;
    }
    if (!state.drawField) { toast("先在右侧选「绘制字段」，再拖框补画目标", true); return; }
    const field = fieldByClass(state.drawField);
    const [nx, ny] = toWorldNorm(mx, my);
    state.drag = { mode: "new", nx, ny };
    draw();
  });

  window.addEventListener("mousemove", (e) => {
    if (!state.drag || !state.img) return;
    const [mx, my] = eventPoint(e);
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
    } else if (drag.mode === "move" || drag.mode === "resize") {
      markDirty(); renderBoxes();
    }
  });
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
  }));
  try {
    const data = await postJson("/api/prompt/item", { image: state.item.image, objects });
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
    meta.prompt_objects = data.prompt_objects;
    meta.prompt_status = data.prompt_status;
    meta.prompt_flags = data.prompt_flags || [];
    meta.prompt_score = data.prompt_score;
  }
  state.item.status = data.prompt_status;
  state.item.flags = data.prompt_flags || [];
  state.item.prompt_status = data.prompt_status;
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
    await postJson("/api/prompt/status", { image: state.item.image, status });
  } catch (err) {
    toast("改状态失败：" + err.message, true);
    return;
  }
  state.item.status = status;
  const meta = libItems().find((i) => i.image === state.item.image);
  if (meta) meta.prompt_status = status;
  const said = status === "approved" ? "已接受" : "已驳回";
  toast(said);
  showUndoBar(`${said} · ${String(state.item.image).split("/").pop()}`);
  if (currentFilter() !== "all") {
    const nextImage = state.list[state.idx + (status === "rejected" ? 0 : 1)];
    rebuildList(nextImage ? nextImage.image : null);
  } else {
    renderFilmstrip(); renderProgress(); renderFlags();
  }
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
    await postJson("/api/prompt/undo", { image: state.item.image });
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
  for (let i = state.idx + dir; i >= 0 && i < state.list.length; i += dir) {
    if ((state.list[i].prompt_flags || []).length) { selectIndex(i); return; }
  }
  toast("没有更多带标记的图片");
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
    case "u": case "U": undoItem(); break;
    case "x": case "X": case "Delete": case "0": deleteBox(); break;
    case "z": case "Z": fitView(); draw(); break;
    case "[": cycleBox(-1); break;
    case "]": cycleBox(1); break;
    case "g": case "G": openRun(true); break;
    case "w": case "W": openUpload(true); break;
    case "m": case "M": location.href = "/models"; break;
    case "p": case "P": location.href = "/pose"; break;
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
    it.video ? `来源视频 ${it.video}` : "来源 图片",
    it.has_backup ? "有可撤销的备份" : "暂无备份",
  ].join("\n");
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
    box.innerHTML = '<div class="muted small">还没有框：空白处拖动画一个，或点「批量提示词标注」让模型找。</div>';
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
    chips.textContent = ` ${(Number(boxItem.score || 0)).toFixed(2)} `;
    for (const flag of boxItem.flags || []) {
      const chip = document.createElement("span");
      chip.className = "chip" + (flag === "manual" ? "" : " warn");
      chip.textContent = flagLabel(flag);
      chip.title = flag + " · " + (flagHint(flag) || "人工确认");
      chips.append(chip);
    }
    head.append(name, chips);
    head.onclick = () => { state.sel = index; renderBoxes(); draw(); };
    const controls = document.createElement("div");
    controls.className = "kpt-row";
    const select = document.createElement("select");
    select.title = "改成别的字段";
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
  return ((state.ds && state.ds.fields && state.ds.fields.fields) || []).slice();
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
      el.textContent = "（还没有字段，先点「字段管理」新建）";
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
  if (open) renderFieldPanel();
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
  const info = (state.ds && state.ds.fields) || {};
  box.innerHTML = "";
  if (!rows.length) {
    box.innerHTML = '<div class="batch-empty muted small">还没有字段：在下面填英文字段名 + 中文显示名 + 描述提示词，点「保存字段」。</div>';
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
      + (row.prompt_hint ? row.prompt_hint : "（还没写描述提示词）");
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
    hint.textContent = "这些类别号在字段登记表里已经没有了（字段被删过或手工改过盘），导出时会被跳过。";
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
  $("field-name").focus();
}

async function saveField() {
  const payload = {
    name: $("field-name").value.trim(),
    label: $("field-label").value.trim(),
    prompt_hint: $("field-hint").value.trim(),
  };
  if ($("field-id").value) payload.id = $("field-id").value;
  if (!payload.name) { fieldLog("先填字段名（英文）", true); return; }
  let data;
  try {
    data = await postJson("/api/fields/save", { field: payload });
  } catch (err) {
    fieldLog("保存失败：" + err.message, true);
    return;
  }
  fieldLog(`已保存「${data.field.name}」 · class_id ${data.field.class_id} · 共 ${data.fields.length} 个字段`);
  await refreshDataset(state.item ? state.item.image : null);
  renderFieldPanel();
  renderRunFields();
  toast(`字段 ${data.field.name} 已保存`);
}

async function deleteField(row) {
  const hasBoxes = Number(row.boxes || 0) > 0;
  const body = hasBoxes
    ? `「${row.name}」还有 ${row.boxes} 个已标注的框。删除字段会连同这些框一起从 prompt_labels/ 里清掉，`
      + "类别号 " + row.class_id + " 之后不会被复用。此操作无法在界面里撤销。"
    : `将删除字段「${row.name}」（class_id ${row.class_id}）。已标注的数据不受影响。`;
  const ok = await confirmBox({ title: "删除字段", body, okLabel: hasBoxes ? "连框一起删" : "确定删除" });
  if (!ok) return;
  try {
    await postJson("/api/fields/delete", { id: row.id, drop_boxes: hasBoxes });
  } catch (err) {
    toast("删除字段失败：" + err.message, true);
    return;
  }
  toast(`已删除字段 ${row.name}`);
  if (String(state.drawField) === String(row.class_id)) state.drawField = "";
  await refreshDataset(state.item ? state.item.image : null);
  renderFieldPanel();
  renderRunFields();
}

function fieldLog(msg, isErr) {
  const box = $("field-log");
  box.textContent = msg || "";
  box.classList.toggle("err", Boolean(isErr));
}

// --------------------------------------------------------------------------- //
// 批量提示词标注（走任务队列，可清除）
// --------------------------------------------------------------------------- //
function openRun(open) {
  openModal("run-modal", open);
  if (open) { renderRunFields(); renderVlmChoices(); }
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
}

function renderRunFields(boxId = "run-fields") {
  const box = $(boxId);
  if (!box) return;
  const previous = new Set(checkedFields(boxId));
  box.innerHTML = "";
  const rows = fieldOptions();
  if (!rows.length) {
    box.innerHTML = '<div class="batch-empty muted small">还没有字段：先点「字段管理」新建要标的物品。</div>';
    return;
  }
  for (const row of rows) {
    const label = document.createElement("label");
    label.className = "check-row";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = row.name;
    input.checked = previous.size ? previous.has(row.name) : true;
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = row.color;
    const text = document.createElement("span");
    text.textContent = `${row.name}（${row.label || row.name}）· 已标 ${row.boxes} 框`;
    const hint = document.createElement("span");
    hint.className = "muted small";
    hint.textContent = row.prompt_hint ? ` — ${row.prompt_hint}` : " — 还没写描述提示词";
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
  const fields = checkedFields();
  if (!fields.length) { runLog("先勾选至少一个字段", true); return; }
  const scope = $("run-scope").value;
  let images = [];
  if (scope === "current") images = state.list.map((i) => i.image);
  else if (scope === "all") images = libItems().map((i) => i.image);
  const payload = {
    fields,
    images,
    limit: parseInt($("run-limit").value, 10) || 0,
    overwrite: $("run-overwrite").checked,
    prompt_extra: $("run-extra").value.trim(),
  };
  const vlm = $("run-vlm").value;
  if (vlm) payload.vlm_profile = vlm;
  if (scope === "unlabeled" && !payload.limit) payload.limit = 0;
  let job;
  try {
    job = await postJson("/api/prompt/run", payload);
  } catch (err) {
    runLog("启动失败：" + err.message, true);
    toast("启动失败：" + err.message, true);
    return;
  }
  runLog(`任务 ${job.id} 已启动 · ${job.file_count} 张 · 字段 ${fields.join("、")}`);
  toast(`提示词标注任务已启动（${job.file_count} 张）`);
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
    const data = await postJson("/api/prompt/export", { val_ratio: 0.2 });
    const info = data.export || {};
    const boxes = info.boxes || {};
    const boxesN = (boxes.train || 0) + (boxes.val || 0);
    const classes = (info.classes || []).map((c) => c.name).join("、") || "无";
    const opened = await revealExport(info.out, { quiet: true });
    runLog(`已导出 ${info.train} 训练 / ${info.val} 验证 · ${boxesN} 框 · 类别 ${classes} · ${info.out}`);
    const verb = info.reused ? "标签没有变化，已复用上次导出" : "YOLO 训练集已导出";
    toast(`${verb}：${info.train} 训练 / ${info.val} 验证 → ${info.out}` + openedSuffix(opened));
  } catch (err) {
    runLog("导出失败：" + err.message, true);
    toast("导出失败：" + err.message, true);
  }
  button.disabled = false;
  button.textContent = label;
  await refreshDataset();
}

function runLog(msg, isErr) {
  const box = $("run-log");
  if (!box) return;
  box.textContent += (box.textContent ? "\n" : "") + `[${new Date().toLocaleTimeString("zh-CN")}] ${msg}`;
  box.classList.toggle("err", Boolean(isErr));
  box.scrollTop = box.scrollHeight;
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
  const promptJobs = jobs.filter((j) => String(j.kind || "").startsWith("prompt"));
  const active = promptJobs.filter((j) => !j.finished);
  for (const job of promptJobs) {
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
    const last = promptJobs[0];
    if (last && last.result && last.result.coords) runLog("坐标文件目录：" + last.result.coords);
    await refreshDataset(state.item ? state.item.image : null);
    renderFieldPanel();
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
    $("job-strip-text").textContent = (cancelling ? "取消中 · " : "")
      + `${job.kind === "prompt_upload" ? "上传并标注" : "提示词标注"}进行中 · 任务 ${job.id}${extra} · `
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
// 本模块独立上传：与姿态标注页互不影响，入库后直接跑提示词标注
// --------------------------------------------------------------------------- //
function openUpload(open) {
  openModal("up-modal", open);
  if (open) { renderRunFields("up-fields"); renderVlmChoices("up-vlm"); renderUpFiles(); }
}

function initUploadPanel() {
  $("btn-upload").onclick = () => openUpload(true);
  $("up-close").onclick = () => openUpload(false);
  $("up-start").onclick = startUpload;
  $("up-clear-jobs").onclick = clearJobs;
  // 素材池里已有的图（多为姿态库先传的）直接引用同一份文件，不必重传
  $("up-pool").onclick = () => openPool(true, "objects", () => refreshDataset());
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
  const box = $("up-log");
  if (!box) return;
  box.textContent += (box.textContent ? "\n" : "") + `[${new Date().toLocaleTimeString("zh-CN")}] ${msg}`;
  box.classList.toggle("err", Boolean(isErr));
  box.scrollTop = box.scrollHeight;
}

async function uploadOne(file) {
  const res = await fetch("/api/upload?name=" + encodeURIComponent(file.name), { method: "POST", body: file });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function startUpload() {
  const files = up.files.slice();
  if (!files.length) { toast("先选择文件或目录", true); return; }
  const labeling = $("up-label").checked;
  const fields = checkedFields("up-fields");
  if (labeling && !fields.length) {
    upLog("先点「字段管理」新建物品字段并在上面勾选，或取消「入库后自动标注」改为只入库", true);
    toast("还没有可标的字段：先去「字段管理」新建", true);
    return;
  }
  $("up-start").disabled = true;
  upLog(`开始上传 ${files.length} 个文件…`);
  const stored = [];
  let sent = 0;
  for (let i = 0; i < files.length; i += 3) {
    const chunk = files.slice(i, i + 3);
    const results = await Promise.all(chunk.map(async (f) => {
      try {
        return { ok: true, name: f.name, info: await uploadOne(f) };
      } catch (err) {
        return { ok: false, name: f.name, error: err.message };
      }
    }));
    for (const r of results) {
      if (r.ok) { stored.push(r.info.stored); upLog(`已上传 ${r.name}（${kb(r.info.bytes)}）`); }
      else upLog(`上传失败 ${r.name}: ${r.error}`, true);
    }
    sent += chunk.length;
    $("up-bar").style.width = Math.round((sent / files.length) * 60) + "%";
  }
  if (!stored.length) { $("up-start").disabled = false; return; }
  let job;
  try {
    job = await postJson("/api/prompt/upload", {
      files: stored,
      fields,
      vlm_profile: $("up-vlm").value,
      label: labeling,
      fps: parseFloat($("up-fps").value) || 0.5,
      max_frames: parseInt($("up-max-frames").value, 10) || 300,
      video_dedupe: $("up-keep-similar").checked ? "none" : "perceptual",
      batch_name: $("up-batch-name").value.trim(),
      prompt_extra: $("up-extra").value.trim(),
    });
  } catch (err) {
    upLog("任务启动失败：" + err.message, true);
    toast("任务启动失败：" + err.message, true);
    $("up-start").disabled = false;
    return;
  }
  upLog(`任务 ${job.id} 已启动 · ${job.file_count} 个文件 · `
    + (labeling ? `字段 ${fields.join("、")}，入库后逐图标注` : "只入库不标注"));
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
  $("btn-undo-bar").onclick = undoItem;
}

/** 命令面板里的提示词标注动作。 */
function objectsCommands() {
  const cmds = [
    { label: "接受当前图", key: "Q", run: () => setStatus("approved") },
    { label: "驳回当前图", key: "E", run: () => setStatus("rejected") },
    { label: "删除选中框", key: "X", run: deleteBox },
    { label: "撤销这张图的上一次保存", key: "U", run: undoItem },
    { label: "跳到下一张待复核", key: "F", run: () => jumpFlagged(1) },
    { label: "适应窗口", key: "Z", run: () => { fitView(); draw(); } },
    { label: "批量提示词标注", key: "G", run: () => openRun(true) },
    { label: "字段管理", key: "1", run: () => openFields(true) },
    { label: "导入素材", key: "W", run: () => openUpload(true) },
    { label: "从素材池添加", run: () => openPool(true, "objects", () => refreshDataset()) },
    { label: "导出坐标 CSV", run: () => downloadHref("/api/prompt/coords?format=csv") },
    { label: "导出坐标 JSONL", run: () => downloadHref("/api/prompt/coords?format=jsonl") },
    { label: "导出数据集", run: () => $("run-export").click() },
    { label: "快捷键速查", key: "?", run: () => openHelp(true) },
    { label: "打开姿态标注", key: "P", run: () => { location.href = "/pose"; } },
    { label: "打开模型管理", key: "M", run: () => { location.href = "/models"; } },
  ];
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
        label: String(it.image).split("/").pop(), tag: statusLabel(it.prompt_status),
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
