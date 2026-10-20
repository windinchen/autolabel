/* 素材池页：素材本体、分类、来处都在这管，三个标注模块只负责"从池里挑要标的"。
 *
 * 为什么单开一页：以前素材只能在各标注库里看，而库里看到的是"本库引用了哪些"，
 * 于是"传上来的视频去哪了""这张图为什么在姿态库有、检测库没有""想按现场批次清一清"
 * 三件事得开三个页面拼答案。这一页只有一个口径——池里有什么、被谁引用、归哪一类。
 */

const POOL = {
  rows: [],          // 池里全部素材（后端 /api/pool?view=all）
  cats: [],          // 分类表
  batches: [],       // 导入批次（池批次 + 各库批次合起来看）
  stats: {},
  sel: new Set(),    // 选中的素材相对路径
  view: "grid",      // grid = 按素材 · batches = 按导入批次
  cat: "",           // "" = 全部；"__none__" = 未分类；其余是分类 ID
  state: "",         // "" / unassigned / in / shared
  batch: "",         // 来源批次 ID
  batchKind: "",     // "" = 全部；import = 只看导入；label = 只看标注
  restore: {},       // 最近一次删除快照（能捞回来的那份）
  query: "",
  busy: false,
  batchSel: new Set(),
  batchFiles: {},
  poll: null,
  active: new Set(),   // 上一轮还在跑的任务 ID：任务结束时要能说清它跑出了什么
};

const UNCATEGORIZED = "__none__";
const BATCH_STATUS = { running: "进行中", done: "已完成", cancelled: "已取消",
                       failed: "失败", interrupted: "已中断", retired: "已退回模块外" };
const BATCH_SOURCE = { upload: "网页上传", cli: "命令行", ui: "网页操作",
                       unknown: "历史", "-": "历史" };

const poolLog = (msg, isErr) => appendLog("pool-log", msg, isErr);

// --------------------------------------------------------------------------- //
// 读池子
// --------------------------------------------------------------------------- //
async function loadPool() {
  const box = $("pool-grid");
  try {
    const data = await getJson("/api/pool?view=all");
    POOL.rows = data.all || [];
    POOL.cats = data.categories || [];
    POOL.batches = data.batches || [];
    POOL.restore = data.restore || {};
    POOL.stats = data.stats || {};
    const name = (data.dataset && data.dataset.name) || "";
    if (name) { $("ds-name").textContent = " · " + name; }
    // 选中的素材可能刚被别的页删掉，选择集要跟磁盘对一次，不然按钮上的数字是假的
    const alive = new Set(POOL.rows.map((row) => row.image));
    POOL.sel = new Set([...POOL.sel].filter((rel) => alive.has(rel)));
  } catch (err) {
    if (box) { box.innerHTML = ''; box.append(poolEmpty("素材池读取失败：" + err.message)); }
    poolLog("素材池读取失败：" + err.message, true);
    return;
  }
  renderPoolStats();
  renderCatStrip();
  renderPoolFilters();
  renderBatchView();
  renderPoolGrid();
}

function renderPoolStats() {
  const stats = POOL.stats || {};
  const files = POOL.rows.length;
  const unassigned = POOL.rows.filter((row) => row.pool_only).length;
  const shared = POOL.rows.filter((row) => (row.libs || []).length > 1).length;
  // 卡面顺序和工作台页一致：大数字 → 名称 → 一句口径，五页的统计卡长得一个样
  const cells = [
    { name: "素材", value: files + " 个", hint: "images/ 里的文件，删这里才是真删" },
    { name: "占用", value: kb(stats.bytes || 0), hint: "同一帧几个库共用，不占双份" },
    { name: "还没入库", value: unassigned + " 个", hint: "三个模块里都看不到，要用就去模块点「从素材池添加」" },
    { name: "几个库共用", value: shared + " 个", hint: "删的时候会问你是退一个库还是整批删" },
    { name: "分类", value: POOL.cats.length + " 个", hint: "分类跟着素材走，三个模块看到的是同一套" },
  ];
  const box = $("pool-stats");
  box.innerHTML = "";
  for (const cell of cells) {
    const el = document.createElement("div");
    el.className = "stat-cell";
    el.innerHTML = '<div class="stat-value"></div><div class="stat-name"></div>'
      + '<div class="stat-hint muted small"></div>';
    el.querySelector(".stat-value").textContent = cell.value;
    el.querySelector(".stat-name").textContent = cell.name;
    el.querySelector(".stat-hint").textContent = cell.hint;
    box.append(el);
  }
  $("pool-note-brief").textContent = `共 ${files} 个素材 · ${kb(stats.bytes || 0)}`
    + ` · 还没入库 ${unassigned} 个`;
}

/** 分类条：全部 + 每个分类 + 未分类。张数直接取后端算好的，页面自己不凑数。 */
function renderCatStrip() {
  const box = $("pool-cats");
  box.innerHTML = "";
  const counts = {};
  for (const row of POOL.rows) {
    const key = row.category || UNCATEGORIZED;
    counts[key] = (counts[key] || 0) + 1;
  }
  const items = [{ id: "", name: "全部素材", count: POOL.rows.length, icon: "ic-pool" }];
  for (const cat of POOL.cats) {
    items.push({ id: String(cat.id), name: cat.name, count: counts[String(cat.id)] || 0,
                 icon: cat.icon || "ic-tag", note: cat.note || "" });
  }
  items.push({ id: UNCATEGORIZED, name: "未分类", count: counts[UNCATEGORIZED] || 0, icon: "ic-list" });
  for (const item of items) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "cat-chip" + (POOL.cat === item.id ? " on" : "");
    chip.title = item.note || (item.id ? "只看这一类" : "看全部素材");
    chip.innerHTML = '<svg class="ic"><use href="#' + item.icon + '"/></svg>';
    const label = document.createElement("span");
    label.textContent = item.name;
    const n = document.createElement("span");
    n.className = "cat-n";
    n.textContent = item.count;
    chip.append(label, n);
    chip.onclick = () => { POOL.cat = item.id; POOL.sel = new Set(); renderCatStrip(); renderPoolGrid(); };
    box.append(chip);
  }
}

function renderPoolFilters() {
  const batch = $("pool-batch");
  const current = POOL.batch;
  batch.innerHTML = '<option value="">全部批次</option>';
  for (const row of POOL.batches.filter((item) => item.action !== "label")) {
    const opt = document.createElement("option");
    opt.value = String(row.id);
    opt.textContent = `${row.name || "未命名批次"}（${row.items || 0} 张）`;
    batch.append(opt);
  }
  batch.value = [...batch.options].some((o) => o.value === current) ? current : "";
  POOL.batch = batch.value;

  const move = $("pool-move-cat");
  const keep = move.value;
  move.innerHTML = '<option value="">未分类</option>';
  for (const cat of POOL.cats) {
    const opt = document.createElement("option");
    opt.value = String(cat.id);
    opt.textContent = cat.name;
    move.append(opt);
  }
  move.value = [...move.options].some((o) => o.value === keep) ? keep : "";
}

// --------------------------------------------------------------------------- //
// 素材网格
// --------------------------------------------------------------------------- //
function visibleRows() {
  const query = String(POOL.query || "").trim().toLowerCase();
  return POOL.rows.filter((row) => {
    if (POOL.cat === UNCATEGORIZED && row.category) { return false; }
    if (POOL.cat && POOL.cat !== UNCATEGORIZED && row.category !== POOL.cat) { return false; }
    if (POOL.state === "unassigned" && !row.pool_only) { return false; }
    if (POOL.state === "in" && row.pool_only) { return false; }
    if (POOL.state === "shared" && (row.libs || []).length < 2) { return false; }
    if (POOL.batch && row.pool_batch !== POOL.batch) { return false; }
    if (query) {
      const hay = `${row.image} ${row.video || ""} ${row.pool_batch_name || ""}`.toLowerCase();
      if (!hay.includes(query)) { return false; }
    }
    return true;
  });
}

function renderPoolGrid() {
  const box = $("pool-grid");
  const rows = visibleRows();
  $("pool-count").textContent = POOL.rows.length
    ? `这一屏 ${rows.length} 个 / 池里共 ${POOL.rows.length} 个` : "池子还是空的";
  box.innerHTML = "";
  if (!POOL.rows.length) {
    box.append(poolEmpty("素材池里还没有东西：点右上角「导入素材」上传图片、目录或视频（U）。"
      + "传进来的东西先落在这，哪个模块要用再去那个模块点「从素材池添加」。"));
    syncPoolBar();
    return;
  }
  if (!rows.length) {
    box.append(poolEmpty("当前筛选下没有素材：换一下分类或入库情况，「⋯」菜单里的「重新读取磁盘」"
      + "可以同步你手工拷进 images/ 的文件。"));
    syncPoolBar();
    return;
  }
  for (const row of rows) { box.append(poolCard(row)); }
  syncPoolBar();
}

function poolEmpty(text) {
  const p = document.createElement("p");
  p.className = "pool-empty muted small";
  p.textContent = text;
  return p;
}

function poolCard(row) {
  const card = document.createElement("div");
  card.className = "pool-card" + (POOL.sel.has(row.image) ? " selected" : "");
  card.title = row.image;
  const img = document.createElement("img");
  img.loading = "lazy";
  img.src = "/thumb/" + encodeURIComponent(row.image);
  img.alt = row.name;
  const check = document.createElement("input");
  check.type = "checkbox";
  check.checked = POOL.sel.has(row.image);
  check.onclick = (e) => e.stopPropagation();
  check.onchange = (e) => toggleSelect(row.image, e.target.checked);
  const meta = document.createElement("div");
  meta.className = "pool-card-meta";
  const name = document.createElement("div");
  name.className = "pool-card-name";
  name.textContent = row.name;
  const bits = [row.width ? `${row.width}×${row.height}` : "尺寸未知", kb(row.bytes || 0)];
  if (row.video) { bits.push("来自 " + row.video); }
  const sub = document.createElement("div");
  sub.className = "muted small";
  sub.textContent = bits.join(" · ");
  const chips = document.createElement("div");
  chips.className = "pool-card-chips";
  chips.append(chipRow(row));
  meta.append(name, sub, chips);
  card.append(img, check, meta);
  card.onclick = () => toggleSelect(row.image, !POOL.sel.has(row.image));
  return card;
}

/** 一张素材身上贴三个签：在哪些库、归哪一类、还有没有只进池没入库。 */
function chipRow(row) {
  const wrap = document.createElement("span");
  wrap.className = "pool-card-chips-inner";
  const owned = (row.libs || []).map(libLabel);
  const ownedChip = document.createElement("span");
  ownedChip.className = "chip " + (owned.length ? "sky" : "warn");
  ownedChip.textContent = owned.length ? owned.join(" · ") : "还没入库";
  ownedChip.title = owned.length ? "这些模块已经在引用它"
    : "只进了池，三个模块都还没有引用它；去模块里点「从素材池添加」才会进去";
  wrap.append(ownedChip);
  const cat = document.createElement("span");
  cat.className = "chip";
  cat.textContent = row.category_name || "未分类";
  wrap.append(cat);
  if ((row.libs || []).length > 1) {
    const shared = document.createElement("span");
    shared.className = "chip ok";
    shared.textContent = "共用";
    wrap.append(shared);
  }
  return wrap;
}

function toggleSelect(rel, on) {
  if (on) { POOL.sel.add(rel); } else { POOL.sel.delete(rel); }
  renderPoolGrid();
}

function selectAllVisible() {
  const rows = visibleRows();
  const all = rows.length && rows.every((row) => POOL.sel.has(row.image));
  POOL.sel = all ? new Set() : new Set(rows.map((row) => row.image));
  renderPoolGrid();
}

function syncPoolBar() {
  const n = POOL.sel.size;
  $("pool-sel-note").textContent = n ? `已选 ${n} 个素材` : "没选中素材（点卡片选，再点取消）";
  $("pool-adopt").disabled = !n || POOL.busy;
  $("pool-assign").disabled = !n || POOL.busy;
  $("pool-delete").disabled = !n || POOL.busy;
  $("pool-select-all").disabled = !visibleRows().length;
}

// --------------------------------------------------------------------------- //
// 三个动作：加进模块 / 改分类 / 删素材
// --------------------------------------------------------------------------- //
async function adoptSelected() {
  const images = [...POOL.sel];
  if (!images.length || POOL.busy) { return; }
  const lib = $("pool-target-lib").value;
  POOL.busy = true;
  syncPoolBar();
  poolLog(`正在把 ${images.length} 个素材加进「${libLabel(lib)}」…`);
  try {
    const data = await postJson("/api/pool/adopt", { images, lib });
    const text = `已加进「${libLabel(data.lib)}」${data.added} 个`
      + (data.already ? `，本来就在里面 ${data.already} 个` : "")
      + (data.missing && data.missing.length ? `，磁盘上找不到 ${data.missing.length} 个` : "");
    poolLog(text);
    toast(text);
    POOL.sel = new Set();
    await loadPool();
  } catch (err) {
    poolLog("加进模块失败：" + err.message, true);
    toast("加进模块失败：" + err.message, true);
  }
  POOL.busy = false;
  syncPoolBar();
}

async function assignSelected() {
  const images = [...POOL.sel];
  if (!images.length || POOL.busy) { return; }
  const category = $("pool-move-cat").value;
  const label = category
    ? (POOL.cats.find((cat) => String(cat.id) === category) || {}).name || "该分类"
    : "未分类";
  POOL.busy = true;
  syncPoolBar();
  try {
    const data = await postJson("/api/pool/assign", { images, category });
    const text = `已把 ${data.set || data.cleared} 个素材移到「${label}」`;
    poolLog(text);
    toast(text);
    await loadPool();
  } catch (err) {
    poolLog("改分类失败：" + err.message, true);
    toast("改分类失败：" + err.message, true);
  }
  POOL.busy = false;
  syncPoolBar();
}

async function deleteSelected() {
  const images = [...POOL.sel];
  if (!images.length || POOL.busy) { return; }
  const shared = images.filter((rel) => {
    const row = POOL.rows.find((item) => item.image === rel);
    return row && (row.libs || []).length > 0;
  });
  const ok = await confirmBox({
    title: `删掉 ${images.length} 个素材`,
    body: "素材文件会从 images/ 删掉，三个标注库里对它的引用一起摘掉"
      + (shared.length ? `（其中 ${shared.length} 个已经被模块引用）` : "")
      + "。标好的结果与索引在 .autolabel/deleted/ 留有快照，误删能捞回来。",
    okLabel: "删掉",
  });
  if (!ok) { return; }
  POOL.busy = true;
  syncPoolBar();
  try {
    const data = await postJson("/api/images/delete",
      { images, lib: "all", mode: "purge", keep_files: false });
    const text = `已删掉 ${data.items || images.length} 个素材（删文件 ${data.files_removed || 0} 个）`;
    poolLog(text);
    toast(text);
    POOL.sel = new Set();
    await loadPool();
  } catch (err) {
    poolLog("删除失败：" + err.message, true);
    toast("删除失败：" + err.message, true);
  }
  POOL.busy = false;
  syncPoolBar();
}

// --------------------------------------------------------------------------- //
// 分类管理
// --------------------------------------------------------------------------- //
let editingCat = "";

function openCategories(open) {
  $("cat-modal").classList.toggle("hidden", !open);
  if (open) { renderCategories(); $("cat-name").focus(); }
}

function renderCategories(message) {
  const box = $("cat-list");
  box.innerHTML = "";
  if (!POOL.cats.length) {
    box.textContent = "还没有分类。建几个分类，比如「户外」「遮挡多」「待复查」，"
      + "以后在分类条上点一下就只看这一类，三个模块里的分类也一起跟着变。";
  }
  for (const cat of POOL.cats) {
    const line = document.createElement("div");
    line.className = "file-row";
    const name = document.createElement("span");
    name.className = "file-name";
    name.textContent = cat.name;
    const meta = document.createElement("span");
    meta.className = "muted small";
    const used = POOL.rows.filter((row) => row.category === String(cat.id)).length;
    meta.textContent = `${used} 个素材${cat.note ? " · " + cat.note : ""}`;
    const rename = document.createElement("button");
    rename.className = "btn";
    rename.textContent = "改名";
    rename.onclick = () => {
      editingCat = String(cat.id);
      $("cat-name").value = cat.name || "";
      $("cat-note").value = cat.note || "";
      $("cat-save").textContent = "保存这个分类";
      $("cat-name").focus();
    };
    const drop = document.createElement("button");
    drop.className = "btn danger";
    drop.textContent = "删除";
    drop.onclick = () => deleteCategory(cat);
    line.append(name, meta, rename, drop);
    box.append(line);
  }
  if (message) { $("cat-log").textContent = message; }
}

async function saveCategory() {
  const name = $("cat-name").value.trim();
  if (!name) { setCatLog("先填分类名，比如「户外」「待复查」。", true); return; }
  try {
    const data = await postJson("/api/pool/category",
      { name, note: $("cat-note").value.trim(), id: editingCat });
    POOL.cats = data.categories || [];
    editingCat = "";
    $("cat-name").value = "";
    $("cat-note").value = "";
    $("cat-save").textContent = "新增分类";
    setCatLog(`已经记好「${data.category.name}」。`);
    await loadPool();
    renderCategories(`已经记好「${data.category.name}」。`);
  } catch (err) {
    setCatLog("没能保存：" + err.message, true);
  }
}

async function deleteCategory(cat) {
  const used = POOL.rows.filter((row) => row.category === String(cat.id)).length;
  const ok = await confirmBox({
    title: `删掉分类「${cat.name}」`,
    body: used ? `这一类下有 ${used} 个素材，删掉分类只会让它们退回「未分类」，素材本身一个都不动。`
      : "这一类下没有素材，直接删掉。",
    okLabel: "删掉分类",
  });
  if (!ok) { return; }
  try {
    const data = await postJson("/api/pool/category/delete", { id: cat.id });
    POOL.cats = data.categories || [];
    if (POOL.cat === String(cat.id)) { POOL.cat = ""; }
    setCatLog(`已删掉分类「${cat.name}」，${data.uncategorized || 0} 个素材退回未分类。`);
    await loadPool();
    renderCategories(`已删掉分类「${cat.name}」，${data.uncategorized || 0} 个素材退回未分类。`);
  } catch (err) {
    setCatLog("没能删除：" + err.message, true);
  }
}

function setCatLog(text, isErr) {
  const el = $("cat-log");
  el.textContent = text;
  el.classList.toggle("bad", Boolean(isErr));
}

// --------------------------------------------------------------------------- //
// 导入批次视图
// --------------------------------------------------------------------------- //
function setPoolView(view) {
  POOL.view = view === "batches" ? "batches" : "grid";
  $("sec-batches").classList.toggle("hidden", POOL.view !== "batches");
  $("sec-materials").classList.toggle("hidden", POOL.view === "batches");
  $("rail-grid").classList.toggle("on", POOL.view === "grid");
  $("rail-batches").classList.toggle("on", POOL.view === "batches");
  renderBatchView();
}

function renderBatchView() {
  const box = $("pool-batch-list");
  if (!box) { return; }
  const rows = POOL.batches.filter((row) => !POOL.batchKind || row.action === POOL.batchKind);
  $("batch-count").textContent = rows.length
    ? `共 ${rows.length} 批` + (POOL.batchKind ? `（已筛，全部 ${POOL.batches.length} 批）` : "")
    : "还没有批次记录";
  syncRestoreButton();
  box.innerHTML = "";
  if (!rows.length) {
    box.innerHTML = '<div class="batch-empty muted small">'
      + (POOL.batchKind ? "这一类还没有批次：换个类型，或去标注模块跑一轮批量标注。"
        : "还没有批次记录：从这一页或各标注模块导入一次素材就会出现在这里。") + '</div>';
    syncBatchButtons();
    return;
  }
  for (const batch of rows) { box.append(batchRow(batch)); }
  syncBatchButtons();
}

/** 「捞回上次删除」：有快照才点亮，并把要捞的东西写在按钮旁边 */
function syncRestoreButton() {
  const button = $("batch-restore");
  const note = $("restore-note");
  if (!button) { return; }
  const info = POOL.restore || {};
  const ok = Boolean(info.name) && Number(info.restorable || 0) > 0;
  button.disabled = !ok || POOL.busy;
  button.title = ok ? `捞回 ${info.deleted_at || ""} 那次删除：${info.items || 0} 张图片`
    : "还没有可捞回的删除快照";
  if (note) {
    note.textContent = ok
      ? `最近一次删除：${info.deleted_at || "时间未知"} · ${info.items || 0} 张（可装回 ${info.restorable} 张）`
        + " · 每次删除都在 .autolabel/deleted/ 留快照，含标签原文"
      : "每次删除都会在 .autolabel/deleted/ 留一份快照（含标签原文），删错了点「捞回上次删除」装回来。";
  }
}

async function restoreLastDelete() {
  const info = POOL.restore || {};
  if (!info.name || POOL.busy) { return; }
  const ok = await confirmBox({
    title: "捞回上次删除",
    body: `把 ${info.deleted_at || "上次"} 删掉的 ${info.items || 0} 张装回各标注库：`
      + "索引条目、标签原文、批次记录一起恢复，原图文件已经删掉的会跳过。",
    okLabel: "装回来",
  });
  if (!ok) { return; }
  POOL.busy = true;
  try {
    const data = await postJson("/api/batches/restore", { name: info.name });
    const text = `已捞回 ${data.items} 张 · 标签 ${data.labels} 份 · 批次记录 ${data.batches} 个`
      + (data.skipped ? ` · 原图没了跳过 ${data.skipped} 张` : "");
    poolLog(text);
    toast(text);
    await loadPool();
  } catch (err) {
    poolLog("捞回失败：" + err.message, true);
    toast("捞回失败：" + err.message, true);
  }
  POOL.busy = false;
  renderBatchView();
}

function batchRow(batch) {
  const id = String(batch.id);
  const wrap = document.createElement("div");
  wrap.className = "batch-block" + (POOL.batchSel.has(id) ? " selected" : "");
  const row = document.createElement("div");
  row.className = "batch-row";
  const check = document.createElement("input");
  check.type = "checkbox";
  check.checked = POOL.batchSel.has(id);
  check.onchange = (e) => {
    if (e.target.checked) { POOL.batchSel.add(id); } else { POOL.batchSel.delete(id); }
    renderBatchView();
  };
  const main = document.createElement("div");
  main.className = "batch-main";
  const name = document.createElement("div");
  name.className = "batch-name";
  const badge = document.createElement("span");
  badge.className = "chip";
  badge.textContent = `${batch.action_name || "导入"} · `
    + (batch.module_name || LIB_LABELS[batch.module] || "素材池");
  name.append(badge, document.createTextNode(" " + (batch.name || "未命名批次")));
  const files = (batch.files || []).slice(0, 3).join("、")
    + (batch.file_count > 3 ? ` 等 ${batch.file_count} 个` : "");
  const meta = document.createElement("div");
  meta.className = "muted small batch-meta";
  const left = Number(batch.in_pool || 0);
  meta.textContent = `${batch.created_at || "时间未知"} · `
    + `${BATCH_SOURCE[batch.source] || batch.source || "历史"} · `
    + (left === Number(batch.items || 0) ? `${left} 个素材`
       : `${left} 个素材还在池里（原报 ${batch.items || 0} 个）`)
    + (files ? ` · ${files}` : "") + (batch.message ? ` · ${batch.message}` : "");
  main.append(name, meta);
  const status = document.createElement("span");
  status.className = "badge " + (batch.status || "unknown");
  status.textContent = BATCH_STATUS[batch.status] || batch.status || "未知";
  row.append(check, main, status);
  wrap.append(row);
  if (POOL.batchFiles[id]) {
    const detail = document.createElement("div");
    detail.className = "batch-files";
    detail.textContent = POOL.batchFiles[id];
    wrap.append(detail);
  }
  row.onclick = (e) => {
    if (e.target === check) { return; }
    loadBatchFiles(id);
  };
  return wrap;
}

async function loadBatchFiles(id) {
  if (POOL.batchFiles[id]) { delete POOL.batchFiles[id]; renderBatchView(); return; }
  const poolBatch = String(id).startsWith("p_");
  const label = [];
  try {
    const data = await getJson("/api/batch/files?id=" + encodeURIComponent(id));
    for (const file of (data.files || []).slice(0, 40)) {
      // 池批次没进过任何库，没有复核状态可写，写它的分类才有用
      label.push(poolBatch ? `${file.name} · ${kb(file.bytes || 0)} · ${file.category_name || "未分类"}`
        : `${file.name} · ${kb(file.bytes || 0)} · ${statusLabel(file.status || "")}`);
    }
    POOL.batchFiles[id] = (data.files || []).length
      ? label.join("\n") + ((data.files || []).length > 40 ? `\n… 还有 ${(data.files || []).length - 40} 个` : "")
      : "这一批的素材已经不在库里了（可能已被删掉）。";
  } catch (err) {
    // 池批次（p_ 开头）没有库侧的文件清单：这些素材还没进任何库，来处本身就是答案
    POOL.batchFiles[id] = String(err && err.message || "").includes("批次")
      ? "这一批只进了素材池，还没有加进任何标注模块。"
      : "文件清单读取失败：" + err.message;
  }
  renderBatchView();
}

function syncBatchButtons() {
  const button = $("batch-delete");
  if (!button) { return; }
  button.disabled = !POOL.batchSel.size || POOL.busy;
  button.textContent = POOL.batchSel.size
    ? `删掉所选批次（${POOL.batchSel.size} 批）` : "删掉所选批次";
}

async function deleteBatches() {
  const ids = [...POOL.batchSel];
  if (!ids.length || POOL.busy) { return; }
  const purge = $("purge-files").checked;
  const picked = POOL.batches.filter((row) => ids.includes(String(row.id)));
  const labels = picked.filter((row) => row.action === "label").length;
  const ok = await confirmBox({
    title: `删掉 ${ids.length} 个批次`,
    body: (purge ? "这批素材在每个标注库里的引用都摘掉，并把 images/ 里的原图一起删掉。"
      : "只摘掉各标注库对这批素材的引用，原图留在池里，以后还能再挑进来。")
      + (labels ? ` 其中 ${labels} 个是标注批次，这一轮标出来的结果也会一起摘掉（可用「捞回上次删除」装回来）。` : ""),
    okLabel: purge ? "删批次和原图" : "只摘引用",
  });
  if (!ok) { return; }
  POOL.busy = true;
  syncBatchButtons();
  try {
    const data = await postJson("/api/batches/delete", { ids, keep_files: !purge });
    // 话术以批次数量打头：只进了池的素材「只摘引用」时 items 本来就是 0，
    // 上来一句「摘掉 0 个素材」会让人以为按钮没生效
    const bits = [`已删掉 ${(data.batches || []).length || ids.length} 个批次`];
    if (purge) {
      if (data.items) { bits.push(`连带 ${data.items} 个素材`); }
      bits.push(`原图删了 ${data.files_removed || 0} 个`);
    } else {
      bits.push(data.items ? `从各库摘掉 ${data.items} 个素材` : "这批素材还没进任何标注库");
      bits.push("原图留在池里");
    }
    const text = bits.join(" · ");
    poolLog(text);
    toast(text);
    POOL.batchSel = new Set();
    POOL.batchFiles = {};
    await loadPool();
  } catch (err) {
    poolLog("删批次失败：" + err.message, true);
    toast("删批次失败：" + err.message, true);
  }
  POOL.busy = false;
  syncBatchButtons();
}

// --------------------------------------------------------------------------- //
// 导入弹层（只进池，不进任何库）
// --------------------------------------------------------------------------- //
const up = { files: [], busy: false };

function openUpload(open) {
  $("up-modal").classList.toggle("hidden", !open);
  if (open) { paintCategoryChoices(); syncUpload(); $("up-batch-name").focus(); }
}

/** 导入弹层的分类选项：这一页已经读过分类表，直接铺，不必再请求一次 */
function paintCategoryChoices() {
  const select = $("up-category");
  if (!select) { return; }
  const keep = select.value;
  select.innerHTML = '<option value="">未分类</option>';
  for (const row of POOL.cats || []) {
    const opt = document.createElement("option");
    opt.value = String(row.id);
    opt.textContent = row.name || "未命名分类";
    select.append(opt);
  }
  if ([...select.options].some((opt) => opt.value === keep)) { select.value = keep; }
}

function addUploadFiles(list) {
  const files = [...list].filter((f) => isImage(f) || isVideo(f));
  const skipped = [...list].length - files.length;
  up.files = up.files.concat(files);
  syncUpload();
  const note = `已选 ${up.files.length} 个文件`;
  $("up-log").textContent = skipped ? `${note}（跳过 ${skipped} 个不支持的类型）` : note;
}

function syncUpload() {
  $("up-start").disabled = !up.files.length || up.busy;
  $("up-start").textContent = up.busy ? "正在送…" : "送进素材池";
  const list = up.files.slice(0, 12).map((f) => `${f.name}（${kb(f.size)}）`).join("、");
  $("up-files").textContent = up.files.length
    ? list + (up.files.length > 12 ? ` 等 ${up.files.length} 个` : "") : "尚未选择文件";
}

function upLog(text, isErr) { appendLog("up-log", text, isErr); }

async function startUpload() {
  const files = up.files.slice();
  if (!files.length) { toast("先选择文件或目录", true); return; }
  up.busy = true;
  syncUpload();
  const stored = { image: [], video: [] };
  const totalBytes = files.reduce((sum, f) => sum + f.size, 0) || 1;
  const acked = new Map();
  const paint = () => {
    let done = 0;
    acked.forEach((value) => { done += value; });
    $("up-bar").style.width = Math.round((done / totalBytes) * 60) + "%";
  };
  for (let i = 0; i < files.length; i += 3) {
    const batch = files.slice(i, i + 3);
    const results = await Promise.all(batch.map(async (file, offset) => {
      const slot = i + offset;
      try {
        const info = await uploadFile(file, (loaded) => { acked.set(slot, loaded); paint(); });
        acked.set(slot, file.size);
        return { ok: true, name: file.name, info };
      } catch (err) {
        acked.set(slot, file.size);   // 失败也要把进度推走，否则进度条永远停在最后一格
        return { ok: false, name: file.name, error: err.message };
      }
    }));
    for (const item of results) {
      if (item.ok) { stored[item.info.kind].push(item.info.stored); upLog(`已上传 ${item.name}`); }
      else { upLog(`上传失败 ${item.name}：${item.error}`, true); }
    }
    paint();
  }
  const params = {
    batch_name: $("up-batch-name").value.trim(),
    category: readImportCategory("up-category"),
    fps: readNumber("up-fps", 0.5),
    max_frames: Math.round(readNumber("up-max-frames", 600)),
    video_dedupe: $("up-keep-similar").checked ? "none" : "perceptual",
  };
  for (const kind of ["video", "image"]) {
    if (!stored[kind].length) { continue; }
    try {
      const res = await fetch("/api/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: "pool", files: stored[kind], params }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `服务返回 ${res.status}`);
      }
      upLog(`已交给后台：${kind === "video" ? "视频抽帧" : "图片"} ${stored[kind].length} 个`);
    } catch (err) {
      upLog("后台任务没建起来：" + err.message, true);
      toast("后台任务没建起来：" + err.message, true);
    }
  }
  up.busy = false;
  up.files = [];
  syncUpload();
  paint();
  startJobPolling();
}

// --------------------------------------------------------------------------- //
// 后台任务条：抽帧要跑一会儿，得让人知道它在动
// --------------------------------------------------------------------------- //
function startJobPolling() {
  if (POOL.poll) { return; }
  POOL.poll = setInterval(pollJobs, 1200);
  pollJobs();
}

async function pollJobs() {
  try {
    const res = await fetch("/api/jobs");
    const data = await res.json();
    const all = data.jobs || [];
    const jobs = all.filter((job) => !job.finished);
    reportFinished(all);
    POOL.active = new Set(jobs.map((job) => job.id));
    const strip = $("job-strip");
    if (!jobs.length) {
      strip.classList.add("hidden");
      if (POOL.poll) { clearInterval(POOL.poll); POOL.poll = null; }
      await loadPool();
      return;
    }
    const job = jobs[0];
    strip.classList.remove("hidden");
    $("job-strip-text").textContent = `素材入库中：${job.message || job.stage}（还有 ${jobs.length} 个任务）`;
  } catch (err) {
    if (POOL.poll) { clearInterval(POOL.poll); POOL.poll = null; }
    poolLog("任务状态读不到：" + err.message, true);
  }
}

/** 任务从"在跑"变成"结束"的那一刻把结果说出口。
 *
 *  以前任务条只是收起、页面重读一遍：抽帧到底进来多少帧、视频为什么一帧没进来，
 *  全都在那一瞬间消失，只剩下一个空池子——"传了视频没显示"就是这么来的。
 */
function reportFinished(all) {
  for (const id of POOL.active) {
    if (all.some((job) => job.id === id && !job.finished)) { continue; }
    const done = all.find((job) => job.id === id);
    if (!done) { continue; }
    const text = done.message || (done.ok ? "导入完成" : "导入失败");
    poolLog(text, !done.ok);
    toast(text, !done.ok);
  }
}

async function clearJobs() {
  const ok = await confirmBox({ title: "清除导入任务",
    body: "正在抽帧的任务会把手上的文件处理完就停下，已经进池的素材保留。", okLabel: "清除" });
  if (!ok) { return; }
  await postJson("/api/jobs/clear", {});
  await pollJobs();
}

// --------------------------------------------------------------------------- //
// 命令面板 / 快捷键
// --------------------------------------------------------------------------- //
function poolCommands() {
  return [
    { label: "导入素材到素材池", key: "U", tag: "素材池", run: () => openUpload(true) },
    { label: "分类管理", key: "C", tag: "素材池", run: () => openCategories(true) },
    { label: "看导入批次", key: "B", tag: "素材池", run: () => setPoolView("batches") },
    { label: "按素材看（缩略图）", tag: "素材池", run: () => setPoolView("grid") },
    { label: "重新读取磁盘", key: "R", tag: "素材池", run: () => loadPool() },
    { label: "全选这一屏素材", tag: "素材池", run: () => selectAllVisible() },
    { label: "把选中的加进姿态标注", tag: "素材池", run: () => adoptTo("pose") },
    { label: "把选中的加进提示词标注", tag: "素材池", run: () => adoptTo("objects") },
    { label: "把选中的加进目标检测标注", tag: "素材池", run: () => adoptTo("detect") },
    { label: "删掉选中的素材", tag: "素材池", run: () => deleteSelected() },
    { label: "看本页操作说明", key: "?", tag: "素材池", run: () => openHelp(true) },
  ];
}

function adoptTo(lib) {
  $("pool-target-lib").value = lib;
  adoptSelected();
}

function poolKeys(e) {
  if (shellKeys(e)) { return; }
  if (isTyping(e)) { return; }
  switch (String(e.key).toLowerCase()) {
    case "u": e.preventDefault(); openUpload(true); break;
    case "c": e.preventDefault(); openCategories(true); break;
    case "b": e.preventDefault(); setPoolView(POOL.view === "batches" ? "grid" : "batches"); break;
    case "a": e.preventDefault(); adoptSelected(); break;
    case "r": e.preventDefault(); loadPool(); break;
    case "escape": POOL.sel = new Set(); renderPoolGrid(); break;
    default:
  }
}

async function bootPool() {
  initNav();
  registerShellCommands(poolCommands());
  initShellChrome({ onBatches: () => setPoolView("batches") });
  document.addEventListener("keydown", poolKeys);
  $("rail-grid").onclick = () => setPoolView("grid");
  $("rail-batches").onclick = () => setPoolView("batches");
  $("rail-cat").onclick = () => openCategories(true);
  $("btn-upload").onclick = () => openUpload(true);
  $("btn-categories").onclick = () => { openMenu(false); openCategories(true); };
  $("btn-view-batches").onclick = () => { openMenu(false); setPoolView("batches"); };
  $("pool-refresh").onclick = () => loadPool();
  $("pool-select-all").onclick = selectAllVisible;
  $("pool-state").onchange = (e) => { POOL.state = e.target.value; renderPoolGrid(); };
  $("pool-batch").onchange = (e) => { POOL.batch = e.target.value; renderPoolGrid(); };
  $("batch-kind").onchange = (e) => { POOL.batchKind = e.target.value; renderBatchView(); };
  $("batch-restore").onclick = () => restoreLastDelete();
  $("pool-query").oninput = (e) => { POOL.query = e.target.value; renderPoolGrid(); };
  $("pool-adopt").onclick = adoptSelected;
  $("pool-assign").onclick = assignSelected;
  $("pool-delete").onclick = deleteSelected;
  $("batch-delete").onclick = deleteBatches;
  $("strip-clear-jobs").onclick = clearJobs;
  $("up-close").onclick = () => openUpload(false);
  $("up-clear").onclick = () => { up.files = []; syncUpload(); $("up-log").textContent = "已清空"; };
  $("up-start").onclick = startUpload;
  $("cat-close").onclick = () => openCategories(false);
  $("cat-save").onclick = saveCategory;
  $("cat-name").onkeydown = (e) => { if (e.key === "Enter") { editingCat = ""; saveCategory(); } };
  const drop = $("up-drop");
  ["dragenter", "dragover"].forEach((name) => drop.addEventListener(name, (e) => {
    e.preventDefault();
    drop.classList.add("hover");
  }));
  ["dragleave", "drop"].forEach((name) => drop.addEventListener(name, (e) => {
    e.preventDefault();
    drop.classList.remove("hover");
    if (name === "drop" && e.dataTransfer) { addUploadFiles(e.dataTransfer.files); }
  }));
  for (const id of ["file-images", "file-dir", "file-video"]) {
    $(id).onchange = (e) => { addUploadFiles(e.target.files); e.target.value = ""; };
  }
  const params = new URLSearchParams(location.search);
  await loadPool();
  if (params.get("upload") === "1") { openUpload(true); }
  if (params.get("view") === "batches") { setPoolView("batches"); }
  pollJobs().then(() => { if ($("job-strip").classList.contains("hidden") === false) { startJobPolling(); } });
}

bootPool();
