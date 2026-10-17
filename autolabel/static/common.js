"use strict";

// 四个页面（门户 / 姿态标注 / 提示词标注 / 模型管理）共用的极小工具集。
// 页面自己的逻辑在 portal.js / app.js / objects.js / models.js 里，各页只加载自己那份。

const $ = (id) => document.getElementById(id);
const mb = (v) => Number(v || 0).toFixed(1);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const kb = (n) => (n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB");

const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"];
const VIDEO_EXTS = [".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".flv", ".wmv",
                    ".mts", ".m2ts", ".ts", ".mpg", ".mpeg", ".3gp"];

const extOf = (name) => {
  const i = String(name).lastIndexOf(".");
  return i < 0 ? "" : name.slice(i).toLowerCase();
};
const isVideo = (f) => VIDEO_EXTS.includes(extOf(f.name)) || /^video\//.test(f.type);
const isImage = (f) => IMAGE_EXTS.includes(extOf(f.name)) || /^image\//.test(f.type);

const MODULES = [
  { id: "pose", href: "/pose", name: "姿态标注", desc: "人体骨骼关键点复核" },
  { id: "objects", href: "/objects", name: "提示词标注", desc: "按描述找物品，出检测框" },
  { id: "models", href: "/models", name: "模型管理", desc: "姿态权重与多模态配置" },
];

// 状态词只在界面层统一，写回后端的仍是 approved / edited 这些原值；两页共用一份，避免各写各的
const STATUS_LABEL = { unreviewed: "待复核", approved: "已接受", edited: "已修改", rejected: "已驳回" };
const statusLabel = (s) => STATUS_LABEL[s] || s || "待复核";

// 质量标记的中文名：qc.py / prompt.py 里的 key 是数据格式，界面只说人话；
// 原始 key 留在 tooltip 里，方便对着 qc/report.md 和导出表查
const FLAG_LABEL = {
  no_person: "没检到人",
  crowded: "多人重叠",
  low_resolution: "分辨率偏低",
  likely_blurred: "疑似运动模糊",
  few_keypoints: "可见关键点太少",
  tiny_box: "目标太小",
  odd_aspect: "框比例异常",
  keypoints_outside_box: "关键点落在框外",
  edge_clipped: "贴边被截断",
  limb_length_mismatch: "左右肢体不一致",
  model_disagreement: "多模型不一致",
  only_secondary_model: "只有辅助模型检出",
  multi_model_confirmed: "多模型结论一致",
  vlm_labeled: "大模型标注",
  no_fields: "还没建字段",
  none_found: "一个框都没有",
  small_box: "框太小",
  edge_box: "框贴边",
  dup_box: "框重复",
  unknown_class: "类别号不认识",
  prompt_stale: "标注设置已变更",
  vlm_failed: "大模型没答上",
  manual: "人工画的框",
};
const flagLabel = (f) => FLAG_LABEL[f] || f;

// --------------------------------------------------------------------------- //
// 外观：白天 / 夜晚 · 字号缩放
// 选择只存在这台浏览器的 localStorage 里（不进数据集、不走接口），四个页面共用；
// 每个 HTML 的头里有一段内联脚本先把 data-theme / data-fs 定好，避免白屏闪一下
// --------------------------------------------------------------------------- //
const APPEAR_KEY = "autolabel.appearance";
const THEME_CHOICES = [
  { value: "auto", label: "跟随系统" },
  { value: "light", label: "白天" },
  { value: "dark", label: "夜晚" },
];
const FONT_CHOICES = [
  { value: "0", label: "小", scale: 0.92 },
  { value: "1", label: "标准", scale: 1 },
  { value: "2", label: "大", scale: 1.12 },
  { value: "3", label: "特大", scale: 1.26 },
];
const appearance = (() => {
  let raw = {};
  try { raw = JSON.parse(localStorage.getItem(APPEAR_KEY) || "{}") || {}; } catch (err) { raw = {}; }
  return {
    theme: THEME_CHOICES.some((c) => c.value === raw.theme) ? raw.theme : "auto",
    fs: FONT_CHOICES.some((c) => c.value === String(raw.fs)) ? String(raw.fs) : "1",
  };
})();

const prefersReducedMotion = () =>
  Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

/** 「跟随系统」在这台机器上到底算白天还是夜晚。 */
function resolveTheme(pref) {
  if (pref === "light" || pref === "dark") { return pref; }
  const light = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  return light ? "light" : "dark";
}

const fontScale = () => (FONT_CHOICES.find((c) => c.value === appearance.fs) || FONT_CHOICES[1]).scale;

/** 把当前选择写进根元素；画布靠 appearancechange 重画。 */
function applyAppearance(options) {
  const opts = options || {};
  const root = document.documentElement;
  const changed = root.dataset.theme !== resolveTheme(appearance.theme);
  root.dataset.theme = resolveTheme(appearance.theme);
  root.dataset.themePref = appearance.theme;
  root.dataset.fs = appearance.fs;
  try { localStorage.setItem(APPEAR_KEY, JSON.stringify(appearance)); } catch (err) { /* 无痕模式忽略 */ }
  if (changed && opts.animate !== false && !prefersReducedMotion() && document.body) {
    document.body.classList.add("theme-fading");
    setTimeout(() => document.body.classList.remove("theme-fading"), 260);
  }
  renderAppearanceControls();
  window.dispatchEvent(new CustomEvent("appearancechange",
    { detail: { theme: root.dataset.theme, scale: fontScale() } }));
}

function setThemePref(pref) {
  if (!THEME_CHOICES.some((c) => c.value === pref)) { return; }
  appearance.theme = pref;
  applyAppearance();
}

function toggleTheme() {
  setThemePref(resolveTheme(appearance.theme) === "light" ? "dark" : "light");
}

function setFontStep(value) {
  if (!FONT_CHOICES.some((c) => c.value === String(value))) { return; }
  appearance.fs = String(value);
  applyAppearance();
}

/** 字号放大 / 缩小一档，返回新的档位说明，好让提示条有话可说。 */
function stepFont(delta) {
  const i = Math.max(0, Math.min(FONT_CHOICES.length - 1,
    FONT_CHOICES.findIndex((c) => c.value === appearance.fs) + delta));
  appearance.fs = FONT_CHOICES[i].value;
  applyAppearance();
  return FONT_CHOICES[i];
}

/** ⋯ 菜单里的分段选择器；没有 #appearance-ctl 的页面直接跳过。 */
function renderAppearanceControls() {
  const slot = $("appearance-ctl");
  if (!slot) { return; }
  slot.innerHTML = "";
  const group = (text, choices, active, onPick, hint) => {
    const label = document.createElement("div");
    label.className = "menu-label";
    label.textContent = text;
    const row = document.createElement("div");
    row.className = "menu-seg";
    choices.forEach((choice) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = choice.label;
      btn.className = choice.value === active ? "on" : "";
      btn.title = hint ? hint(choice) : choice.label;
      btn.onclick = () => onPick(choice.value);
      row.appendChild(btn);
    });
    slot.append(label, row);
  };
  group("外观", THEME_CHOICES, appearance.theme, setThemePref,
    (c) => c.value === "auto" ? "跟随系统的深色设置" : "切到" + c.label);
  group("字号", FONT_CHOICES, appearance.fs, setFontStep,
    (c) => "字号" + c.label + "（" + Math.round(c.scale * 100) + "%）");
}

/** 四个页面都能用 ⌘K 搜到这些动作；由 registerShellCommands 自动带上。 */
function appearanceCommands() {
  const said = (what) => { toast(what, false); };
  return [
    { label: "外观：切到白天", tag: "外观", key: "⌥T", run: () => { setThemePref("light"); said("已切到白天"); } },
    { label: "外观：切到夜晚", tag: "外观", key: "⌥T", run: () => { setThemePref("dark"); said("已切到夜晚"); } },
    { label: "外观：跟随系统", tag: "外观", run: () => { setThemePref("auto"); said("外观改为跟随系统"); } },
    { label: "字号放大一档", tag: "字号", key: "⌥=", run: () => said("字号「" + stepFont(1).label + "」") }
      ,
    { label: "字号缩小一档", tag: "字号", key: "⌥-", run: () => said("字号「" + stepFont(-1).label + "」") },
  ].concat(FONT_CHOICES.map((c) => ({
    label: "字号设为" + c.label, tag: "字号", run: () => { setFontStep(c.value); said("字号「" + c.label + "」"); },
  })));
}

/** 画布上的字号：按 --fs 档位缩放，两个标注页共用。 */
function cfont(size, weight) {
  const n = Math.round(size * canvasPalette().scale * 10) / 10;
  return (weight ? weight + " " : "") + n + "px ui-monospace, monospace";
}

/** 给实色加透明度（画布不认 color-mix，只能自己换算）。 */
function rgbaOf(hex, alpha) {
  const n = parseInt(String(hex).replace("#", ""), 16);
  return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + alpha + ")";
}

/** 画布用的配色：CSS 变量变了就重取，画布跟着主题走。 */
let canvasCache = null;
function canvasPalette() {
  const root = document.documentElement;
  const stamp = root.dataset.theme + "|" + root.dataset.fs;
  if (canvasCache && canvasCache.stamp === stamp) { return canvasCache; }
  const cs = getComputedStyle(root);
  const v = (name, fallback) => String(cs.getPropertyValue(name) || fallback).trim() || fallback;
  canvasCache = {
    stamp, scale: fontScale(),
    ink: v("--ink", "#F1EFEA"),
    amber: v("--amber", "#F5A524"),
    jade: v("--jade", "#46D39A"),
    coral: v("--coral", "#FF6B57"),
    sky: v("--sky", "#79B8FF"),
    veil: v("--c-canvas-veil", "rgba(10,12,14,.78)"),
    ringIdle: v("--c-ring-idle", "rgba(8,10,11,.85)"),
    keyHidden: v("--c-key-hidden", "rgba(163,170,175,.5)"),
    onAccent: v("--on-accent", "#14171A"),
    photoAmber: v("--c-photo-amber", "#F5A524"),
    photoSky: v("--c-photo-sky", "#79B8FF"),
    photoCoral: v("--c-photo-coral", "#FF8A78"),
    stage: v("--stage", "#080A0B"),
    fieldFallback: v("--c-field-fallback", "#8b94a5"),
  };
  return canvasCache;
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function getJson(url) {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

let toastTimer = null;
function toast(msg, isErr) {
  const el = $("toast");
  if (!el) { return; }
  el.textContent = msg;
  el.className = isErr ? "err" : "";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = "hidden"; el.classList.add("hidden"); }, 1800);
}

const confirmBoxState = { resolve: null };

function openConfirm(open) {
  const el = $("confirm-modal");
  if (el) el.classList.toggle("hidden", !open);
}

/** 通用确认弹窗，resolve(true) 表示用户点了「确定」 */
function confirmBox({ title = "确认操作", body = "", okLabel = "确定" } = {}) {
  $("confirm-title").textContent = title;
  $("confirm-body").textContent = body;
  $("confirm-ok").textContent = okLabel;
  openConfirm(true);
  return new Promise((resolve) => { confirmBoxState.resolve = resolve; });
}

function closeConfirm(value) {
  if (!confirmBoxState.resolve) return;
  openConfirm(false);
  const resolve = confirmBoxState.resolve;
  confirmBoxState.resolve = null;
  resolve(Boolean(value));
}

function bindConfirmBox() {
  const ok = $("confirm-ok"), cancel = $("confirm-cancel");
  if (ok) ok.onclick = () => closeConfirm(true);
  if (cancel) cancel.onclick = () => closeConfirm(false);
}

function visible(id) {
  const el = $(id);
  return Boolean(el && !el.classList.contains("hidden"));
}

/** 弹层的键盘兜底：确认框 Esc 取消 / Enter 确定，素材池弹层 Esc 关闭。
 *  用捕获阶段抢在页面自己的快捷键前面，四个模块里行为一致。 */
function bindModalKeys() {
  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" && e.key !== "Enter") return;
    if (visible("confirm-modal")) {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === "Escape") { closeConfirm(false); return; }
      const tag = (document.activeElement || {}).tagName;
      if (tag !== "INPUT" && tag !== "TEXTAREA" && tag !== "SELECT") closeConfirm(true);
      return;
    }
    if (e.key === "Escape" && visible("pool-modal")) {
      e.preventDefault();
      e.stopPropagation();
      openPool(false);
    }
  }, true);
}

/** 顶栏模块导航：高亮当前页，并在按 M/B/O 时跳到对应模块。 */
function initNav() {
  const page = document.body.dataset.page || "";
  const host = $("nav-modules");
  if (host) {
    host.innerHTML = "";
    for (const item of MODULES) {
      const link = document.createElement("a");
      link.href = item.href;
      link.className = "nav-module" + (item.id === page ? " active" : "");
      link.title = item.desc;
      link.textContent = item.name;
      host.append(link);
    }
  }
  const brand = $("nav-portal");
  if (brand) brand.href = "/";
  bindConfirmBox();
  bindModalKeys();
}

// --------------------------------------------------------------------------- //
// 导出结果：产物是目录，路径不能只在 toast 里闪一下，这里给常驻行 + 一键复制
// --------------------------------------------------------------------------- //
const EXPORT_COMMANDS = {
  pose: { verb: "pose", model: "yolo11s-pose.pt" },
  detect: { verb: "detect", model: "yolo11s.pt" },
};

/** 复制了目录还得手敲训练命令，这里直接给整条。 */
function exportCommand(kind, row) {
  const spec = EXPORT_COMMANDS[kind] || EXPORT_COMMANDS.pose;
  return `yolo ${spec.verb} train data=${row.data_yaml} model=${spec.model} epochs=100 imgsz=640`;
}

function latestExport(ds, kind) {
  return ((ds && ds.exports_recent) || []).find((row) => row.kind === kind) || null;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    const box = document.createElement("textarea");
    box.value = text;
    box.style.position = "fixed";
    box.style.opacity = "0";
    document.body.append(box);
    box.select();
    let done = false;
    try { done = document.execCommand("copy"); } catch (ignored) { done = false; }
    box.remove();
    return done;
  }
}

function exportCopyButton(label, value) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn tiny";
  button.textContent = label;
  button.onclick = async () => {
    const done = await copyText(value);
    toast(done ? "已复制：" + value : "复制失败，请手动选中路径", !done);
  };
  return button;
}

/** 按系统说人话：mac 叫访达，Windows 叫资源管理器。 */
function revealLabel() {
  const ua = navigator.userAgent;
  if (/Mac|iPhone|iPad|iPod/.test(ua)) return "在访达中显示";
  if (/Windows/.test(ua)) return "在资源管理器中显示";
  return "打开所在目录";
}

/** 导出完自动帮忙开一次目录时用的后缀文案。 */
function openedSuffix(opened) {
  if (!opened || !opened.ok) return "";
  const ua = navigator.userAgent;
  if (/Mac|iPhone|iPad|iPod/.test(ua)) return "（已在访达中打开）";
  if (/Windows/.test(ua)) return "（已在资源管理器中打开）";
  return "（已打开所在目录）";
}

/** 浏览器没权限直接开本地目录，交给后端调系统命令；quiet 时由调用方自己拼提示。 */
async function revealExport(path, opts) {
  if (!path) return { ok: false, message: "没有导出路径" };
  try {
    const res = await postJson("/api/export/reveal", { path });
    if (!(opts && opts.quiet)) toast(res.message || ("已打开：" + path));
    return res;
  } catch (err) {
    if (!(opts && opts.quiet)) toast("打开目录失败：" + err.message, true);
    return { ok: false, message: err.message };
  }
}

function exportRevealButton(path) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn tiny";
  button.textContent = revealLabel();
  button.onclick = () => revealExport(path);
  return button;
}

/** 一条「上次导出」提示：时间 + 完整路径 + 份数 + 打开目录 + 复制。 */
function buildExportNote(ds, kind, idleText) {
  const row = latestExport(ds, kind);
  const el = document.createElement("div");
  el.className = "export-note muted small";
  if (!row) {
    el.textContent = idleText || "还没有导出过：导出后产物目录会长期显示在这里。";
    return el;
  }
  const text = document.createElement("span");
  text.className = "export-note-path";
  text.textContent = `上次导出（${row.at}）：${row.path} · ${row.train} 训练 / ${row.val} 验证`;
  text.title = row.path;
  el.append(text, exportRevealButton(row.path), exportCopyButton("复制路径", row.path),
    exportCopyButton("复制训练命令", exportCommand(kind, row)));
  return el;
}

/**
 * 把提示行塞进某个容器（容器不存在就 quietly 跳过）。
 * 顶栏那条（.bar）只在真的导出过之后才出来占位，平时不必拿一句「还没导出过」挡路。
 */
function renderExportNote(containerId, ds, kind) {
  const host = $(containerId);
  if (!host) return;
  host.innerHTML = "";
  host.append(buildExportNote(ds, kind));
  const idle = !latestExport(ds, kind);
  host.classList.toggle("hidden", idle && host.classList.contains("bar"));
}

/** 复核类页面共用的数字格式化：把 null/undefined 显示成占位符。 */
const dash = (v) => (v === null || v === undefined || v === "" ? "—" : v);

// --------------------------------------------------------------------------- //
// 素材池：两个库共用 images/ 下的素材，这里提供"把池里已有的图添加进本库"的选择器
// --------------------------------------------------------------------------- //
const LIB_LABELS = { pose: "姿态标注", objects: "提示词标注" };
const libLabel = (lib) => LIB_LABELS[lib] || String(lib || "本库");

const poolState = { lib: "pose", files: [], sel: new Set(), loading: false, busy: false, onDone: null };

function ensurePoolModal() {
  if ($("pool-modal")) { return; }
  const wrap = document.createElement("div");
  wrap.id = "pool-modal";
  wrap.className = "modal hidden";
  wrap.innerHTML = `
    <div class="modal-card">
      <div class="modal-title">从素材池添加<span class="muted small"> · 素材已在 <span class="mono">images/</span> 里，只是还没进本库；添加不复制文件</span></div>
      <div class="muted small" id="pool-note"></div>
      <div class="pick-row">
        <button id="pool-refresh" class="btn">重新载入</button>
        <label class="muted small grow">批次名（选填）
          <input type="text" id="pool-batch-name" placeholder="例如：补标 10 月旧素材" spellcheck="false">
        </label>
      </div>
      <div id="pool-files" class="pool-list muted small">加载中…</div>
      <div class="btn-row">
        <button id="pool-adopt" class="primary" disabled>添加到本库</button>
        <button id="pool-all">全选</button>
        <button id="pool-close">关闭</button>
      </div>
      <div class="muted small" id="pool-log"></div>
    </div>`;
  document.body.append(wrap);
  $("pool-refresh").onclick = () => loadPool();
  $("pool-all").onclick = poolToggleAll;
  $("pool-adopt").onclick = poolAdopt;
  $("pool-close").onclick = () => openPool(false);
}

function openPool(open, lib, onDone) {
  ensurePoolModal();
  if (typeof lib === "string" && lib) { poolState.lib = lib; }
  if (typeof onDone === "function") { poolState.onDone = onDone; }
  const el = $("pool-modal");
  el.classList.toggle("hidden", !open);
  if (open) loadPool();
}

async function loadPool() {
  const box = $("pool-files");
  if (poolState.loading) { return; }
  poolState.loading = true;
  box.textContent = "加载中…";
  try {
    const data = await getJson("/api/pool?lib=" + encodeURIComponent(poolState.lib));
    poolState.files = data.files || [];
    poolState.sel = new Set();
    poolState.stats = data.stats || {};
    $("pool-note").textContent = `素材池共 ${poolState.stats.files || 0} 个文件（`
      + `姿态库 ${poolState.stats.pose || 0} · 提示词库 ${poolState.stats.objects || 0} · `
      + `两库共用 ${poolState.stats.shared || 0}）· ${kb(poolState.stats.bytes || 0)}；`
      + `其中可添加到「${libLabel(poolState.lib)}」的 ${poolState.files.length} 张。`;
  } catch (err) {
    box.textContent = "素材池读取失败：" + err.message;
    poolState.files = [];
  }
  poolState.loading = false;
  renderPool();
}

function renderPool() {
  const box = $("pool-files");
  box.innerHTML = "";
  if (!poolState.files.length) {
    box.textContent = "素材池里没有可添加的图了：本库已引用全部素材，或池子本来就是空的。";
  }
  for (const file of poolState.files) {
    const line = document.createElement("div");
    line.className = "file-row";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = poolState.sel.has(file.image);
    check.onchange = (e) => {
      if (e.target.checked) { poolState.sel.add(file.image); } else { poolState.sel.delete(file.image); }
      syncPoolButtons();
    };
    const name = document.createElement("span");
    name.className = "file-name";
    name.textContent = file.name;
    name.title = file.image;
    const meta = document.createElement("span");
    meta.className = "muted small";
    meta.textContent = `${file.imported_at || ""}${file.video ? " · 来自 " + file.video : ""}`
      + ` · ${file.width || "?"}×${file.height || "?"} · ${kb(file.bytes || 0)}`;
    const chips = document.createElement("span");
    chips.className = "chip";
    chips.textContent = (file.in_pose ? "已在姿态库 " : "") + (file.in_objects ? "已在提示词库" : "")
      || "两库都没有";
    line.append(check, name, meta, chips);
    box.append(line);
  }
  syncPoolButtons();
}

function syncPoolButtons() {
  const all = poolState.files.length;
  $("pool-all").textContent = all && poolState.sel.size === all ? "清空选择" : "全选";
  $("pool-all").disabled = !all;
  $("pool-adopt").disabled = !poolState.sel.size || poolState.busy;
  $("pool-adopt").textContent = poolState.busy ? "添加中…"
    : (poolState.sel.size ? `添加到${libLabel(poolState.lib)}（${poolState.sel.size} 张）` : "添加到本库");
}

function poolToggleAll() {
  const all = poolState.files.length;
  poolState.sel = all && poolState.sel.size === all
    ? new Set() : new Set(poolState.files.map((f) => f.image));
  renderPool();
}

async function poolAdopt() {
  const images = Array.from(poolState.sel);
  if (!images.length || poolState.busy) { return; }
  poolState.busy = true;
  syncPoolButtons();
  try {
    const data = await postJson("/api/pool/adopt", {
      images, lib: poolState.lib, batch_name: $("pool-batch-name").value.trim(),
    });
    const added = `已添加 ${data.added} 张到「${libLabel(data.lib)}」`
      + (data.already ? `（本来就在库里 ${data.already} 张）` : "");
    $("pool-log").textContent = added
      + (data.missing && data.missing.length ? `，池里查无 ${data.missing.length} 张` : "") + "。";
    toast(added);
    poolState.sel = new Set();
    await loadPool();
    if (poolState.onDone) { await poolState.onDone(data); }
  } catch (err) {
    $("pool-log").textContent = "添加失败：" + err.message;
  }
  poolState.busy = false;
  syncPoolButtons();
}

// --------------------------------------------------------------------------- //
// 应用壳（四个页面共用）：命令面板 / 溢出菜单 / 快捷键速查 / 撤销条
// 页面只需 registerShellCommands() 登记动作，再 initShellChrome() 接线，
// 并在自己的 keydown 里先问一句 shellKeys(e)。缺哪个元素就跳过哪个，不会报错。
// --------------------------------------------------------------------------- //
const shell = { commands: [], rows: [], sel: 0, matchFrames: null, hideUndo: null, cd: null };

function registerShellCommands(commands, matchFrames) {
  shell.commands = (commands || []).concat(appearanceCommands());
  if (typeof matchFrames === "function") { shell.matchFrames = matchFrames; }
}

function isTyping(e) {
  return ["INPUT", "SELECT", "TEXTAREA"].includes(((e || {}).target || {}).tagName);
}

function shellRows(query) {
  const q = String(query || "").trim().toLowerCase();
  const hits = shell.commands.filter((c) => !q || String(c.label).toLowerCase().includes(q));
  const frames = q && shell.matchFrames ? shell.matchFrames(q) : [];
  return (frames || []).concat(hits);
}

function renderPalette() {
  const input = $("palette-input"), list = $("palette-list");
  if (!input || !list) { return; }
  shell.rows = shellRows(input.value);
  shell.sel = clamp(shell.sel, 0, Math.max(0, shell.rows.length - 1));
  list.innerHTML = "";
  if (!shell.rows.length) {
    const tip = document.createElement("div");
    tip.className = "palette-hint";
    tip.textContent = "没有匹配的动作或帧";
    list.appendChild(tip);
    return;
  }
  shell.rows.forEach((row, i) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "palette-item" + (i === shell.sel ? " on" : "");
    const label = document.createElement("span");
    label.textContent = row.label;
    label.title = row.label || "";
    item.appendChild(label);
    if (row.tag) {
      const tag = document.createElement("span");
      tag.className = "chip";
      tag.textContent = row.tag;
      item.appendChild(tag);
    }
    if (row.key) {
      const key = document.createElement("span");
      key.className = "kbd k";
      key.textContent = row.key;
      item.appendChild(key);
    }
    item.onclick = () => runPalette(i);
    list.appendChild(item);
  });
}

function openPalette(open) {
  const el = $("palette-modal");
  if (!el) { return; }
  const on = open === undefined ? el.classList.contains("hidden") : Boolean(open);
  el.classList.toggle("hidden", !on);
  if (!on) { return; }
  $("palette-input").value = "";
  shell.sel = 0;
  renderPalette();
  $("palette-input").focus();
}

function runPalette(i) {
  const row = shell.rows[i];
  if (!row) { return; }
  openPalette(false);
  try {
    Promise.resolve(row.run()).catch((err) => toast("失败：" + err.message, true));
  } catch (err) {
    toast("失败：" + err.message, true);
  }
}

function openMenu(open) {
  const menu = $("more-menu");
  if (!menu) { return; }
  const on = open === undefined ? menu.hasAttribute("hidden") : Boolean(open);
  menu.toggleAttribute("hidden", !on);
  $("btn-more").classList.toggle("on", on);
  $("btn-more").setAttribute("aria-expanded", on ? "true" : "false");
}

function openHelp(open) {
  const el = $("help-modal");
  if (!el) { return; }
  el.classList.toggle("hidden", !(open === undefined ? el.classList.contains("hidden") : Boolean(open)));
}

/** 顶栏/左导航条按钮的统一接线；回调动作由页面传进来。 */
function initShellChrome(options) {
  const opts = options || {};
  const bind = (id, fn) => {
    const el = $(id);
    if (el && typeof fn === "function") { el.onclick = fn; }
  };
  applyAppearance({ animate: false });
  if (window.matchMedia) {
    const mq = window.matchMedia("(prefers-color-scheme: light)");
    const follow = () => { if (appearance.theme === "auto") { applyAppearance(); } };
    if (mq.addEventListener) { mq.addEventListener("change", follow); }
    else if (mq.addListener) { mq.addListener(follow); }
  }
  bind("btn-palette", () => openPalette(true));
  bind("btn-more", (e) => { e.stopPropagation(); openMenu(); });
  bind("btn-help", () => openHelp(true));
  bind("help-close", () => openHelp(false));
  bind("rail-queue", opts.onQueue);
  bind("rail-upload", opts.onUpload);
  bind("rail-batches", opts.onBatches);
  bind("rail-fields", opts.onFields);
  bind("rail-run", opts.onRun);
  if ($("palette-input")) {
    $("palette-input").addEventListener("input", () => { shell.sel = 0; renderPalette(); });
    $("palette-input").addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        shell.sel = Math.min(shell.sel + 1, Math.max(0, shell.rows.length - 1));
        renderPalette();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        shell.sel = Math.max(shell.sel - 1, 0);
        renderPalette();
      } else if (e.key === "Enter") {
        e.preventDefault();
        runPalette(shell.sel);
      }
    });
  }
  const closeOnBackdrop = (id, close) => {
    const el = $(id);
    if (el) { el.addEventListener("click", (e) => { if (e.target.id === id) close(); }); }
  };
  closeOnBackdrop("palette-modal", () => openPalette(false));
  closeOnBackdrop("help-modal", () => openHelp(false));
  document.addEventListener("click", (e) => {
    const menu = $("more-menu");
    if (menu && !menu.hasAttribute("hidden")
      && !menu.contains(e.target) && !$("btn-more").contains(e.target)) { openMenu(false); }
  });
}

/** 壳层按键：⌘K 开面板、? 开速查、Esc 关壳层。返回 true 表示这个键已被消费。 */
function shellKeys(e) {
  if ((e.metaKey || e.ctrlKey) && String(e.key).toLowerCase() === "k") {
    e.preventDefault();
    openPalette();
    return true;
  }
  const pal = $("palette-modal");
  if (pal && !pal.classList.contains("hidden")) {
    if (e.key === "Escape") { openPalette(false); }
    return true;
  }
  const help = $("help-modal");
  if (help && !help.classList.contains("hidden")) {
    if (e.key === "Escape" || e.key === "?") { openHelp(false); }
    return true;
  }
  if ($("more-menu") && !$("more-menu").hasAttribute("hidden") && e.key === "Escape") {
    openMenu(false);
    return true;
  }
  if (e.altKey && !e.metaKey && !e.ctrlKey && !isTyping(e)) {
    const code = String(e.code || "");
    if (code === "KeyT") { e.preventDefault(); toggleTheme(); return true; }
    if (code === "Equal" || code === "NumpadAdd") { e.preventDefault(); stepFont(1); return true; }
    if (code === "Minus" || code === "NumpadSubtract") { e.preventDefault(); stepFont(-1); return true; }
  }
  if (e.key === "?" && !isTyping(e)) {
    e.preventDefault();
    openHelp(true);
    return true;
  }
  return false;
}

/** 撤销条：接受/驳回后停 5 秒，手滑不用去面板里翻「撤销」。 */
function showUndoBar(text) {
  const bar = $("undo-bar");
  if (!bar) { return; }
  $("undo-label").textContent = text;
  bar.classList.add("on");
  clearInterval(shell.cd);
  clearTimeout(shell.hideUndo);
  let left = 5;
  $("undo-cd").textContent = left + "s";
  shell.cd = setInterval(() => {
    left -= 1;
    $("undo-cd").textContent = Math.max(0, left) + "s";
    if (left <= 0) { clearInterval(shell.cd); }
  }, 1000);
  shell.hideUndo = setTimeout(() => bar.classList.remove("on"), 5000);
}

function hideUndoBar() {
  clearInterval(shell.cd);
  clearTimeout(shell.hideUndo);
  const bar = $("undo-bar");
  if (bar) { bar.classList.remove("on"); }
}

/** 下载一个后端生成的文件（坐标 CSV / JSONL 这类一次性产物）。 */
function downloadHref(url, note) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
  toast(note || "已开始下载");
}
