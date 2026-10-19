"use strict";

// 五个页面（门户 / 姿态标注 / 提示词标注 / 目标检测 / 模型管理）共用的极小工具集。
// 页面自己的逻辑在 portal.js / app.js / objects.js / models.js 里，各页只加载自己那份；
// 「目标检测」与「提示词标注」共用同一份 objects.js，靠 <body data-lib> 决定吃哪一库，
// 绝不再复制一份：同一件事有两个入口，改天就只有一个跟着改。

const $ = (id) => document.getElementById(id);
const mb = (v) => Number(v || 0).toFixed(1);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
// 数字输入框：留空才回落默认值。`parseFloat(v) || 默认` 会把用户填的 **0** 当假值吞掉
// ——最多帧数填 0、置信度填 0 都会悄悄变回默认，与服务端"0 是实义值"的口径相反。
const readNumber = (id, fallback) => {
  const node = $(id);
  const raw = node ? parseFloat(node.value) : NaN;
  return Number.isFinite(raw) ? raw : fallback;
};

// 导出训练集时留作验证集的比例。三个导出入口（门户面板 / 姿态页菜单 / 两个库的面板）
// 共用这一个默认，只有门户面板上有输入框可改；写死三遍迟早有三个数。
const DEFAULT_VAL_RATIO = 0.2;

// 日志区只留最后这么多行。一次拖两百个文件就是两百行，而
// ``box.textContent += ...`` 每次都要把整段字符串重造一遍，越滚越慢；
// 更前面的行本来也滚出可视区了，留着只是让页面越来越卡。
const LOG_TAIL = 400;
const logState = new WeakMap();

/** 往某个日志区追加一行（带时间戳），超出上限就折成「更早 N 条」。 */
function appendLog(id, msg, isErr) {
  const box = $(id);
  if (!box) return;
  let state = logState.get(box);
  // 有人清空过日志区（「清空」按钮直接写 textContent），行缓存必须跟着归零，
  // 否则下一次追加会把已经清掉的那批又写回来。
  if (!state || state.rendered !== box.textContent) state = { lines: [], dropped: 0, rendered: "" };
  state.lines.push(`[${new Date().toLocaleTimeString("zh-CN")}] ${msg}`);
  if (state.lines.length > LOG_TAIL) {
    state.dropped += state.lines.length - LOG_TAIL;
    state.lines = state.lines.slice(-LOG_TAIL);
  }
  state.rendered = (state.dropped ? `（更早 ${state.dropped} 条已折叠）\n` : "")
    + state.lines.join("\n");
  logState.set(box, state);
  box.textContent = state.rendered;
  box.classList.toggle("err", Boolean(isErr));
  box.scrollTop = box.scrollHeight;
}
const kb = (n) => (n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB");

/** 上传单个文件，带**真实字节进度**（只有 XHR 有 upload.onprogress，fetch 没有）。
 *
 * 进度回调收 ``(已传字节, 总字节)``；失败时 reject 的 Error 消息一律中文——
 * 上传 500 MB 录像时那条进度条就是这页唯一的表现，卡在 0% 和走到 60% 是两种体验。
 */
function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload?name=" + encodeURIComponent(file.name));
    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(Math.min(event.loaded, file.size), file.size);
      };
    }
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText || "{}"); } catch (err) { data = {}; }
      if (xhr.status >= 200 && xhr.status < 300) { resolve(data); return; }
      // 超限 / 类型不支持时服务端回的就是中文 JSON，优先用它，剩下的才按状态码兜底
      const reason = data.error || (xhr.status === 413 ? "超过单文件大小上限" : `上传失败（HTTP ${xhr.status}）`);
      reject(new Error(reason));
    };
    xhr.onerror = () => reject(new Error("网络中断，这一条没传完"));
    xhr.ontimeout = () => reject(new Error("上传超时"));
    xhr.send(file);
  });
}

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
  { id: "objects", href: "/objects", name: "提示词标注", desc: "大模型按你写的描述找物品" },
  { id: "detect", href: "/detect", name: "目标检测", desc: "本地 YOLO 权重按自带类别出框" },
  { id: "models", href: "/models", name: "模型管理", desc: "搜模型、下载权重、多模态配置" },
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
  model_stale: "标注用的模型或设置已变更",
  model_failed: "这次推理失败",
  masked: "分割轮廓已随框保存",
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
  renderSettingsPanel();
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

// --------------------------------------------------------------------------- //
// 系统设置：白天/夜晚、字号这些界面偏好，和「标注跑在哪块卡上」收进同一个面板。
// 两半存的地方不一样是有意的：外观是"我这台电脑怎么看"，存浏览器 localStorage，
// 切页首屏还有一段内联脚本先定主题防闪烁；推理算力决定标注算在哪，必须落到
// 服务器那份 ~/.autolabel/settings.json，换台电脑打开同一个数据集才不会被偷偷改回去。
// --------------------------------------------------------------------------- //
const settingsState = { data: null, loading: false, saving: false };

const settingsSaved = () => (settingsState.data || {}).saved || {};
const settingsDevice = (value) => ((settingsState.data || {}).devices || {})[value] || {};
const deviceLabel = (value) => ((settingsState.data || {}).choices || [])
  .map((c) => c.value === value ? c.label : "").filter(Boolean)[0] || value || "—";

function buildSettingsModal() {
  const existed = $("settings-modal");
  if (existed) { return existed; }
  const wrap = document.createElement("div");
  wrap.id = "settings-modal";
  wrap.className = "modal hidden";
  wrap.innerHTML = `
<div class="modal-card">
  <div class="modal-title">系统设置<span class="muted small"> · 界面外观存这台浏览器，推理参数存这台机器</span></div>
  <div class="settings-grid">
    <section>
      <div class="panel-title">界面外观</div>
      <div class="set-row"><span class="set-name">白天 / 夜晚</span></div>
      <div class="seg" id="set-theme"></div>
      <div class="set-row"><span class="set-name">字体大小</span></div>
      <div class="seg" id="set-font"></div>
      <div class="set-note muted small">点一下就生效，不用保存；只影响这个浏览器，不会写进数据集。</div>
    </section>
    <section>
      <div class="panel-title">推理参数</div>
      <div class="set-note muted small">标注跑在哪块卡上。选了这台机器没有的设备不会报错，跑的时候自动降级并说明原因。</div>
      <div class="set-note muted small" id="set-env" hidden></div>
      <div id="set-devices"></div>
      <div class="set-row">
        <span class="set-name">半精度加速<span class="muted small">只有 NVIDIA 显卡提速，CPU 上会自动按普通精度跑</span></span>
        <button class="switch" id="set-half" type="button" role="switch" aria-checked="false"><span></span></button>
      </div>
      <div class="set-row"><span class="set-name">输入分辨率<span class="muted small">越大越容易检出小目标与服装类，代价是更慢</span></span></div>
      <div class="seg" id="set-imgsz"></div>
      <div class="set-row">
        <span class="set-name">默认置信度<span class="muted small">低于这个分数的框直接丢掉，标服装/配件时调低它</span></span>
        <input class="set-num mono" id="set-conf" type="number" min="0.01" max="0.95" step="0.01" inputmode="decimal">
      </div>
      <div class="set-note muted small" id="set-openvocab" hidden></div>
      <div class="set-note muted small" id="set-status">读取中…</div>
    </section>
  </div>
  <div class="set-note muted small">配置文件 <span class="mono" id="set-path">—</span> ·
    命令行 <span class="mono">--device</span> / <span class="mono">--config</span> 里写明的值优先于这里</div>
  <div class="btn-row"><button id="set-refresh">重新探测显卡</button><button id="set-close">关闭</button></div>
</div>`;
  document.body.appendChild(wrap);
  $("set-close").onclick = () => openSettings(false);
  $("set-refresh").onclick = () => loadSettings(true);
  $("set-half").onclick = () => saveSettings({ half: !settingsSaved().half });
  // 置信度是打字输入的：失焦或回车才存，边打字边存会每敲一位就发一次请求
  const confInput = $("set-conf");
  confInput.onchange = () => saveSettings({ conf: confInput.value });
  confInput.onkeydown = (e) => { if (e.key === "Enter") { confInput.blur(); } };
  wrap.addEventListener("click", (e) => { if (e.target === wrap) { openSettings(false); } });
  return wrap;
}

function setSettingsStatus(text, kind) {
  const el = $("set-status");
  if (!el) { return; }
  el.textContent = text;
  el.classList.toggle("warn", kind === "warn");
  el.classList.toggle("ok", kind === "ok");
}

/** 推理参数的一行摘要：分辨率与置信度是"为什么没框"的第一现场，得常驻显示。 */
function inferenceSummary(saved) {
  const data = saved || {};
  const bits = [];
  if (data.imgsz != null) { bits.push("分辨率 " + data.imgsz); }
  if (data.conf != null) { bits.push("置信度 " + data.conf); }
  return bits.length ? " · " + bits.join(" · ") : "";
}

/** 分段选择器（外观 / 字号）与设备清单一并刷新；面板没建就直接跳过。 */
function renderSettingsPanel() {
  if (!$("settings-modal")) { return; }
  const seg = (id, choices, active, pick) => {
    const host = $(id);
    if (!host) { return; }
    host.innerHTML = "";
    choices.forEach((choice) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = choice.label;
      btn.className = choice.value === active ? "on" : "";
      btn.title = choice.value === "auto" ? "跟随系统的深色设置"
        : (choice.hint || (choice.scale ? "字号 " + Math.round(choice.scale * 100) + "%" : choice.label));
      btn.onclick = () => pick(choice.value);
      host.appendChild(btn);
    });
  };
  seg("set-theme", THEME_CHOICES, appearance.theme, setThemePref);
  seg("set-font", FONT_CHOICES, appearance.fs, setFontStep);
  seg("set-imgsz", (settingsState.data || {}).imgsz_choices || [], settingsSaved().imgsz,
    (value) => saveSettings({ imgsz: value }));
  const confBox = $("set-conf");
  // 正在这个框里打字时别把用户刚敲的数字盖回去
  if (confBox && document.activeElement !== confBox) {
    confBox.value = settingsSaved().conf == null ? "" : settingsSaved().conf;
  }
  renderDeviceRows();
  const half = $("set-half");
  if (half) {
    half.classList.toggle("on", Boolean(settingsSaved().half));
    half.setAttribute("aria-checked", settingsSaved().half ? "true" : "false");
  }
  const path = $("set-path");
  if (path) { path.textContent = (settingsState.data || {}).path || "—"; }
  renderOpenVocabRow();
}

/** 开放词表权重（YOLOE / YOLO-World）的依赖状况。
 *
 *  缺依赖不是故障：第一次用会自动装包并下载编码器权重，说清楚免得用户
 *  对着"缺依赖"以为坏了要来报 bug；已缓存时把体积报出来，磁盘账好对。
 */
function renderOpenVocabRow() {
  const host = $("set-openvocab");
  if (!host) { return; }
  const info = (settingsState.data || {}).open_vocab || {};
  if (!info || !Object.keys(info).length) { return; }
  const cached = Math.round((info.cached_bytes || 0) / 1048576);
  host.hidden = false;
  host.textContent = "开放词表权重（YOLOE / YOLO-World）· "
    + (info.installed ? "依赖已就绪" : "第一次用时自动装依赖")
    + (cached ? ` · 文本编码器权重已缓存 ${cached} MB` : "")
    + (info.note ? `：${info.note}` : "");
}

/** 设备清单：能用/不能用都说清楚，按不下去的绝不摆成能按的样子。 */
function renderDeviceRows() {
  const host = $("set-devices");
  if (!host) { return; }
  const data = settingsState.data || {};
  const probed = Boolean(data.devices && Object.keys(data.devices).length);
  const picked = settingsSaved().device;
  const envNote = $("set-env");
  if (envNote) {
    // 启动参数指定的算力只属于这一个实例：不写出来，改了这里却没换卡会很费解
    const env = data.env || {};
    const parts = [];
    if (env.device) { parts.push(`推理设备被启动参数指定为「${env.device}」`); }
    if (typeof env.half === "boolean") { parts.push(`半精度被指定为「${env.half ? "开" : "关"}」`); }
    if (env.imgsz) { parts.push(`输入分辨率被指定为「${env.imgsz}」`); }
    if (typeof env.conf === "number") { parts.push(`默认置信度被指定为「${env.conf}」`); }
    envNote.hidden = !parts.length;
    envNote.textContent = parts.length
      ? `${parts.join("，")}：这个实例以它为准，在这里改动只对其它实例生效`
      : "";
  }
  const rows = (data.choices || []).map((choice) => {
    const facts = settingsDevice(choice.value);
    const row = document.createElement("button");
    row.type = "button";
    const unusable = probed && facts.available === false;
    row.className = "dev-item" + (choice.value === picked ? " on" : "") + (unusable ? " off" : "");
    row.title = choice.hint || choice.label;

    const head = document.createElement("span");
    head.className = "dev-head";
    const name = document.createElement("span");
    name.className = "dev-name";
    name.textContent = choice.label;
    const chip = document.createElement("span");
    chip.className = "chip " + (choice.value === "auto" ? "" : unusable ? "bad" : "ok");
    chip.textContent = choice.value === "auto" ? "推荐" : unusable ? "这台机器用不了" : "可用";
    head.append(name, chip);

    const note = document.createElement("span");
    note.className = "dev-note";
    const names = (facts.names || []).filter(Boolean);
    note.textContent = unusable ? (facts.reason || "这台机器上没有")
      : (names[0] || choice.hint || "");
    row.append(head, note);
    row.onclick = () => saveSettings({ device: choice.value });
    return row;
  });
  host.innerHTML = "";
  rows.forEach((row) => host.appendChild(row));
}

async function saveSettings(patch) {
  if (settingsState.saving) { return; }
  settingsState.saving = true;
  setSettingsStatus("保存中…");
  try {
    const data = await postJson("/api/settings", patch);
    settingsState.data = Object.assign({}, settingsState.data, { saved: data.saved, path: data.path });
    renderSettingsPanel();
    const saved = data.saved || {};
    if (data.note) {
      setSettingsStatus(data.note, "warn");
      toast(data.note, true);
    } else {
      setSettingsStatus("已保存 · " + deviceLabel(saved.device)
        + " · 半精度" + (saved.half ? "开" : "关") + inferenceSummary(saved), "ok");
    }
  } catch (err) {
    setSettingsStatus("保存失败：" + err.message, "warn");
    toast("保存失败：" + err.message, true);
  } finally {
    settingsState.saving = false;
  }
}

async function loadSettings(probe) {
  if (settingsState.loading) { return; }
  settingsState.loading = true;
  setSettingsStatus(probe ? "正在探测这台机器的显卡…" : "读取中…");
  try {
    settingsState.data = await getJson("/api/settings" + (probe ? "?probe=1" : ""));
    renderSettingsPanel();
    const saved = settingsSaved();
    const note = ((settingsState.data || {}).effective || {}).note || "";
    setSettingsStatus("当前：" + deviceLabel(saved.device) + " · 半精度" + (saved.half ? "开" : "关")
      + inferenceSummary(saved) + (note ? " · " + note : ""), note ? "warn" : "");
  } catch (err) {
    setSettingsStatus("读不到系统设置：" + err.message, "warn");
  } finally {
    settingsState.loading = false;
  }
}

function openSettings(open) {
  const modal = buildSettingsModal();
  const on = open === undefined ? modal.classList.contains("hidden") : Boolean(open);
  modal.classList.toggle("hidden", !on);
  if (!on) { return; }
  renderSettingsPanel();
  loadSettings(true);
}

function initSettings() {
  const btn = $("btn-settings");
  if (btn) { btn.onclick = (e) => { e.stopPropagation(); openSettings(true); }; }
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
    { label: "打开系统设置（外观 / 推理参数）", tag: "设置", key: "⌥S", run: () => openSettings(true) },
    { label: "推理设备：改用 Apple 加速（MPS）", tag: "设置", run: () => saveSettings({ device: "mps" }) },
    { label: "推理设备：改用 NVIDIA 显卡（CUDA）", tag: "设置", run: () => saveSettings({ device: "cuda" }) },
    { label: "推理设备：改用纯 CPU", tag: "设置", run: () => saveSettings({ device: "cpu" }) },
    { label: "推理设备：自动选择", tag: "设置", run: () => saveSettings({ device: "auto" }) },
    { label: "检查更新（比对 GitHub）", tag: "关于", run: () => checkForUpdates() },
    { label: "打开 GitHub 仓库", tag: "关于", run: () => openExternal((versionState.info || {}).repo_url) },
    { label: "打开开发者 GitHub 主页", tag: "关于",
      run: () => openExternal((versionState.info || {}).developer_url) },
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

/** 左侧竖条里补上缺的模块入口。
 *
 *  竖条是各页写死的 HTML，加一个二级模块就要改五个文件，漏掉的那页等于没有入口
 *  （上次加提示词标注时模型管理页就漏了）。这里按 MODULES 兜底补齐：
 *  页面自己写好了就不动，缺了才插到「提示词标注」后面，位置和各页保持一致。
 */
function ensureRailModules() {
  const rail = $("rail");
  if (!rail) { return; }
  for (const item of MODULES) {
    if (rail.querySelector('a[href="' + item.href + '"]')) { continue; }
    const link = document.createElement("a");
    link.className = "iconbtn";
    link.href = item.href;
    link.title = item.name + " · " + item.desc;
    link.innerHTML = '<svg class="ic"><use href="#ic-boxes"/></svg>';
    const anchor = rail.querySelector('a[href="/objects"]') || rail.querySelector("a[href]");
    if (anchor) { anchor.after(link); } else { rail.prepend(link); }
  }
}

/** 「⋯」菜单里的模块跳转写死在各页 HTML 里，新增加一个库时在这里补齐：
 *
 *  否则哪页的 HTML 还没来得及改，用户从那页就跳不去新模块。已有同 href 的不重复加。
 */
function ensureMenuModules() {
  const menu = $("more-menu");
  if (!menu) { return; }
  const page = document.body.dataset.page || "";
  const missing = MODULES.filter((item) => item.id !== page
    && !menu.querySelector('a[href="' + item.href + '"]'));
  if (!missing.length) { return; }
  const anchor = $("btn-settings") || menu.lastElementChild;
  if (!anchor) { return; }
  const label = document.createElement("div");
  label.className = "menu-label";
  label.textContent = "其他模块";
  anchor.before(label);
  for (const item of missing) {
    const link = document.createElement("a");
    link.className = "menu-item";
    link.href = item.href;
    link.title = item.desc;
    link.innerHTML = '<svg class="ic"><use href="#ic-boxes"/></svg>';
    link.append(document.createTextNode(item.name));
    anchor.before(link);
  }
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
  ensureRailModules();
  ensureMenuModules();
  bindConfirmBox();
  bindModalKeys();
  initVersionWidget();
  initSettings();
}

// --------------------------------------------------------------------------- //
// 版本与更新提示：四个页面顶栏都挂着版本号，数字只来自后端 autolabel/version.py，
// 前端一个都不写死——写死就会出现"界面说 1.2、命令行说 1.1"这种没人信的数字。
// --------------------------------------------------------------------------- //
const UPDATE_DISMISS_KEY = "autolabel.updateDismissed";
const versionState = { info: null };

const dismissedVersion = () => {
  try { return localStorage.getItem(UPDATE_DISMISS_KEY) || ""; } catch (e) { return ""; }
};

function rememberDismissed(version) {
  try { localStorage.setItem(UPDATE_DISMISS_KEY, String(version || "")); } catch (e) { /* 无痕模式只是记不住 */ }
}

function openExternal(url) {
  if (url) window.open(url, "_blank", "noopener");
}

/** 「上次检查」这行要说实话：没查到就写没查到，别留空让人以为软件坏了。 */
function versionNote(info) {
  if (!info.checked) return "GitHub 版本没查到" + (info.note ? "：" + info.note : "");
  if (info.update_available) return "GitHub 最新 v" + info.latest + "，可以升级";
  if (info.ahead) return "GitHub 最新 v" + info.latest + "，本地是开发中的版本";
  return "GitHub 最新 v" + info.latest + " · 已是最新版";
}

function renderVersion(info) {
  versionState.info = info || {};
  const current = info.current || "";
  const chip = $("version-chip");
  if (chip) {
    chip.textContent = "v" + (current || "—");
    chip.href = info.repo_url || "#";
    chip.title = "当前 v" + current + " · 开发者 " + (info.developer || "—") + " · 点开看 GitHub 仓库";
    chip.classList.toggle("update", !!info.update_available);
  }
  const version = $("about-version");
  if (version) {
    version.textContent = "v" + (current || "—")
      + (info.checked && info.latest && info.latest !== current ? " · GitHub v" + info.latest : "");
  }
  const dev = $("about-dev");
  if (dev) {
    dev.textContent = info.developer || "—";
    dev.href = info.developer_url || "#";
    dev.title = "打开 " + (info.developer || "") + " 的 GitHub 主页";
  }
  const home = $("about-home");
  if (home) home.href = info.repo_url || "#";
  const note = $("about-note");
  if (note) note.textContent = versionNote(info);
  // 「知道了」只按版本生效：升到 GitHub 那一版之前不再刷这条，但真出了更新版还会来
  const bar = $("update-bar");
  const show = !!info.update_available && info.latest !== dismissedVersion();
  if (bar) bar.hidden = !show;
  if (show) {
    const text = $("update-text");
    if (text) text.textContent = "GitHub 上有新版本 v" + info.latest + "（当前 v" + current + "）";
    const open = $("update-open");
    if (open) open.href = info.releases_url || info.repo_url || "#";
  }
}

async function loadVersion(force) {
  try {
    const info = await getJson("/api/version" + (force ? "?refresh=1" : ""));
    renderVersion(info);
    return info;
  } catch (e) {
    return null;   // 版本徽标不是关键路径：取不到就安静待着，别在顶栏报错
  }
}

/** 菜单 / 命令面板里的「检查更新」：真的联网比一次，有没有结果都当面说清。 */
async function checkForUpdates() {
  toast("正在比对 GitHub 上的版本…");
  const info = await loadVersion(true);
  if (!info) return toast("没查到 GitHub 上的版本，这台机器可能没联网", true);
  if (info.update_available) return toast("有新版本 v" + info.latest + "（当前 v" + info.current + "）");
  if (!info.checked) return toast("没查到 GitHub 上的版本：" + (info.note || "未知原因"), true);
  return toast(info.ahead ? "本地 v" + info.current + " 比 GitHub 的 v" + info.latest + " 更新"
                          : "已经是最新版 v" + info.current);
}

function initVersionWidget() {
  const later = $("update-later");
  if (later) {
    later.addEventListener("click", () => {
      rememberDismissed((versionState.info || {}).latest);
      const bar = $("update-bar");
      if (bar) bar.hidden = true;
    });
  }
  const check = $("about-check");
  if (check) check.addEventListener("click", () => checkForUpdates());
  loadVersion(false);
}

// --------------------------------------------------------------------------- //
// 导出结果：产物是目录，路径不能只在 toast 里闪一下，这里给常驻行 + 一键复制
// --------------------------------------------------------------------------- //
const EXPORT_COMMANDS = {
  pose: { verb: "pose", model: "yolo11s-pose.pt" },
  detect: { verb: "detect", model: "yolo11s.pt" },
  // 目标检测库导出的 kind 与提示词库不同名，少了这行训练命令会错拿 pose 的那条
  yolo_detect: { verb: "detect", model: "yolo11n.pt" },
};

/** 复制了目录还得手敲训练命令，这里直接给整条。 */
function exportCommand(kind, row) {
  const spec = EXPORT_COMMANDS[kind] || EXPORT_COMMANDS.pose;
  return `yolo ${spec.verb} train data="${row.data_yaml}" model=${spec.model} epochs=100 imgsz=640`;
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
// 素材池：三个库共用 images/ 下的素材，这里提供"把池里已有的图添加进本库"的选择器
// --------------------------------------------------------------------------- //
const LIB_LABELS = { pose: "姿态标注", objects: "提示词标注", detect: "目标检测" };
const libLabel = (lib) => LIB_LABELS[lib] || String(lib || "本库");

// 素材池里每张图被哪些库引用：后端按下发 in_<lib>，界面按这张表念名字。
// 写成一等数据而不是三处 if，是为了以后加库只改这张表——上一版加检测库时，
// "已在姿态库 / 已在提示词库" 那句就漏了新库，新库的图看起来像"哪都没去"。
const POOL_LIBS = [
  { lib: "pose", name: "姿态库" },
  { lib: "objects", name: "提示词库" },
  { lib: "detect", name: "检测库" },
];

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
    const owned = (data.libs || POOL_LIBS)
      .map((row) => `${row.name} ${poolState.stats[row.lib] || 0}`).join(" · ");
    $("pool-note").textContent = `素材池共 ${poolState.stats.files || 0} 个文件（`
      + owned + (owned ? " · " : "") + `几个库共用 ${poolState.stats.shared || 0}）`
      + `· ${kb(poolState.stats.bytes || 0)}；`
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
    const owned = POOL_LIBS.filter((row) => file["in_" + row.lib]).map((row) => "已在" + row.name);
    chips.textContent = owned.join(" · ") || "各库都没有";
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
  const own = commands || [];
  shell.commands = own.concat(moduleNavCommands(own), appearanceCommands());
  if (typeof matchFrames === "function") { shell.matchFrames = matchFrames; }
}

/** 每个模块都必须能在 ⌘K 里搜到：页面自己登记过的跳转（文案里带模块名）不重复补。 */
function moduleNavCommands(own) {
  const labels = (own || []).map((row) => String(row.label || ""));
  return MODULES
    .filter((item) => !labels.some((label) => label.includes(item.name)))
    .map((item) => ({
      label: "去" + item.name, tag: "页面",
      run: () => { location.href = item.href; },
    }));
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
  const settings = $("settings-modal");
  if (settings && !settings.classList.contains("hidden")) {
    if (e.key === "Escape") { openSettings(false); }
    return true;   // 面板开着的时候别把 Q / E / R 这些页面快捷键漏到后面去
  }
  if ($("more-menu") && !$("more-menu").hasAttribute("hidden") && e.key === "Escape") {
    openMenu(false);
    return true;
  }
  if (e.altKey && !e.metaKey && !e.ctrlKey && !isTyping(e)) {
    const code = String(e.code || "");
    if (code === "KeyT") { e.preventDefault(); toggleTheme(); return true; }
    if (code === "KeyS") { e.preventDefault(); openSettings(true); return true; }
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
function showUndoBar(text, action) {
  const bar = $("undo-bar");
  if (!bar) { return; }
  $("undo-label").textContent = text;
  // 撤销条承诺的是"退回刚才那一步"。以前只带一句话，按钮固定绑在页面的"撤销坐标修改"上，
  // 而 Q / E 之后界面会自动翻页——点撤销就把**下一帧**的坐标改回去了，状态原样留着。
  shell.undoAction = action || null;
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
  shell.undoAction = null;
  const bar = $("undo-bar");
  if (bar) { bar.classList.remove("on"); }
}

/** 撤销条上的按钮与 `U` 键都走这里；没有待撤销动作时返回 false，让页面退回常规撤销。 */
function runPendingUndo() {
  const action = shell.undoAction;
  if (!action) { return false; }
  shell.undoAction = null;
  hideUndoBar();
  action();
  return true;
}

// ---- 「这一帧要不要人看」的口径：由后端下发，前端不再自己猜 ----
const flagPolicy = { positive: new Set(["multi_model_confirmed"]), lowScore: 0.4 };

/** 数据集加载后同步后端口径：正向标记清单 + 置信度阈值。 */
function setFlagPolicy(ds) {
  if (ds && Array.isArray(ds.positive_flags)) flagPolicy.positive = new Set(ds.positive_flags);
  if (ds && typeof ds.low_score === "number") flagPolicy.lowScore = ds.low_score;
}

/** 真正要人处理的标记：剔掉「两个模型结论一致」这类正向标记（后端权重是负的）。 */
function issueFlags(item, key) {
  const list = (item && item[key || "flags"]) || [];
  return list.filter((f) => !flagPolicy.positive.has(f));
}

/** 被剔掉的正向标记，单独显示成加分项，别让用户以为是漏了。 */
function positiveFlags(item, key) {
  const list = (item && item[key || "flags"]) || [];
  return list.filter((f) => flagPolicy.positive.has(f));
}

/** 导出范围：与后端 /api/export 的 scope 一一对应，界面上终于能选。 */
const EXPORT_SCOPES = [
  ["reviewed", "剔除已驳回（含未复核）"],
  ["approved", "只要已接受 / 已修改"],
  ["all", "全部帧（含已驳回）"],
];
const scopeLabel = (value) => (EXPORT_SCOPES.find((row) => row[0] === value) || ["", value])[1];

/** 导出结果说人话：收了多少 + 为什么少了这些张（后端算好的剔除明细）。 */
function exportSummary(info, kind) {
  const counts = (info && info.counts) || {};
  const bits = [`${info.train || 0} 训练 / ${info.val || 0} 验证`];
  if (kind === "pose") {
    const people = info.people || {};
    bits.push(`${(people.train || 0) + (people.val || 0)} 人`);
  } else {
    const boxes = info.boxes || {};
    bits.push(`${(boxes.train || 0) + (boxes.val || 0)} 框`);
  }
  if (info.segment_note) bits.push(info.segment_note);
  if (info.skipped_note) bits.push(`没收进来的 ${info.skipped_note}`);
  if (info.mode_note) bits.push(info.mode_note);
  const dropped = info.empty_dropped !== undefined ? info.empty_dropped : counts.empty_dropped;
  if (dropped) bits.push(`其中背景图按上限少收 ${dropped} 张`);
  return bits.join(" · ");
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
