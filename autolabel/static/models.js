"use strict";

// 二级模块「模型管理」：官方模型目录的搜索与下载、本机权重的查看/删除、
// 本地权重登记，以及多模态大模型（本地服务 / 云端 API）配置与连通性验证。
//
// 页面上的两个列表各管一件事，别再混：
//   「模型目录」= 能下什么（默认只显示本机已下载的那几档，要看官方全量就关掉开关或搜一下）；
//   「本机权重」= models/ 里真有什么文件（未下载的预置档不占这里的行）。

const modelState = {
  weights: [], weightsDir: "", downloads: {}, presets: [], profiles: [], defaultId: null,
  selected: new Set(), polling: null,
  rows: [], catalog: [], defaults: {}, filter: "", secretsPath: "", configPath: "",
  // 目录搜索：query 只做本地模糊匹配；remote 打开后请求才带 remote=1 去搜 GitHub
  query: "", remote: false, searching: false, searchSeq: 0, searchTimer: null,
  // loaded：成功拿到过一次目录结果；没拿到之前只能退回本机登记表粗筛
  hub: { ok: true, loaded: false, query: "", task: "", local: [], results: [], note: "",
         error: "", remote_state: "skipped" },
  // installedOnly：目录默认只摆本机已下载的模型。官方一摆就是一百六十档，
  // 全列出来真正能点的下载按钮反而找不到；要看全量随时能关。
  installedOnly: true,
  // 目标检测库当前用的权重：存在数据集 .autolabel/detect/prefs.json，跟姿态默认是两件事
  detectPrefs: {},
};

const MODEL_HINTS = { "yolo11n-pose.pt": "快", "yolo11s-pose.pt": "更准", "yolo11m-pose.pt": "最准" };
const VLM_KIND_LABEL = { local: "本地", cloud: "云端" };
const MODEL_BADGE_CLASS = { available: "done", incomplete: "failed", missing: "", incompatible: "failed" };

/** 统一模型清单：优先用后台的 model_rows（带中文说明与兼容性），退化成权重行也要能列出来。 */
function registryRows() {
  if (Array.isArray(modelState.rows) && modelState.rows.length) return modelState.rows;
  return (modelState.weights || []).map((w) => ({
    id: String(w.name).replace(/\.pt$/, ""), display_name: w.name, filename: w.name,
    capability: "pose", ready: Boolean(w.ready), size_mb: w.size_mb || 0, expected_mb: 0,
    status: w.ready ? "available" : (w.size_mb ? "incomplete" : "missing"),
    status_label: w.ready ? "可用" : (w.size_mb ? "文件不完整" : "未下载"),
    compatible: true, compatibility_reasons: [], description: "",
  }));
}

async function refreshModels() {
  let data;
  try {
    data = await getJson("/api/models");
  } catch (err) {
    toast("模型信息读取失败：" + err.message, true);
    return;
  }
  modelState.weights = data.weights || [];
  modelState.weightsDir = data.weights_dir || "";
  modelState.downloads = {};
  for (const task of data.downloads || []) modelState.downloads[task.name] = task;
  modelState.rows = data.models || modelState.rows;
  modelState.catalog = data.catalog || [];
  modelState.defaults = data.defaults || modelState.defaults;
  modelState.secretsPath = data.secrets_path || modelState.secretsPath;
  modelState.configPath = data.model_config_path || modelState.configPath;
  const secretsBox = $("model-secrets-path");
  if (secretsBox && modelState.secretsPath) secretsBox.textContent = modelState.secretsPath;
  const configBox = $("model-config-path");
  if (configBox && modelState.configPath) configBox.textContent = modelState.configPath;
  await loadDetectPrefs();
  if (data.vlm) {
    modelState.presets = data.vlm.presets || modelState.presets;
    modelState.profiles = data.vlm.profiles || [];
    modelState.defaultId = data.vlm.default || null;
    renderProviderOptions();
    // 首次打开时按预设把地址/模型名填好，用户已经在编辑的话不动它
    if (!$("vlm-id").value && !$("vlm-base").value.trim()) applyVlmPreset();
  }
  for (const name of Array.from(modelState.selected)) {
    if (!modelState.weights.some((w) => w.name === name)) modelState.selected.delete(name);
  }
  $("model-dir").textContent = modelState.weightsDir || "models/";
  renderWeights();
  renderCatalog();
  hubSearch({ immediate: true });   // 本机装了什么，搜索结果里的「已在本机」要跟着变
  renderVlmList();
  renderModelNote();
  if (Object.values(modelState.downloads).some((t) => t.stage === "running")) startModelPolling();
  else stopModelPolling();
}

/** 目标检测页当前用哪个权重：接口不存在或没选过都算正常，界面就不显示检测默认徽标。 */
async function loadDetectPrefs() {
  try {
    const data = await getJson("/api/detect/models");
    modelState.detectPrefs = data.prefs || {};
  } catch (err) {
    modelState.detectPrefs = {};   // 还没建检测库 / 老版本后端，静默降级
  }
}

/** 设为检测默认：写的是数据集里的检测偏好，不是这台机器的姿态默认权重。 */
async function setDetectDefault(filename) {
  let data;
  try {
    data = await postJson("/api/detect/prefs", { model: filename });
  } catch (err) {
    // 后端已经把话说清楚了（「本机没有 X：先在模型管理页搜到它并下载」「这是「人体姿态」权重…」），照原文显示
    const msg = String(err.message || "");
    toast(msg.includes("not found")
      ? "设为检测默认失败：这个服务还是旧版本，重启后才有该接口"
      : msg || "设为检测默认失败", true);
    return;
  }
  const model = (data && data.model) || filename;
  modelState.detectPrefs = (data && data.prefs) || { model };
  modelState.detectPrefs.model = model;
  toast(`已设为检测默认：${model}`);
  renderCatalog();
}

/** 顶栏副标题里报一句「默认模型 / 多模态配置」，省得用户去目录里翻。 */
function renderModelNote() {
  const rows = registryRows();
  const def = rows.find((r) => r.default && (r.capability || "pose") !== "vlm");
  const vlm = rows.find((r) => r.default && (r.capability || "pose") === "vlm");
  const bits = [];
  if (def) bits.push(`默认姿态模型 ${def.display_name || def.id}`);
  if (vlm) bits.push(`默认多模态配置 ${vlm.display_name || vlm.id}`);
  const el = $("model-default-note");
  if (el) el.textContent = bits.length ? " · " + bits.join(" · ") : " · 还没设默认模型";
}

/** 把只在 downloads 里出现的权重（下载中 / 下载失败）补成清单行，否则进度无处显示。 */
function syncWeightRows() {
  for (const [name, task] of Object.entries(modelState.downloads)) {
    if (modelState.weights.some((w) => w.name === name)) continue;
    modelState.weights.push({
      name, ready: false, size_mb: 0, custom: !MODEL_HINTS[name], mtime: null,
      partial_mb: Math.round((task.done || 0) / 104857.6) / 10,
    });
  }
}

/** 本机权重 = models/ 里真存在的文件（含下载中、半截残留）；未下载的预置档不在这里。 */
function localWeightRows() {
  return (modelState.weights || []).filter((w) => w.ready || (w.size_mb || 0) > 0
    || (w.partial_mb || 0) > 0 || modelState.downloads[w.name]);
}

function renderWeights() {
  syncWeightRows();
  const box = $("model-weights");
  const rows = localWeightRows();
  box.innerHTML = "";
  if (!rows.length) {
    box.innerHTML = '<div class="batch-empty muted small">models/ 里还没有权重文件：'
      + '在上面「模型目录」里搜一个（yolo11n.pt 只有 5 MB），点「下载」就能用。</div>';
  }
  for (const info of rows) {
    const task = modelState.downloads[info.name] || null;
    const running = Boolean(task && task.stage === "running");
    const state = running
      ? { cls: "running", text: task.message }
      : info.ready
        ? { cls: "done", text: "已就绪 " + mb(info.size_mb) + " MB" }
        : task && task.stage === "failed"
          ? { cls: "failed", text: "下载失败" }
          : info.size_mb > 0
            ? { cls: "failed", text: "不完整，仅 " + mb(info.size_mb) + " MB" }
            : { cls: "", text: "未下载" };
    const deletable = info.size_mb > 0 || info.partial_mb > 0;
    const weightTask = info.task || "";
    const openVocab = String(info.task_name || "").includes("开放词表");
    const row = document.createElement("div");
    row.className = "model-row" + (modelState.selected.has(info.name) ? " selected" : "");
    row.innerHTML = `
      <input type="checkbox" ${modelState.selected.has(info.name) ? "checked" : ""} ${deletable ? "" : "disabled"}>
      <div class="model-main">
        <div class="model-title-line">
          <span class="badge task-badge ${TASK_BADGE_CLASS[weightTask] || "module"}"></span>
          <span class="model-name"></span>
          <span class="chip ${openVocab ? "" : "hidden"}">开放词表·填词</span>
        </div>
        <div class="muted small model-meta"></div>
      </div>
      <span class="badge ${state.cls}">${state.text}</span>
      <button class="dl">${running ? "下载中…" : info.ready ? "重新下载" : "下载"}</button>`;
    row.querySelector(".task-badge").textContent = TASK_LABEL[weightTask] || "任务未知";
    row.querySelector(".model-name").textContent = info.name;
    const lead = info.custom ? "自定义权重 · "
      : MODEL_HINTS[info.name] ? MODEL_HINTS[info.name] + " · " : "";
    row.querySelector(".model-meta").textContent = lead
      + (info.mtime ? "更新于 " + info.mtime : "首次标注时自动经镜像下载")
      + (info.partial_mb ? ` · 残留半截 ${mb(info.partial_mb)} MB` : "")
      + (task && task.error ? " · " + task.error : "");
    row.querySelector("input").onchange = (e) => {
      if (e.target.checked) modelState.selected.add(info.name);
      else modelState.selected.delete(info.name);
      renderWeights();
    };
    const button = row.querySelector(".dl");
    button.disabled = running;
    button.onclick = (e) => { e.preventDefault(); downloadWeight(info.name, Boolean(info.ready)); };
    box.appendChild(row);
  }
  const ready = rows.filter((w) => w.ready);
  const total = ready.reduce((a, w) => a + (w.size_mb || 0), 0);
  const tally = new Map();
  for (const w of ready) {
    const label = TASK_LABEL[w.task || ""] || "任务未知";
    tally.set(label, (tally.get(label) || 0) + 1);
  }
  const parts = Array.from(tally, ([label, n]) => `${label} ${n}`).sort();
  $("model-weight-note").textContent = `已就绪 ${ready.length} 个 · 合计 ${total.toFixed(0)} MB`
    + (parts.length ? " · " + parts.join(" · ") : "");
  const chosen = rows.filter((w) => modelState.selected.has(w.name));
  $("model-del-weight").disabled = !chosen.length;
  $("model-all").checked = rows.length > 0 && chosen.length === rows.length;
}

async function downloadWeight(name, force) {
  let data;
  try {
    data = await postJson("/api/models/weight/download", { name, force });
  } catch (err) {
    toast("启动下载失败：" + err.message, true);
    return;
  }
  modelState.downloads[name] = data.task;
  toast(force ? `重新下载 ${name}` : `开始下载 ${name}（进度在本页「姿态权重」里）`);
  renderWeights();
  renderCatalog();
  startModelPolling();
}

// --------------------------------------------------------------------------- //
// 模型目录搜索：本地目录模糊匹配（离线也能用），联网搜 GitHub 是可选增强
// --------------------------------------------------------------------------- //

const TASK_LABEL = {
  pose: "姿态估计", detect: "目标检测", segment: "实例分割", obb: "旋转框",
  classify: "图像分类", vlm: "多模态", custom_task: "自定义任务",
};
// 任务徽标只复用 app.css 里已有的 badge module 变体，本页不新造配色
const TASK_BADGE_CLASS = {
  pose: "module pose", segment: "module objects", obb: "module objects",
  vlm: "module objects", detect: "module", classify: "module", custom_task: "module",
};
// 这几个字段只有 /api/models 登记表里有，搜索结果要按文件名把它们并回来
const REGISTRY_KEYS = ["ready", "status", "status_label", "size_mb", "partial_mb", "default",
  "last_used", "registered", "compatible", "compatibility_reasons", "map_pose",
  "keypoint_set", "source", "source_label"];
const MODEL_EMPTY_HINT = "本地目录没有匹配的权重。可以换个说法（分割 / 旋转框 / 骨骼 / 分类 / nano / v8）；"
  + "也可以关掉「只看已下载」看官方完整目录，或打开「联网搜 GitHub」找社区权重——"
  + "从 GitHub 下下来丢进 models/ 再点「登记本地模型」一样能用。";

/** 搜索请求失败时的兜底：拿本机登记表粗筛一遍，列表不至于整块空白。 */
function fallbackRows(query, task) {
  const q = String(query || "").trim().toLowerCase();
  return registryRows().filter((row) => {
    const capability = row.capability || "pose";
    if (task && capability !== task) return false;
    if (!q) return true;
    return [row.id, row.filename, row.display_name, row.description, row.recommended_for,
      row.keypoint_set, TASK_LABEL[capability] || ""]
      .join(" ").toLowerCase().includes(q);
  });
}

/** 搜索接口没回来时，拿上一次成功取到的目录在本机再粗筛一遍，列表不会整块消失。 */
function refilterLocal(rows, query, task) {
  const q = String(query || "").trim().toLowerCase();
  return (rows || []).filter((row) => {
    if (task && String(row.task || "") !== task) return false;
    if (!q) return true;
    return [row.filename, row.display_name, row.task_label, row.task_name, row.intro,
      row.description, row.recommended_for, row.class_note]
      .join(" ").toLowerCase().includes(q);
  });
}

/** 目录行 = 搜索结果的中文介绍/下载地址 ∪ 登记表的本机状态/备注/默认标记。 */
function catalogRows() {
  const known = new Map();
  for (const row of registryRows()) {
    known.set(String(row.filename || "").toLowerCase(), row);
    known.set(String(row.id || "").toLowerCase(), row);
  }
  const source = modelState.hub.loaded
    ? (modelState.hub.local || []) : fallbackRows(modelState.query, modelState.filter);
  const rows = [];
  for (const item of source) {
    const hit = known.get(String(item.filename || "").toLowerCase())
      || known.get(String(item.id || "").toLowerCase()) || null;
    const row = Object.assign({}, item);
    row.known = Boolean(hit);
    for (const key of REGISTRY_KEYS) if (hit && key in hit) row[key] = hit[key];
    row.task = item.task || (hit && (hit.capability || hit.task)) || "pose";
    row.task_label = item.task_label || item.task_name || TASK_LABEL[row.task] || row.task;
    row.registry_capability = hit ? (hit.capability || "pose") : "";
    row.filename = item.filename || (hit && hit.filename) || `${row.id || ""}.pt`;
    row.installed = Boolean(item.installed || row.ready);
    if (!row.known) row.ready = row.installed;
    row.intro = item.intro || item.description || (hit && hit.description) || "";
    rows.push(row);
  }
  return rows;
}

/** 多模态 / 自定义任务是另一条链路：没搜索、没筛任务时原样附在目录末尾。 */
function extraCatalogRows(covered) {
  if (String(modelState.query || "").trim() || modelState.filter) return [];
  return registryRows()
    .filter((row) => ["vlm", "custom_task"].includes(row.capability || "pose"))
    .filter((row) => !covered.has(String(row.id || "").toLowerCase()))
    .map((row) => Object.assign({}, row, {
      known: true,
      registry_capability: row.capability || "pose",
      task: row.capability || "pose",
      task_label: TASK_LABEL[row.capability || "pose"] || row.capability,
      intro: row.description || "",
      installed: Boolean(row.ready),
    }));
}

/** 状态徽标文案：登记表认识的沿用后台判定，纯目录条目只说在本机 / 没下过。 */
function catalogStateText(row, isVlm) {
  if (row.known && (row.status_label || row.status)) return row.status_label || row.status;
  if (isVlm) return row.ready ? "可用" : "未就绪";
  if (row.installed) {
    const short = row.expected_mb && row.size_mb && row.size_mb + 1 < row.expected_mb;
    return short ? "文件不完整" : "已在本机";
  }
  return row.downloadable ? "未下载" : "需自行获取";
}

/** 「取消登记」只对本地登记进来的模型出现：内置目录项没有可取消的登记。 */
function bindForget(row, button) {
  if (!row.registered) { button.classList.add("hidden"); return; }
  button.classList.remove("hidden");
  button.onclick = (e) => { e.preventDefault(); forgetLocalModel(row.id); };
}

/** 目录卡片：任务标签 + 中文名 + 中文介绍 + 能标哪些类别 + 参数/体积/速度 + 下载/设为默认。 */
function catalogCard(row) {
  const task = row.task || row.capability || "pose";
  const isVlm = task === "vlm";
  const detectDefault = String(modelState.detectPrefs.model || "").toLowerCase();
  const el = document.createElement("div");
  el.className = "model-row model-catalog-row" + (row.default ? " selected" : "");
  el.innerHTML = `
    <div class="model-main">
      <div class="model-title-line">
        <span class="badge task-badge ${TASK_BADGE_CLASS[task] || "module"}"></span>
        <span class="model-name"></span>
        <span class="chip ok installed-flag hidden">已在本机</span>
        <span class="chip detect-flag hidden">当前检测默认</span>
        <button class="forget danger hidden">取消登记</button>
        <span class="muted small mono model-file"></span>
        <a class="home hidden" target="_blank" rel="noopener noreferrer">官方文档</a>
      </div>
      <div class="model-desc muted small"></div>
      <div class="muted small model-class-note hidden"></div>
      <div class="muted small model-meta"></div>
      <div class="progress model-progress hidden"><div class="bar"></div></div>
    </div>
    <span class="badge state"></span>
    <button class="dl"></button>
    <button class="def"></button>`;
  const q = (sel) => el.querySelector(sel);
  q(".task-badge").textContent = row.task_label || TASK_LABEL[task] || task;
  q(".model-name").textContent = row.display_name || row.filename || row.id;
  const file = q(".model-file");
  file.textContent = isVlm ? (row.base_url || "") : (row.filename || "");
  if (!file.textContent) file.classList.add("hidden");
  if (row.installed) q(".installed-flag").classList.remove("hidden");
  if (row.homepage) {
    const home = q(".home");
    home.href = row.homepage;
    home.classList.remove("hidden");
  }
  q(".model-desc").textContent = row.intro || "（目录里没写说明）";
  // 最要紧的一行：这个权重到底能标哪些类别，单独摆出来，别埋进参数里
  if (row.class_note) {
    const line = q(".model-class-note");
    line.classList.remove("hidden");
    line.textContent = "能标什么：";
    const strong = document.createElement("b");
    strong.textContent = row.class_note;
    line.appendChild(strong);
  }
  const facts = isVlm
    ? [`${VLM_KIND_LABEL[row.kind] || row.kind} · ${row.base_url || "未填地址"} · ${row.model || "未填模型名"}`,
      `Key ${row.api_key_masked || "未设置"}`, row.last_used ? `最后使用 ${row.last_used}` : "还没用过"]
    : [row.params_m == null ? "参数量官方未公布" : `参数约 ${row.params_m}M`,
      row.score == null ? "精度官方未公布" : `${task === "classify" ? "top-1" : "mAP50-95"} 约 ${row.score}`,
      row.installed ? `本机已占 ${mb(row.size_mb || row.expected_mb || 0)} MB`
        : `下载约 ${mb(row.expected_mb || 0)} MB`,
      row.speed ? `速度 ${row.speed}` : "",
      row.recommended_for ? `适合 ${row.recommended_for}` : "",
      row.size_verified === false ? "体积待核对" : "",
      row.last_used ? `最后使用 ${row.last_used}` : "",
      row.compatible === false ? (row.compatibility_reasons || []).join("；") : ""];
  q(".model-meta").textContent = facts.filter(Boolean).join(" ｜ ");
  q(".state").className = "badge state " + (MODEL_BADGE_CLASS[row.status] || "");
  q(".state").textContent = (row.default ? "默认 · " : "") + catalogStateText(row, isVlm);
  bindForget(row, q(".forget"));
  const download = modelState.downloads[row.filename] || null;
  const running = Boolean(download && download.stage === "running");
  if (running && download.total) {
    q(".model-progress").classList.remove("hidden");
    const pct = Math.round((download.done || 0) * 100 / download.total);
    q(".model-progress .bar").style.width = `${clamp(pct, 0, 100)}%`;
  }
  const dl = q(".dl");
  if (isVlm || !row.downloadable) dl.classList.add("hidden");
  else {
    dl.textContent = running ? "下载中…" : (row.installed ? "重新下载" : "下载");
    dl.disabled = running;
    dl.onclick = (e) => { e.preventDefault(); downloadWeight(row.filename, Boolean(row.installed)); };
  }
  const def = q(".def");
  const detectTask = ["detect", "segment", "obb", "classify"].includes(task);
  if (detectTask) {
    // 检测库的默认权重存在数据集里，和这台机器的姿态默认不是同一个概念，文案分开写
    const active = detectDefault === String(row.filename || "").toLowerCase();
    if (active) q(".detect-flag").classList.remove("hidden");
    if (!row.installed) def.classList.add("hidden");   // 没下进本机之前只能先点「下载」
    else {
      def.textContent = active ? "当前检测默认" : "设为检测默认";
      def.disabled = active;
      def.title = "目标检测页下次打开就选它；不选也用它跑批量标注";
      def.onclick = (e) => { e.preventDefault(); setDetectDefault(row.filename); };
    }
  } else {
    const canDefault = row.known && Boolean(row.registry_capability);
    def.textContent = row.default ? "默认中" : "设为默认";
    def.disabled = !canDefault || Boolean(row.default);
    def.title = canDefault
      ? `设为这台机器${TASK_LABEL[task] || "该任务"}的默认权重（存 configs/models.json）`
      : "这个权重不在登记表里，暂时设不了默认";
    def.onclick = (e) => { e.preventDefault(); setModelDefault(row.id, row.registry_capability || ""); };
  }
  return el;
}

/** 只看已下载：默认只留本机已有的模型 + 多模态配置（那是配置，不是下载物）。
 *
 *  一旦用户开始搜关键词或筛任务，开关就地失效：那是「找一档下载」的动作，
 *  这时候把没下载的挡掉，等于让用户搜了个寂寞。
 */
function visibleCatalogRows(rows) {
  if (!modelState.installedOnly || String(modelState.query || "").trim() || modelState.filter)
    return { shown: rows, hidden: 0 };
  const shown = rows.filter((row) => row.installed || row.ready
    || ["vlm", "custom_task"].includes(row.task));
  return { shown, hidden: rows.length - shown.length };
}

function toggleInstalledOnly() {
  modelState.installedOnly = !modelState.installedOnly;
  renderCatalog();
  syncInstalledOnlySwitch();
}

function syncInstalledOnlySwitch() {
  const toggle = $("model-installed-only");
  if (!toggle) return;
  toggle.classList.toggle("on", Boolean(modelState.installedOnly));
  toggle.setAttribute("aria-checked", modelState.installedOnly ? "true" : "false");
}

/** 「模型目录」区块主体：先渲染本地结果，再叠 GitHub 结果与搜索状态提示。 */
function renderCatalog() {
  const box = $("model-catalog");
  if (!box) return;
  const rows = catalogRows();
  const covered = new Set(rows.map((row) => String(row.id || "").toLowerCase()));
  const all = rows.concat(extraCatalogRows(covered));
  const view = visibleCatalogRows(all);
  modelState.hiddenCount = view.hidden;
  box.innerHTML = "";
  for (const row of view.shown) box.appendChild(catalogCard(row));
  if (!view.shown.length) box.innerHTML = `<div class="batch-empty muted small">${MODEL_EMPTY_HINT}</div>`;
  renderCatalogNote(view.shown);
  renderGitHub();
  renderSearchNote();
}

/** 计数按任务分开报：把「分割 0/5」和「姿态 3/13」揉成一个数会让人以为权重下好了。 */
function renderCatalogNote(rows) {
  const note = $("model-catalog-note");
  if (!note) return;
  const tally = new Map();
  for (const row of rows) {
    const task = row.task || row.capability || "pose";
    const bucket = tally.get(task) || { total: 0, ready: 0 };
    bucket.total += 1;
    if (row.ready || row.installed) bucket.ready += 1;
    tally.set(task, bucket);
  }
  const parts = [];
  for (const [task, bucket] of tally) {
    parts.push(`${TASK_LABEL[task] || task} ${bucket.ready}/${bucket.total}`);
  }
  const head = String(modelState.query || "").trim() ? `匹配 ${rows.length} 个` : `共 ${rows.length} 个`;
  const detectDefault = String(modelState.detectPrefs.model || "");
  const tail = detectDefault ? ` · 检测默认 ${detectDefault}` : "";
  // 被开关藏掉多少条必须说出来，否则「共 4 个」会被当成官方目录就只有 4 档
  const hidden = Number(modelState.hiddenCount || 0);
  const extra = hidden ? ` · 另有 ${hidden} 个未下载没显示（搜关键词或关掉「只看已下载」看全量）` : "";
  note.textContent = head + (parts.length ? " · " + parts.join(" · ") : "") + tail + extra;
}

/** GitHub 栏：社区仓库级结果没有可下载地址，所以只给「打开 GitHub」和「复制仓库名」。 */
function renderGitHub() {
  const box = $("model-github");
  const note = $("model-github-note");
  if (!box || !note) return;
  if (!modelState.remote) {
    box.classList.add("hidden");
    box.innerHTML = "";
    note.textContent = "";
    return;
  }
  const items = modelState.hub.results || [];
  if (!items.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    note.textContent = modelState.searching
      ? "联网检索中（可能要几秒）…"
      : "GitHub 上没有更多匹配，上面的本地结果已经够用了。";
    return;
  }
  note.textContent = `GitHub 检索到 ${items.length} 个社区仓库 · `
    + (modelState.hub.note || "社区权重未经校验，下载前建议核对文件大小");
  box.classList.remove("hidden");
  box.innerHTML = "";
  for (const item of items) box.appendChild(githubCard(item));
}

function githubCard(item) {
  const el = document.createElement("div");
  el.className = "model-row";
  el.innerHTML = `
    <div class="model-main">
      <div class="model-title-line">
        <span class="badge task-badge ${TASK_BADGE_CLASS[item.task] || "module"}"></span>
        <span class="model-name"></span>
        <span class="chip ok hidden">已在本机</span>
      </div>
      <div class="model-desc muted small"></div>
    </div>
    <button class="gh-open">打开 GitHub</button>
    <button class="gh-copy">复制仓库名</button>`;
  const q = (sel) => el.querySelector(sel);
  q(".task-badge").textContent = item.task_label || TASK_LABEL[item.task] || item.origin || "社区";
  q(".model-name").textContent = item.name || item.html_url || "";
  if (item.installed) q(".chip").classList.remove("hidden");
  q(".model-desc").textContent = item.intro || item.description || "（仓库没写说明）";
  q(".gh-open").onclick = (e) => {
    e.preventDefault();
    if (item.html_url) window.open(item.html_url, "_blank", "noopener");
    else toast("这条结果没带仓库地址", true);
  };
  q(".gh-copy").onclick = (e) => { e.preventDefault(); copyText(item.name || ""); };
  return el;
}

/** 复制仓库名：浏览器不给 clipboard 权限时退回临时输入框，别让人点了没反应。 */
function copyText(text) {
  const fallback = () => {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
    document.body.removeChild(area);
    toast(ok ? `已复制 ${text}` : "复制失败，请手动选中仓库名", !ok);
  };
  if (!text) { toast("这条结果没带仓库名", true); return; }
  if (!navigator.clipboard || !navigator.clipboard.writeText) { fallback(); return; }
  navigator.clipboard.writeText(text).then(() => toast(`已复制 ${text}`), fallback);
}

/** 搜索栏下方的状态提示：搜索中 / 联网不可用的原因 / GitHub 命中条数。 */
function renderSearchNote() {
  const clear = $("model-search-clear");
  if (clear) clear.hidden = !String(modelState.query || "").trim();
  const toggle = $("model-remote");
  if (toggle) {
    toggle.classList.toggle("on", Boolean(modelState.remote));
    toggle.setAttribute("aria-checked", modelState.remote ? "true" : "false");
  }
  const note = $("model-search-note");
  if (!note) return;
  note.classList.remove("err");
  if (modelState.searching) {
    note.textContent = modelState.remote ? "联网检索中（可能要几秒）…" : "搜索中…";
    return;
  }
  const hub = modelState.hub;
  // remote_state 才说得清联网走到哪一步：没开联网、没填关键词都不该喊「连不上」
  const state = hub.remote_state || (hub.ok ? "ok" : "failed");
  if (state === "failed") {
    note.classList.add("err");
    note.textContent = "联网检索不可用：" + (hub.note || hub.error || "未知原因")
      + "（下面仍是本地目录结果）";
    return;
  }
  if (state === "needs_query") {
    note.textContent = hub.note || "联网检索要先输入关键词，下面先是本地目录结果";
    return;
  }
  const bits = [];
  if (modelState.remote) bits.push(`GitHub ${(hub.results || []).length} 条`);
  if (String(modelState.query || "").trim()) bits.push(`本地 ${(hub.local || []).length} 条`);
  note.textContent = bits.join(" · ");
}

/** 打一次目录搜索：400ms 内的连续输入并成一次请求，回来的旧结果按序号丢掉。 */
function hubSearch(options) {
  const opts = options || {};
  if (modelState.searchTimer) {
    clearTimeout(modelState.searchTimer);
    modelState.searchTimer = null;
  }
  const run = async () => {
    modelState.searchTimer = null;
    const seq = ++modelState.searchSeq;
    modelState.searching = true;
    const params = new URLSearchParams();
    if (modelState.query.trim()) params.set("q", modelState.query.trim());
    if (modelState.filter) params.set("task", modelState.filter);
    if (modelState.remote) params.set("remote", "1");
    const url = "/api/modelhub/search" + (params.toString() ? "?" + params.toString() : "");
    let data = null;
    let failure = "";
    try {
      data = await getJson(url);
    } catch (err) {
      failure = err.message || String(err);   // 后端挂了也照常显示本地目录，别整块空掉
    }
    if (seq !== modelState.searchSeq) return;
    modelState.searching = false;
    if (data) {
      modelState.hub = {
        ok: Boolean(data.ok), loaded: true, query: data.query || "", task: data.task || "",
        local: data.local || [], results: data.results || [], note: data.note || "",
        error: data.error || "", remote_state: data.remote_state || (data.ok ? "ok" : "failed"),
      };
    } else {
      // 接口挂了不清空列表：沿用上次的目录按新关键词本地粗筛，只把提示换成红色
      const keep = refilterLocal(modelState.hub.local, modelState.query, modelState.filter);
      modelState.hub = { ok: false, loaded: modelState.hub.loaded, query: modelState.query,
        task: modelState.filter, local: keep, results: [], error: failure,
        remote_state: "failed", note: `模型搜索接口没响应（${failure}）` };
    }
    renderCatalog();
  };
  if (opts.immediate) { run(); return; }
  renderSearchNote();
  modelState.searchTimer = setTimeout(run, 400);
}

/** 搜索框相关的小动作：聚焦、清空、开关联动，命令面板和快捷键共用。 */
function focusModelSearch() {
  const input = $("model-search");
  if (!input) return;
  input.focus();
  input.select();
}

function clearModelSearch() {
  modelState.query = "";
  const input = $("model-search");
  if (input) input.value = "";
  hubSearch({ immediate: true });
  if (input) input.focus();
}

function toggleModelRemote() {
  modelState.remote = !modelState.remote;
  toast(modelState.remote ? "已开启联网搜 GitHub：第一次会慢几秒" : "已关闭联网检索：只搜本地目录");
  hubSearch({ immediate: true });
}

function setModelTask(task) {
  const select = $("model-filter");
  if (select) select.value = task || "";
  modelState.filter = select ? select.value : (task || "");
  hubSearch({ immediate: true });
}

async function setModelDefault(id, capability) {
  try {
    await postJson("/api/models/default", { id, capability: capability || "" });
  } catch (err) {
    toast("设为默认失败：" + err.message, true);
    return;
  }
  toast(`默认${capability === "vlm" ? "多模态配置" : "模型"}已设为 ${id}`);
  await refreshModels();
}

async function registerLocalModel() {
  const path = $("model-reg-path").value.trim();
  if (!path) { toast("先填本地权重路径", true); return; }
  let data;
  try {
    data = await postJson("/api/models/local/add", {
      path,
      display_name: $("model-reg-name").value.trim(),
    });
  } catch (err) {
    toast("登记本地模型失败：" + err.message, true);
    return;
  }
  const saved = data.model || {};
  toast(saved.already_in_catalog
    ? `${path} 本来就在目录里（${saved.id}），不用重复登记`
    : `已登记本地模型 ${saved.id}`);
  $("model-reg-path").value = "";
  $("model-reg-name").value = "";
  await refreshModels();
}

async function forgetLocalModel(id) {
  const ok = await confirmBox({
    title: "取消登记本地模型",
    body: `只把 ${id} 从登记表里去掉，权重文件留在原地不动；要真删文件请到下面「本机权重」里勾选删除。`,
    okLabel: "取消登记",
  });
  if (!ok) return;
  try {
    await postJson("/api/models/local/forget", { id });
  } catch (err) {
    toast("取消登记失败：" + err.message, true);
    return;
  }
  toast(`已取消登记 ${id}`);
  await refreshModels();
}

function startModelPolling() {
  if (modelState.polling) return;
  modelState.polling = setInterval(pollDownloads, 1500);
  pollDownloads();
}

function stopModelPolling() {
  if (modelState.polling) { clearInterval(modelState.polling); modelState.polling = null; }
}

async function pollDownloads() {
  const running = Object.keys(modelState.downloads).filter((n) => modelState.downloads[n].stage === "running");
  if (!running.length) { stopModelPolling(); return; }
  let changed = false;
  for (const name of running) {
    let data;
    try {
      data = await postJson("/api/models/download", { name });
    } catch (err) {
      continue;  // 单次失败下一轮再试
    }
    const before = modelState.downloads[name].message;
    modelState.downloads[name] = data.task;
    if (data.task.stage !== "running") {
      toast(data.task.ok ? `${name} 已就绪` : `${name} 下载失败：${data.task.error || data.task.message}`,
            !data.task.ok);
      await refreshModels();
      return;
    }
    if (data.task.message !== before) changed = true;
  }
  if (changed) {
    renderWeights();
    renderCatalog();   // 目录卡片里的进度条跟上面这块是同一个 downloads 状态
  }
}

async function deleteSelectedWeights() {
  const names = Array.from(modelState.selected);
  if (!names.length) { toast("先勾选要删除的权重", true); return; }
  const size = modelState.weights
    .filter((w) => names.includes(w.name))
    .reduce((a, w) => a + (w.size_mb || 0), 0);
  const ok = await confirmBox({
    title: "删除权重文件",
    body: `将删除 ${names.length} 个权重文件（约 ${size.toFixed(1)} MB）：${names.join("、")}。`
      + "已标注的数据不受影响，下次用到该权重时会自动经镜像重新下载。正在标注的权重会被拒绝删除。",
    okLabel: "确定删除",
  });
  if (!ok) return;
  let removed = 0;
  let freed = 0;
  for (const name of names) {
    try {
      const data = await postJson("/api/models/weight/delete", { name });
      removed += 1;
      freed += data.freed_mb || 0;
    } catch (err) {
      toast(`删除 ${name} 失败：${err.message}`, true);
    }
    modelState.selected.delete(name);
  }
  if (removed) toast(`已删除 ${removed} 个权重，释放 ${freed.toFixed(1)} MB`);
  await refreshModels();
}

// --------------------------------------------------------------------------- //
// 多模态大模型配置（本地多模态 / 云端多模态）
// --------------------------------------------------------------------------- //
function renderProviderOptions() {
  const select = $("vlm-provider");
  if (!select) return;
  const previous = select.value;
  select.innerHTML = "";
  for (const preset of modelState.presets) {
    const option = document.createElement("option");
    option.value = preset.id;
    option.textContent = preset.label;
    select.appendChild(option);
  }
  if (previous && modelState.presets.some((p) => p.id === previous)) select.value = previous;
}

function applyVlmPreset() {
  const preset = modelState.presets.find((p) => p.id === $("vlm-provider").value);
  if (!preset) return;
  $("vlm-kind").value = preset.kind;
  if ($("vlm-id").value) return;  // 编辑已有配置时不覆盖用户填的内容
  const short = preset.label.replace(/（[^）]*）/g, "").trim();
  if (!$("vlm-name").value.trim()) $("vlm-name").value = short;
  if (!$("vlm-base").value.trim()) $("vlm-base").value = preset.base_url || "";
  if (!$("vlm-model").value.trim()) $("vlm-model").value = preset.model || "";
}

function applyVlmState(data) {
  if (!data) return;
  if (Array.isArray(data.profiles)) modelState.profiles = data.profiles;
  if ("default" in data) modelState.defaultId = data.default || null;
  if (Array.isArray(data.presets)) modelState.presets = data.presets;
  renderVlmList();
  renderProviderOptions();
}

function renderVlmList() {
  const box = $("vlm-list");
  if (!box) return;
  box.innerHTML = "";
  if (!modelState.profiles.length) {
    box.innerHTML = '<div class="batch-empty muted small">还没有多模态配置：在下面填一条，点「保存并测试」验证连通。</div>';
  }
  for (const row of modelState.profiles) {
    const test = row.last_test || null;
    const isDefault = row.id === modelState.defaultId;
    const el = document.createElement("div");
    el.className = "vlm-row-item" + (isDefault ? " default" : "");
    el.innerHTML = `
      <div class="model-main">
        <div class="model-name"></div>
        <div class="muted small model-meta"></div>
      </div>
      <span class="badge ${test ? (test.ok ? "done" : "failed") : ""}">${
        test ? (test.ok ? "可用 " + test.latency_ms + " ms" : "不可用") : "未测试"}</span>
      <span class="badge">${VLM_KIND_LABEL[row.kind] || row.kind}</span>
      <button class="test">测试</button>
      <button class="edit">编辑</button>
      <button class="def${isDefault ? " active" : ""}">${isDefault ? "默认中" : "设为默认"}</button>
      <button class="danger del">删除</button>`;
    el.querySelector(".model-name").textContent = row.name
      + (row.has_key ? ` · Key ${row.api_key_masked}` : " · 无 Key");
    el.querySelector(".model-meta").textContent = row.base_url + " · " + row.model
      + (row.remark ? " · " + row.remark : "")
      + (test ? ` · ${test.checked_at || "最近测试"} ${test.message}` : "");
    el.querySelector(".test").onclick = (e) => { e.preventDefault(); testVlm(row.id); };
    el.querySelector(".edit").onclick = (e) => { e.preventDefault(); fillVlmForm(row); };
    el.querySelector(".def").onclick = (e) => { e.preventDefault(); setVlmDefault(row.id); };
    el.querySelector(".del").onclick = (e) => { e.preventDefault(); deleteVlm(row.id); };
    box.appendChild(el);
  }
}

function fillVlmForm(row) {
  $("vlm-id").value = row ? row.id : "";
  $("vlm-name").value = row ? row.name : "";
  $("vlm-kind").value = row ? row.kind : "local";
  $("vlm-base").value = row ? row.base_url : "";
  $("vlm-model").value = row ? row.model : "";
  $("vlm-timeout").value = row ? row.timeout || 60 : 60;
  $("vlm-remark").value = row ? row.remark || "" : "";
  $("vlm-max-tokens").value = row ? row.max_tokens || 4096 : 4096;
  $("vlm-image-limit").value = row ? row.image_limit || 1024 : 1024;
  $("vlm-json-mode").checked = Boolean(row && row.json_mode);
  $("vlm-key").value = "";
  $("vlm-key").placeholder = row && row.has_key
    ? `已保存 ${row.api_key_masked}（留空表示不修改）` : "本地留空；云端填 sk-…";
  if (row) $("vlm-provider").value = row.provider || "custom";
  else applyPresetDefault();
}

function applyPresetDefault() {
  if (!modelState.presets.length) return;
  $("vlm-provider").value = modelState.presets[0].id;
  applyVlmPreset();
}

function vlmFormPayload() {
  const payload = {
    name: $("vlm-name").value.trim(),
    provider: $("vlm-provider").value,
    kind: $("vlm-kind").value,
    base_url: $("vlm-base").value.trim(),
    model: $("vlm-model").value.trim(),
    timeout: parseInt($("vlm-timeout").value, 10) || 60,
    max_tokens: parseInt($("vlm-max-tokens").value, 10) || 4096,
    image_limit: parseInt($("vlm-image-limit").value, 10) || 1024,
    json_mode: $("vlm-json-mode").checked,
    remark: $("vlm-remark").value.trim(),
    api_key: $("vlm-key").value,
  };
  if ($("vlm-id").value) payload.id = $("vlm-id").value;
  return payload;
}

function vlmLog(msg, isErr) {
  const box = $("vlm-log");
  box.textContent = msg || "";
  box.classList.toggle("err", Boolean(isErr));
}

async function saveVlm(runTest) {
  const payload = vlmFormPayload();
  let data;
  try {
    data = await postJson("/api/models/vlm/save", { profile: payload });
  } catch (err) {
    vlmLog("保存失败：" + err.message, true);
    toast("保存失败：" + err.message, true);
    return;
  }
  applyVlmState(data);
  $("vlm-id").value = data.profile.id;
  $("vlm-key").value = "";
  $("vlm-key").placeholder = data.profile.has_key
    ? `已保存 ${data.profile.api_key_masked}（留空表示不修改）` : "本地留空；云端填 sk-…";
  vlmLog(`已保存「${data.profile.name}」 · ${data.profile.base_url} · ${data.profile.model}`);
  if (runTest) await testVlm(data.profile.id);
}

async function testVlm(profileId) {
  vlmLog("正在测试连通性…");
  let data;
  try {
    data = await postJson("/api/models/vlm/test", { id: profileId });
  } catch (err) {
    vlmLog("测试失败：" + err.message, true);
    return;
  }
  applyVlmState(data);
  const result = data.result;
  vlmLog(`${result.ok ? "可用" : "不可用"}（${result.latency_ms} ms · ${result.checked_at}）：${result.message}`,
         !result.ok);
}

async function setVlmDefault(profileId) {
  try {
    applyVlmState(await postJson("/api/models/vlm/default", { id: profileId }));
    toast("已设为默认多模态配置");
    await refreshModels();
  } catch (err) {
    toast("设置默认失败：" + err.message, true);
  }
}

async function deleteVlm(profileId) {
  const row = modelState.profiles.find((p) => p.id === profileId);
  const ok = await confirmBox({
    title: "删除多模态配置",
    body: `将删除「${row ? row.name : profileId}」的连接信息（含 API Key）。已标注的数据不受影响。`,
    okLabel: "确定删除",
  });
  if (!ok) return;
  try {
    applyVlmState(await postJson("/api/models/vlm/delete", { id: profileId }));
    if ($("vlm-id").value === profileId) fillVlmForm(null);
    vlmLog("");
    toast("已删除多模态配置");
  } catch (err) {
    toast("删除失败：" + err.message, true);
  }
}

/** 命令面板条目：这一页能做的事，顶栏只留一个主按钮。 */
function modelCommands() {
  return [
    { label: "下载 yolo11s-pose（默认推荐）", tag: "姿态权重", run: () => downloadWeight("yolo11s-pose.pt", false) },
    { label: "下载 yolo11m-pose（更准）", tag: "姿态权重", run: () => downloadWeight("yolo11m-pose.pt", false) },
    { label: "去姿态标注", tag: "姿态标注", run: () => { location.href = "/pose"; } },
    { label: "去提示词标注", tag: "提示词标注", run: () => { location.href = "/objects"; } },
    { label: "回工作台总览", tag: "工作台", run: () => { location.href = "/"; } },
    { label: "在模型目录里搜索", key: "/", tag: "模型目录", run: () => focusModelSearch() },
    { label: "联网搜 GitHub（开 / 关）", tag: "模型目录", run: () => toggleModelRemote() },
    { label: modelState.installedOnly ? "看官方完整目录（含未下载）" : "只看本机已下载的模型",
      tag: "模型目录", run: () => toggleInstalledOnly() },
    { label: "清空模型搜索", tag: "模型目录", run: () => clearModelSearch() },
    { label: "只看姿态估计权重", tag: "模型目录", run: () => setModelTask("pose") },
    { label: "只看实例分割权重", tag: "模型目录", run: () => setModelTask("segment") },
    { label: "只看旋转框权重", tag: "模型目录", run: () => setModelTask("obb") },
    { label: "只看图像分类权重", tag: "模型目录", run: () => setModelTask("classify") },
    { label: "看全部任务的权重", tag: "模型目录", run: () => setModelTask("") },
    { label: "重新读取权重目录", key: "R", run: () => refreshModels() },
    { label: "看本页操作说明", key: "?", run: () => openHelp(true) },
  ];
}

function onModelKey(e) {
  if (shellKeys(e)) { return; }
  if (isTyping(e)) { return; }
  switch (String(e.key).toLowerCase()) {
    case "r": e.preventDefault(); refreshModels(); break;
    case "/": e.preventDefault(); focusModelSearch(); break;
    default:
  }
}

async function bootModels() {
  initNav();
  registerShellCommands(modelCommands());
  initShellChrome({});
  document.addEventListener("keydown", onModelKey);
  $("rail-help").onclick = () => openHelp(true);
  $("model-refresh").onclick = refreshModels;
  $("model-del-weight").onclick = deleteSelectedWeights;
  // 任务筛选直接走后端匹配：中文任务词（分割 / 旋转框…）也由后端统一判定
  $("model-filter").onchange = (e) => setModelTask(e.target.value);
  const search = $("model-search");
  search.oninput = () => { modelState.query = search.value; hubSearch(); };
  search.onkeydown = (e) => {
    if (e.key !== "Escape") return;
    e.preventDefault();
    if (search.value) clearModelSearch();   // 有内容先清空，已经空了才交出焦点
    else search.blur();
  };
  $("model-search-clear").onclick = (e) => { e.preventDefault(); clearModelSearch(); };
  // 联网检索默认关：开着之后每次搜索都要等 GitHub 几秒，本地搜不该被它拖慢
  $("model-remote").onclick = (e) => { e.preventDefault(); toggleModelRemote(); };
  // 「只看已下载」默认开：一屏全是没下过的档位，真正能用的那几行反而要翻半天
  $("model-installed-only").onclick = (e) => { e.preventDefault(); toggleInstalledOnly(); };
  syncInstalledOnlySwitch();
  $("model-register").onclick = registerLocalModel;
  $("model-reg-path").onkeydown = (e) => { if (e.key === "Enter") registerLocalModel(); };
  $("model-download-new").onclick = () => {
    const name = $("model-new-name").value.trim();
    if (!name) { toast("先填权重文件名，如 yolo11l-pose.pt", true); return; }
    $("model-new-name").value = "";
    downloadWeight(name.endsWith(".pt") ? name : name + ".pt", false);
  };
  $("model-all").onchange = (e) => {
    modelState.selected = e.target.checked
      ? new Set(modelState.weights.filter((w) => w.size_mb > 0 || w.partial_mb > 0).map((w) => w.name))
      : new Set();
    renderWeights();
  };
  $("vlm-provider").onchange = applyVlmPreset;
  $("vlm-save").onclick = () => saveVlm(false);
  $("vlm-save-test").onclick = () => saveVlm(true);
  $("vlm-reset").onclick = () => { fillVlmForm(null); vlmLog(""); };
  applyPresetDefault();
  await refreshModels();
}

bootModels();
