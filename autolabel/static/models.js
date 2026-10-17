"use strict";

// 二级模块「模型管理」：姿态权重的查看/下载/删除、模型目录备注、
// 本地权重登记，以及多模态大模型（本地服务 / 云端 API）配置与连通性验证。

const modelState = {
  weights: [], weightsDir: "", downloads: {}, presets: [], profiles: [], defaultId: null,
  selected: new Set(), polling: null,
  rows: [], catalog: [], defaults: {}, filter: "", secretsPath: "", configPath: "",
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
    compatible: true, compatibility_reasons: [], note: "", description: "",
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
  renderVlmList();
  renderModelNote();
  if (Object.values(modelState.downloads).some((t) => t.stage === "running")) startModelPolling();
  else stopModelPolling();
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

function renderWeights() {
  syncWeightRows();
  const box = $("model-weights");
  box.innerHTML = "";
  if (!modelState.weights.length) {
    box.innerHTML = '<div class="batch-empty muted small">权重目录里什么都没有，点「下载」拉一个开始。</div>';
  }
  for (const info of modelState.weights) {
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
    const row = document.createElement("div");
    row.className = "model-row" + (modelState.selected.has(info.name) ? " selected" : "");
    row.innerHTML = `
      <input type="checkbox" ${modelState.selected.has(info.name) ? "checked" : ""} ${deletable ? "" : "disabled"}>
      <div class="model-main">
        <div class="model-name"></div>
        <div class="muted small model-meta"></div>
      </div>
      <span class="badge ${state.cls}">${state.text}</span>
      <button class="dl">${running ? "下载中…" : info.ready ? "重新下载" : "下载"}</button>`;
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
  const ready = modelState.weights.filter((w) => w.ready);
  const total = modelState.weights.reduce((a, w) => a + (w.size_mb || 0), 0);
  $("model-weight-note").textContent = `已就绪 ${ready.length}/${modelState.weights.length} · 合计 ${total.toFixed(1)} MB`;
  const chosen = modelState.weights.filter((w) => modelState.selected.has(w.name));
  $("model-del-weight").disabled = !chosen.length;
  $("model-all").checked = modelState.weights.length > 0 && chosen.length === modelState.weights.length;
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
  startModelPolling();
}

/** 「模型目录」区块：中文名 + 中文说明 + 参数/体积/关键点 + 备注输入框 + 下载/设为默认。 */
function renderCatalog() {
  const box = $("model-catalog");
  if (!box) return;
  const rows = registryRows().filter((row) => !modelState.filter
    || (row.capability || "pose") === modelState.filter);
  box.innerHTML = "";
  if (!rows.length) {
    box.innerHTML = '<div class="batch-empty muted small">没有符合条件的模型。</div>';
  }
  const tally = {};
  for (const row of rows) {
    const capability = row.capability || "pose";
    const bucket = tally[capability] || (tally[capability] = { total: 0, ready: 0 });
    bucket.total += 1;
    if (row.ready) bucket.ready += 1;
    const el = document.createElement("div");
    el.className = "model-row model-catalog-row" + (row.default ? " selected" : "");
    el.innerHTML = `
      <div class="model-main">
        <div class="model-name"></div>
        <div class="model-desc muted small"></div>
        <div class="muted small model-meta"></div>
        <div class="model-note-row">
          <input type="text" class="note" spellcheck="false"
            placeholder="备注说明：例如「只在内网机器下」「显存吃紧别用」">
          <button class="note-save" disabled>保存备注</button>
          <button class="forget danger hidden">取消登记</button>
        </div>
      </div>
      <span class="badge ${MODEL_BADGE_CLASS[row.status] || ""}"></span>
      <button class="dl"></button>
      <button class="def"></button>`;
    el.querySelector(".model-name").textContent = row.display_name || row.id;
    el.querySelector(".model-desc").textContent = row.description || "（目录里没写说明）";
    const facts = capability === "vlm"
      ? [`${VLM_KIND_LABEL[row.kind] || row.kind} · ${row.base_url || "未填地址"} · ${row.model || "未填模型名"}`,
        `Key ${row.api_key_masked || "未设置"}`,
        row.last_used ? `最后使用 ${row.last_used}` : "还没用过"]
      : [row.params_m ? `参数约 ${row.params_m}M` : "参数官方未列明",
        row.map_pose ? `mAP50-95 约 ${row.map_pose}` : "精度官方未列明",
        row.ready ? `本地 ${mb(row.size_mb || 0)} MB` : `预计 ${mb(row.expected_mb || 0)} MB`,
        `${row.num_keypoints || 17} 关键点（${row.keypoint_set || "未标注"}）`,
        row.speed ? `速度 ${row.speed}` : "",
        row.last_used ? `最后使用 ${row.last_used}` : "还没用过"];
    if (row.recommended_for) facts.push(`适合 ${row.recommended_for}`);
    if (!row.compatible) facts.push((row.compatibility_reasons || []).join("；"));
    el.querySelector(".model-meta").textContent = facts.filter(Boolean).join(" ｜ ");
    el.querySelector(".badge").textContent = (row.default ? "默认 · " : "")
      + (row.status_label || row.status || "");
    const noteInput = el.querySelector(".note");
    noteInput.value = row.note || "";
    const noteSave = el.querySelector(".note-save");
    // 启用条件只能绑在输入框上：绑在按钮自身的 change 永远不触发，备注会一直存不了
    const syncNoteState = () => {
      noteSave.disabled = noteInput.value.trim() === String(row.note || "").trim();
    };
    noteInput.oninput = syncNoteState;
    noteInput.onchange = syncNoteState;
    noteInput.onkeydown = (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      if (!noteSave.disabled) saveModelNote(row.id, noteInput.value.trim());
    };
    noteSave.disabled = true;
    noteSave.onclick = (e) => { e.preventDefault(); saveModelNote(row.id, noteInput.value.trim()); };
    const dl = el.querySelector(".dl");
    if (capability === "vlm" || !row.downloadable) dl.classList.add("hidden");
    else {
      const task = modelState.downloads[row.filename];
      const running = Boolean(task && task.stage === "running");
      dl.textContent = running ? "下载中…" : (row.ready ? "重新下载" : "下载");
      dl.disabled = running;
      dl.onclick = (e) => { e.preventDefault(); downloadWeight(row.filename, Boolean(row.ready)); };
    }
    const def = el.querySelector(".def");
    def.textContent = row.default ? "默认中" : "设为默认";
    def.disabled = Boolean(row.default);
    def.onclick = (e) => { e.preventDefault(); setModelDefault(row.id, capability); };
    const forget = el.querySelector(".forget");
    if (row.registered) {
      forget.classList.remove("hidden");
      forget.onclick = (e) => { e.preventDefault(); forgetLocalModel(row.id); };
    }
    box.appendChild(el);
  }
  // 姿态权重和多模态配置的「可用」不是一回事，混在一起数会让人以为权重下好了
  const CAP_LABEL = { pose: "姿态权重", detect: "检测权重", segment: "分割权重",
                      obb: "旋转框权重", vlm: "多模态配置" };
  const parts = Object.keys(tally).map((cap) => {
    const word = cap === "vlm" ? "可用" : "就绪";
    return `${CAP_LABEL[cap] || cap}${word} ${tally[cap].ready}/${tally[cap].total}`;
  });
  $("model-catalog-note").textContent = `共 ${rows.length} 个` + (parts.length ? " · " : "") + parts.join(" · ");
}

async function saveModelNote(id, note) {
  try {
    await postJson("/api/models/note", { id, note });
  } catch (err) {
    toast("备注保存失败：" + err.message, true);
    return;
  }
  toast(note ? "备注已保存" : "备注已清空");
  await refreshModels();
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
      note: $("model-reg-note").value.trim(),
    });
  } catch (err) {
    toast("登记本地模型失败：" + err.message, true);
    return;
  }
  const saved = data.model || {};
  toast(saved.already_in_catalog
    ? `${path} 本来就在目录里（${saved.id}），备注已写进去`
    : `已登记本地模型 ${saved.id}`);
  $("model-reg-path").value = "";
  $("model-reg-name").value = "";
  $("model-reg-note").value = "";
  await refreshModels();
}

async function forgetLocalModel(id) {
  const ok = await confirmBox({
    title: "取消登记本地模型",
    body: `只把 ${id} 从登记表里去掉，权重文件留在原地不动；要真删文件请到下面「姿态权重」里勾选删除。`,
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
  if (changed) renderWeights();
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
    { label: "重新读取权重目录", key: "R", run: () => refreshModels() },
    { label: "看本页操作说明", key: "?", run: () => openHelp(true) },
  ];
}

function onModelKey(e) {
  if (shellKeys(e)) { return; }
  if (isTyping(e)) { return; }
  switch (String(e.key).toLowerCase()) {
    case "r": e.preventDefault(); refreshModels(); break;
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
  $("model-filter").onchange = (e) => { modelState.filter = e.target.value; renderCatalog(); };
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
