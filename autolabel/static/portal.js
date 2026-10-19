"use strict";

// 上层门户：下一步行动流 + 概览数字 + 三个标注库入口 + 坐标/训练集导出 + 批次管理。

/** 模块卡片配置：文案与统计口径都在这里，避免和 common.js 的导航漂移。 */
const MODULE_CARDS = [
  {
    id: "pose", href: "/pose", name: "姿态标注", tag: "标注库",
    desc: "YOLO-pose 人体骨骼关键点的复核与微调：拖点、改可见性、补目标、接受/驳回。",
    upload: true, lib: "pose",
  },
  {
    id: "objects", href: "/objects", name: "提示词标注", tag: "标注库",
    desc: "自定义字段名 + 中文描述提示词，让多模态大模型在画面里找物品出检测框，复核时删错框、补漏框。",
    upload: true, lib: "objects",
  },
  {
    id: "detect", href: "/detect", name: "目标检测标注", tag: "标注库",
    desc: "本机 YOLO 权重按它自带的类别表（COCO-80 等）出检测框：不联网、单张几十毫秒，"
      + "类别由权重决定，复核时删错框、补漏框。",
    upload: true, lib: "detect",
  },
  {
    id: "models", href: "/models", name: "模型管理", tag: "公共配置",
    desc: "按关键词搜五种任务的模型（姿态 / 检测 / 分割 / 旋转框 / 分类）、下载与删除权重、"
      + "本地权重登记，以及本地多模态服务与云端多模态 API 的配置和连通性测试。",
  },
];

/** 概览数字：只留看得懂且会用的这几条，实现细节词一律不上界面。 */
const STAT_STRIP = [
  { key: "files", name: "图片文件", unit: "张" },
  { key: "pose_items", name: "姿态库", unit: "帧" },
  { key: "object_items", name: "提示词库", unit: "帧" },
  { key: "detect_items", name: "检测库", unit: "帧" },
  { key: "pending", name: "等你复核", unit: "帧" },
  { key: "batches", name: "导入批次", unit: "批" },
];


/** 概览数字：口径由后端 `/api/dataset` 的 `review` 一次算清，前端只做兜底。
 *
 * 以前这里自己数一遍，而且两个库数的是两件事——姿态库数「所有没定论的」、
 * 提示词库只数「带标记的」，于是「等你复核」是个拼起来的数；正向标记
 * （两个模型结论一致）还被当成待办。现在后端算，兜底逻辑也照同一口径写。 */
function buildStats(ds) {
  const pose = ds.pose_items || ds.items || [];
  const objects = ds.object_items || [];
  const detect = ds.detect_items || [];
  const pool = ds.pool || {};
  const positive = new Set(ds.positive_flags || ["multi_model_confirmed"]);
  const lowScore = typeof ds.low_score === "number" ? ds.low_score : 0.4;
  const settled = (status) => ["approved", "rejected"].includes(status);
  const values = {
    files: pool.files !== undefined ? pool.files : (ds.pool_files || []).length,
    pool_shared: pool.shared || 0,
    pose_items: pose.length, object_items: objects.length, detect_items: detect.length,
    pose_labeled: 0, pose_unreviewed: 0, pose_flagged: 0, pending: 0,
    prompt_labeled: 0, prompt_boxes: 0, prompt_flagged: 0, prompt_unreviewed: 0,
    detect_labeled: 0, detect_boxes: 0, detect_flagged: 0, detect_unreviewed: 0,
    batches: (ds.batches || []).length,
  };
  for (const item of pose) {
    if (item.objects !== null && item.objects !== undefined) values.pose_labeled += 1;
    if (!settled(item.status)) values.pose_unreviewed += 1;
    const issues = (item.flags || []).filter((f) => !positive.has(f));
    const low = item.score !== null && item.score !== undefined && item.score < lowScore;
    if (!settled(item.status) && (issues.length || low)) values.pose_flagged += 1;
  }
  for (const item of objects) {
    if (item.prompt_objects !== null && item.prompt_objects !== undefined) {
      values.prompt_labeled += 1;
      values.prompt_boxes += Number(item.prompt_objects || 0);
    }
    const issues = (item.prompt_flags || []).filter((f) => !positive.has(f));
    const unlabeled = item.prompt_objects === null || item.prompt_objects === undefined;
    if (!settled(item.prompt_status) && (issues.length || unlabeled)) values.prompt_flagged += 1;
  }
  for (const item of detect) {
    if (item.detect_objects !== null && item.detect_objects !== undefined) {
      values.detect_labeled += 1;
      values.detect_boxes += Number(item.detect_objects || 0);
    }
    const issues = (item.detect_flags || []).filter((f) => !positive.has(f));
    const unlabeled = item.detect_objects === null || item.detect_objects === undefined;
    if (!settled(item.detect_status) && (issues.length || unlabeled)) values.detect_flagged += 1;
  }
  values.pending = values.pose_unreviewed + values.prompt_unreviewed + values.detect_unreviewed;
  const review = ds.review || {};
  for (const key of ["pose_pending", "pose_flagged", "prompt_pending", "prompt_flagged",
                     "detect_pending", "detect_flagged", "pending"]) {
    if (typeof review[key] === "number") values[key === "pose_pending" ? "pose_unreviewed"
      : key === "prompt_pending" ? "prompt_unreviewed"
        : key === "detect_pending" ? "detect_unreviewed" : key] = review[key];
  }
  return values;
}
function renderStats(ds) {
  const values = buildStats(ds);
  const hints = {
    files: "几个库共用同一份，不占双份",
    pose_items: "已标 " + values.pose_labeled + " · 带标记 " + values.pose_flagged,
    object_items: "已标 " + values.prompt_labeled + " · 框 " + values.prompt_boxes,
    detect_items: "本机权重出的框 已标 " + values.detect_labeled + " · 框 " + values.detect_boxes,
    pending: "三个库都没点接受 / 驳回的，与页内队列同口径",
    batches: "每次导入算一批",
  };
  const box = $("portal-stats");
  box.innerHTML = "";
  for (const item of STAT_STRIP) {
    const cell = document.createElement("div");
    cell.className = "stat-cell";
    cell.innerHTML = '<div class="stat-value"></div><div class="stat-name"></div>'
      + '<div class="stat-hint muted small"></div>';
    cell.querySelector(".stat-value").textContent = values[item.key] + " " + item.unit;
    cell.querySelector(".stat-name").textContent = item.name;
    cell.querySelector(".stat-hint").textContent = hints[item.key] || "";
    box.appendChild(cell);
  }
}

/** 下一步：按「先能跑起来、再挑毛病」排序，最多三条，每条带直达按钮。 */
function nextActions(ds, values) {
  const acts = [];
  const poseReady = (ds.models || []).filter((m) => m.ready && m.capability === "pose").length;
  const untouched = Math.max(0, values.object_items - values.prompt_labeled);
  if (!values.files) {
    acts.push({ icon: "ic-upload", title: "先导入图片或视频", sub: "传完模型会自动预标，再回来复核",
      href: "/pose?upload=1", go: "去导入" });
  }
  if (values.pose_items && !poseReady) {
    acts.push({ icon: "ic-cube", title: "姿态权重还没下载", sub: "没有权重就没法自动预标关键点",
      href: "/models", go: "去下载" });
  }
  if (values.pose_unreviewed) {
    acts.push({ icon: "ic-target", chip: "姿态标注", title: "还有 " + values.pose_unreviewed + " 帧骨骼没复核",
      sub: "其中 " + values.pose_flagged + " 帧被标了可疑，进去按 J / K 一条条跳",
      href: "/pose", go: "去复核" });
  }
  if (untouched) {
    acts.push({ icon: "ic-boxes", chip: "提示词标注", title: "有 " + untouched + " 帧还没画过检测框",
      sub: "可以先批量跑一轮提示词，再手工补漏框",
      href: "/objects", go: "去标注" });
  }
  if (values.prompt_flagged) {
    acts.push({ icon: "ic-boxes", chip: "提示词标注", title: "有 " + values.prompt_flagged + " 帧检测框待复核",
      sub: "漏标 / 小框 / 贴边的排在最前面",
      href: "/objects", go: "去复核" });
  }
  if (values.detect_unreviewed) {
    acts.push({ icon: "ic-boxes", chip: "检测标注", title: "有 " + values.detect_unreviewed + " 帧本机权重出的框没复核",
      sub: "本地图已入库的，点批量检测几十秒就出一轮",
      href: "/detect", go: "去复核" });
  }
  if (!acts.length) {
    acts.push({ icon: "ic-export", title: "都处理完了，可以导出训练集",
      sub: "导出目录里有 images / labels / data.yaml，能直接开训",
      jump: "sec-export", go: "去导出" });
  }
  return acts.slice(0, 3);
}

function renderNextActions(ds) {
  const box = $("next-actions");
  if (!box) { return; }
  box.innerHTML = "";
  for (const act of nextActions(ds, buildStats(ds))) {
    const row = document.createElement("div");
    row.className = "flow-row";
    row.innerHTML = '<span class="flow-ic"><svg class="ic"><use/></svg></span>'
      + '<div class="flow-main"><div class="flow-title"></div>'
      + '<div class="flow-sub muted small"></div></div>';
    row.querySelector("use").setAttribute("href", "#" + act.icon);
    const title = row.querySelector(".flow-title");
    title.textContent = act.title;
    if (act.chip) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = act.chip;
      title.append(chip);
    }
    row.querySelector(".flow-sub").textContent = act.sub;
    const go = document.createElement(act.href ? "a" : "button");
    go.className = "btn primary";
    go.textContent = act.go;
    if (act.href) {
      go.href = act.href;
    } else {
      go.onclick = () => scrollToSection(act.jump);
    }
    row.append(go);
    box.appendChild(row);
  }
}

/** 页内跳转：顶栏「⋯」与左导航条都指向正文区块。 */
function scrollToSection(id) {
  const el = $(id);
  if (el) { el.scrollIntoView({ behavior: "smooth", block: "start" }); }
}

function renderCards(ds) {
  const prompt = ds.prompt || {};
  const pose = ds.stats || {};
  const status = pose.status_counts || {};
  const counts = {
    pose: { 图片: (ds.pose_items || ds.items || []).length,
            待复核: (status.unreviewed || 0) + (status.edited || 0) },
    objects: { 图片: (ds.object_items || []).length, 已标: prompt.labeled || 0,
               框: prompt.boxes || 0, 字段: ((ds.fields || {}).fields || []).length },
    detect: { 图片: (ds.detect_items || []).length, 已标: (ds.detect || {}).labeled || 0,
              框: (ds.detect || {}).boxes || 0,
              类别: ((ds.detect_classes || {}).fields || []).length },
    models: { 权重: (ds.models || []).length, 多模态: ((ds.vlm || {}).profiles || []).length },
  };
  const box = $("module-cards");
  box.innerHTML = "";
  for (const card of MODULE_CARDS) {
    const el = document.createElement("a");
    el.className = "module-card";
    el.href = card.href;
    el.innerHTML = '<div class="module-tag muted small"></div><div class="module-name"></div>'
      + '<div class="module-desc muted small"></div><div class="module-nums"></div>'
      + (card.upload ? '<div class="module-extra"><span class="btn primary" data-upload="1">'
        + '导入图片 / 视频</span></div>' : "");
    el.querySelector(".module-tag").textContent = card.tag;
    el.querySelector(".module-name").textContent = card.name;
    el.querySelector(".module-desc").textContent = card.desc;
    const nums = el.querySelector(".module-nums");
    for (const [name, value] of Object.entries(counts[card.id] || {})) {
      const bit = document.createElement("span");
      bit.textContent = name + " ";
      const b = document.createElement("b");
      b.textContent = String(value);
      bit.append(b);
      nums.append(bit);
    }
    const extra = el.querySelector("[data-upload]");
    if (extra) {
      extra.onclick = (e) => { e.preventDefault(); e.stopPropagation(); location.href = card.href + "?upload=1"; };
    }
    box.appendChild(el);
  }
}

function renderExports(ds) {
  const box = $("export-rows");
  box.innerHTML = "";
  const rows = [
    {
      name: "人体骨骼（姿态标注）",
      desc: "坐标：一行一个人的宽表（归一化框 + 17 关键点 x, y, v）。"
        + "训练集：images/{train,val} + labels/{train,val} + data.yaml，可直接 yolo pose train。",
      links: [["CSV", "/api/coords?format=csv"], ["JSONL", "/api/coords?format=jsonl"]],
      action: { id: "export-pose", label: "导出 YOLO 训练集", kind: "pose" },
      count: (ds.stats && ds.stats.persons_total) || 0, unit: "人",
    },
    {
      name: "检测框（提示词标注）",
      desc: "坐标：一行一个框（class_id + 英文类名 + 中文显示名 + 像素/归一化两套）。"
        + "训练集：train/val 切分 + data.yaml（类别名取字段登记表的英文类名）。",
      links: [["CSV", "/api/prompt/coords?format=csv"], ["JSONL", "/api/prompt/coords?format=jsonl"]],
      action: { id: "export-detect", label: "导出 YOLO 训练集", kind: "detect" },
      count: (ds.prompt && ds.prompt.boxes) || 0, unit: "框",
    },
    {
      name: "检测框（目标检测标注）",
      desc: "坐标：一行一个框（class_id + 权重自带英文类名 + 中文显示名 + 两套坐标）。"
        + "训练集：train/val 切分 + data.yaml（类别名取本库类别表），与提示词库那份互不覆盖。",
      links: [["CSV", "/api/prompt/coords?format=csv&lib=detect"],
              ["JSONL", "/api/prompt/coords?format=jsonl&lib=detect"]],
      action: { id: "export-yolo-detect", label: "导出 YOLO 训练集", kind: "yolo_detect", lib: "detect" },
      count: (ds.detect && ds.detect.boxes) || 0, unit: "框",
    },
  ];
  for (const row of rows) {
    const el = document.createElement("div");
    el.className = "export-row";
    el.innerHTML = '<div class="model-main"><div class="model-name"></div>'
      + '<div class="muted small model-meta"></div></div>'
      + '<span class="chip"></span><div class="export-actions"></div>';
    el.querySelector(".model-name").textContent = row.name;
    el.querySelector(".model-meta").textContent = row.desc;
    el.querySelector(".chip").textContent = `${row.count} ${row.unit}`;
    const actions = el.querySelector(".export-actions");
    if (row.links) {
      for (const [label, href] of row.links) {
        const a = document.createElement("a");
        a.className = "btn";
        a.href = href;
        a.download = "";
        a.textContent = label;
        actions.append(a);
      }
    }
    if (row.action) {
      el.querySelector(".model-main").append(buildExportNote(ds, row.action.kind));
      // 以前界面只有一个按钮，导出的永远是"剔除已驳回"这一档，
      // 想要"只导人工确认过的"只能回命令行敲 --only-approved。
      const scope = document.createElement("select");
      scope.className = "export-scope";
      scope.title = "导出范围：哪些帧进训练集";
      for (const [value, label] of EXPORT_SCOPES) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        scope.append(option);
      }
      const b = document.createElement("button");
      b.className = "btn primary";
      b.id = row.action.id;
      b.textContent = row.action.label;
      b.onclick = () => exportDataset(b, row.action.kind, scope, row.action.lib);
      actions.append(scope, b);
    }
    box.append(el);
  }
}

async function exportDataset(button, kind, scopeSelect, lib) {
  if (button.disabled) return;
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "导出中…";
  const scope = scopeSelect ? scopeSelect.value : "reviewed";
  try {
    const url = kind === "pose" ? "/api/export" : "/api/prompt/export";
    // 提示词库与目标检测库共用同一个导出接口，不带 lib 就永远导出提示词库那份
    const body = { val_ratio: clamp(readNumber("export-val-ratio", DEFAULT_VAL_RATIO), 0, 0.9), scope };
    if (lib) { body.lib = lib; }
    const info = (await postJson(url, body)).export || {};
    const opened = await revealExport(info.out, { quiet: true });
    const verb = info.reused ? "内容没变，已复用上次那份" : "YOLO 训练集已导出";
    toast(`${verb}（${scopeLabel(scope)}）：${exportSummary(info, kind)} → ${info.out}`
      + openedSuffix(opened));
  } catch (err) {
    toast("导出失败：" + err.message, true);
  }
  button.disabled = false;
  button.textContent = label;
  await refresh();
}

const BATCH_STATUS = {
  running: "标注中", done: "已完成", cancelled: "已取消", failed: "失败",
  interrupted: "已中断", "-": "历史数据", unknown: "未知",
};
const BATCH_SOURCE = { upload: "网页上传", cli: "命令行", unknown: "历史", "-": "历史" };
const REVIEW_LABEL = { unreviewed: "待复核", approved: "已接受", edited: "已修改", rejected: "已驳回" };
const batchState = { ds: null, module: "all", selected: new Set(), expanded: "",
                     files: {}, fileSel: new Set(), loading: "" };

/** 批次汇总：三个模块各自推进到哪一步，一行说清。 */
function renderBatchSummary(ds) {
  const info = (ds.batches || []).length ? (ds.by_module || {}) : {};
  const parts = [];
  for (const key of ["pose", "objects", "detect"]) {
    const row = info[key];
    if (!row) continue;
    // 老键名 prompt_labeled 只有提示词库有，检测库读的是 labeled / boxes
    parts.push(`${row.label} ${row.batches} 批 / ${row.items} 张`
      + (key === "pose" ? ` / 已标 ${row.labeled} 张`
        : ` / 已标 ${row.prompt_labeled !== undefined ? row.prompt_labeled : row.labeled} 张`
          + ` · ${row.prompt_boxes !== undefined ? row.prompt_boxes : row.boxes} 框`));
  }
  $("batch-summary").textContent = parts.length ? "按模块汇总：" + parts.join(" · ") : "";
}

function renderBatches(ds) {
  batchState.ds = ds;
  const all = ds.batches || [];
  const rows = batchState.module === "all" ? all
    : all.filter((b) => (b.module || "unknown") === batchState.module);
  $("batch-count").textContent = all.length ? `共 ${all.length} 个批次` : "还没有批次记录";
  renderBatchSummary(ds);
  const box = $("portal-batches");
  box.innerHTML = "";
  if (!all.length) {
    box.innerHTML = '<div class="batch-empty muted small">还没有批次：去「姿态标注」「提示词标注」或「目标检测标注」点「导入素材」上传图片和视频。</div>';
  } else if (!rows.length) {
    box.innerHTML = '<div class="batch-empty muted small">这个模块还没有批次记录。</div>';
  }
  for (const batch of rows) box.append(batchRow(batch));
  syncBatchButtons();
}

function batchRow(batch) {
  const wrap = document.createElement("div");
  wrap.className = "batch-block" + (batchState.selected.has(batch.id) ? " selected" : "");

  const row = document.createElement("div");
  row.className = "batch-row";
  const check = document.createElement("input");
  check.type = "checkbox";
  check.checked = batchState.selected.has(batch.id);
  check.onchange = (e) => {
    if (e.target.checked) batchState.selected.add(batch.id);
    else batchState.selected.delete(batch.id);
    renderBatches(batchState.ds);
  };

  const main = document.createElement("div");
  main.className = "batch-main";
  const name = document.createElement("div");
  name.className = "batch-name";
  const badge = document.createElement("span");
  const module = batch.module || "unknown";
  badge.className = "badge module " + module;
  badge.textContent = batch.module_name || module;
  name.append(badge, document.createTextNode(" " + batch.name));
  const meta = document.createElement("div");
  meta.className = "muted small batch-meta";
  const files = (batch.files || []).slice(0, 3).join("、")
    + (batch.file_count > 3 ? ` 等 ${batch.file_count} 个` : "");
  meta.textContent = `${batch.created_at || "时间未知"} · ${BATCH_SOURCE[batch.source] || batch.source || "历史"} · `
    + (files ? `${files} · ` : "") + (batch.message || "");
  main.append(name, meta);

  const nums = document.createElement("div");
  nums.className = "batch-nums";
  const pose = batch.pose || {}, prompt = batch.prompt || {}, detect = batch.detect || {};
  const boxBits = module === "detect" ? detect : prompt;
  const bits = [
    ["素材", batch.items],
    ["骨骼", `${pose.labeled || 0}/${batch.items} 张 · ${pose.people || 0} 人`],
    [module === "detect" ? "检测框（本机权重）" : "检测框",
     `${boxBits.labeled || 0}/${batch.items} 张 · ${boxBits.boxes || 0} 框`],
  ];
  for (const [label, value] of bits) {
    const span = document.createElement("span");
    span.append(document.createTextNode(label + " "));
    const b = document.createElement("b");
    b.textContent = String(value);
    span.append(b);
    nums.append(span);
  }

  const status = document.createElement("span");
  status.className = "badge " + (batch.status || "unknown");
  status.textContent = BATCH_STATUS[batch.status] || batch.status || "未知";

  const peek = document.createElement("button");
  peek.className = "peek";
  const expanded = batchState.expanded === batch.id;
  peek.textContent = expanded ? "收起文件" : "看文件";
  peek.onclick = () => toggleBatchFiles(batch.id);

  const link = document.createElement("a");
  link.className = "btn";
  link.href = module === "objects" ? "/objects" : module === "detect" ? "/detect" : "/pose";
  link.textContent = "去复核";

  row.append(check, main, nums, status, peek, link);
  wrap.append(row);

  const detail = document.createElement("div");
  detail.className = "batch-files";
  if (expanded) detail.append(batchFileList(batch));
  wrap.append(detail);
  return wrap;
}

function batchFileList(batch) {
  const box = document.createElement("div");
  const files = batchState.files[batch.id];
  if (!files) {
    box.textContent = batchState.loading === batch.id ? "文件清单读取中…" : "点「看文件」加载清单";
    return box;
  }
  if (!files.length) {
    box.textContent = "这个批次里已经没有图片了（可能都删掉了），可以直接删除批次记录。";
    return box;
  }
  for (const file of files) {
    const line = document.createElement("div");
    line.className = "file-row";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = batchState.fileSel.has(file.image);
    check.onchange = (e) => {
      if (e.target.checked) batchState.fileSel.add(file.image);
      else batchState.fileSel.delete(file.image);
      syncBatchButtons();
    };
    const name = document.createElement("span");
    name.className = "file-name";
    name.textContent = file.name;
    name.title = file.image;
    const meta = document.createElement("span");
    meta.className = "muted small";
    meta.textContent = `${file.imported_at || ""}${file.video ? " · 来自 " + file.video : ""}`
      + ` · ${file.width || "?"}×${file.height || "?"} · ${kb(file.bytes || 0)}`;
    const pose = document.createElement("span");
    pose.className = "chip" + (file.flags && file.flags.length ? " warn" : "");
    pose.textContent = `骨骼 ${REVIEW_LABEL[file.status] || file.status} · `
      + `${file.objects === null || file.objects === undefined ? "未标" : file.objects + " 人"}`;
    const prompt = document.createElement("span");
    prompt.className = "chip" + (file.prompt_flags && file.prompt_flags.length ? " warn" : "");
    prompt.textContent = `检测框 ${REVIEW_LABEL[file.prompt_status] || file.prompt_status} · `
      + `${file.prompt_objects === null || file.prompt_objects === undefined ? "未标" : file.prompt_objects + " 框"}`;
    const detect = document.createElement("span");
    detect.className = "chip" + (file.detect_flags && file.detect_flags.length ? " warn" : "");
    detect.textContent = `本机框 ${REVIEW_LABEL[file.detect_status] || file.detect_status || "待复核"} · `
      + `${file.detect_objects === null || file.detect_objects === undefined ? "未标" : file.detect_objects + " 框"}`;
    const shared = document.createElement("span");
    shared.className = "chip" + (file.shared ? " warn" : "");
    shared.textContent = file.shared ? "另一个库也在用" : "只在这个库";
    const del = document.createElement("button");
    del.className = "btn danger";
    del.textContent = purgeMode() ? "删素材" : "从本库移除";
    del.title = purgeMode()
      ? "两个库一起移除，并删掉 images/ 里的这份文件"
      : "只把这张从当前库移除，素材文件与另一个库的标注都留着";
    del.onclick = () => deleteFiles([file.image], file.name, file.lib);
    line.append(check, name, meta, pose, prompt, detect, shared, del);
    box.append(line);
  }
  return box;
}

function syncBatchButtons() {
  const chosen = (batchState.ds && batchState.ds.batches || []).filter((b) => batchState.selected.has(b.id));
  const images = chosen.reduce((a, b) => a + (b.items || 0), 0);
  $("batch-delete").disabled = !chosen.length;
  $("batch-delete").textContent = chosen.length
    ? `删除所选批次（${chosen.length} 批 · ${images} 张）` : "删除所选批次";
  const count = batchState.fileSel.size;
  $("file-delete").disabled = !count;
  $("file-delete").textContent = count
    ? `${purgeMode() ? "删除所选素材" : "从本库移除所选"}（${count} 张）`
    : (purgeMode() ? "删除所选素材" : "从本库移除所选图片");
}

async function toggleBatchFiles(batchId) {
  if (batchState.expanded === batchId) {
    batchState.expanded = "";
    renderBatches(batchState.ds);
    return;
  }
  batchState.expanded = batchId;
  batchState.fileSel = new Set();
  if (!batchState.files[batchId]) {
    batchState.loading = batchId;
    renderBatches(batchState.ds);
    try {
      const data = await getJson("/api/batch/files?id=" + encodeURIComponent(batchId));
      batchState.files[batchId] = data.files || [];
    } catch (err) {
      toast("文件清单读取失败：" + err.message, true);
      batchState.expanded = "";
    }
    batchState.loading = "";
  }
  renderBatches(batchState.ds);
}

async function deleteSelectedBatches() {
  const ids = Array.from(batchState.selected);
  if (!ids.length) { toast("先勾选要删除的批次", true); return; }
  const chosen = (batchState.ds && batchState.ds.batches || []).filter((b) => ids.includes(b.id));
  const images = chosen.reduce((a, b) => a + (b.items || 0), 0);
  const ok = await confirmBox({
    title: "删除这些批次的数据",
    body: `将删除 ${chosen.length} 个批次、共 ${images} 张素材，连同本库标签、复核记录、QC 叠加图与批次记录一起清掉。`
      + "素材文件只有在另一个库不再引用时才会删，否则只摘本库引用（素材留在池里给对方用）。"
      + "删除前会在 .autolabel/deleted/ 写快照（含标签原文），界面里不能一键恢复。",
    okLabel: "确定删除",
  });
  if (!ok) return;
  try {
    const data = await postJson("/api/batches/delete", { ids });
    const libs = (data.libs || []).map(libLabel).join("、") || "本库";
    toast(`已从「${libs}」移除 ${data.items} 张素材 · 连带清理 ${data.files_removed} 个标签/素材文件`);
    batchState.selected = new Set();
  } catch (err) {
    toast("删除失败：" + err.message, true);
    return;
  }
  await refresh();
}

/** 勾选框决定删除粒度：不勾=只摘本库引用，勾上=两个库一起摘并删素材文件。 */
function purgeMode() {
  const box = $("purge-files");
  return Boolean(box && box.checked);
}

/** 这批图当前属于哪个库：展开的批次行说了算，取不到时退回清单里第一条的 lib。 */
function currentLib(fallback) {
  const row = (batchState.ds && batchState.ds.batches || []).find((b) => b.id === batchState.expanded);
  return fallback || (row && row.lib) || "pose";
}

async function deleteFiles(images, label, lib) {
  if (!images.length) { toast("先勾选要删的图片", true); return; }
  const target = currentLib(lib);
  const purge = purgeMode();
  const ok = await confirmBox({
    title: purge ? "删除素材（两个库都受影响）" : `从「${libLabel(target)}」移除`,
    body: `共 ${images.length} 张${label && images.length === 1 ? `（${label}）` : ""}。`
      + (purge
        ? "两个库一起移除，并删掉 images/ 里的素材文件与各自的标签、复核记录、QC 叠加图——另一个库也会立刻看不到这些图。"
        : `只从「${libLabel(target)}」移除它们的索引条目与标签，images/ 里的素材文件和另一个库的标注都不动。`)
      + "同批次其余图片保留，删除前会在 .autolabel/deleted/ 写快照。",
    okLabel: purge ? "确定删素材" : "确定移除",
  });
  if (!ok) return;
  let data;
  try {
    data = await postJson("/api/images/delete", { images, lib: target, mode: purge ? "purge" : "lib" });
  } catch (err) {
    toast("删除失败：" + err.message, true);
    return;
  }
  const libs = (data.libs || []).map(libLabel).join("、") || libLabel(target);
  toast(`已从「${libs}」移除 ${data.items} 张素材`
    + (data.files_removed ? ` · 连带清理 ${data.files_removed} 个标签/素材文件`
      : " · 素材文件留在池里，另一个库还能用"));
  for (const key of Object.keys(batchState.files)) delete batchState.files[key];
  batchState.fileSel = new Set();
  await refresh();
}

async function deleteSelectedFiles() {
  const images = Array.from(batchState.fileSel);
  if (!images.length) { toast("先在展开的批次里勾选要删的图片", true); return; }
  await deleteFiles(images, "");
}

/** 命令面板条目：门户能做的事都收在这儿，顶栏因此不用堆按钮。 */
function portalCommands() {
  return [
    { label: "导入图片 / 视频 / 目录", tag: "姿态标注", run: () => { location.href = "/pose?upload=1"; } },
    { label: "去姿态标注复核", tag: "姿态标注", run: () => { location.href = "/pose"; } },
    { label: "去提示词标注", tag: "提示词标注", run: () => { location.href = "/objects"; } },
    { label: "去目标检测标注", tag: "目标检测标注", run: () => { location.href = "/detect"; } },
    { label: "去模型管理", tag: "模型管理", run: () => { location.href = "/models"; } },
    { label: "跳到「导出训练用文件」", run: () => scrollToSection("sec-export") },
    { label: "跳到「批次管理」", run: () => scrollToSection("sec-batches") },
    { label: "重新读取磁盘上的文件", key: "R", run: () => refresh() },
    { label: "看本页操作说明", key: "?", run: () => openHelp(true) },
  ];
}

function onPortalKey(e) {
  if (shellKeys(e)) { return; }
  if (isTyping(e)) { return; }
  switch (String(e.key).toLowerCase()) {
    case "r": e.preventDefault(); refresh(); break;
    default:
  }
}

async function bootPortal() {
  initNav();
  registerShellCommands(portalCommands());
  initShellChrome({});
  document.addEventListener("keydown", onPortalKey);
  $("portal-refresh").onclick = refresh;
  $("jump-export").onclick = () => { openMenu(false); scrollToSection("sec-export"); };
  $("jump-batches").onclick = () => { openMenu(false); scrollToSection("sec-batches"); };
  $("rail-batches").onclick = () => scrollToSection("sec-batches");
  $("rail-help").onclick = () => openHelp(true);

  $("batch-module").onchange = (e) => { batchState.module = e.target.value; renderBatches(batchState.ds); };
  $("batch-delete").onclick = deleteSelectedBatches;
  $("file-delete").onclick = deleteSelectedFiles;
  $("purge-files").onchange = () => { if (batchState.ds) renderBatches(batchState.ds); };
  await refresh();
}

async function refresh() {
  let ds;
  try {
    ds = await getJson("/api/dataset");
  } catch (err) {
    toast("数据集读取失败：" + err.message, true);
    return;
  }
  $("ds-name").textContent = " · " + ds.name;
  $("portal-note").textContent = "数据集目录 " + (ds.root || "");
  renderStats(ds);
  renderNextActions(ds);
  renderCards(ds);
  renderExports(ds);
  renderBatches(ds);
  renderRecovery(ds);
}

/** 上次任务被中断时，后端会按磁盘补回素材，这里把结果说清楚。 */
function renderRecovery(ds) {
  const box = $("recovery-note");
  const info = ds.recovery || {};
  const parts = [];
  if (info.recovered) parts.push(`补回 ${info.recovered} 张已导入但没进索引的图片`);
  if ((info.interrupted || []).length) parts.push(`把 ${info.interrupted.length} 个还在"标注中"的批次判定为中断`);
  if (info.missing_files) parts.push(`${info.missing_files} 条记录的原图已不在磁盘上`);
  if (!parts.length) {
    box.classList.add("hidden");
    box.textContent = "";
    return;
  }
  box.textContent = "启动对账：" + parts.join("，") + "。中断批次里没跑完的图，重新上传一次即可。";
  box.classList.remove("hidden");
}

bootPortal();
