"use strict";

// 上层门户：下一步行动流 + 概览数字 + 素材池入口 + 三个标注库入口 + 坐标/训练集导出。
//
// 分工（别再往回改）：**素材归池，标注归库**。导入的图片视频一律先落素材池，
// 分类和「哪一批传进来的」也只在池里记一本账；三个标注库只存引用和标好的结果。
// 所以门户这边只留一句「池子现在什么样」加两个入口，批次清单去素材池看。

/** 模块卡片配置：文案与统计口径都在这里，避免和 common.js 的导航漂移。 */
const MODULE_CARDS = [
  {
    id: "pose", href: "/pose", name: "姿态标注", tag: "标注库",
    desc: "YOLO-pose 人体骨骼关键点的复核与微调：拖点、改可见性、补目标、接受/驳回。",
    lib: "pose",
  },
  {
    id: "objects", href: "/objects", name: "提示词标注", tag: "标注库",
    desc: "自定义字段名 + 中文描述提示词，让多模态大模型在画面里找物品出检测框，复核时删错框、补漏框。",
    lib: "objects",
  },
  {
    id: "detect", href: "/detect", name: "目标检测标注", tag: "标注库",
    desc: "本机 YOLO 权重按它自带的类别表（COCO-80 等）出检测框：不联网、单张几十毫秒，"
      + "类别由权重决定，复核时删错框、补漏框。",
    lib: "detect",
  },
  {
    id: "models", href: "/models", name: "模型管理", tag: "公共配置",
    desc: "按关键词搜五种任务的模型（姿态 / 检测 / 分割 / 旋转框 / 分类）、下载与删除权重、"
      + "本地权重登记，以及本地多模态服务与云端多模态 API 的配置和连通性测试。",
  },
];

/** 概览数字：只留看得懂且会用的这几条，实现细节词一律不上界面。 */
const STAT_STRIP = [
  { key: "files", name: "素材池", unit: "个" },
  { key: "pose_items", name: "姿态库", unit: "帧" },
  { key: "object_items", name: "提示词库", unit: "帧" },
  { key: "detect_items", name: "检测库", unit: "帧" },
  { key: "pending", name: "等你复核", unit: "帧" },
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
    files: "素材都在池里，下面「管理素材池」能按分类和批次管",
    pose_items: "已标 " + values.pose_labeled + " · 带标记 " + values.pose_flagged,
    object_items: "已标 " + values.prompt_labeled + " · 框 " + values.prompt_boxes,
    detect_items: "本机权重出的框 已标 " + values.detect_labeled + " · 框 " + values.detect_boxes,
    pending: "三个库都没点接受 / 驳回的，与页内队列同口径",
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
    acts.push({ icon: "ic-upload", title: "先把图片或视频传进素材池",
      sub: "素材先进池子统一管，哪个模块要用再去那个模块点「从素材池添加」",
      href: "/pool?upload=1", go: "去导入" });
  }
  if (values.pose_items && !poseReady) {
    acts.push({ icon: "ic-cube", title: "姿态权重还没下载", sub: "没有权重就没法自动预标关键点",
      href: "/models", go: "去下载" });
  }
  // 先分清「还没标」和「标了没复核」：从素材池刚添加进来的库一张都没标，
  // 这时候说「N 帧骨骼没复核」是假话——连骨骼都还没有，该做的是先批量标一轮
  const poseUntouched = Math.max(0, values.pose_items - values.pose_labeled);
  if (poseUntouched) {
    acts.push({ icon: "ic-pose", chip: "姿态标注", title: "有 " + poseUntouched + " 帧还没生成骨骼",
      sub: "点「批量姿态标注」按当前权重先出一轮，再人工改错的（快捷键 G）",
      href: "/pose", go: "去标注" });
  }
  if (values.pose_unreviewed && values.pose_labeled) {
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
  const detectUntouched = Math.max(0, values.detect_items - values.detect_labeled);
  if (detectUntouched) {
    acts.push({ icon: "ic-boxes", chip: "目标检测标注", title: "有 " + detectUntouched + " 帧本机权重还没跑过",
      sub: "点「批量目标检测标注」几十秒出一轮框，再进来删错补漏（快捷键 G）",
      href: "/detect", go: "去标注" });
  }
  if (values.detect_unreviewed && values.detect_labeled) {
    acts.push({ icon: "ic-target", chip: "目标检测标注", title: "有 " + values.detect_unreviewed + " 帧检测框没复核",
      sub: "画错的删掉、漏的补上，其中 " + values.detect_flagged + " 帧被权重自己报了问题",
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
      + '<div class="module-desc muted small"></div><div class="module-nums"></div>';
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
      action: { id: "export-detect", label: "导出 YOLO 训练集", kind: "detect", lib: "objects" },
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

/** 命令面板条目：门户能做的事都收在这儿，顶栏因此不用堆按钮。 */
function portalCommands() {
  return [
    { label: "导入图片 / 视频 / 目录（进素材池）", tag: "素材池", run: () => { location.href = "/pool?upload=1"; } },
    { label: "去素材池管素材与分类", tag: "素材池", run: () => { location.href = "/pool"; } },
    { label: "去姿态标注复核", tag: "姿态标注", run: () => { location.href = "/pose"; } },
    { label: "去提示词标注", tag: "提示词标注", run: () => { location.href = "/objects"; } },
    { label: "去目标检测标注", tag: "目标检测标注", run: () => { location.href = "/detect"; } },
    { label: "去模型管理", tag: "模型管理", run: () => { location.href = "/models"; } },
    { label: "跳到「导出训练用文件」", run: () => scrollToSection("sec-export") },
    { label: "跳到「素材池」摘要", run: () => scrollToSection("sec-pool") },
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
  // 菜单里的「素材池」是直接跳转的链接，点的时候把菜单收掉，别留一层浮层
  document.querySelectorAll("#more-menu a[href='/pool']").forEach((link) => {
    link.onclick = () => openMenu(false);
  });
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
  renderPoolBrief(ds);
  renderRecovery(ds);
}

/** 素材池摘要：只报「有多少、占用、几个还没入库」，剩下的事去素材池页做。 */
function renderPoolBrief(ds) {
  const pool = ds.pool || {};
  const files = pool.files || 0;
  const orphan = pool.orphan || 0;
  const chip = $("pool-chip");
  chip.textContent = files ? files + " 个素材" : "池子空的";
  chip.className = "chip" + (files ? " sky" : " warn");
  const bits = [`共 ${files} 个素材 · 占用 ${kb(pool.bytes || 0)}`,
                `姿态 ${pool.pose || 0} / 提示词 ${pool.objects || 0} / 检测 ${pool.detect || 0} 个库在用`];
  if (orphan) { bits.push(`${orphan} 个还没入库，模块里看不到`); }
  if (pool.shared) { bits.push(`${pool.shared} 个被几个库共用`); }
  $("pool-brief").textContent = bits.join(" · ");
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
