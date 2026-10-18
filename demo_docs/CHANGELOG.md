# 变更记录

## 2026-10-18 18:10 — 合理性检查：两万张的库启动不再像卡死，接口不再白胖一倍

第五轮看的是**资源与健壮性**：句柄与临时文件泄漏、大目录性能、启动耗时、日志噪音、异常兜底、`.autolabel/` 膨胀。做法是先造一个 1500 张的合成数据集，把每个热路径单独计时，再对着数字找根因。**测试 364 → 373**（新增 `tests/test_resource_robustness.py` 9 条；跳过项仍是缺 ffmpeg 的那条）。

- **① 启动对账是 O(N²)（`autolabel/batches.py`）**：`_reconcile_lib()` 判断"这张图是不是对面库已经认领了"时，在**逐张遍历的循环里**调 `load_index_or_none(root, _other(lib))`——每扫到一张图就把对面库整个 `index.json` 读一遍、解析一遍。cProfile 实测 1500 张的数据集启动对账要把索引解析 **1507 次**、耗时 1.03 秒；按这条曲线两万张要几分钟，用户看到的就是"点了启动，终端一直不动"。改成每轮对账把对面库的索引读一次提到循环外，解析次数降到 **4 次**、同一数据集耗时 **0.04 秒**。「两边都没记才算孤儿」「某库在标注时另一个库不抢孤儿素材」两条语义不变，并有专门一条测试盯着。
- **② `/api/dataset` 把同一份清单序列化两遍（`autolabel/review.py`）**：返回值里 `items` 和 `pose_items` 指向**同一个 Python 列表**，`json.dumps` 老老实实各写一遍；提示词库的行还同时给 `prompt_status` 与 `status`、`prompt_flags` 与 `flags` 等同义键。实测 1500 张（只有姿态库）响应体 **725 KB**，其中 **305 KB 纯属第二份拷贝**；两库各 1500 张时约 1.17 MB。界面两处读姿态清单写的都是 `ds.pose_items || ds.items`、提示词页只读 `prompt_*`，于是删掉 `pose_items` 与那五个同义键，前端一行不用改：**725 KB → 420 KB**、两库 1503 张时 **1166 KB → 751 KB**。这是每次进页面 / 每次接受后刷新列表都要走的接口，省的是实打实的等待。
- **③ `.autolabel/deleted/` 只增不减（`autolabel/batches.py`）**：每次删除动作都落一份快照，界面上逐张删两万张就攒两万个文件，而启动对账其实只回头看最近 20 份。改成只保留最近 **200 份**，留 25 份余量再一次清一批（不是每次删除都去扫目录）。刚写的快照一定在最近一窗里，误删找回不受影响；更早的删除要靠自己的备份，已写进常见问题。
- **④ 上传临时文件没人认领（`autolabel/upload.py` + `serve()`）**：上传的原始文件先落 `.autolabel/uploads/`，**任务成功后**才删。上传到一半 Ctrl+C / 崩溃 / 重启电脑，这批原文件就永远留在数据集里，一次视频上传就是几百 MB，界面上还完全看不见它。新增 `prune_stale_uploads()`，复核服务启动时清掉超过 24 小时没被碰过的残留并打印释放了多少 MB；只看 mtime、且留 24 小时余量，另一个还在跑的服务刚收下的文件不会被动。
- **⑤ 访问日志把终端刷满（`autolabel/review.py`）**：`log_message()` 只静音了 `/api/`，`/image/`、`/thumb/` 照打——焦点队列一屏就要拉几十张缩略图，翻一屏队列终端全是 `GET /thumb/...`，真正要看的启动告警和报错被冲走。静音前缀扩到 `("/api/", "/image/", "/thumb/")`，页面访问照常留痕（否则出问题时没线索）。
- **这一轮查过但干净的**：210 个请求（概览 + 100 张缩略图 + 100 张原图）打完，进程句柄数 **6 → 6**，没有泄漏；`video.py` / `coords.py` / `prompt.py` 的临时目录一律 `try/finally` + `rmtree`，`ffprobe` / `ffmpeg` 的 `subprocess.run` 都带 `timeout`，超时报的是人话还顺带告诉用户 `AUTOLABEL_FFMPEG_TIMEOUT` 怎么调；撤销历史每图封顶 10 份、批次表 300 条、批次里的文件名 30 个、`manifest.skipped` 2000 条，都有上限；`_kept_file_tombstones()` 只读最近 20 份快照，不会因为快照多而变慢；HTTP 层四类异常收口齐全（缺文件 404 / 未知对象 404 / 参数不对 400 / 其余 500），上传超限先把请求体读干净再回 413；缩略图是 320px 缓存并带 30 天清理。
- **已知边界**：`/api/dataset` 仍然一次给整本清单，两万张的库省掉重复之后仍有几百 KB，真正的解法是分页 / 虚拟列表（属 UI 重构方案里的 P2），这轮只把白胖的那一半省掉；删一张图、接受一张图都是整本 `index.json` 重写（单文件 JSON 的固有代价，实测约 6 ms/张），要变记得先决定"要不要拆成按张存储"；删除快照只留最近 200 份；上传残留按 mtime 判旧，同时开两个复核服务指向同一个数据集、且某个任务在队列里排了 24 小时以上，它的临时文件会被当成残留清掉（本地单用户工具，先接受）；缩略图清理只在启动时跑，服务连开几周不会中途清。
- **防回归测试**：`tests/test_resource_robustness.py`（新建 9 条）——造一个"图已落盘、索引里没有"的数据集，用猴子补丁数 `load_index_or_none` 的调用次数，超过 8 次就判定退化成按张重读（旧实现是 40 次起）；对账耗时上限；提示词库已认领的 10 张必须不被姿态库抢走、另外 30 张真空闲孤儿必须补回；`/api/dataset` 不许再有 `pose_items` 且 `items` 六个键齐全；提示词行不许再带 `status`/`objects`/`score`/`flags`/`priority`；界面里每个读 `pose_items` 的位置必须留 `|| ds.items` 兜底（防的就是后端删键、前端白屏）；预置 260 份快照后删一张图，目录必须正好剩 200 份且刚写的快照还在、最老的已清、最近一窗没被误删；起真实服务后 `/thumb/`、`/image/`、`/api/` 不得出现在终端而 `/pose` 必须在；`uploads/` 里三天前的文件被清、刚收下的必须留、启动流程确实接了这一步。

校验：`373 passed, 1 skipped`（跳过项缺 ffmpeg，约 45 秒）、`ruff check --select E4,E7,E9,F` 全绿、五个 JS 文件 `node --check` 全绿；起真实服务实测 `/`、`/pose`、`/objects`、`/models`、`/api/dataset`、`/thumb/`、`/image/` 全部 200，`/api/dataset` 的 `items` 正常、终端只剩页面与启动行。性能对照：1500 张数据集启动对账 **1.03s → 0.04s**（索引解析 1507 → 4 次）、概览响应体 **725 KB → 420 KB**。


## 2026-10-18 17:20 — 合理性检查：导出目录不再吃掉同名素材，flip_idx 不再越界，软链能搬家

第四轮看的是**数据格式与导出的正确性**：导出目录里的文件名、`data.yaml` 的关键点配对、软链能不能搬家、COCO JSON 是否自洽。都用小数据集实测复现过。**测试 348 → 364**（新增 `tests/test_export_names.py` 16 条；跳过项仍是缺 ffmpeg 的那条）。

- **① 带子目录的素材在导出目录里互相盖掉（`autolabel/export.py`）**：导出目录是平的两层（`images/train/`、`images/val/`），旧代码取 `os.path.basename` 当输出名，`images/a/frame_0001.jpg` 与 `images/b/frame_0001.jpg` 都写成 `frame_0001.jpg`，而 `_place()` 落盘前会先 `os.remove(dst)`——后放的把前面的连图片带标签一起盖掉。实测 4 张图导完磁盘上只剩 2 个文件，`manifest.json` 的 `counts` 与 COCO 的 `images` 却都记 4 张：训练照跑、loss 照降，只有样本数悄悄少一截，等发现时模型已经练完了。新增 `export_stem(rel)` 把子目录压进输出名（`a/frame_0001.jpg` → `a__frame_0001.jpg`），姿态导出、提示词导出（`prompt.export_detection_dataset`）、坐标文件的 `stem`/`split` 三处共用同一套规则——只改导出不改坐标文件的话，`export --coords` 带过来的 `split` 列会整列对不上。素材平铺（绝大多数真实数据集）算出来和以前一字不差，不给老用户换一批文件名。
- **② `data.yaml` 的 `flip_idx` 越界（同上）**：以前是 `COCO17_FLIP_IDX[:num_kpt]` 直接切片。14 点模型（COCO 去掉双眼的常见裁剪）会拿到 `[..., 14]`——14 点数据集里没有第 14 号点，翻转增强时关键点直接指错位置；点数多于 17 时又短一截。改成 `flip_idx_for(num_kpt)`：配对落在范围外的点自己映自己，多于 17 的部分补齐，结果永远是当前点数的一个合法排列（`sorted(idx) == range(n)` 且左右配对对称），17 点仍旧逐位等于 COCO 标准表。
- **③ 软链一搬家就整批断（同上 `_place()`）**：`--mode symlink`（默认）以前写绝对路径，数据集或导出目录一改名、换挂载点、拷去另一台机器就整批断链，只能重导一次。改成相对路径（`../../../images/a/frame_0001.jpg`），实测把整个数据集目录改名之后仍然读得出图片头两字节。取舍见「已知边界」：相对链接记的是层级差，导出目录**单独**挪到不同深度仍会断。
- **④ COCO 关键点没补齐会整体错位（同上 `write_coco_keypoints()`）**：标签点数少于 `kpt_shape` 声明（人工删过点、老版本写的标签）时，以前只写实际给的那几组 `(x, y, v)`，从缺失点开始后面每一组都往左挪一位，`num_keypoints` 与坐标一起错，而 JSON 看起来完全合法。现在补零到 `num_kpt × 3`，缺的点变成 `v=0`（不参与 YOLO 监督），语义与标签文件一致。
- **这一轮查过但干净的**：COCO JSON 的自洽性实测无问题——每条 annotation 的 `image_id` 都在 `images` 里、`bbox` 是 xywh 且没越出图片边界、`category_id` 从 1 起、`num_keypoints` 与实际 `v≠0` 的组数相等、`keypoints` 长度恒为 3 的倍数；检测集导出的标签每行恒为 5 列、类别号重映射后连续且与 `data.yaml` 的 `names` 一一对应、`counts.boxes` 与磁盘上的行数相等、归一化值都在 0~1；空标签按 YOLO 背景图约定写空文件；`--mode hardlink` 失败自动退回 `copy`；`--mode` 给了未知值直接报错而不是静默产出半个目录。
- **已知边界**：相对软链只保证「导出目录跟着数据集一起搬家」，把某一份导出单独拷到别的层级（如 `~/datasets/x`）仍要重导——真要跨机器搬走，正确姿势一直是 `--mode copy`；自定义关键点的 `flip_idx` 仍按 COCO 的左右配对推（索引里没保存模型自带的 flip_idx），非左右对称的自定义骨架还得手工改 `data.yaml`；`a/x.jpg` 与 `x.jpg` 同时存在时导出名不同（`a__x.jpg` / `x.jpg`）但内容可能相同，那是两条素材记录，不算命名冲突。
- **防回归测试**：`tests/test_export_names.py`（新建 16 条）——两个子目录各放同名素材后，导出目录必须凑齐 4 个文件且图片/标签同名、`manifest.split` 四条名字唯一、每个软链都指向读得通的真实图片；COCO 的 `file_name` 集合必须与磁盘文件集合相等；平铺数据集的导出名必须还是 `img0.jpg`/`img1.jpg`（防的是「修 bug 顺手改了所有人的文件名」）；默认导出的软链必须非绝对路径，且数据集改名后仍能读出 JPEG 头；`flip_idx_for()` 在 17/14/9/5/2/1/24/0 上必须是合法排列且配对对称，17 点逐位等于 `COCO17_FLIP_IDX`；把 `kpt_shape` 改成 14 后 `data.yaml` 解析出的 `flip_idx` 长度必须是 14、最大值 < 14；一张 14 点标签混在 17 点库里时每条 annotation 的 `keypoints` 必须都是 51；`export --coords` 的 `split` 列必须与导出目录里的真实文件名对得上；提示词库带子目录时同样不得撞名。

校验：`364 passed, 1 skipped`（跳过项缺 ffmpeg，约 44 秒）、`ruff check --select E4,E7,E9,F` 全绿；COCO / 检测集格式自洽性单独跑脚本实测无问题；`models/bus.jpg` + `models/yolo11n-pose.pt` 的端到端集成测试仍真实通过（说明导出改动没破坏上一轮的翻转融合结果）。


## 2026-10-18 16:40 — 合理性检查：快捷键不再和浏览器抢键，翻页读不到图会说人话

第三轮看的是**复核界面的交互合理性**：快捷键归谁、翻页时读到失败怎么办、并发操作留没留痕迹。都在本地起服务实测过。**测试 342 → 348**（新增 `tests/test_review_keys.py` 6 条；跳过项仍是缺 ffmpeg 的那条）。

- **① `⌘`/`Ctrl` 组合键被页面快捷键抢走（`static/app.js` / `static/objects.js`）**：两个标注页的 `onKeyDown()` 只看 `e.key`，不看修饰键，`INPUT` 的守卫之后直接进 `switch (e.key)`。实测在标注页按 `⌘A`（全选）会翻上一张、`⌘D`（收藏）翻下一张、`⌘F`（页内查找）跳去下一张带标记、`⌘J`（Chrome 下载页）跳待办下一条、`⌘B`（书签栏）弹出批次管理、`⌘M` 直接 `location.href = "/models"` 跳页、`⌘1/2/3`（切标签页）顺手把关键点可见性改成 1/2/3——浏览器该做的动作 and 页面动作**两边一起执行**，用户找书签栏却看见画面换了人。改法：在类型守卫之后、`switch` 之前加修饰键守卫，`⌘` / `Ctrl` 组合只保留两个（`⌘K` 命令面板由 `shellKeys()` 先接走、`⌘Z` 撤销新接），其余原样 `return` 交回浏览器。`⌥` 那几个外观键不受影响（本来就在 `shellKeys()` 里且带 `isTyping()` 判断）。
- **② `⌘Z` 不是撤销（同上）**：`z` 是「适应窗口」，撤销只有 `U`。按通用直觉按 `⌘Z` 的人会得到视图被重置、撤销没发生——两页都改成 `⌘Z` = 撤销（姿态页 `undo()`、提示词页 `undoItem()`），裸 `Z` 保持适应窗口。速查表两行、`README.md` 与 `04-复核界面指南.md` 的快捷键清单一并补上「`U` 或 `⌘Z`」。
- **③ 读不到标注时白屏不报错（`static/app.js` 的 `selectIndex()`）**：`const res = await fetch("/api/item?...")` 之后直接 `state.item = await res.json()`，**没看状态码**。这张图在另一个标签页被删（或刚按批次删掉）时，接口回的是 404 + `{"error": "未知对象: ..."}`，前端把这个对象当标注数据用，下一行 `state.item.objects.length` 抛 `TypeError`，Promise 无人认领：画布空白、右侧面板空白、只有控制台里一行英文，用户只能刷新碰运气。改成走 `common.js` 里已有的 `getJson()`（非 2xx 抛错），失败时清空当前张、把接口那句中文原样写在画面下方（`读不到这张图的标注：…`）并弹横幅，队列与进度照常刷新，按 `→` 就能继续。`boot()` 里同样裸奔的 `/api/dataset` 也补了兜底（拉不到就写「数据集读不出来：…」，不再白着一页）。
- **④ 连续翻页时旧请求后到会覆盖新页（同上）**：`selectIndex()` 是 async，键盘连按 / 长按方向键时多个请求同时在飞，`state.idx` 已经翻到第 N+3 张，早先发出的那张图却可能最后才回来，把 `state.item` 换回旧的一张——界面显示的是 A 的骨架、高亮和 HUD 写着 D，此时点接受，队列里被打勾的是另一行。加 `loadSeq` 请求序号，`fetch` 前后与图片加载回来后各校验一次，过期的结果直接丢弃。
- **⑤ 接口错误文案里的 Python 味（`autolabel/review.py`）**：`read_item()` 抛 `KeyError(image_rel)`，处理器写的是 `f"未知对象: {exc}"`，`KeyError` 的字符串表示自带引号，界面上一显示就是 `未知对象: 'images/a.jpg'`，看着像程序炸了。取 `exc.args[0]`，`GET` / `POST` 两条路径一起改。
- **这一轮查过但干净的**：`beforeunload` 两个标注页都挂了（`state.dirty` 时拦关页，站内跳页同样会拦）；`setStatus()` 与 `selectIndex()` 都会先 `await save()` 再动，自动保存 600ms 防抖期间翻页不会把改动落到下一张（`save()` 第一件事就是按引用取好 `state.item`）；`shellKeys()` 里 `⌘K`、速查表、`⋯` 菜单、命令面板的优先级顺序正确，弹层开着时页面快捷键不会被误触发；`portal.js` 的 `r`（刷新）与模型页快捷键都有 `isTyping()` 守卫；提示词页 `selectIndex()` 本来就走 `getJson()` 并提示「读取标注失败」，这次是把姿态页拉到同一水位；接口侧 `ValueError → 400`、`FileNotFoundError → 404`、超限 `413` 的收口上一轮已就位。
- **已知边界**：两个标签页同时编辑**同一张图**仍是后写的赢（接口没有版本号，服务端只有写入锁，不会写坏文件但不会提醒「这张被别人改过」）——本地单用户工具按投入产出先不动，真要做得给 `/api/item` 加基线版本。`⌘W`（关标签页）由浏览器直接处理，页面无从拦截，靠的仍是 `beforeunload` 那句确认。快速连翻时若前一张还有未保存改动，`await save()` 会串行等它写完，翻页手感会跟着慢一拍，这是保数据而不是 bug。
- **防回归测试**：`tests/test_review_keys.py`（新建 6 条）——两页的修饰键守卫必须存在、必须在 `switch (e.key)` 之前、且守卫到 `switch` 之间有 `return`；`⌘Z` 分支必须调到各页自己的撤销函数；姿态页读标注必须走 `getJson` 且不许再出现 `state.item = await res.json()`，失败路径要写「读不到这张图的标注」并弹提示；`loadSeq` 的过期检查至少出现两处（`fetch` 之后与图片加载之后各挡一次）；`boot()` 必须有「数据集读不出来」兜底；起真实服务 `GET /api/item?image=ghost.jpg` 断言 404 且 `error == "未知对象: ghost.jpg"`（不带引号）。

校验：`348 passed, 1 skipped`（跳过项缺 ffmpeg，约 44 秒）、`ruff check --select E4,E7,E9,F` 全绿；五个 JS 文件 `node --check` 全部通过。


## 2026-10-18 15:40 — 合理性检查：重标不再吃掉人工成果，QC 的几何判断回到像素空间

第二轮的对象是**标注流水线与 QC**：断点续跑到底续没续上、重标覆盖了谁、QC 标记在说什么。四条都在实测里复现过（前两条共用一个根因，一起修）。**测试 334 → 342**（新增 `tests/test_label_pipeline.py` 8 条；跳过项仍是缺 ffmpeg 的那条）。

- **① 换参数重标会静默冲掉人工骨骼（`autolabel/pipeline.py`）**：`label_dataset()` 里有一句 `only_missing = only_missing and previous.get("signature") == signature`——只要索引头的参数指纹和本次不一致，就把「断点续跑」整个关掉，**全部图片进重标队列**，包括人工点过接受的那些。`write_label_file()` 无条件覆盖标签文件，而 `fields["status"]` 又把 `approved` 原样保留：结果是人手改好的骨架被模型输出替换、状态却还是「已接受」，这张图**不会再出现在复核队列里**，用户完全看不见自己丢了什么，只能去 `.autolabel/history/` 翻快照。代码里 `stats["skipped_reviewed"]` 和「人工已复核的保留」那句注释说明本意就是要保，只是没落到标签文件上；`README.md` 与 `02-命令行参考.md` 也都早就写着「已 approved 的图不会被覆盖」——文档和实现第一次对上。改法：新增 `HUMAN_STATUSES = (approved, edited)` 与 `_human_reviewed()`，`only_missing` 时状态是这两种**且标签文件确实存在**的图直接跳过（`edited` 也算，手改还没点接受同样是人工成果）；`--force` 是用户明确说「连人工的一起重标」，才照常覆盖并退回未复核。新统计 `skipped_human` 一路带到 `label` 的收尾输出（`人工已复核 N 张未覆盖（连它们一起重标加 --force）`），全被跳过时也把这句话说清楚，不再只回一句「全部已完成」。`--force` 的 help 一并改成描述真实规则。
- **② 参数变更后中断，重跑会把整批从头再来（同一处）**：`_already_labeled()` 本来就是**逐张**比 `label_signature` 的，上面那句整批关掉续跑属于多余叠加。跑 5000 张时中途改了 `--imgsz` 又中断（强杀 / OOM / 关终端都会跳过收尾的 `finish()`，索引头仍是旧指纹），重跑会把中断前已经按新参数标好的几千张再标一遍。删掉那句覆盖，跳过与否完全交给逐张判定：一致的跳过、不一致的重标，与索引头是否写回无关。提示语也跟着收敛成「N 张需要重标（人工已复核的不计入）」。
- **③ `odd_aspect` 在归一化空间算长宽比（`autolabel/yolo_pose.py`）**：`ar = person.h / person.w` 用的是归一化宽高，等于把**图片自己的宽高比**乘了进去（`ar_norm = ar_px × 图宽/图高`）。实测同一副骨架：放进 16:9 横屏（1920×1080）像素比例 4.5 的正常人被拉成 8.0 → **误报**长宽比异常；放进竖屏手机视频（1080×1920）像素比例 9 的断链只剩 5.1 → **漏报**。这个软件的素材大量来自手机视频，等于在最常见方向上把这条标记废掉了。改成按像素算 `px_h / px_w`（阈值 0.25~6.0 不变，本来就是按像素定的），横屏竖屏同一个结论。这条标记同时被 `coords.py` 的逐框标记复用，导出的 CSV 一起受益。
- **④ 先扩框、后判可见性，bbox 与标签自相矛盾（`autolabel/backends.py`）**：`PoseBackend.infer()` 原先是 `_predict → fuse_flips → finalize_box → assign_visibility`，而 `_predict` 里的 `v` 还是占位的「全可见」（真实 `v` 由 `assign_visibility` 按 `--visible-conf` / `--occluded-conf` 现算）。`expand_bbox_with_keypoints()` 只圈 `v != 0` 的点，在这个顺序下它看到的是「全部可见」，于是把 conf 低到即将被判 `v=0` 的杂点也圈进包围盒：框被撑大，且和落盘标签里的 `v` 打架（标签说这个点不参与监督，框却为它扩了边）。`box_mode="keypoints"` 同理。改成**先定可见性、再定框**（`_predict` 出来的每一路过 `assign_visibility`，再融合、再定框；`assign_visibility` 是幂等的，融合取 `max(v)` 也仍然一致）。
- **⑤ 多模型融合后只留一路的框（`autolabel/yolo_pose.py`）**：`merge_person_flips()` 关键点按 conf 加权平均，框却整份沿用 `primary` 的——骨架往两个模型的中间挪了，框没挪，于是自己造出 `keypoints_outside_box`，再让人工去修软件自己造成的不匹配。改成框按两路 `score` 加权一起融合。翻转路径上后面还跟着 `finalize_box` 兜底，这次主要是补多模型共识那条路（融合之后没有再定框）。bus.jpg 上真跑一轮：`flip_tta=True` 时三个人的可见关键点全在框内，QC 标记从 `limb_length_mismatch` 变成空。
- **这一轮查过但干净的**：`tiny_box` / `limb_length_mismatch` / `keypoints_outside_box` / `edge_clipped` 本来就是像素口径；`few_keypoints` 读的是落盘标签的真实 `v`，不受 ④ 影响；`review_priority` 对未知标记给 1.0 权重、正向标记 `multi_model_confirmed` 不计入「待复核」，语义自洽；`_visible_ratio` 把 `v=1` 记为可见与「v=1 同样参与监督」的文档口径一致；标注期间复核界面写索引靠 `updates` 合并而不是整体覆盖，不会互相抹（上一轮已修）；`qc --render` 在未命中任何标记时退回按文件名渲染而不是空跑；提示词库那条路径早就有人工保护（跳过 approved 与手动画框，`--overwrite` 才覆盖并退回 `edited`），这次是把骨骼库拉到同一水位。
- **已知边界**：人工保护看的是**状态**，状态由人在界面里点出来；如果有人绕过界面手改标签文件（状态仍是 `unreviewed`），重标照样会覆盖它，这类改动请配合 `.autolabel/history/` 快照。`_human_reviewed` 不区分「谁改的」，将来若要区分人工与 VLM 自动接受需要另加字段。`qc` 的统计与优先清单目前只覆盖骨骼库，提示词库的标记在 `prompt --status` 与提示词复核页各看各的。
- **防回归测试**：`tests/test_label_pipeline.py`（新建 8 条，用假后端，不依赖权重）——人工接受 + 手改的标签文件与状态在 `imgsz` 变更后都保持原样、只有未碰过的另一张被重标；`--force` 仍按字面意思连人工一起重标；只到 `edited` 的同样受保护；索引头指纹被人为改回旧值（模拟中断）时已标好的仍跳过、`total == 0`；全是人工图时输出那句解释；`odd_aspect` 在 16:9 横屏不误报、在竖屏不漏报且同样像素比例结论一致；conf=0.02 的杂点判 `v=0` 后不再把框撑到 0.95 处；两模型融合后框中心落在两路中间、不再触发 `odd_aspect`。

校验：`342 passed, 1 skipped`（跳过项缺 ffmpeg，约 43 秒）、`ruff check --select E4,E7,E9,F` 全绿；`models/bus.jpg` 真实权重端到端复跑（`flip_tta` 开与关）通过，关键点全部落在框内。


## 2026-10-18 14:20 — 合理性检查：错参数当场拦、路径打错不再「导入成功」，报错不再甩 Python 栈

这是「合理性检查」五轮里的第一轮，对象是**命令行这一层**：参数填得合不合理、错了以什么姿态告诉你。五处都实测复现过再动手。**测试 317 → 334**（新增 `tests/test_cli_validation.py` 17 条：取值范围 8 + 合法写法不被误挡 2 + 导入源校验 4 + 错误收口 2 + 坐标格式 1；跳过项仍是缺 ffmpeg 的那条）。

- **① 数值参数没有取值范围（`autolabel/cli.py`）**：`--val-ratio 1.5`、`--conf 5`、`--imgsz 0`、`--jpeg-quality 101`、`--port 70000` 全部**静默接受**——`argparse` 只写了 `type=float`，只管「能不能转成数字」，不管「这个数讲不讲理」。最难发现的是阈值类：`--conf 5` 会一路跑完 8000 张、一个框都不产出（阈值高到永远达不到），用户等到最后才看见空报告，一小时显卡白烧。新增八个语义化校验器 `_positive_int` / `_non_negative_int` / `_positive_float` / `_non_negative_float` / `_fraction`（0~1 不含两端）/ `_unit_interval`（0~1 含端点）/ `_quality`（1~100）/ `_port`（1~65535），按语义挂到约 30 处 `add_argument(type=...)`，越界直接 argparse 中文报错、退出码 2。两类边界特意留着：`--limit 0` / `--render 0` 的 0 是「不限制」的实义、`--start 0` 是最常见写法，所以它们走 `_non_negative_*` 而不是 `_positive_*`；`--max-empty-ratio` 允许 0（不要负样本）所以用 `_unit_interval`。另外 `argparse` 的 `type` 不作用于 `default`，内置默认值不经过校验、不会被自己新加的规则挡掉。
- **② 路径打错却「导入成功」（`cmd_import`）**：`autolabel import <数据集> <打错的路径>` 实测打印「[ok] 扫描 0 · 导入 0」、退出码 0，还在批次表里**留下一个 0 张的空批次**——`batches` 里从此多一行看着像有一批数据、其实什么都没有的东西。新增 `_missing_sources()`（先展开 `~` 与相对路径再判存在）：逐条 `[warn] 路径不存在，跳过: ...`；**全部**不存在直接 `FileNotFoundError` 退出 1，并提示「相对路径请写成绝对路径或先 cd 到上级目录」。同时把**建批次挪到导入之后**：入库为 0 张就只打印「扫描 N 张，没有新图入库（完全重复 X · 近似重复 Y · 失败 Z，已入库的要重标用 `label --force`）」并返回 0，不再产生空批次。部分路径写错仍然继续导能用的那些。
- **③ `auto` 空跑到底（`cmd_auto`）**：`auto` 是「导入 → 标注 → QC → 坐标」一条链，源不存在 / 一张都没进来时它照跑不误，最后交出一堆空文件，退出码还是 0。改成源全缺当场失败；`imported` 为空则把批次 `finish_batch(..., "failed", ...)` 收尾再抛 `ValueError`——批次状态里留下「没有可导入的图片」的说明，而不是一个永远 `running` 的半成品。
- **④ 报错甩 Python 栈（`main()`）**：`status` / `qc` / `coords` / `export` 指到一个不存在的数据集，以及 `batches --delete b_ghost`，实测都是八行 Traceback，真正有用的「数据集没初始化」埋在最后一行英文里。`main()` 兜住 `FileNotFoundError` / `NotADirectoryError` / `ValueError` / `KeyError` → 一行 `[error] 中文说明` 进 stderr、退出码 1；这类错误的特点是**用户自己就能改对**，栈只是噪音。想排查栈本身（怀疑是软件的锅）用 `AUTOLABEL_TRACEBACK=1`，原样再抛一次。`KeyboardInterrupt` 仍回 130 不变。
- **⑤ `coords --formats xml` 假装成功（`autolabel/coords.py`）**：命令行给「导出成功」但一个文件都没生成，网页端 `/api/coords?format=xml` 却是规规矩矩的 400——同一个参数两个口径。加 `KNOWN_FORMATS` 并在 `write_coordinate_files()` 校验，未知格式抛 `ValueError("未知坐标格式: xml（可选 csv、long、jsonl）")`，空选择也顺手挡掉（`--formats ""` 以前同样静默产出 0 个文件）。
- **这一轮查过但干净的**：`--seed` 为负仍然合法（`random.Random` 接受任意整数，切分哈希也一样有效），不强加非负限制；`--device` 传错值由 torch 报错、错误文案本身够直白；`--model` 权重缺失走的是下载逻辑而不是静默降级；子命令的 `--lib` 一律 `choices=[all,pose,objects]`，没有漏网的自由字符串；`pool --limit` / `label --limit` / `prompt --limit` 三个同名参数的默认值（30 / 0 / 0）与各自 help 对得上。
- **已知边界**：校验只看类型和范围，不看语义组合——`--start 300 --duration 5` 落在一个 10 秒的视频上仍然抽不到帧（这属于素材与参数的关系，只有 ffprobe 之后才知道），现在的表现是「导入 0 张 → 批次 failed」而不是静默；`--formats csv,,jsonl` 里的空段会被 `strip()` 后丢掉，不报错。`_missing_sources()` 只在命令行入口判存在，`import_images()` 被网页上传路径直接调用时不经过它（那条路径的源是刚落盘的临时文件，本来不存在打错的问题）。
- **防回归测试**：`tests/test_cli_validation.py`（新建 17 条）——8 种越界写法逐个确认退出码 2 且错误行含参数名；`--val-ratio 0.25` / `--seed -3` / `--render 0` / `--limit 0` / `--start 0` 这些合法写法确认没被误挡（`--start` 那条断言的是「报的是文件不存在、不是参数不合法」）；路径打错退出 1 且**批次数量不变**；重复导入退出 0 且提示「没有新图入库」且不新增批次；部分路径错时新图照常入库（`[ok] 扫描 1 · 导入 1`）；`auto` 无图可导时退出 1、末批状态为 `failed`；未知坐标格式报错且带上可选值。`tests/conftest.py` 把导入源目录拆成 `source_dir` 独立 fixture（`dataset` 依赖它），行为不变，方便别的测试复用同一批合成图。

校验：`334 passed, 1 skipped`（跳过项缺 ffmpeg，43 秒）、`ruff check --select E4,E7,E9,F` 全绿。


## 2026-10-18 12:50 — 合理性检查：验证集不再换人、超限上传说人话、抽帧有超时、索引写入不再互相抹

按「这软件自己讲不讲理」把读写路径过了一遍，修掉四处会真实咬人的问题，另外给队列缩略图开了一条小图通道。**测试 304 → 317**（新增 13 条：切分稳定性 2、上传 3、ffmpeg 超时 4、索引并发 1、缩略图 5；跳过项仍是缺 ffmpeg 的那条）。

- **① 导出切分不稳定（同一帧既训练又算指标）**：`export_dataset()` / `export_detection_dataset()` 原先是 `random.Random(seed).shuffle(整表)` 再取前 N 张当 val——**名次取决于当次一共有多少张**。实测 3 张时 `b.jpg` 在 val，后来补到 12 张重导，它掉进了 train：同一帧既参与训练又参与验证，两次导出的指标根本不可比，`--val-ratio` 也会跟着飘。改成按 `(seed, 相对路径)` 的 SHA-256 分桶（`export.split_val_set()`，百万分位比较），**每张图的归属只跟它自己有关**：数据集怎么长，老验证集的成员都不动。「至少留 1 张验证图」「验证集不许吃光训练集」两条老保证保留（空集时塞最小的那张、全进 val 时踢掉最大的那张）。`seed` 仍然有效——换种子会换一批人，但换完之后依旧稳定可复现。
- **② 超限上传客户端看到的是断链**：`_handle_upload` 在 `Content-Length` 超上限时**先回 413 再断开**，请求体一个字节都没读。HTTP/1.1 复用连接时客户端还在往里塞剩下几 MB，实测收到的是 `URLError: [Err32] Broken pipe`，界面上只剩「网络错误」四个字，用户不知道是体积超限。同时正常路径 `rfile.read(length)` 整份进内存，默认上限 1024 MB 意味着一个上传能顶出 1 GB 瞬时内存、跟 torch 抢地方。新增 `upload.save_upload_stream()`：按 1 MB 分块边读边写，**超限也先把 `Content-Length` 字节读干净**才回 413（半截文件删掉），收不满就报「上传中断：收到 X / Y 字节」。现在浏览器拿到的是「单个文件超过 1024 MB 上限」。
- **③ ffmpeg / ffprobe 没有超时**：`probe_video()` 与 `_run_ffmpeg()` 的 `subprocess.run` 都不带 `timeout`，一个卡死的 ffmpeg（损坏文件、挂载盘失联）会**永久占住后台任务线程**，界面上一路转圈且没有报错。ffprobe 给 60 秒（超时只降级成「元信息未知」，不中断任务）；抽帧按片段长度推导——`min(3600, max(300, 300 + 片段秒数 × 4))`，`AUTOLABEL_FFMPEG_TIMEOUT` 可整体覆盖，超时抛「抽帧超过 N 秒被中止，视频可能损坏或过长」并附环境变量提示；两处都补了 `stdin=DEVNULL`，免得 ffmpeg 等标准输入。
- **④ 索引的读-改-写不在同一把锁里**：`write_item` / `set_status` / `write_prompt_item` / `undo` 等 6 处是「`reload()`（拿锁→放锁）→ 算 → `with lock:` 改+存」。两次拿锁之间别人改过盘，手里的内存副本就成了旧值，而索引是**整表覆盖写**、没有字段级合并，后写的会把对方整批抹掉。实测：一张图在写标签时删掉另外一批图，**删掉的 20 张会连同整批删除一起被旧副本盖回来**（不是只复活那一张）。新增 `ReviewStore._mutate()` 把「对齐磁盘 → 改 → 落盘」收进同一把锁；删除/认领/删字段这三处直接改盘的助手原先完全不参与这把锁，现在也挤进来（`_mutate(resync=True)`，收尾按磁盘对齐内存）；`ReviewStore.lock` 换 `RLock` 以允许同线程重入。锁序固定为 `ReviewStore.lock → batches.lock`，任务线程只在没有持锁时才调 `reload()`，不会反向咬。
- **⑤ 队列缩略图拉原图**：焦点队列一次渲染 121 张，每张都拉全分辨率原图，而且所有响应都是 `Cache-Control: no-store`，翻页就重传。新增 `thumbs.py` + `GET /thumb/<相对路径>`：长边 320 的 JPEG（质量 72），按 `(相对路径, mtime_ns, 字节数)` 做文件名，**内容一换 URL 就变**，所以敢给 `max-age=604800` 一周长缓存；两个标注页的队列改拉 `/thumb/`，主图仍走 `/image/` 全分辨率（同名重导不能看到旧图）。坏图/缺 Pillow 一律回退原图而不是 500；`.autolabel/thumbs/` 定位成可随手删的缓存，服务启动时清掉 30 天没再生成的（`prune_thumbs()`）。

- **一处环境教训值回票价**：缩略图第一版直接 `Image.open()`，全套测试里坏图那条卡了十几秒——ultralytics 会给 PIL 挂懒加载插件，认不出的文件触发它**联网装 `pi-heif`**，图片请求线程就这么挂在网络上了。改成先读 16 字节文件头（JPEG/PNG/BMP/WEBP）再决定要不要交给 PIL。
- **审计里干净的部分**（也记一下，免得下次重查）：索引/批次/登记表全部 tmp + `os.replace` 原子写（仅 `batches._write_snapshot` 是直写，低危）；`resolve_image()` 与 `reveal_export()` 的越界路径都挡得住；撤销快照 12 次修改保留 10 份、连续撤销正常；`dataset_payload()` 400 项 32 ms；队列只渲染当前张 ±60 且已经带 `loading="lazy"`。
- **已知边界**：后台标注任务写索引走的是 `pipeline`/`prompt` 自己的 `load_index → save`，不持 `ReviewStore.lock`（持锁会横跨整个推理过程，界面会被锁死几秒）。任务运行期间恰好删图仍有小概率互相覆盖，`_mutate()` 已把窗口从「读索引到写盘的全部计算时间」压到「写盘那几毫秒」；批次表那侧本来就有 `flock` 跨进程保护。
- **防回归测试**：`tests/test_review_export.py`（+2 切分：数据集 40→100 张验证集成员不搬家、换种子换人但可复现；+1 并发：写标签时批量删除不复活）、`tests/test_upload.py`（+3：4 MB 体积超限拿到 413 而不是断链且不留半截文件、多 MB 上传字节数分毫不差、类型不支持也要把请求体读干净）、`tests/test_video_timeout.py`（新建 4 条，用假 ffmpeg/ffprobe 二进制验证超时真的会中止、错误文案带逃生阀、快路径不受影响）、`tests/test_thumbs.py`（新建 5 条：小图尺寸与长缓存、缓存复用与图片变更后重生成、坏图回退原图、过期清理只动缩略图、两个页面确实改拉 `/thumb/`）。

校验：`317 passed, 1 skipped`（跳过项缺 ffmpeg）、`ruff check --select E4,E7,E9,F` 全绿。

## 2026-10-17 21:15 — 外观：白天 / 夜晚 · 四档字号（四个页面通用）

界面加了一层外观设置：**白天 / 夜晚两套主题** + **四档字号**（小 / 标准 / 大 / 特大）。选择记在本机浏览器（`localStorage` 的 `autolabel.appearance`），切页、刷新、重启服务都保持。默认「跟随系统」，跟着 macOS 的浅色深色自动走。三种切法：顶栏 `⋯` → 底部**外观**分段选择器、`⌘K` 搜「外观」、快捷键 `⌥T`（切昼夜）/ `⌥=` / `⌥-`（字号）。纯前端改动，接口与数据格式一字未动。

- **主题走 CSS 变量**（`app.css`）：`data-theme` / `data-fs` 两个属性驱动，`:root[data-theme="light"]` 覆盖一整套色值。白天不是把背景涂白——底色改暖纸白（`--base #E7E4DC` / `--surface #F7F6F2`）、主文改墨色、强调色整体压深到浅色底也能读（琥珀 `#F5A524` → `#94600A`、玉 `#46D39A` → `#0E8559`、珊瑚 → `#C8442A`），发丝线与阴影基准跟着 `--ink` / `--shadow` 一起反色。收口之后 `:root` 之外零色值字面量。
- **画布跟着主题走**（`common.js` 新增 `canvasPalette()`）：按 `data-theme` + `data-fs` 缓存、从 CSS 变量取色，两个标注页画布上的框、骨骼、关键点、标签底色与字号全部改走 token，并监听 `appearancechange` 立刻重画，不用刷新。
- **修掉一处白天专用的隐形 bug**：压在照片上的深色底板（HUD、框标签底色）按设计**两种主题都保持深**，但里面的字原先复用主题的 `--ink-2` / `--amber`——白天这些色是为读浅色面板而压深的，搬到深色底板上就成深字压深底，实拍时 HUD 的「4 处待确认」和画布上的 `#1 k13/17` 几乎读不出。改为一组恒定偏亮的「亮墨」token（`--veil-ink` / `--veil-ink-2` / `--veil-ink-3` / `--veil-hairline` / `--veil-jade` / `--veil-amber` / `--veil-coral` + 画布用的 `--c-photo-amber` / `--c-photo-sky` / `--c-photo-coral`），两种主题下 HUD 文字恒为 `rgb(174,181,185)`、「N 处待确认」恒为 `rgb(255,138,120)`、「无需修改」恒为 `rgb(70,211,154)`（`--jade` 在白天是 `#0E8559`，压在深底上同样读不出，`#hud .chip.ok` 一起改走 `--veil-jade`）。
- **字号是真缩放**：81 条 `font-size` 全部改成 `calc(Npx * var(--fs))`，`body` 的 `font` 简写同样接缩放，档位 0.92 / 1 / 1.12 / 1.26；按钮 `height` 改 `min-height` 以免撑破，画布标签高度按缩放算。特大档实拍 1440 宽仍无横向溢出。
- **不闪一下**：四个 HTML 在 `</head>` 前各内联一段脚本，首屏先按本地偏好写 `data-theme` / `data-fs` 再渲染；切换给 200ms 柔和过渡，`prefers-reduced-motion` 时跳过。
- **防回归测试**（`tests/test_review_shell.py` 新增 5 条）：四页必须有首屏外观脚本与 `#appearance-ctl` 控件位、速查表必须宣传 `⌥T` / `⌥=` / `⌥-`；每条 `font-size` 必须接 `--fs`；`:root` 之外不许有硬编码色值；画布必须用 `canvasPalette()` / `cfont()` 且监听 `appearancechange`；压在深色底板上的选择器必须走亮墨 token。

校验：`302 passed, 1 skipped`（跳过项缺 ffmpeg）、`node --check` 三个 JS 全绿。无头 Chrome 实拍白天 / 夜晚 × 小 / 标准 / 特大 × 四个页面：主题切换即时生效、画布重绘、`⌘K` 出三条外观动作、刷新后记住偏好，全程无脚本错误、无横向溢出。

## 2026-10-17 20:10 — 真实数据集首屏：纯姿态数据集不再打不开网页，质量标记改说中文

把新界面拿到真实数据集（`datasets/样本视频`）上跑，撞出两件事：一是网页根本起不来，二是界面还在对用户念英文变量名。

- **纯姿态数据集打不开复核网页（后端根因）**：这类数据集从没跑过提示词标注，`split_legacy_index()` 没有 `prompt_*` 条目可搬，`migrate` 直接判定「已经是分库布局」，`.autolabel/objects/index.json` 永远不会被创建；而 `ReviewStore.__init__` 要求两本索引都在 → `FileNotFoundError`，`autolabel review` 直接崩。新增 `library.ensure_objects_index()`：迁移时缺索引就补一本空的（数据集名沿用姿态库），`migrate()` 结果新增 `created` 字段，`autolabel migrate` 据此打印「新建 .autolabel/objects/index.json」并把这次算成有变更。**现在老数据集不分类型，`review` 起来就能用**，不必先跑一遍提示词标注。
- **质量标记改说中文**：`common.js` 新增 `FLAG_LABEL` / `flagLabel()`，14 个骨骼标记 + 9 个检测框标记全给了中文短名（`limb_length_mismatch` → 左右肢体不一致、`small_box` → 框太小）。姿态页的标记块与缩略图悬停、提示词页的标记块与框上的小标签统一取中文名，原始 key 退到 `title` 里，方便对着 `qc/report.md` 和导出表查。
- **防回归测试**：`tests/test_library_split.py` 新增一条——纯姿态数据集（无 `prompt_*`）迁移后必须存在 `.autolabel/objects/index.json`、`ReviewStore` 必须能构造、再执行两次必须回「已经是分库布局」；`tests/test_review_shell.py` 新增一条——后端 `qc.FLAG_HINTS` 与 `prompt.PROMPT_FLAG_HINTS` 里每个 key 都必须在 `FLAG_LABEL` 里有中文标签，且两页都必须调用 `flagLabel()`（新增标记忘了给中文名会立刻红）。

验收期间顺手排除一处误会：门户概览全是 0 不是统计坏了，是界面上刚把唯一那个批次删掉了（勾了「连图片文件一起删」）。用仓库自带样例补了个可整批删除的「示例素材（可删）」批次（2 张 / 6 人）后实拍：门户行动流正确指向「还有 2 帧骨骼没复核 → 去复核」，姿态页待办清单 4 条与画布编号牌一一对应、标记块显示「左右肢体不一致」，四页均无脚本错误。另外确认权重缺失时命令行标注会自动下权重并继续（`models/yolo11n-pose.pt`，6.0 MB）。

校验：`297 passed, 1 skipped`（跳过项缺 ffmpeg；权重与 ultralytics 就位时原先静默跳过的 4 项集成测试会真的跑起来）、`node --check` 三个 JS 全绿。

## 2026-10-17 18:20 — UI 重构 P1 追加：壳层铺满四个页面，门户改成「下一步」行动流

接着 P0 往下推，把「两个标注页换了新壳、门户和模型管理还穿着旧皮」补平，并按方案主张②重写门户。仍然**零后端改动**，`/api/*` 与数据格式一字未动。

- **四个页面同一套壳**：门户 `/` 与模型管理 `/models` 换成左侧 52px 竖排导航条 + 顶栏（页名 · 模块切换 · `⌘K` 搜索 · 一个主按钮 `导入素材` · `⋯` 溢出菜单），正文包在 `.shell-row` 里与导航条并排。旧顶栏的「刷新」「去标注素材」两个按钮收进 `⋯`：工作台是「重新读取磁盘（`R`）+ 数据集目录 + 跳到导出/批次」，模型管理是「重新读取权重（`R`）+ 当前默认 + 跳到其他页」。两页各加一份 `⌘K` 命令表（`registerShellCommands`）与 `?` 速查表，并接上 `initShellChrome` / `shellKeys`。
- **门户不再是术语卡墙**：11 张统计卡 → **「下一步」行动流 + 五格概览**。行动流按「先能跑起来、再挑毛病」排序，最多三条（没图→去导入 / 缺权重→去下载 / 有帧没复核→去复核 / 没画框→去标注 / 都清完→去导出），每条右侧按钮直达，`nextActions()` 现算不缓存。概览只留图片文件、姿态库、提示词库、等你复核、导入批次五格，单位跟数字同处。
- **界面主词去掉实现细节**：`素材池`「两库共用」「二级模块」不再出现在门户界面（磁盘上的历史批次名不动），库卡片标签改成「标注库 / 公共配置」，逐张清单的提示改成「另一个库也在用 / 只在这个库」，批量删除的勾选文案改「连图片文件一起删」。
- **同一动作只留一个入口**：库卡片删掉 `从素材池添加`（认领已传素材只在上传弹层做），门户不再放第二个入口。
- **防回归测试扩到四页**（`tests/test_review_shell.py` 新增 3 条）：文档型页面必须带 `#rail` / `#palette-modal` / `#help-modal` / `#more-menu` 并接线三个壳层函数；速查表里宣传的单键必须在同页 JS 有 `case`；门户界面文案禁止再出现那三个术语词，且行动流必须限量、必须被调用。

校验：`291 passed, 5 skipped`（跳过项需 torch / ultralytics）、`node --check` 五个 JS 全绿、无头 Chrome 实拍门户 / 模型管理换壳前后 + `⌘K` / `⋯` / 速查表三个弹层 + 姿态页与提示词页回归，四页均无脚本错误；门户正文左偏移 52px（导航条不压正文）、1440 宽无横向溢出。


## 2026-10-17 16:40 — UI 重构 P0 落地：一套石墨壳 + 「编号 ↔ 待办清单」复核动线

按 `10-UI交互重构方案.md` 的 P0 范围改前端，后端 `/api/*` 一字未动。风格采纳**石墨 Graphite** 骨架，并把**纸本 Index** 的「编号 ↔ 待办清单」机制搬来做复核主线。

- **`⌘K` 命令面板四页通用**：动作、筛选、按文件名搜帧都能搜（`common.js` 提供 `registerShellCommands` / `renderPalette` / `runPalette`，姿态页与提示词页各注册自己那套）。顶栏因此瘦身成「数据集胶囊 + 一个主按钮 + `⋯`」，筛选 / 排序 / 显示项 / 批次或字段管理 / 三个导出全收进 `⋯` 溢出菜单，1440 宽不再换行。
- **待办清单是新的主线**：`attentionList()` 把这一帧值得动手的关键点（贴边、被标遮挡、置信度 <0.4）按「先修错的、再看可疑的」排序，最多 9 条；面板里的编号 1…9 与画布上的编号牌一一对应，`J` / `K` 逐条跳并自动把点居中，`V` 聚焦模式淡去非选中目标，画布左下角 HUD 给可见性色块 + `N 处待确认` / `无需修改`。
- **接受 / 驳回后弹 5 秒撤销条**（`showUndoBar` 带倒计时），取代「提示一闪就没」；`↑` `↓` 在竖排焦点队列里移动，队列行显示 `待复核 · 3 项标记`。
- **状态词收进 `common.js` 一份**（`STATUS_LABEL` / `statusLabel`），两库统一显示 待复核 / 已修改 / 已接受 / 已驳回；筛选默认「待复核」= 未复核 **+ 已修改**，另补「已修改」独立出口，提示词页多一项「未标注」。
- **设计系统落地**：石墨底 `#0E1113` + 骨白正文 + 琥珀 `#F5A524` 单一强调色（玉绿=接受、珊瑚红=待确认、蓝=选中），4px 间距阶、圆角只用 2/4/7/胶囊、动效只有 120/200/500ms 三段并支持 `prefers-reduced-motion`；检测框只画四角括号；所有数字走等宽与 `tabular-nums`。没引入网络字体，本地离线照用。顺带修掉一个隐性 bug：`--border` 这个变量此前从未定义。
- **目标列表改为只展开选中目标**：七个人 × 十七个点全摊开（119 行）根本没法看，未展开的行只留「均置信 0.82 · 2 处待办」一行摘要。
- **两处「常驻但没用」的条改成有内容才出现**：顶栏 3px 进度条（一张都没处理时不挂灰线）、顶栏下「上次导出」条（没导出过时那句「还没有导出过」白占一行）。
- 响应式补齐 1360 / 1140 / 900 三段：窄屏先收左右列宽，再收左导航条，最后面板换行到画布下方（900 宽仍能完成一次完整复核）。

实机截图逐屏验收顺手修掉四处回归：

- 提示词页顶栏缺 `#btn-upload`，`initUploadPanel()` 往 `null` 上挂 `onclick` 抛错，**整页脚本从那一行起全部没执行**（队列、画布、快捷键、数据加载全哑）。
- `statusLabel` 只在姿态页定义，提示词页 `renderFilmstrip()` 第 206 行引用它抛 `ReferenceError`，队列渲染中断——根因是共用状态词当初只搬了一半，现已统一由 `common.js` 提供。
- 右侧动作区用 `position: sticky` 钉底，把上方 1432px 高的目标列表整个盖住，列表实际滚不到；改成面板分「可滚动区 + 固定动作区」两段。
- 撤销条与画布 HUD 在窄画布上重叠，遮住 `第 N / M`；撤销条抬高到 HUD 上方。

新增 `tests/test_review_shell.py` 三条契约：状态词只允许定义一份（重复 `const` 会让整页脚本不执行）、两页壳层锚点齐全且动作区不得再用 sticky、下拉里的每个筛选值都必须被 JS 认识。`04-复核界面指南.md` 补「界面骨架」一节并重写快捷键与状态筛选，`09-提示词标注.md` 同步壳层与键位（顺手修掉文档里 `Z` 撤销 / 「补画字段」两处与代码不符的旧说法），根 `README.md` 与 `demo_docs/README.md` 索引同步。

校验：`288 passed, 5 skipped`（跳过项需 torch / ultralytics）、`node --check` 四个 JS 全绿、无头 Chrome 1440×900 逐屏截图复核（姿态 / 提示词 / 门户 / 模型管理 + 命令面板 / 速查表 / `⋯` 菜单 / 撤销条 / 聚焦模式）。

## 2026-10-17 09:48 — 走查复测：导出复用实测生效、补回缺失权重、消掉图标 404

接上一条的 9 项修复做回归实测（浏览器逐项点 + 全量自动化），确认结果并补两个小口子：

- **导出复用实测通过**：连点两次「导出 YOLO 训练集」，第二次提示变成「标签没有变化，已复用上次导出：8 训练 / 2 验证 · 56 人 → `exports/2026-10-17_091833`（已在访达中打开）」，`exports/` 下没有多出目录；导出条常驻显示上次路径，带「在访达中显示 / 复制路径 / 复制训练命令」三个按钮。
- **模型页数字口径复测通过**：标题实际渲染为 `共 15 个 · 姿态权重就绪 3/13 · 多模态配置可用 2/2`，与旁边「已就绪 3/13」不再打架。
- **补回 3 个权重**：`yolo11n-pose.pt`、`yolo11m-pose.pt`、`yolo26m-pose.pt` 之前不在 `models/` 里，导致 5 项端到端测试静默跳过、模型页默认档显示未下载。已走镜像重新下好（6.0 / 40.5 / 46.8 MB），端到端测试恢复执行。
- **图标不再报 404**：新增 `autolabel/static/favicon.ico` 与服务端 `/favicon.ico` 路由。页面本身已内联 SVG 图标，这里只兜住浏览器自动发起的那次请求，避免控制台留红。新增测试 `test_favicon_route_serves_icon_instead_of_404()`。
- **素材池空池表现确认正常**：11 张图全部已在本库时，弹层说明写「其中可添加的 0 张」，「全选」「添加到本库」按预期置灰，不是卡死。

校验：`290 passed`（0 跳过）、`ruff check` 全绿、前端 5 个 JS 文件 `node --check` 全绿。

## 2026-10-17 09:21 — 全项目走查：提示不打架、数字不说谎、重复导出不堆目录

按用户旅程把「导入 → 标注 → 复核 → 批次 → 导出 → 训练」和三个二级模块 + 工作台完整走了一遍（干净数据集 11 张 / 56 人实测，浏览器逐项点），修掉 9 个交互与口径问题：

- **成功提示不再被顶掉**：`refreshDataset()` 结尾原本无条件 `toast("列表已更新")`，把刚弹出的导出路径、删除张数、任务结果全部覆盖掉（实测导出成功后界面只剩一句「列表已更新」）。改成默认静默，各操作自己的提示留着；素材池添加完成补了一条 toast（原来只写在弹层日志里，关掉就看不见）。
- **撤销提示不再甩 Python 原文**：`undo()` / `undo_prompt()` 没历史时抛 `KeyError`，被 HTTP 层包成 `未知对象: '没有可回滚的历史版本'`。改成 `ValueError("这张图还没有可撤销的修改：它没有被保存过")` → 400 + 原句上屏。
- **重复导出不再造重复目录**：新增 `export.content_fingerprint()` 与 `export.find_reusable_export()`，把「收哪些图 + 标签原文 + 导出参数」压成 SHA-1，与 `exports/` 最近一份比对，命中就复用并返回 `reused: true`，提示语变成「标签没有变化，已复用上次导出」。指纹刻意读文件内容而不是大小/mtime——手工把 `0.500` 改成 `0.501` 大小不变，只看大小会把用户刚改的结果当旧导出。姿态与提示词两条链路同口径。
- **同秒导出不再互相污染**：`export.unique_export_dir()` 给撞名目录加 `-2`、`-3`。原先同一秒内两次不同参数的导出写进同一个目录，第一次的软链会残留在第二份里当数据。
- **模型页数字口径拆开**：目录标题原本写「共 15 个 · 可用 8 个」，其中 8 = 6 个姿态权重 + 2 个多模态配置，和旁边「已就绪 6/13」看着像矛盾。改成 `共 15 个 · 姿态权重就绪 6/13 · 多模态配置可用 2/2`。
- **快捷键说明与实现对齐**：`04` / `09` 两份文档都把 `Z` 写成撤销（实际是适应窗口）、`[ ]` 写成翻页（实际切换选中目标）、`G` 写成跳待复核（实际打开批量标注弹层）、还列了提示词页根本没绑定的 `O`；页内 `kbd-help` 承诺的 `0 删框` 压根没绑定。现已全部改正并补齐 `←/→`、`A/D`、`P`、`M`，同时给 `objects.js` 真的绑上 `0` 删框、`P` 回姿态页、`M` 去模型页。新增测试 `test_shortcut_help_only_advertises_bound_keys()` 自动比对「帮助里写的单字母键」与 JS 里的 `case`，防止再飘。
- **姿态导入也能命名批次**：提示词页早有「批次名」，姿态页没有，同一个动作两套体验。后端 `upload.py` 本来就读 `params["batch_name"]`，这次给 `review.html` 补输入框并在 `app.js` 透传。
- **术语统一为「导入素材」**：`/pose` 顶栏原本叫「上传素材」，同页快捷键说明却写「W 导入素材」，工作台又叫「去导入素材」。顶栏、弹层标题、CLI 提示、文档全部统一；顺手给四个页面加了内联 SVG 图标，消掉每次加载都有的 favicon 404。
- **`objects.js` 的 `refreshDataset(keepImage)` 参数不再被忽略**：调用方一直在传「继续停在哪张图」，函数签名却没接，导致删除框后选中项会跳。
- **测试**：`tests/test_review_export.py` 新增 5 项（撤销文案、重复导出复用、改一个关键点必须重导、换 `scope` 不算重复、检测集同样复用、快捷键说明与绑定自动比对），全量 **289 passed**；`ruff check` 与逐文件 `node --check` 全绿。

改动文件：`autolabel/export.py`、`autolabel/prompt.py`、`autolabel/review.py`、`autolabel/cli.py`、`autolabel/static/app.js`·`objects.js`·`common.js`·`models.js`·`review.html`·`objects.html`·`portal.html`·`models.html`、`tests/test_review_export.py`、`README.md`、`demo_docs/01·02·03·04·05·06·07·08·09·README`。

**走查中确认没问题的部分**：删除类操作（删批次 / 删单张 / 删字段 / 删权重 / 删多模态配置 / 清除任务）全部有确认弹窗且文案写清影响范围；`resolve_image()` 与新的 `reveal` 接口都做了路径越界校验；弹层打开时不会误触发复核快捷键；图片懒加载缓存有上限（12 张）；`--lib` 与素材池共用文件的「只摘本库引用」逻辑在删除时判定正确；模型页空路径登记、多模态连通性测试的报错都直接可读。

## 2026-10-17 08:55 — 导出目录一键定位：在访达中显示

上一轮把路径常驻到界面后，"没找到导出的文件在哪里"还剩最后一步：看得见路径，但还得自己复制、自己开访达。这轮直接补上"打开目录"这一跳。

- **新接口** `POST /api/export/reveal`（`ReviewStore.reveal_export()`，autolabel/review.py:623）：按系统调 `open -R` / `explorer` / `xdg-open` 在文件管理器里定位导出目录。只允许本数据集 `exports/` 下的路径（realpath 前缀校验），路径缺失、越界、目录已被清理都返回 400；无桌面环境时不抛异常，返回 `ok=false` 并附可手动执行的命令。
- **导出行加按钮**：`common.js` 常驻行现在是「完整路径 + **在访达中显示** + **复制路径** + **复制训练命令**」，按钮文案按系统变化（访达 / 资源管理器 / 打开所在目录）。
- **导出成功自动打开一次**：`app.js`、`objects.js`、`portal.js` 三个入口导出成功后都调 `revealExport(out, {quiet:true})`，把「（已在访达中打开）」拼进原本那条提示，不另弹 toast 盖掉张数。
- **测试**：`tests/test_review_export.py` 新增 2 项（reveal 的缺路径/越界/目录不存在/正常四种分支；三个入口的自动打开与按钮在位检查），全量 **284 passed**。

改动文件：`autolabel/review.py`、`autolabel/static/common.js`·`app.js`·`objects.js`·`portal.js`、`tests/test_review_export.py`、`README.md`、`demo_docs/03·04·06·07·README`。

## 2026-10-17 08:45 — 导出产物路径常驻可见（附一键复制训练命令）

导出按钮上线后收到的第一条反馈是「没找到导出的文件在哪里」：产物是目录，路径只在提示条里出现几秒，用户找不到就反复点导出——一个测试数据集两分钟内堆出 12 份内容相同的 `exports/<时间戳>/`。这轮把路径变成界面里长期可见的东西：

新增：

- **最近导出清单** `export.list_exports()`（autolabel/export.py:166）：扫 `exports/` 目录，每份给出 `{name, kind, path, data_yaml, mtime, at, train, val, total}`，`kind` 按 `prompt_meta.json`（detect）/ `manifest.json`（pose）判定，图片数用 `scandir` 现数（symlink 也数，默认导出就是软链），按 `(mtime, name)` 倒序取最近 8 份。
- **`/api/dataset` 出 `exports_recent`**：走 `ReviewStore.exports_recent()`，函数内导入 `export` 模块，保持导入期不拉重依赖。
- **常驻提示行** `common.js`：模块页顶栏下面、门户两条导出行里各有一行「上次导出（时间）：完整路径 · N 训练 / M 验证」，右侧 **复制路径** 与 **复制训练命令** 两个按钮，后者给整条 `yolo pose train data=<...>/data.yaml model=yolo11s-pose.pt epochs=100 imgsz=640`（检测那套换成 `yolo detect train` + `yolo11s.pt`）。没导出过时显示「还没有导出过：点上面的导出按钮，产物目录会长期显示在这里」。
- **复制助手** `copyText()`：优先 `navigator.clipboard`，非安全上下文退回隐藏 textarea + `execCommand("copy")`；导出成功后页面自动重取数据集，提示行立刻跟上新那份。

测试：`python3 -m pytest tests -q` **282 passed**（上一轮 280）。新增两条：导出后 `/api/dataset` 的 `exports_recent` 里能查到且最新一份路径与接口返回一致、训练/验证数相加对得上、`data.yaml` 真实存在；检测框导出会被正确标成 `kind=detect`。

改动文件：`autolabel/export.py`（`list_exports` / `count_split_images`）、`autolabel/review.py`（`exports_recent` 并入数据集快照）、`autolabel/static/common.js`（`buildExportNote` / `exportCommand` / `copyText` / `renderExportNote`）、`autolabel/static/portal.js`·`app.js`·`objects.js`（渲染与导出后刷新）、`autolabel/static/review.html`·`objects.html`（`#export-note` 容器）、`autolabel/static/app.css`（`.export-note`、`.btn.tiny`、导出按钮换行）、`tests/test_review_export.py`、`README.md`、`demo_docs/03·04·06·07`、`demo_docs/README.md`。

说明：`/tmp` 下的数据集访达默认看不见，长期用建议放项目里的 `datasets/`；反复点导出会堆出多份同内容目录，现在能看到路径了可以按需清理。


## 2026-10-17 08:28 — 界面一键导出 YOLO 标准训练集（姿态库）

之前 `autolabel export` 生成的 YOLO pose 训练集只能从命令行拿，界面里只有坐标 CSV / JSONL 能点，提示词标注却有 `导出数据集` 按钮，两条链路不对称。这轮补齐姿态侧：

新增：

- **新接口** `POST /api/export`：调用 `export.export_dataset()` 现场生成 `images/{train,val}` + `labels/{train,val}` + `data.yaml` + `manifest.json`，参数 `{out?, val_ratio?, seed?, tag?, mode?, scope?, coco?, coords?}`。`scope` 三档——`reviewed`（默认，剔掉 `rejected`）、`approved`（只要人工接受过的）、`all`（全都要），非法值报 400。返回体除 `export_dataset` 原有字段外补 `scope` 与 `data_yaml` 路径。
- **空库不空跑** `ReviewStore.pose_labelled_count()`：导出前先按 `scope` 数一遍真正带非空标签的图，为 0 就直接返回「姿态库还没有可导出的骨骼标签：先跑一次自动标注，或在复核界面把几张图点成「接受」」，不再落一个空目录让人以为导出成功。
- **三个入口，产物同一个**：姿态标注页右上角 `YOLO 训练集`（与 `坐标 CSV` / `坐标 JSONL` 并排）、工作台「人体骨骼」行的 `导出 YOLO 训练集`；提示词标注页右上角同样补了 `YOLO 训练集`，与批量弹层里的 `导出数据集` 走同一个 `POST /api/prompt/export`。按钮点下时置灰并显示「导出中…」，完成后提示里给出目录路径。
- **测试** `python3 -m pytest tests -q` **280 passed**（上一轮 274）：`tests/test_review_export.py` 新增 6 项——标准目录结构与 `data.yaml`/`manifest.json` 落地、`coco=true` 附 COCO 关键点文件（51 个分量）、`scope=approved` 只出人接受过的、没标签与全是未复核两种情况都给人话报错、非法 `scope` 报 400、两个模块页与门户的按钮与接口引用在位。

改动文件：`autolabel/review.py`（`export_pose_dataset` / `pose_labelled_count` / `/api/export` 路由）、`autolabel/static/review.html`·`objects.html`（顶栏按钮）、`autolabel/static/app.js`（`initExports` / `exportYoloDataset`）、`autolabel/static/objects.js`（`exportDataset` 改收按钮参数，两个入口共用）、`autolabel/static/portal.js`（导出行改成两库对称，`exportDetect` → `exportDataset(button, kind)`）、`tests/test_review_export.py`、`README.md`、`demo_docs/01·03·04·06·07·09`、`demo_docs/README.md`。

说明：门户原来的第三行「YOLO 数据集目录」只有检测框一个按钮，现已并入两条链路各自的行，两行都是 `CSV` / `JSONL` / `导出 YOLO 训练集` 三件套。


## 2026-10-17 04:55 — 分库后第二轮走查：全链路对账 + 三处引导文案

按「两个库各自独立、素材池共用」的口径，把 上传 → 认领 → 复核 → 两种粒度删除 整条链路在浏览器里重新跑了一遍，并逐个批次核对「行内小计 == 展开清单汇总」。数字全部对得上（素材池 / 姿态库 / 提示词库三本账：13/10/12 删一张素材后 12/10/11；只摘本库引用时素材池与姿态库纹丝不动）。本轮只发现三处引导文案还停留在分家之前，已修正：

- **提示词库空态** `autolabel/static/objects.js:134`：原来让人「去「姿态标注」导入素材」，现在两个模块各管各的上传，改成指向本页右上角的「导入素材」与上传弹层里的「从素材池添加」。
- **没有字段就上传** `autolabel/static/objects.js:1182`：提示补上去哪建字段——「先点「字段管理」新建物品字段并在上面勾选，或取消「入库后自动标注」改为只入库」，不再只说「先勾选至少一个字段」。
- **按张删除的回执** `autolabel/static/portal.js:462`：原来的「已移除 N 张（M 处库引用）」太像日志，改为「已从「提示词标注」移除 1 张素材 · 素材文件留在池里，另一个库还能用」，删素材那一档则写「连带清理 N 个标签/素材文件」。

验证：`python3 -m pytest tests -q` 274 passed；`ruff check autolabel tests` 全绿；`node --check` 四个静态脚本通过；三处新文案均在真实浏览器里复现确认。


## 2026-10-17 04:35 — 姿态库与提示词库彻底分家，素材池两库共用

上一轮把上传链路拆成两条，但底层仍是一套索引：在提示词库里删一张图会把姿态库的骨骼标注一起带走，两个模块的统计也互相掺水。这轮把**存储层真正分成两个库**，素材池保持共用，并顺带做了两轮全项目交互与口径走查。

新增：

- **两个标注库** `autolabel/library.py`：库名归一化（`pose` / `objects`）、两套索引 / 批次表 / 原件 / 撤销快照的路径由它统一给出。姿态库沿用老位置（`.autolabel/index.json`、`batches.json`、`originals/`、`history/`）**零迁移**；提示词库整体住进 `.autolabel/objects/`，标签仍在 `prompt_labels/`。批次 ID 全局唯一，`resolve_lib()` 按 ID 自动定库，接口不必额外传库名。
- **共享素材池**：`images/` 不再属于任何一库，同一帧只落盘一次，两库各持引用。跨库导入命中池内指纹就直接复用（命令行输出与批次备注写「复用素材池 N」）；新接口 `GET /api/pool?lib=`（谁引用了哪张图、池内总数/共用数/体积）与 `POST /api/pool/adopt`（认领进某库，建 `kind=pool` 批次，不复制文件）。
- **两种删除粒度**：`delete_images(mode=...)` —— `lib`（默认）只摘本库引用与本库标签，素材和对面标注都不动；`purge` 两库一起摘并删 `images/` 里的素材（`keep_files=True` 保留文件）。素材只在**对面库不再引用**时才真删（`_maybe_remove_pool()`）。返回体补 `removals`（库×图）、`libs`、`missing`、`batches_emptied`。界面对应「连素材文件一起删（两个库都移除）」勾选框，按钮文案随之在「从本库移除 / 删素材」之间切换。
- **命令行选库**：`import` / `video` / `auto` 加 `--lib pose|objects`（`auto --lib objects` 入库后直接收尾并打印下一步的 `prompt` 命令，不跑 YOLO-pose）；新增 `pool`（看池、`--add` 认领）与 `migrate`（老数据集拆库，幂等）两个子命令；`batches` 加 `--lib all|pose|objects` 与 `--mode lib|purge`，表格新增「库 / 检测框」列，`--json` 输出结构对齐接口（`{"batches": [...], "libs": ...}`）；`status` 分两库报进度并给出「素材池 N 个文件 · 两库共用 M · 无人引用 K」。
- **前端两库分账**：工作台统计卡拆成素材池 / 姿态库 / 提示词库三本账；模块卡各带 **去导入素材** 与 **从素材池添加**；`common.js` 新增可复用的素材池选择器（四页通用，运行时注入弹层）；姿态页顶栏显示「姿态库 N 张（素材池另有 M 张可添加）」并新增 `W` 打开导入弹层，提示词页同口径；两页都支持 `?upload=1` / `?pool=1` 深链。「未分组」按库拆成 `__ungrouped__:pose` / `__ungrouped__:objects`。
- **弹层键盘兜底** `bindModalKeys()`：确认框 `Enter` 确定 / `Esc` 取消、素材池弹层 `Esc` 关闭，用捕获阶段抢在页面快捷键之前，四个模块行为一致（此前主页的弹层只能用鼠标点）。

修复：

- **对账抢素材**：启动对账原本不分库，姿态库会把提示词库刚落盘、正在标注的孤儿图片认领到自己的批次上。改为逐库对账并加 `_should_claim()`——某库仍有在跑任务时，另一个库不许带走它的孤儿素材。
- **批次行与展开清单自相矛盾**：分库后批次行只读本库字段，出现「文件清单写已标 2 框，批次行写 0 框」。新增 `_cross_view()` 跨库视图，骨骼列取姿态库、检测框列取提示词库，保证行内小计恒等于展开清单的逐张汇总（库级统计仍各算各的，不掺水）。
- **迁移后的幽灵批次行**：`split_legacy_index()` 拆分老索引时没按批次归属归位 `batch_id`，姿态库仍指向已划给提示词库的批次，主页多出「记录缺失」虚行。现在拆分时同时归位，并把 `stripped_batch_ids` 写进迁移结果。
- **批次备注少报入库张数**：CLI 的批次备注写 `导入 0 张`（`imported` 不含复用），实际入库 9 张。新增 `_import_note()`，`import` / `video` 都写「导入 N 张（复用素材池 M）」，视频行还补上「复用素材池 M 帧」与目标库名。
- 门户混选两个库的文件时删除会漏库：整批/按张删除按各行的 `lib` 取数，跨库时确认文案与结果提示分别写明；`pool --lib` 的帮助文案不再沿用 `import` 的措辞（认领不落盘，不是复用落盘）。

测试：

- `python3 -m pytest tests -q` **274 passed**（上一轮 256 passed / 5 skipped）。新增 `tests/test_library_split.py` 13 项：跨库导入复用素材池不复制文件、`adopt_from_pool` 幂等且不泄漏复核状态、`lib` 删 / `purge` 删 / `keep_files` 三种语义、`batch_file_rows` 的 `shared` 标记、跨库批次进度对账、`migrate` 拆索引 + 拆批次 + 搬目录 + 幂等、CLI `migrate` 与列表一致、对账优先权、`status` 分库文案、`auto --lib objects` 只入库不跑骨骼、空库 `status` 直接写明往哪个库灌素材；`tests/test_batches.py`（24 项）把批次行断言改成跨库一致并加一条「行内小计 == 展开清单汇总」；`tests/_helpers.write_label()` 支持 `lib` 参数。
- 服务实测（Playwright 四页走查 + 两库真实数据）：默认「从本库移除」只让提示词库 7→6、素材池 8 不变；勾「删素材」后素材池 8→7、两库同步掉数；主页「从素材池添加」认领 1 张后提示词库 6→7 且素材池文件数不变；确认框 `Esc` 取消不动数据、`Enter` 确认生效；批次行两列与展开清单逐张加总一致；四页无 JS 报错与 404。
- `/opt/homebrew/bin/ruff check autolabel tests` 全通过，前端 JS `node --check` 全通过。

文档：`06-架构与数据格式.md` 重写数据集布局、新增「两个标注库与素材池」小节（对照表 + 跨库视图 + 删除粒度表）、索引与批次章节改两库口径、接口表补 `/api/pool`、`/api/pool/adopt` 与 `mode`/`lib` 参数；`02-命令行参考.md` 命令表补 `--lib`/`--mode`/`pool`/`migrate`，重新生成 `import`/`video`/`batches`/`status` 帮助快照并新增 `pool`/`migrate` 两节；`04-复核界面指南.md` 重写「批次管理（门户）」、补素材池入口与两种删除力度；`05-视频与批量导入.md` 上传入口改成「在哪上传」对照表；`07-常见问题.md` 重写 5 条 FAQ 并新增「同一批素材两个模块都要用，会占两份磁盘吗」「老数据集怎么升级到两个库」；`09-提示词标注.md` 对照表补库名/索引/批次/撤销目录与 `pool` 任务类型，新增素材池认领命令；`README.md`、`demo_docs/README.md`、`demo_docs/01-快速上手.md` 补两库与素材池说明、测试数改到 267、子命令数改到 18。

改动文件：`autolabel/library.py`、`autolabel/batches.py`、`autolabel/dataset.py`、`autolabel/review.py`、`autolabel/upload.py`、`autolabel/video.py`、`autolabel/prompt.py`、`autolabel/cli.py`、`autolabel/static/{common.js,portal.html,portal.js,review.html,app.js,objects.html,objects.js,app.css}`、`tests/{_helpers.py,test_batches.py,test_library_split.py}`、`README.md`、`demo_docs/{README.md,01-快速上手.md,02-命令行参考.md,04-复核界面指南.md,05-视频与批量导入.md,06-架构与数据格式.md,07-常见问题.md,09-提示词标注.md,CHANGELOG.md}`。

## 2026-10-17 03:22 — 两个模块各自上传素材，主页统一管批次与文件

之前提示词标注只能借用姿态标注的上传口：素材从 `/pose` 传进来、先跑一遍 YOLO-pose，再到 `/objects` 跑检测框，两个模块的素材混在一堆批次里，主页也看不出哪批是骨骼哪批是物品。这轮把上传链路彻底拆开，并把「管批次」这件事统一到主页。

新增：

- **提示词标注模块独立上传**：`/objects` 右上角 **导入素材**（快捷键 `W`）——拖文件/选图片/选图片目录/选视频，勾字段、选多模态配置、抽帧密度、批次名、补充说明，提交后走新任务类型 `prompt_upload`：入库（视频抽帧 + 感知去重）→ 逐图问多模态大模型 → 写 `prompt_labels/` → 出检测框坐标 → 收尾批次，**全程不跑 YOLO-pose**，`labels/` 一个字节不动。取消「入库后自动标注」就只入库，字段还没建也能先传（只入库时不校验字段与多模态配置）。新接口 `POST /api/prompt/upload`，后端 `_run_prompt_upload()`。
- **批次归模块**：批次记录新增 `module`（`pose` / `prompt`），写入时按任务类型推导，历史批次没这个字段也在读取时按 `kind` 现补（`module_for()`），认不出归「未识别」，不会有漏网批次。`/api/batches` 与 `/api/dataset` 一并返回按模块汇总的 `by_module`（批数 / 张数 / 已标 / 框数）。
- **主页「批次管理」**：原来只是批次摘要前 8 条，现在是完整的管理处——模块筛选下拉框、按模块汇总一行、每行模块徽章 + 骨骼与检测框两套进度 + 状态 + `看文件` + `去复核`（按模块跳对应页），行首勾选整批删。`看文件` 展开该批次逐张图片（导入时间 / 尺寸 / 体积 / 两套复核状态 / 框数 / 标记），行尾 **删这张** 删单张，行首勾选配合「删除选中图片」一次删多张。新接口 `GET /api/batch/files`、`POST /api/images/delete`。
- **按张删除后端 `batches.delete_images()`**：与整批删除同口径——删前写 `.autolabel/deleted/` 快照（含两套标签原文）、从所属批次 `items` 里摘除、删空的那批连批次记录一起移除、重算 `coords/` 与 QC 报告；批次还在标注中时拒绝并提示先「清除任务」。命令行同步加 `autolabel batches <数据集> --delete-images images/a.jpg`（可重复），与界面同一套实现。
- 姿态页的老「批次管理」弹层（快捷键 `B`）保留，`_row()` 同时给出 `pose{}` / `prompt{}` 分组和原来的扁平字段，老界面照旧能读。

修复：

- **删图留一堆垃圾**：`_remove_files()` 以前只清 `labels/` 与 `prompt_labels/`，漏了 `.autolabel/prompt_originals/` 与 `.autolabel/prompt_history/`，删过的图在这两个目录里越积越多，还会被下一次撤销当成可用版本。现在一并清理。
- **批次被误删空**：`delete_images` 判「批次是否已空」原本只看登记表 `items`，而 `items` 会因为导入中断比索引少——实测删一张图就把还有 10 张图的批次整条抹掉，主页表现为「共 0 个批次」。改为**以索引为准**统计每个批次还剩几张，且只处理本次真被删到图的批次，没碰过的批次（包括 0 张的空批次）一律不动。
- **两个进程互相覆盖批次表**：批次表是「读全表 → 改 → 写全表」，命令行 `autolabel import` 与复核服务是两个进程，撞上就把对方刚写的批次记录覆盖掉（`_LOCK` 只是进程内线程锁，挡不住）。写路径统一换成 `_file_lock(root)`：`.autolabel/batches.lock` 上的 `flock` + 进程内 `RLock`，并且**按线程可重入**——`reconcile_with_disk()` 握锁期间还要调 `attach_items()`，而 flock 按文件描述符判归属，第一版不可重入直接把 `tests/test_batches.py` 挂死在启动对账那条用例上。
- **文件行的「删这张」被拉成整行宽**：全局 `button { flex: 1 }` 蔓延到新加的文件列表，每个按钮变成一条红杠；`.file-row button` 补 `flex: 0 0 auto`，文件名改为 `flex: 1 1 220px`。

测试：

- `python3 -m pytest tests -q` **256 passed / 5 skipped**（上一轮 241 passed）。`tests/test_batches.py` 增至 24 项：模块归属推导、批次行按模块拆分进度、逐张明细两套状态、按张删除只动选中的图、删空连带移除批次记录、`--keep-files` 保留文件不被对账长回、护栏（标注中 / 不在索引 / 空列表）、**登记表 `items` 偏少时删一张不许带走整批**、删一个批次不碰别的批次、`batches --delete-images` 命令行口径；`tests/test_prompt_web.py` 增至 31 项：`by_module` 分组、逐张文件接口与单张删除、`prompt_upload` 端到端（入库 + 标注 + 出坐标，打桩多模态请求）、只入库不标注、入参校验（没文件 / 字段缺失的中文报错）。
- 服务实测：重启后命令行导入 9 张 → 主页「姿态标注 1 批 / 8 张」，展开逐张 `删这张` 删掉 1 张后批次同步变 8→8 且索引一致；提示词页 `导入素材` 传 8 张跑真实 Ollama 配置（`Qwen3.8-Flash-Next-Uncensored-oQ6e-mtp`），完成「标注 8 张，产出 46 个框」并自动出 `coords/prompt-*.jsonl`；重复图入库如实回报「没有新图入库（可能与已有图片重复…）」不谎报张数；门户 / 姿态 / 提示词 / 模型四页无 JS 与 404 报错。
- `/opt/homebrew/bin/ruff check autolabel tests` 全通过，五个前端 JS `node --check` 全通过。

文档：`04-复核界面指南.md` 新增「批次管理（门户）」小节并保留姿态页弹层说明、提示词页补「导入素材」；`09-提示词标注.md` 新增「素材从哪来：本页自己上传」，对照表补上传入口与 `prompt_upload` 任务类型；`06-架构与数据格式.md` 补批次 `module` 字段与推导规则、两种删除粒度、跨进程可重入锁、任务类型 `prompt_upload`、三个新接口；`07-常见问题.md` 新增「两个模块各自上传，主页能一起管吗」「只想删传错的某一张图」；`02-命令行参考.md` 补 `--delete-images` 并重新生成 `batches --help` 快照；`README.md`、`demo_docs/01-快速上手.md` 测试条数改到 256。

改动文件：`autolabel/batches.py`、`autolabel/upload.py`、`autolabel/review.py`、`autolabel/cli.py`、`autolabel/static/{portal.html,portal.js,objects.html,objects.js,app.js,app.css}`、`tests/{test_batches.py,test_prompt_web.py}`、`README.md`、`demo_docs/{01-快速上手.md,02-命令行参考.md,04-复核界面指南.md,06-架构与数据格式.md,07-常见问题.md,09-提示词标注.md,CHANGELOG.md}`。

## 2026-10-17 02:30 — 批量重标不再抹掉人工成果 + 命令行导入的图不用重启

顺手实测提示词页时撞到一个真问题，属于上一轮「人碰过的数据要被尊重」这条原则的遗漏，这轮补上。

现象：图片已经被人工点过「接受」、甚至手动画了框，再跑一次批量提示词标注（换了描述或换了模型导致签名变化），模型对这张回了 0 个框，于是人工成果被整张抹平；更糟的是索引里还留着 `prompt_objects: 0` 但 `prompt_manual: 1`、`prompt_manual_boxes: [0]`、`prompt_classes: [4]` 这种悬空标记——指向一个已经不存在的框，撤销和导出都会被带偏。

修复：

- `prompt.run_prompt_labeling()` 在非 `overwrite` 模式下的跳过条件从「同签名已标过」扩成三条：同签名已标过、`prompt_status == approved`、`prompt_manual > 0`。跳过张数记进新的 `stats["protected"]`，界面/命令行汇总显示为「人工保护 N 张」。
- 真被重标的图（勾了覆盖）在写模型结果时同步重置 `prompt_manual=0`、`prompt_manual_boxes=[]`、`prompt_classes` 按实际框重算，不再留悬空的人工标记。
- `autolabel prompt --overwrite` 的帮助文案改为「已经标过的也重标（默认跳过同签名已标的、人工接受过的、人工画过框的图）」，界面「覆盖已有框」勾选框同口径。
- 计数要看得见：`upload.JobStore._run_prompt()` 把 `protected` 放进 `job.result`，完成消息在有保护对象时追加「N 张人工接受过或画过框的图未覆盖（勾「覆盖已有框」可重标）」——否则全是保护时界面只显示「完成：标注 0 张」，像任务没跑起来；`autolabel prompt` 命令行汇总同样补一行 `[info] 人工保护 N 张…`。提示词页的任务日志直接透印任务消息，不用改前端。

顺带修的第二条（实测导入示例素材时撞到）：

- **服务跑着的时候用命令行导入图片，界面一直显示 0 张**：`ReviewStore.dataset_payload()` 读的是常驻内存的索引，而 `reload()` 只在写接口和后台任务结束时被调用，`GET /api/dataset` 从不重读磁盘。于是 `autolabel import` 写完 `index.json`（实测磁盘 4 条、接口 0 条），界面既看不到新图也不报错，只能重启服务。改为 `dataset_payload()` 先比对 `index.json` 的 `mtime_ns + size`，变了才重读（`_disk_stamp()`/`_refresh_if_changed()`，`reload()` 同步更新指纹）；文件没变就零开销，不会给每次刷新加一次 JSON 解析。

测试：

- `python3 -m pytest tests -q` **241 passed / 5 skipped**（上一轮 237 passed）；`tests/test_prompt.py` 增至 36 项，新增 `test_run_protects_human_work_without_overwrite`（换签名后已接受/画过框的两张都被跳过、`protected==2`、不发请求；`overwrite=True` 才重标）与 `test_relabel_resets_stale_manual_marks`（重标后 `prompt_manual`/`prompt_manual_boxes`/`prompt_classes` 与磁盘上的框对齐，复核状态退回 `edited`）；`tests/test_prompt_web.py` 增至 26 项，新增 `test_prompt_run_reports_protected_human_work`（走真实 HTTP 与任务队列：接受一张 + 手动画框一张后换描述重跑，`labeled==0`、`protected==2`、完成消息含「未覆盖」）、
  `test_dataset_endpoint_picks_up_external_index_writes`（绕开服务直接 `import_images` 落盘，再 `GET /api/dataset` 必须看到新图；把 `_refresh_if_changed` 打桩成空函数可复现旧行为的漏图）。
- 服务实测：重启后按命令行导入 1 张，接口从 4 张变 5 张，无需再重启；提示词页跑真实 Ollama 配置（`Qwen3.8-Flash-Next-Uncensored-oQ6e-mtp`）时如实回报「完成：标注 0 张，产出 0 个框；1 张人工接受过或画过框的图未覆盖（勾「覆盖已有框」可重标）」；门户/姿态/模型/提示词四页无 JS 报错；删掉 3 个批次后索引、批次、对账提示全部归零，`.autolabel/deleted/` 快照生效没把文件当孤儿补回来。
- `/opt/homebrew/bin/ruff check autolabel tests` 全通过。

文档：`09-提示词标注.md`「跑标注」补「人碰过的图默认不动」；`07-常见问题.md` 新增「批量重标会不会把我手工画的框冲掉？」；`01-快速上手.md`、`README.md` 测试条数改到 244。

改动文件：`autolabel/prompt.py`、`autolabel/cli.py`、`autolabel/upload.py`、`autolabel/review.py`、`tests/{test_prompt.py,test_prompt_web.py}`、`README.md`、`demo_docs/{01-快速上手.md,02-命令行参考.md,07-常见问题.md,09-提示词标注.md,CHANGELOG.md}`。

## 2026-10-17 02:15 — 上层工作台 + 提示词标注二级模块：一句话描述换一批检测框

这一轮把"只有复核页"的单页工具拆成了**门户 + 三个二级模块**，并新增第二条标注链路：用自然语言描述让多模态大模型找物品，产出检测框给 `yolo detect train`。

新增能力：

- **门户工作台 `/`**：八个统计卡（图片总数 / 骨骼已标注·待复核·带标记 / 提示词已标注·检测框总数·待复核 / 批次数，全部从当前索引现算，不做缓存）、三张模块卡片（姿态卡带「去导入素材」深链 `/pose?upload=1`）、两套导出入口（骨骼坐标、检测框坐标、YOLO 数据集目录）、批次摘要前 8 条、启动对账提示条。`/pose`（旧地址 `/review` 仍可用）、`/objects`、`/models` 顶栏互相切换，高亮按 `data-page` 走。
- **字段登记表 `autolabel/fields.py`**：一个字段 = 英文类名（进 `data.yaml` 的 `names`，只允许 ASCII：字母开头 + 字母/数字/`_`/`-`，≤32 字符）+ 中文显示名（≤40）+ 描述提示词（≤600，超出直接报错不静默截断）。存 `.autolabel/fields.json`，**`class_id` 分配后永不复用**（防止删字段后历史标签串类），颜色按调色板顺序分配。**人工框的 `manual` 标记现在能存住**：YOLO 5 列标签存不下"这条是人工画的"，改为在索引里记 `prompt_manual_boxes`（框下标），重读磁盘时按 `apply_manual_marks()` 把标记贴回去——代价是别手改标签文件的行顺序。
- **提示词标注 `autolabel/prompt.py`**：`build_prompt()` 把字段表拼成标注指令（坐标按 0-1000 整数回报、类名必须原样照抄、"看不清/像/遮挡过半不要标"）；`signature_for()` 按「字段集合 + 描述文本 + 多模态配置」出签名，同签名跳过不重复跑，改了描述或换模型自动判定需要重标（打 `prompt_stale`）；`parse_objects()` 兜住模型的坏毛病（fenced 代码块包裹 / 裸数组 / `detections|items|results` 外层键 / 散文里掏 `{}` / `xywh`·`{x1,y1,x2,y2}`·`{cx,cy,w,h}` 三种写法 / 0-1 与 0-1000 与像素三种量纲 / 中文与模糊类名回捞，认不出的进 `guesses` 不硬塞，纯文字答复按"没找到"处理）；`box_flags()`/`image_flags()` 打 `small_box`/`edge_box`/`dup_box`/`unknown_class`/`vlm_failed`/`manual` 等标记并算复核优先级；`export_detection_dataset()` 出 `yolo detect train` 目录，**类别号自动重映射成连续 `0..N-1`**（因为字段号不再复用）并在 `data.yaml` 注释写明。
- **提示词标注页 `/objects`**：画布支持**空白处拖动补画新框**（字段取「补画字段」下拉框，打 `manual`、置信度按 1 计）、框内拖动移动、拖角点缩放、滚轮缩放、空格平移；右侧框列表每行可**改字段**（画错类不必删了重画）与删除；筛选（待复核/带标记/已接受/已驳回/未标注/全部）、排序（复核优先级/文件名）、接受驳回与撤销、跳到待复核；「字段管理」弹层做新建/改名/改描述/删除（仍有框时要确认「连带删除框」，并提示孤儿类别号）；「批量提示词标注」弹层选字段多选、多模态配置、范围（仅未标注/当前筛选/整个数据集）、限张数、是否覆盖、补充说明，提交后走 `kind="prompt"` 任务队列，沿用顶部任务条与「清除任务」。快捷键 `Q/E/F/U/X/Z/[/]/G/O/1`。
- **模型管理独立页 `/models`**：原复核页的 `#model-modal` 整块迁出为独立页面（复核页瘦了 600 行），按 `M` 直达；`bootModels()` / `registryRows()` 不再依赖复核页的 `state.ds`，四个页面各自 `boot`。
- **启动对账 `batches.reconcile_with_disk()`**：`ReviewStore` 构造时按磁盘补回 `images/` 里已落盘但索引缺席的图片（挂到那个还"标注中"的批次上，状态未复核），并把无人认领的 running 批次判定为 **`interrupted`**；`--keep-files` 删批次留下的文件按 `.autolabel/deleted/` 删除快照排除，不会被误当成孤儿重长；索引有记录但原图已消失的条数一并汇报。结果经 `/api/dataset` 的 `recovery` 字段显示在工作台黄色提示条，同时打启动日志。起因是实测中撞到：导入 62 张跑到一半服务被重启，49 张图片已落盘却再也没出现在界面里，批次永远显示"标注中"。
- 新接口：`GET /`·`/portal`·`/pose`·`/objects`·`/models` 四个页面与 `/common.js`·`/portal.js`·`/objects.js`·`/models.js`；`GET /api/fields`、`POST /api/fields/save`、`/api/fields/delete`；`GET /api/prompt/item`、`POST /api/prompt/item`·`/api/prompt/status`·`/api/prompt/undo`·`/api/prompt/run`(202)·`/api/prompt/export`、`GET /api/prompt/summary`·`/api/prompt/coords`；`POST /api/undo` 支持 `{kind:"prompt"}` 只回滚检测框不碰姿态标签；`/api/dataset` 增出 `fields`·`prompt`·`recovery`。
- 命令行：新增 `autolabel fields`（`--add 字段名=描述` / `--set-label` / `--delete` / `--drop-boxes` / `--json`）与 `autolabel prompt`（`--fields` / `--vlm` / `--limit` / `--overwrite` / `--prompt-extra` / `--coords` / `--export` / `--status` / `--quiet`）。

修复：

- **第一次画框保存后撤销报"没有可回滚的历史版本"**：`_backup()` 在标签文件不存在时直接返回，既不备份也不留快照，于是"空白图 → 画第一个框 → 撤销"这条最常见路径根本没得撤销。改成首次保存前先落一个**空白快照**，撤销能回到"一个框都没有"；`has_backup` 同时改为「有原始备份 **或** 有历史快照」，姿态与检测框两套目录各自判定。
- **`GET` 的参数错误被判成 500**：`/api/prompt/coords?format=xml` 这类未知格式抛 `ValueError` 却被兜底成 `500 Internal Server Error`（`do_POST` 早就映射成 400，`do_GET` 漏了）。补齐后浏览器里直接显示「未知格式: xml（可选 csv, jsonl）」。
- **刚改名的字段删不掉**：`fields.upsert_field()` 认 `bottle` 与 `BOTTLE` 是同一条，但 `find_field()` 严格区分大小写，于是把字段改名后再按原名删除会报"字段不存在"。名称比较统一改成大小写不敏感。

验证：

- `python3 -m pytest tests -q` **237 passed / 5 skipped**（上一轮 163 passed）；新增 `tests/test_fields.py`（18 项：`class_id` 分配与永不复用、重名合并、名称校验矩阵、描述超长抛错、删除保护、孤儿类别号、5 列往返、坏行跳过、`resolve_fields`、`prompt_labels/` 与 `labels/` 隔离）、`tests/test_prompt.py`（34 项：提示词文案、签名变化、`parse_objects` 容错矩阵、`apply_manual_marks`、三类标记与优先级、`run_prompt_labeling` 跳过/覆盖/失败处理、`render_object_coords`、`write_object_coordinate_files`、`export_detection_dataset` 类别重映射与 rejected 排除）、`tests/test_prompt_web.py`（24 项：四个页面与 `data-page` 标记、静态资源白名单、四页都含顶栏与确认弹窗、姿态页不再有模型弹层、字段 CRUD 与中文报错、检测框写读撤销与 manual 往返、未知类别号与零面积框 400、`/api/undo?kind=prompt` 不碰姿态标签、`/api/prompt/run` 端到端（打桩多模态请求）、`/api/jobs/clear`、`/api/prompt/coords` 下载头与未知格式 400、`/api/prompt/export`、门户单次 `/api/dataset` 数据完整性）、`tests/test_batches.py` 新增 3 项对账测试（补回孤儿图片并挂到中断批次、`--keep-files` 保留文件不重长、原图丢失条数汇报）。
- `/opt/homebrew/bin/ruff check autolabel tests` 全通过；五个 JS 文件逐个 `node --check` 通过。
- headless Chrome 实跑：门户八卡计数与批次摘要正确、三张模块卡片链接正确、顶栏高亮随页面切换；姿态页模型下拉 14 项（13 档权重 + 1 条多模态配置）分组渲染、`/pose?upload=1` 自动开弹层并把地址洗回 `/pose`；模型页目录 14 行 / 权重 13 行 / 多模态 1 行 / 6 个预设、默认值与权重目录路径显示正确；提示词页新建 `bottle`·`backpack` 两个字段、非法名返回中文提示、拖框→保存→`prompt_status=edited`、改字段下拉、再画第二个框、删框、接受后进度 `0/4 → 1/4` 且列表筛掉该图、批量标注任务真跑起来并按多模态地址不可达如实报错且无 JS 异常；全程无 JS 报错、无意外 4xx。

改动文件：`autolabel/fields.py`、`autolabel/prompt.py`、`autolabel/batches.py`、`autolabel/review.py`、`autolabel/static/{common.js,portal.html,portal.js,objects.html,objects.js,models.html,models.js,app.js,app.css,review.html}`、`tests/{test_fields.py,test_prompt.py,test_prompt_web.py,test_batches.py}`、`README.md`、`demo_docs/{README.md,02-命令行参考.md,04-复核界面指南.md,06-架构与数据格式.md,07-常见问题.md,09-提示词标注.md,CHANGELOG.md}`。


## 2026-10-17 01:02 — 备注在界面上真正存得住了，空数据集不再看着像坏页

修复：

- **「模型管理」的备注保存按钮永远是灰的**：启用判断绑在了按钮自身的 `change` 事件上，而按钮不会派发 `change`，所以输入多少字按钮都不亮，界面上根本存不了备注（后端 `/api/models/note` 与命令行 `models --note` 本身没问题）。改成绑在输入框的 `input` / `change` 上、初始置灰、内容与已存值不同才亮；额外支持在输入框里按回车直接保存（全局快捷键本来就会跳过输入框，不冲突）。
- **空数据集看着像页面坏了**：不分情况都显示「当前筛选条件下没有图片，试试切换右上角筛选」，把用户往筛选器上引。现在分成两句——数据集为空写「这个数据集还没有图片：把文件拖进本页、点「导入素材」，或用命令行 autolabel import 导入后即可开始复核」，只有真被筛选清空时才提筛选。
- **少了 `GET /favicon.ico` 的 404**：`review.html` 改为内联 SVG 图标（绿色小骨架人），零额外请求，控制台干净。

验证（这一轮改用真实浏览器实测，不再只做语法检查）：

- headless Chrome 实跑：模型目录 14 行（13 档 YOLO-pose + 1 条多模态配置）中文名 / 中文说明 / 参数·mAP·体积·关键点·速度 / 状态徽章 / 下载·设为默认按钮全部正常渲染，「共 14 个 · 可用 14 个」计数正确；备注端到端通过（输入 → 按钮启用 → 回车保存提示「备注已保存」→ 刷新后备注仍在 → 清空提示「备注已清空」）；控制台无 JS 报错、无 4xx 响应。
- `python3 -m pytest tests -q` 163 passed；`ruff check autolabel tests` 全通过；`node --check autolabel/static/app.js` 通过。
- 改动文件：`autolabel/static/app.js`、`autolabel/static/review.html`、`demo_docs/07-常见问题.md`（新增「网页打不开，或者打开后一张图都没有」小节）、`demo_docs/CHANGELOG.md`。

## 2026-10-17 00:42 — 模型目录可浏览、可下载、有中文说明与备注

新增能力：

- `autolabel/models.py` 重写为「三层数据 + 一份清单」：代码里的 `PRESET_CATALOG`（13 档 YOLO-pose 可下载目录：YOLO11 n/s/m/l/x → YOLOv8 n/s/m/l/x → YOLO26 n/s/m）+ 新增本机登记表 `configs/models.json` + 原有 `configs/vlm.json`，由 `ModelRegistry` 合成界面与命令行共用的统一模型清单。每档带**中文说明**、参数规模、mAP50-95、预计体积、SHA256、关键点集、速度与适合场景；官方没公布的字段留空（YOLO26 的参数与精度、部分档位的 SHA256），文案直说「官方暂未公布」，不编数字。
- 新增 **API Key 独立存放** `autolabel/secrets.py`：明文只落 `~/.autolabel/secrets.json`（0600，`AUTOLABEL_SECRETS` 可改），`configs/vlm.json` 里只留 `key_ref: vlm:<配置ID>`；`load_vlm()` 读盘时才把明文回填内存，`_migrate_keys()` 把老配置里残留的明文自动搬走并回写。`configs/*.json` 与 `index.json` 里不再有 Key 明文，可以放心贴给别人。
- **备注说明**：任意模型（含多模态配置）都能写一句人话，界面上是每行的备注输入框，命令行 `models --note ID=说明`（可重复、等号后留空即清空、ID 写模型 ID 或权重文件名都认、上限 500 字）；多模态备注沿用 `remark` 字段，两处看到同一句话。
- **默认模型**：`models --default <ID>` 按能力（pose / vlm / custom_task）分别记默认值，跨能力设默认直接拒绝；没显式设过时按「已就绪的推荐档 → 任一已就绪 → 目录第一档」兜底，保证下拉框永远有得选。
- **登记本地模型**：`models --register <路径> [--id --display-name --origin --note]` 只记路径不动文件，要求姿态权重是完整 zip 包（半截文件当场拒），文件名本就在目录里的只写备注不重复登记，ID 撞车会提示换名；`models --forget <ID>` 只删登记条目，内置目录项拒绝并提示改用备注或 `catalog.<ID>.hidden` 隐藏。
- **下载留痕**：`start_download()` 每次下载写一条审计记录（`model_id` / `url` / `mirror` / SHA256 期望与实际值及是否一致 / `expected_bytes` / `bytes` / `size_mb` / `temp_name` / `seconds` / `ok` / `error` / 起止时间，最多留 50 条），`models --info <ID>` 列出最近 3 条；SHA256 不一致只在提示里提醒「建议删除后重下」，不判失败。
- **兼容性判定**：`models --dataset <数据集>` 时按关键点数量比对，不匹配的行状态转「不兼容/暂不可用」并给中文理由；模型加载后把实测关键点数 / 设备 / ultralytics 版本 / 输入尺寸回写进登记表（`--info` 里的「加载实测」），排查不靠猜。
- 多模态配置新增三个运行参数并在界面表单可调：`max_tokens`（回复上限，默认 4096）、`image_limit`（送图最长边，默认 1024）、`json_mode`（强制 `response_format` JSON，默认关）。
- 命令行 `models` 新增 `--dataset` `--browse/--catalog` `--info` `--capability` `--note` `--register` `--id` `--display-name` `--origin` `--forget` `--default`，并新增按显示宽度对齐的表格打印（中文列不再错位）、`--browse` 每个模型一段中文说明含下载命令、`--info` 竖排详情。
- 新接口 `POST /api/models/note`、`/api/models/default`、`/api/models/local/add`、`/api/models/local/forget`；`GET /api/models` 增出统一清单 `models`、可下载目录 `catalog`、各类默认值 `defaults`、下载记录与登记表 / Key 文件路径；`/api/dataset` 增出 `model_rows`（带兼容性结论）。
- 界面「模型管理」面板顶部新增**模型目录**区块（能力筛选下拉框 + 中文名 + 中文说明 + 参数/体积/关键点/适合场景 + 备注输入框 + 下载 / 设为默认 / 取消登记 + 登记本地模型表单 + Key 文件路径提示）；上传弹层选定模型后紧跟一行中文说明，说明这档适合什么场景。

修复与健壮性：

- `--note` 早于 `--register` 处理，导致 `--register 新权重 --note 新ID=说明` 必然报「未知模型」：改成先登记后写备注，同一条命令搞定。
- `models --default <ID> --capability 拼错` 把 `ValueError` 抛成 traceback：捕获后按 `[error] 能力只能是 pose/vlm/custom_task，收到 xxx` 输出并返回退出码 2。
- `models --info` 里「下载地址 / SHA256」两行永远不显示（`describe()` 走的统一清单没带 `url`）：详情里补上下载地址与镜像名，直接可抄。
- `--dataset` 只影响 `--info`，普通清单与 `/api/models` 不带兼容性结论：`vlm_payload(dataset_root=...)` 打通，复核服务的 `/api/models` 按当前数据集给结论，界面切面板不再抹掉兼容性。
- 未下载的关键点不匹配模型能在上传弹层里被选中，跑起来才失败：下拉框直接置灰并写明「与当前数据集关键点数量不匹配」，全部没下载 / 全不兼容时才放开第一条兜底。
- 状态、来源、能力以前是散落的英文串：改成 `capability` / `source` / `status` / `origin` 四组枚举 + 中文标签（`capability_label` 等），`weight_info()` 的老键 `source`（`disk|preset`）与 `custom` 语义保留，前端与既有测试不受影响。

测试：163 项（新增 `tests/test_model_registry.py` 32 项，覆盖 13 档目录顺序与中文说明、体积/SHA256/说明字段、自定义权重占位行、备注写入与清空与未知 ID、备注与多模态 `remark` 同步、默认值兜底与跨能力拒绝、本地登记与半截文件拒绝与 ID 撞车、`forget` 拒绝内置目录项并清默认、关键点不匹配与不完整权重的不兼容结论、下载审计记录字段齐全与「无官方 SHA256 不下结论」、明文 Key 自动迁移与掩码、`--browse/--info/--note/--default/--register/--forget` 命令行、四个新接口的成功与 400/404 分支）；`tests/conftest.py` 新增 autouse fixture，把登记表与 Key 文件指向 tmp，测试不再污染 `configs/` 与 `~/.autolabel`。

文档：新增 `08-模型管理.md`（目录浏览 / 下载记录 / 备注 / 默认值 / 本地登记 / 兼容性 / Key 存放 / 换机器要拷哪几份）；`README.md` 与 `demo_docs/README.md` 更新模型管理条目与文档索引；`02-命令行参考.md` 补 `models` 新参数与示例并重新生成帮助快照、多模态字段表补 `key_ref` / `max_tokens` / `image_limit` / `json_mode`；`04-复核界面指南.md` 改写模型管理为三块并更新上传弹层模型选择说明；`06-架构与数据格式.md` 补 `secrets.py`、`ModelRegistry` 三层数据说明、依赖方向与 4 条新路由；`07-常见问题.md` 重写 Key 存放问答，新增「备注与默认值存在哪、换机器拷什么」「怎么判断模型能不能用在当前数据集」「下拉框灰着点不动」三条。

修复的现场问题：

- 上一轮遗留的复核服务仍在后台跑**旧代码**，按旧格式反复回写 `configs/vlm.json`，把已迁移的 `key_ref` 冲掉、又塞回一个空 `api_key`（本机那份 Ollama 配置的 Key 引用因此丢过一次）。已停掉旧进程并补回 `key_ref: vlm:vlm_a13bdd38`；`tests/conftest.py` 的 autouse fixture 顺带把 `AUTOLABEL_VLM_CONFIG` 也指到 tmp，任何没显式隔离的测试都碰不到开发者真实配置了。

界面实测：起新代码的复核服务打 `GET /api/models` 返回 13 条目录 + 14 行清单、`defaults` 三类齐、中文说明与状态正常；`POST /api/models/note` 写入与读取正常、未知 ID 返回 404；`POST /api/models/default` 生效，跨能力返回 400 并写明「YOLO11-medium 姿态 属于姿态估计，不能当多模态大模型的默认模型」；登记 `/tmp/al-smoke/gym.pt` 后清单里立刻带出中文名与备注，取消登记后重复取消返回 404；`/api/dataset` 的 `model_rows` 14 行，`/app.css`、`/app.js`、首页元素全部为新版。

其它：`.gitignore` 增加 `configs/models.json`（本机状态不入库）。

## 2026-10-16 23:45 — 多模态大模型接进标注流水线

新增能力：

- 新增模块 `autolabel/vlm.py`：多模态视觉大模型作为**可选标注后端**，接口与 `PoseEnsemble` 对齐（`load` / `infer` / `display_names`），流水线不加分支。
- `backends.build_backend(settings)` 按 `BackendSettings.vlm_profile` 挑后端：为空走 YOLO-pose，非空走 `VLMBackend`；`BackendSettings` 新增 `vlm_profile` 字段，参数指纹自动随之变化（换后端会判定需要重标，`approved` 的图不动）。
- 上传素材的模型下拉框改成两组：`姿态权重（YOLO-pose，快且稳）` 与 `多模态大模型（逐图询问，慢但灵活）`，后者直接列「模型管理」里配好的本地/云端配置并标注 `已连通 / 测试失败 / 未测试`；没配过则显示「未配置：先在模型管理添加」。
- 命令行 `label` / `auto` 新增 `--vlm <配置ID或名称>`（`find_profile()` 支持 ID、精确名与前缀匹配）。
- 提示词写死 COCO-17 顺序与可见性语义，只收 JSON：`{"people":[{"bbox":[x1,y1,x2,y2],"keypoints":[[x,y,v]×17]}]}`，坐标按 0-1000 整数回报，模型回 0-1 小数也认。
- 标注结果每条打 `vlm_labeled`（QC 权重 1.5）、索引 `labeled_by` 记 `vlm:<配置ID>`，整批进人工复核清单，与 YOLO 标注可区分；`/api/dataset` 新增 `vlm` 字段（Key 已掩码）。
- 实测：本地 8B 级视觉模型（`http://127.0.0.1:7979/v1`）标 `models/bus.jpg` 用 38.8 秒、出 2 人，标签通过 `validate`；同一张图 YOLO11n 约 0.03 秒出 4 人，所以定位是补充手段而非替代。

修复与健壮性：

- 思考型模型把输出额度花在推理上导致正文为空：`chat()` 输出上限提到 4096 token、等待下限放宽到 60 秒，正文为空时回落到 `reasoning_content` 继续解析；仍为空才报「模型返回空内容」并说明原因。
- 模型输出的 JSON 有瑕疵就整张判失败：解析改成两级——先整段 `json.loads`，失败再按 `keypoints` 做括号平衡扫描 + 三元组正则提取，漏逗号、缺尾花括号、被截断都能救回一部分；连 `{` 都没有才报「模型没返回 JSON」。
- 多模态配置默认超时从 15 秒提到 60 秒（界面表单、`normalize_profile` 同步），否则单张 25~40 秒的标注必然超时。
- 单张推理失败会中断整批：`label_dataset` 现在捕获 `infer` 异常，打印 `[warn] 推理失败 <图片>: <原因>` 并计入 `failed`，其余继续。
- `_reply_text` 兼容 `content` 为字符串或多段数组两种写法；bbox 缺失时用关键点包围盒兜底，再经 `expand_bbox_with_keypoints()` 撑开。
- 后台任务日志与批次记录都带上所用多模态配置（任务开头写明「用多模态大模型 X（本地 · 模型名）逐图标注…比 YOLO 慢很多属正常」），`Job.payload()` 暴露 `vlm_profile` / `vlm_name`。
- 提交任务时就解析多模态配置（`JobStore._normalize_params`）：按 ID 或名称定到唯一配置，把批次与任务里显示的 `vlm:vlm_xxx` 换成配置名，配置不存在直接 400 拒收并列出可选项，不再等任务跑到标注阶段才失败。

测试：131 项（新增 `tests/test_vlm.py` 24 项，覆盖提示词与 17 点顺序、图片缩边编码、0-1000/0-1 尺度识别、点数补齐与全零点丢弃、非法可见性收敛、残缺与截断 JSON 抢救、chat 请求头与错误透传、后端选择、按 ID/名称加载配置、端到端标注写入 `labeled_by` 与 `vlm_labeled`、单张失败不中断整批、`vlm_choices` 不泄露 Key、`--vlm` 参数落进 `BackendSettings`、提交任务时解析与拒绝未知配置）。

界面实测：用已配置的 `Ollama → http://127.0.0.1:7979/v1`（`Qwen3.8-Flash-Next-Uncensored-oQ6e-mtp`）从上传弹层提交 `models/bus.jpg`，46 秒完成，日志写明所用配置，标出 2 人并带 `vlm_labeled`，坐标文件与 QC 报告同步生成。

文档：`README.md` 复核特性与参数建议补多模态标注、目录树加 `vlm.py`；`02-命令行参考.md` 新增「用多模态大模型标注」小节与 `--vlm` 参数、重新生成帮助快照；`04-复核界面指南.md` 上传参数表改成两组模型来源并补耗时与精度说明；`06-架构与数据格式.md` 补 `vlm.py` 模块行、新增「多模态标注后端」小节与依赖方向；`07-常见问题.md` 补「能不能替代 YOLO-pose」与「特别慢/一直失败」排查表。

## 2026-10-16 23:25 — 模型管理界面（权重 + 多模态大模型）

新增能力：

- 新增模块 `autolabel/models.py`：权重清单/删除/后台下载 + 多模态大模型配置与连通性验证，一个模块管两类模型。
- 复核界面新增「模型管理」面板（右上角按钮，快捷键 `M`）。上半部分是**姿态权重**：预置 `n/s/m` 与 `models/` 里自定义 `.pt` 一并列出（体积、来源、修改时间、半截 `.part` 残留也算），填文件名即可下载非预置权重（如 `yolo11l-pose.pt`），后台下载带百分比进度条，勾选后批量删除；下半部分是**多模态大模型**：本地（Ollama / LM Studio / vLLM / OpenAI 兼容）与云端（OpenAI、阿里云 DashScope、自定义 Base URL）两类，类型切换自动带出默认地址与模型名建议，保存前一键 **测试** 连通性。
- 多模态配置存 `configs/vlm.json`（`AUTOLABEL_VLM_CONFIG` 可换路径）。**API Key 永不外泄**：`/api/models` 与界面只回掩码（`sk-…1234`），重存时输入框留空或仍是掩码即视为不修改，不会覆盖成真值。
- 连通性验证先 `GET {base_url}/models`，拿不到再退化成一次 `max_tokens=1` 的 `POST /chat/completions` 探测；失败文案翻译成可执行动作（本地服务没启动 / 地址要写成 `…/v1` / `401` 是 Key 问题 / 超时把上限调到 60–300 秒），结果连同耗时与可见模型列表记进 `last_test`。
- 新命令 `autolabel models`：不带参数即清单；`--download-weight`（`--force` 强制重下）、`--delete-weight --yes`、`--add-vlm '<JSON>'`、`--test-vlm` / `--default-vlm` / `--delete-vlm`、`--json`。
- 新接口 `GET /api/models` 与 `POST /api/models/{weight/download,weight/delete,download,vlm/save,vlm/test,vlm/default,vlm/delete}`；`/api/dataset` 的 `models` 字段改为走同一份清单，自定义权重从此也出现在上传弹层的模型下拉框里（下拉框按清单重建，不再写死三项）。

修复与健壮性：

- 删权重会误删正在用的：`Job.payload()` 补上 `model` 字段，删除前按当前任务判定占用，被占用直接拒绝并提示「正在被标注任务使用」。
- 权重下载失败在后台线程里静默消失：`ensure_weights()` 新增 `on_error` 回调透传真实网络错误，`_download_to()` 的进度回调异常不再打断下载。
- 权重名做过注入防护：`resolve_weight_in_dir()` 拒绝带路径分隔符、绝对路径、非 `.pt` 的名字，越界路径一律 400。
- 上传弹层模型体积小数位不一致（一位/两位混着显示），统一由 `mb()` 处理成 1 位小数。

测试：107 项（新增 `tests/test_models.py` 26 项，覆盖清单与自定义权重识别、半截残留、越界路径拒绝、删除释放空间、占用保护、后台下载成功/失败与错误透传、Key 掩码与回写不覆盖、字段校验与默认回落、`last_test` 保留/作废、连通性三种分支、`/api/models` 全链路与越界拒绝）。

文档：`README.md` 补「模型管理」条目、`models` 命令速查行与多模态问答，目录结构树补 `models.py` / `batches.py` / `upload.py` / `configs/vlm.json`；`02-命令行参考.md` 补 `models` 行、行为要点与「多模态大模型配置」小节并重新生成帮助快照；`04-复核界面指南.md` 快捷键表补 `B` / `M` 与模型管理说明；`06-架构与数据格式.md` 补模块表与「模型管理」小节及 9 条新路由；`07-常见问题.md` 补多模态配置、测试失败排查、Key 存储与权重增删四条。

说明：本轮只交付「配置 + 连通性验证」，多模态大模型尚未接入标注流水线（自动补标 / 校验关键点）；下载并保留 `models/yolo11n/s/m-pose.pt`、`yolov8n-pose.pt` 四个权重，界面与 CLI 均可自行补下其它权重。

## 2026-10-16 22:45 — 权重下载走镜像 + 就绪状态

新增能力：

- 权重下载改由 `backends.ensure_weights()` 负责：默认经 `https://gh-proxy.com/` 镜像（`AUTOLABEL_WEIGHTS_MIRROR` 可覆盖，留空回到直连），支持 Range 断点续传，先写 `*.part`、校验长度与 zip 完整性后再改名，失败自动重试 3 次。
- `weights_ready()` / `model_inventory()` 判断并报出候选权重状态；`/api/dataset` 新增 `models` 字段，上传弹层的模型下拉框实时显示「已就绪 40.5 MB / 未下载 / 权重不完整」。
- 提交任务时若所选权重未就绪，任务日志先写明「正在下载权重（已下载 X MB）」，不再看着像卡死。

修复与健壮性：

- `PoseBackend.load()` 遇到损坏权重会自动挪成 `*.bad` 并经镜像重下一次，再失败才抛 `weights_hint`；此前 `yolo11m-pose.pt` 被并发写入拼坏（40.5 MB 但 zip 内部错位，`is_zipfile` 查不出来），已重新下载并用 `models/bus.jpg` 实测出 4 人（0.93/0.91/0.89/0.77）。
- 下载进度在非交互式终端下按 25% 整行输出，不再刷一长串 `
`。
- 已下载 `models/yolo11s-pose.pt`（19.4 MB）、`models/yolo11m-pose.pt`（40.5 MB），三个 COCO-pose 权重全部就绪。

测试：81 项（新增 `tests/test_weights.py` 8 项，覆盖就绪判定、镜像 URL 拼接、断点续传、镜像不可用退回直连、显式路径不代下载）。

## 2026-10-16 22:25 — 批次分组与整批删除

新增能力：

- 新增批次登记表 `.autolabel/batches.json` 与模块 `autolabel/batches.py`：网页上传、命令行 `import` / `video` / `auto` 每次都登记为一个批次，索引里每张图带 `batch_id`。
- 复核界面新增「批次管理」面板（快捷键 `B`）：按批次显示图片数 / 已标注 / 人体数 / 待复核数与状态，支持「只看这批」筛选、全选、勾选后整批删除（带确认弹窗）。
- 整批删除会清掉图片、标签、QC 叠加图、`originals/`、`history/` 与索引条目，删除前写快照到 `.autolabel/deleted/<时间戳>.json`（含标签原文），随后自动重算坐标文件与 QC 报告。
- 新接口 `GET /api/batches`、`POST /api/batches/delete`；`/api/dataset` 的每条图片与 `/api/item` 都带上 `batch_id`，并附带批次列表。
- 新命令 `batches <数据集>`：列出批次（`--json`），`--delete <批次ID>` 整批删除，`--keep-files` 只删记录保留图片；`import` / `video` / `auto` 新增 `--batch-name`。
- `import_images` 返回值新增 `imported_files`（源路径 → 入库相对路径），批次挂载依赖它。

修复与健壮性：

- 模型权重被下载中断（服务重启/断网）时报错难懂，现在 `backends.weights_hint` 会给出文件实际体积与 `rm <权重>` 修复命令；已重新下载被截断的 `models/yolo11m-pose.pt`。
- 正在标注的批次禁止删除，避免边写边删产生孤儿文件；判定改为看**当前是否真有任务在跑**（`delete_batches` 新增 `active_ids`），服务重启遗留的「标注中」批次在界面里可以直接删，命令行加 `--force` 放行。
- 确认弹窗与「批次管理」面板同时打开时，确认按钮被下层卡片挡住点不到（`.modal` 共用 `z-index: 50`，批次面板在 DOM 里更靠后）；现给 `#confirm-modal` 提到 `z-index: 60`，叠层弹窗一律可点。

文档：

- `README.md` 补「批次管理」条目与 `batches` 命令速查行；`demo_docs/02-命令行参考.md` 补 `batches` 行、`--batch-name` 参数与重新生成的帮助快照；`04-复核界面指南.md`、`05-视频与批量导入.md`、`06-架构与数据格式.md`、`07-常见问题.md` 同步批次格式、生命周期与恢复方式。

测试：73 项（新增 `tests/test_batches.py` 11 项；`server` fixture 移入 `tests/conftest.py` 供多文件复用）。

## 2026-10-16 21:45 — 清除任务按钮

新增能力：

- 复核界面新增 **清除任务** 按钮（上传弹层 + 顶部任务进度条各一个），点击弹确认框，`确定`/回车执行、`取消`/`Esc` 放弃。
- 新接口 `POST /api/jobs/clear`，返回 `{"ok":true,"cancelled":N,"cancelling":N,"dropped":N}`。
- 排队中的任务立即删除，并清理它在 `.autolabel/uploads/` 的临时文件；历史记录一并清空。
- 运行中的任务协作式取消：`label_dataset` 新增 `should_stop` 回调，每张图前检查；标完当前图才停，已写入的标注保留，并照常重新生成 `coords/` 坐标文件。
- 界面在取消期间显示「取消中…」并禁用按钮，任务真正退出后自动刷新列表。

测试：62 项（新增清除任务的 HTTP 用例 2 项、`should_stop` 立即停止用例 1 项）。

## 2026-10-16 21:25 — 坐标导出 + 网页上传

新增能力：

- `coords` 子命令：宽表 CSV（一行一个人，73 列）、长表 CSV（一行一个关键点）、JSONL（一行一张图）+ `meta.json` / `summary.json`，同时给像素与归一化坐标。
- `auto` 子命令：视频/图片混合提交 → 入库 → 自动标注 → QC 报告 → 坐标导出，一条命令跑完。
- `export --coords`：导出训练集时顺带生成坐标文件，多一列 `split`。
- 复核界面「上传素材」：拖拽或选择视频、图片、整个图片目录，可选模型 / 抽帧密度 / 最多帧数 / 保留近重复帧；后台任务串行执行，进度条 + 日志 + 完成后自动刷新列表。
- 复核界面「坐标 CSV / 坐标 JSONL」：按最新标签现场生成并下载。
- 上传接口：`POST /api/upload`（原始字节）、`POST /api/jobs`、`GET /api/jobs`、`GET /api/coords`。
- `review --max-upload-mb` 控制上传大小上限。
- `label_dataset` 新增 `progress` 回调，落盘改为"重读磁盘 + 按图合并"，避免覆盖并发修改。
- `review` 服务支持 `reload()`，后台新入库图片立即出现在界面。

修复：

- `.modal` 定义在 `.hidden` 之后导致隐藏弹层仍拦截整页点击，界面点不动（已加 `.modal.hidden`）。
- `export` 的 `manifest.json` 之前没写入 `coco` 字段（先写文件后加键）。
- `status` 命令里重复加载索引的冗余代码。

测试：59 项（新增 `tests/test_coords.py` 7 项、`tests/test_upload.py` 6 项、`tests/test_integration.py` 2 项端到端）。

## 早前已实现

- COCO-17 YOLO pose 标签读写与校验、翻转镜像、bbox 自愈、`validate` 检查孤儿标签。
- 双模型共识与分歧标记、翻转 TTA、可见性三档判定、断点续跑（参数指纹失效自动重标）。
- QC 报告（`report.md` / `stats.json` / `flags.csv`）、骨架叠加图、复核优先级排序。
- 复核界面：Canvas 拖点、`0/1/2` 可见性、`Q/E/F/X/U`、缩放平移、防抖自动保存、备份与 10 级历史撤销。
- ffmpeg 抽帧（定帧率 / 场景切点）并记录 `time_sec` 回溯。
- 导出 train/val + `data.yaml`（`kpt_shape` / `flip_idx`）+ COCO keypoint JSON + manifest。
- 修复过的历史缺陷：`mirror_person` 未交换左右索引（曾污染翻转 TTA）、复核保存把 `score` 归零、导出可能产生空 val 集、界面默认筛选导致首屏空白、图片加载后不重绘。
