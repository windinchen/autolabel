# 自动标注系统 · 文档集（姿态 · 提示词 · 目标检测）

给一堆图片（或一段视频），用现成的姿态模型批量生成人体骨骼关键点标签，人工只在浏览器里修正少数可疑帧，最后同时产出两样东西：可直接 `yolo pose train` 的数据集，以及带像素/归一化坐标的 CSV / JSONL 文件。

另两条链路出的是检测框：**提示词标注**用一句中文描述让多模态大模型逐张图找物品（类别随你写、可联网、慢），**目标检测**让本机 YOLO 权重按它自带的类别表出框（不联网、单张几十毫秒、类别由权重定），复核后都导出 `yolo detect train` 数据集（见 `09-提示词标注.md`、`11-目标检测.md`）。选哪个：权重认识的东西走目标检测，权重不认识的东西走提示词标注。

三条链路是**三个独立的标注库**——各自一套索引、批次表、标签目录、原件与撤销快照，删图与复核互不影响，只有 `images/` 素材池共用一份（同一帧只落盘一次，所有引用方都不要了才删）。命令行用 `--lib pose|objects|detect` 选库，界面上每个模块有自己的上传口与「从素材池添加」（这就是各库的独立素材库），主页统一管三库的批次。老数据集用 `autolabel migrate <数据集>` 一键拆分。

- 生成时间：2026-10-18 12:50（Asia/Shanghai）
- 代码位置：`../autolabel/`
- 根目录速览：`../README.md`
- 本目录是完整文档，与代码同步维护；改代码请同步改这里

## 文档索引

| 文档 | 内容 |
| --- | --- |
| `01-快速上手.md` | 环境要求、一条命令跑完、分步流程、开始训练 |
| `02-命令行参考.md` | 19 个子命令的用途与全部参数、目标检测命令（含自动生成的帮助快照）、多模态大模型配置与 `--vlm` 标注、提示词标注命令（含自动生成的帮助快照） |
| `03-坐标文件说明.md` | CSV / 长表 / JSONL 字段字典、可见性语义、pandas 读取示例、YOLO 标准训练集导出对照 |
| `04-复核界面指南.md` | **一套壳 · 五个页面（左侧导航条 + 顶栏 `⌘K` 与 `⋯`）**、**界面骨架（顶栏胶囊 / 竖排焦点队列 / 画布 HUD 与撤销条 / 待办清单 / 工作台行动流）**、快捷键与命令面板、**外观（白天 / 夜晚 · 四档字号）**、上传面板、坐标下载、YOLO 训练集导出与「上次导出」路径、批次管理、提示词标注页操作、模型管理（模型目录 / 权重文件 / 多模态配置）、撤销与备份、启动对账 |
| `05-视频与批量导入.md` | 抽帧策略、去重、时间戳回溯、大小上限 |
| `06-架构与数据格式.md` | 模块划分、索引与批次结构、标签格式、模型管理、后台任务与并发策略（索引写入的锁范围与已知边界）、HTTP 接口 |
| `07-常见问题.md` | FAQ 与踩坑记录 |
| `08-模型管理.md` | **模型搜索（本机目录 + 联网搜 GitHub）**、可下载模型目录与中文说明、按任务筛选（姿态/检测/分割/旋转框/分类）、下载记录、备注说明、默认模型与「设为检测默认」、本地登记、兼容性判定、Key 存放 |
| `11-目标检测.md` | 非姿态的目标检测标注：选权重 → 勾它自带的类别 → 批量出框 → 复核 → 导出；三库分工表、类别号规则（同名不换号 / 删号不回收 / 允许词间空格）、`/detect` 页操作、`autolabel detect` 全参数、独立素材库、产物与接口 |
| `09-提示词标注.md` | 字段与描述提示词怎么写、批量标注、删错框与补画漏框（焦点队列 / HUD / 撤销条一套壳）、质量标记、实际提示词、坐标与 detection 训练集导出 |
| `CHANGELOG.md` | 已实现能力与本次新增 |

方案与实施记录：

| 文档 | 内容 |
| --- | --- |
| `10-UI交互重构方案.md` | 界面现状诊断（逐条对应代码行）、应用壳信息架构、五个视图设计、命令面板与统一快捷键表、术语统一表、接口增量、P0~P3 分期与验收标准、拍板结果；**P0 已落地**等小节记录实际改动、四处回归修复、视觉方向定稿（石墨骨架 + 编号待办）与外观层（白天 / 夜晚 · 字号）的取舍 |

## 30 秒版本

```bash
# 一条命令：视频/图片入库 → 自动标注 → QC 报告 → 坐标文件
python3 -m autolabel auto datasets/myset 录像.mp4 ~/Pictures/人像 --fps 0.5

# 浏览器复核（右上角导入素材；导出在 `⋯` 菜单，或按 `⌘K` 搜「导出」）
python3 -m autolabel review datasets/myset

# 导出训练集
python3 -m autolabel export datasets/myset --coco --coords

# 提示词标注：先建字段 → 素材进提示词库 → 试跑 20 张 → 复核后出检测框训练集
python3 -m autolabel fields datasets/myset --add "bottle=桌上或手里拿着的塑料水瓶，含纸杯、保温杯"
python3 -m autolabel pool    datasets/myset --lib objects --add images/a.jpg   # 姿态库传过的素材直接复用，不重传
python3 -m autolabel prompt  datasets/myset --vlm 本地\ Ollama --limit 20
python3 -m autolabel prompt  datasets/myset --coords --export

# 目标检测：本机 YOLO 权重按自带类别出框（先挑权重 → 勾类别 → 批量 → 复核 → 导出）
python3 -m autolabel import  datasets/myset ~/Pictures/货架 --lib detect
python3 -m autolabel detect  datasets/myset --model-classes yolo11n.pt
python3 -m autolabel detect  datasets/myset --model yolo11n.pt --add-classes bus,person --coords
python3 -m autolabel detect  datasets/myset --export --out datasets/myset/exports/goods
```

## 核心结论（先看这几条）

- 标签是 YOLO pose 文本格式：`class cx cy w h (x y v)*17`，坐标归一化。
- ultralytics 的可见性判定是 `kpt_mask = v != 0`，所以 **`v=1`（遮挡）同样参与训练监督**；只想训高置信点就把 `v` 设成 0。
- 空标签文件代表背景图（负样本），导出默认把背景图比例压到 15% 以内（`--max-empty-ratio`）。
- 上传的临时文件落在 `datasets/<名称>/.autolabel/uploads/`，任务成功后自动清理。
- 后台标注与人工复核同时发生时，落盘采用"重新读盘 + 按图合并"，不会整体覆盖索引。
- 权重与多模态大模型都在界面「模型管理」（快捷键 `M`）或 `autolabel models` 里管：`--browse` 浏览 13 档可下载模型的中文说明，`--note ID=说明` 写备注，`--default` 设默认，`--register` 登记仓库外的权重，详见 `08-模型管理.md`。
- 本机状态分三份：权重在 `models/`，登记表（备注、默认、下载记录）在 `configs/models.json`，API Key 明文在 `~/.autolabel/secrets.json`；`configs/vlm.json` 里只有掩码与引用名，贴给别人看不会泄 Key。
- 配好的多模态大模型能直接当标注后端：上传弹层的模型下拉框选它，或命令行 `--vlm <配置名>`。约 25~40 秒/张、精度不如 YOLO-pose，结果一律打 `vlm_labeled` 进人工复核清单。
- 三个库三套产物，互不覆盖：`labels/` + `.autolabel/index.json` + `export`（`yolo pose train`）、`prompt_labels/` + `.autolabel/objects/index.json` + `prompt --export`、`detect_labels/` + `.autolabel/detect/index.json` + `detect --export`（都是 `yolo detect train`）。索引键按库分前缀（`status/objects/flags` 与 `prompt_*` 与 `detect_*`），接受驳回、撤销、删除都互不影响。
- 目标检测的类别号由**本库登记表** `.autolabel/detect/classes.json` 决定，不是模型自带的号：**同名类别永不换号**（换权重不会让老框指错东西）、删掉的号不回收、只登记这个权重真的报得出的类别。COCO 里 `potted plant` 这类带空格的名字照收（提示词库仍不允许空格）。
- 素材池 `images/` 三库共用：跨库导入命中已有指纹就复用（输出里的「复用素材池 N」），素材要等所有引用它的库都不再引用才真删。按张删默认只摘本库引用，`--mode purge` 或界面勾「连素材文件一起删」才连文件一起清。
- 类别/字段的 `class_id` 分配后永不复用（防止删字段后历史标签串类），导出时自动重映射成连续 `0..N-1` 并在 `data.yaml` 注释里写明。
- 复核服务启动时先按库 `migrate`（老布局自动升级，幂等），再逐库对账：中断导入留下的图片补回它自己库的索引，无人认领的"标注中"批次判定为已中断；某库在标注时另一个库不会抢走它的孤儿素材。
