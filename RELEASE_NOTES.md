## 人体骨骼自动标注系统 v1.1

按「这软件自己讲不讲理」把读写路径过了一遍，修掉四处会真咬人的问题，另给队列缩略图开了一条小图通道。

**测试 304 → 318**（本机实测 `318 passed in 45.77s`，`ruff check` 全绿）。上一次 Release 里写的 303 项这里一并更正。

---

### ① 导出切分不稳定：同一帧既训练又算指标

原来 `random.Random(seed).shuffle(整表)` 再取前 N 张当 val——**名次取决于当次一共有多少张**。

实测：3 张时 `b.jpg` 在 val，后来补到 12 张重导，它掉进了 train。同一帧既参与训练又参与验证，两次导出的指标根本不可比，`--val-ratio` 也会跟着飘。

现在改成按 `(seed, 相对路径)` 的 SHA-256 分桶（`export.split_val_set()`）：**每张图的归属只跟它自己有关**，数据集怎么长，老验证集的成员都不动。换 `seed` 仍会换一批人，但换完之后依旧稳定可复现。

「至少留 1 张验证图」「验证集不许吃光训练集」两条老保证保留。

### ② 超限上传：客户端看到的是断链

`_handle_upload` 在 `Content-Length` 超上限时**先回 413 再断开**，请求体一个字节都没读。HTTP/1.1 复用连接时客户端还在往里塞剩下几 MB，实测收到的是 `URLError: [Err32] Broken pipe`，界面上只剩「网络错误」四个字。

同时正常路径 `rfile.read(length)` 整份进内存，默认上限 1024 MB 意味着一个上传能顶出 1 GB 瞬时内存、跟 torch 抢地方。

新增 `upload.save_upload_stream()`：按 1 MB 分块边读边写，**超限也先把 `Content-Length` 字节读干净**才回 413（半截文件删掉），收不满就报「上传中断：收到 X / Y 字节」。现在浏览器拿到的是「单个文件超过 1024 MB 上限」。

### ③ ffmpeg / ffprobe 没有超时

`probe_video()` 与 `_run_ffmpeg()` 的 `subprocess.run` 都不带 `timeout`——一个卡死的 ffmpeg（损坏文件、挂载盘失联）会**永久占住后台任务线程**，界面上一路转圈且没有报错。

- **ffprobe** 给 60 秒，超时只降级成「元信息未知」，不中断任务
- **抽帧** 按片段长度推导：`min(3600, max(300, 300 + 片段秒数 × 4))`，`AUTOLABEL_FFMPEG_TIMEOUT` 可整体覆盖
- 超时抛「抽帧超过 N 秒被中止，视频可能损坏或过长」并附环境变量提示
- 两处都补了 `stdin=DEVNULL`，免得 ffmpeg 等标准输入

### ④ 索引的读-改-写不在同一把锁里

`write_item` / `set_status` / `write_prompt_item` / `undo` 等 6 处是「`reload()`（拿锁→放锁）→ 算 → `with lock:` 改+存」。两次拿锁之间别人改过盘，手里的内存副本就成了旧值，而索引是**整表覆盖写**、没有字段级合并，后写的会把对方整批抹掉。

实测：一张图在写标签时删掉另外一批图，**删掉的 20 张会连同整批删除一起被旧副本盖回来**（不是只复活那一张）。

新增 `ReviewStore._mutate()`，把「对齐磁盘 → 改 → 落盘」收进同一把锁；删除 / 认领 / 删字段这三处直接改盘的助手也挤进来（`_mutate(resync=True)`）。`ReviewStore.lock` 换 `RLock` 以允许同线程重入。锁序固定为 `ReviewStore.lock → batches.lock`。

### ⑤ 队列缩略图拉原图

焦点队列一次渲染 121 张，每张都拉全分辨率原图，而且所有响应都是 `Cache-Control: no-store`，翻页就重传。

新增 `thumbs.py` + `GET /thumb/<相对路径>`：

- 长边 320 的 JPEG（质量 72）
- 文件名按 `(相对路径, mtime_ns, 字节数)` 做指纹，**内容一换 URL 就变**，所以敢给 `max-age=604800` 一周长缓存
- 两个标注页的队列改拉 `/thumb/`，主图仍走 `/image/` 全分辨率（同名重导不能看到旧图）
- 坏图 / 缺 Pillow 一律回退原图而不是 500
- `.autolabel/thumbs/` 定位成可随手删的缓存，服务启动时清掉 30 天没再生成的

> **一个环境教训**：缩略图第一版直接 `Image.open()`，坏图那条测试卡了十几秒——ultralytics 会给 PIL 挂懒加载插件，认不出的文件触发它**联网装 `pi-heif`**，图片请求线程就挂在网上了。改成先读 16 字节文件头（JPEG/PNG/BMP/WEBP）再决定要不要交给 PIL。

---

### 防回归测试

| 文件 | 新增 |
| --- | --- |
| `tests/test_review_export.py` | +2 切分（40→100 张验证集成员不搬家、换种子换人但可复现）、+1 并发（写标签时批量删除不复活） |
| `tests/test_upload.py` | +3（4 MB 超限拿到 413 而非断链且不留半截文件、多 MB 上传字节数分毫不差、类型不支持也要把请求体读干净） |
| `tests/test_video_timeout.py` | 新建 4 条，用假 ffmpeg/ffprobe 二进制验证超时真的会中止 |
| `tests/test_thumbs.py` | 新建 5 条（尺寸与长缓存、缓存复用与变更后重生成、坏图回退、过期清理、两页确实改拉 `/thumb/`） |

### 已知边界

后台标注任务写索引走的是 `pipeline` / `prompt` 自己的 `load_index → save`，不持 `ReviewStore.lock`（持锁会横跨整个推理过程，界面会被锁死几秒）。任务运行期间恰好删图仍有小概率互相覆盖，`_mutate()` 已把窗口从「读索引到写盘的全部计算时间」压到「写盘那几毫秒」；批次表那侧本来就有 `flock` 跨进程保护。记进 `demo_docs/06-架构与数据格式.md`。

---

### 安装与升级

```bash
git clone https://github.com/windinchen/autolabel.git
cd autolabel
python3 -m pip install -r requirements.txt
```

已在用的直接 `git pull` 即可，数据格式未变（无需 `migrate`）。

**模型权重不随仓库分发**，首次使用先下载：

```bash
python3 -m autolabel models --browse
python3 -m autolabel models --download-weight yolo11n-pose.pt
```

### 校验

本机（macOS / Apple Silicon / Python 3.11）实测：

```
318 passed in 45.77s
ruff check --select E4,E7,E9,F  →  All checks passed
```

完整变更细节见 `demo_docs/CHANGELOG.md` 2026-10-18 条目。
