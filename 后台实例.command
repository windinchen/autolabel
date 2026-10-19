#!/bin/zsh
# 复核工作台的后台启动器：launchd 用它起实例，你也可以在终端里直接跑。
# 与 运行工作台.command 的差别只有两条——不开浏览器、不占终端窗口；
# 起不来的原因写进日志，launchd 每 10 秒重试一次，不用有人盯着。
#
# 用法：
#   ./后台实例.command mps                          按 configs/后台实例.conf 里 mps 那行起
#   ./后台实例.command --port 8767 --dataset datasets/demo --device cuda
#                                                   不读清单，临时起一个（调试用）
#
# 为什么 plist 里不直接写本脚本的路径：这个项目目录带中文，而 launchd 把
# ProgramArguments 里的非 ASCII 路径传成乱码（/bin/zsh: can't open input file:
# /Users/…/项…注/…），任务会每 10 秒崩一次。所以 plist 只写 ~/.autolabel/services/
# <名字>.sh 这个纯 ASCII 启动片，由它再交给本脚本——所以本脚本不管被软链还是
# 副本调起来，都得按"解析后的真实位置"定位项目（${0:P:h}），dirname "$0" 会指错。
export PATH="$HOME/Library/Python/3.9/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
cd -- "${0:P:h}" || exit 1
ROOT="$PWD"
SERVICE="$ROOT/后台服务.command"
UV_BIN="$HOME/Library/Python/3.9/bin/uv"

NAME=""
PORT=""
DATASET=""
DEVICE=""
HALF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)    [[ $# -ge 2 ]] || { print -P "%F{red}✗%f --name 后面要跟实例名"; exit 2; }
               NAME="$2"; shift 2 ;;
    --port)    [[ $# -ge 2 ]] || { print -P "%F{red}✗%f --port 后面要跟端口"; exit 2; }
               PORT="$2"; shift 2 ;;
    --dataset) [[ $# -ge 2 ]] || { print -P "%F{red}✗%f --dataset 后面要跟目录"; exit 2; }
               DATASET="$2"; shift 2 ;;
    --device)  [[ $# -ge 2 ]] || { print -P "%F{red}✗%f --device 后面要跟 auto/mps/cuda"; exit 2; }
               DEVICE="$2"; shift 2 ;;
    --half)    HALF="yes"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    -*)        print -P "%F{red}✗%f 不认识的参数：$1（./后台实例.command --help 看用法）"; exit 2 ;;
    *)         NAME="$1"; shift ;;
  esac
done

fail() { print -P "%F{red}✗%f $*"; exit 1; }

# 实例名是唯一喂给 launchd 的信息：端口、数据集、算力每次启动现从清单里读，
# 所以改完 configs/后台实例.conf 只要重启就生效，不用重新注册
if [[ -n "$NAME" ]]; then
  [[ -x "$SERVICE" ]] || fail "找不到 $SERVICE，清单没法解析（项目被挪走了？）"
  if ! INST_ENV="$("$SERVICE" print-env "$NAME" 2>&1)"; then
    fail "${INST_ENV%%$'\n'*}"
  fi
  eval "$INST_ENV"
fi

[[ -n "$PORT" ]] || PORT=8765
[[ -n "$DATASET" ]] || DATASET="datasets/demo"
LOG="/tmp/autolabel-$PORT.log"
DEPS=(--with numpy --with pillow --with pyyaml --with requests)
LIGHT=(run --offline --quiet "${DEPS[@]}" --python 3.11)
FULL=(run --offline --quiet "${DEPS[@]}" --with ultralytics --python 3.11)

# 端口已经有人听着就别硬上：多半是用户自己双击开了工作台。后台这个实例再起只会
# 撞端口、非正常退出，然后被 launchd 每 10 秒拉起来撞一次，日志全是没用的报错
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  print -P "%F{yellow}!%f 端口 $PORT 已经有程序在听了（多半是工作台已经开着），这个后台实例先不起。"
  print -P "  想让后台接管：先关掉那个终端窗口，再跑 ./后台服务.command。"
  exit 0
fi

[[ -x "$UV_BIN" ]] || fail "没找到 uv（$UV_BIN），后台服务起不来。装一次：curl -LsSf https://astral.sh/uv/install.sh | sh"
[[ -d autolabel ]] || fail "脚本所在目录里没有 autolabel 代码：$ROOT"
# access() 被隐私拦截时照样回"可读"，只有真读一次才知道读不读得到
head -1 autolabel/version.py >/dev/null 2>&1 || {
  # 项目在 ~/Desktop 下时，macOS 的隐私设置可能不让后台任务读它，而且不弹窗问权限。
  # 这时候报清楚原因再正常退出（exit 0），launchd 就不会每 10 秒空转重试一次
  print -P "%F{red}✗%f 这个进程读不到项目目录：$ROOT"
  print -P "  macOS 不让后台任务读「桌面 / 文稿 / 下载」这三个文件夹，也不会弹窗问你要权限。"
  print -P "  两条路，第一条实测可靠："
  print -P "  ① 把整个项目挪出这三个文件夹（比如 ~/work/自动标注），再跑 ./后台服务.command；"
  print -P "  ② 或「系统设置 → 隐私与安全性 → 完全磁盘访问权限」里按 + 加入 /bin/zsh"
  print -P "     （列表里默认没有它，按 ⌘⇧G 手输路径）。给「终端」授权多半没用——"
  print -P "     后台任务是 launchd 直接起的，不是终端的子进程。"
  print -P "  在此之前双击 运行工作台.command 照样能用——前台进程不受这条限制。"
  exit 0
}
[[ -d "$DATASET" ]] || fail "数据集目录不存在：$DATASET（先 python3 -m autolabel init $DATASET）"

# 日志滚存：超过 2MB 先挪去 .1，免得后台跑几个月吃掉磁盘。
# 关键是启动时不清空——崩溃重试时最需要看到的正是上一次留下的那几行
if [[ -f "$LOG" ]] && (( $(stat -f%z "$LOG" 2>/dev/null || echo 0) > 2097152 )); then
  mv -f "$LOG" "$LOG.1"
fi

# 后台服务没有浏览器弹窗可看，缺证书等于 HTTPS 起不来，所以先补一张再启动
if [[ ! -f certs/autolabel.local.pem ]]; then
  print -P "%F{cyan}▸%f 缺本地证书，先签一张（细节写进 $LOG）"
  "$UV_BIN" "${LIGHT[@]}" python3 -m autolabel ca >>"$LOG" 2>&1 || true
fi

# 清单里设备留空（写成 -）就一个 --device 都不传：让「系统设置」里存的那档算力生效
EXTRA=(--no-browser)
[[ -n "$DEVICE" ]] && EXTRA+=(--device "$DEVICE")
[[ "$HALF" == "yes" || "$HALF" == "true" ]] && EXTRA+=(--half)

LABEL="后台启动"
[[ -n "$NAME" ]] && LABEL="后台启动（实例 $NAME）"
print -P "%F{cyan}▸%f $LABEL：数据集 $DATASET · 端口 $PORT · 日志 $LOG"
exec "$UV_BIN" "${FULL[@]}" python3 -m autolabel review "$DATASET" --port "$PORT" \
    "${EXTRA[@]}" >>"$LOG" 2>&1
