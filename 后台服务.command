#!/bin/zsh
# 把复核工作台注册成 macOS 后台服务：开机自起、崩了自动拉起，不占终端窗口。
#
# 用法（双击本文件＝跑第一种；带参数就在终端里跑）：
#   ./后台服务.command                    注册并启动清单里的全部实例
#   ./后台服务.command --status           谁在跑、端口、退出原因、日志尾几行
#   ./后台服务.command --restart [名字]    改完代码重启（不给名字就全部重启）
#   ./后台服务.command --remove [名字]     注销并删掉生成的 plist 与软链
#   ./后台服务.command plist <名字>        只打印这个实例的 plist（排查用）
#   ./后台服务.command print-env <名字>    只打印这个实例的参数（启动器自己用）
#   ./后台服务.command --dry-run          只把 plist 写出来，不碰 launchd
#
# 实例清单在 configs/后台实例.conf，一行一个：名字 端口 数据集 设备 半精度。
# 每个实例生成一个 ~/Library/LaunchAgents/com.<开发者>.autolabel.<名字>.plist，
# 登录后由 launchd 自己加载；启动命令指向 ~/.autolabel/services/<名字>.sh，
# 它是一根指回项目里 后台实例.command 的软链。这样绕开两个坑：启动脚本以前躺在
# /tmp，重启就被清掉，服务第二天起不来；而项目路径带中文，直接写进 plist 会被
# launchd 传成乱码，任务每 10 秒崩一次。项目放在 ~/Desktop 下还有一道 macOS
# 权限要过（默认不让后台任务读桌面），卡住时脚本会把该点哪里直接写出来。
export PATH="$HOME/Library/Python/3.9/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
cd "$(dirname "$0")" || exit 1

ROOT="$PWD"
RUNNER="$ROOT/后台实例.command"
# 清单默认在项目里，测试或第二份清单可以用 AUTOLABEL_SERVICE_CONF 指到别处
CONF="${AUTOLABEL_SERVICE_CONF:-$ROOT/configs/后台实例.conf}"
AGENT_DIR="$HOME/Library/LaunchAgents"
SHIM_DIR="$HOME/.autolabel/services"
DOMAIN="gui/$(id -u)"

say() { print -P "  $*"; }
ok() { print -P "%F{green}✓%f $*"; }
step() { print -P "%F{cyan}▸%f $*"; }
bad() { print -P "%F{red}✗%f $*"; }
warn() { print -P "%F{yellow}!%f $*"; }
bail() {
  local head="$1"; shift
  bad "$head"
  local line
  for line in "$@"; do say "$line"; done
  exit 1
}

# 标签前缀跟 autolabel/version.py 里的开发者名走，改名时后台任务不会变成孤儿
DEV_NAME="$(sed -n 's/^DEVELOPER = "\([^"]*\)".*/\1/p' autolabel/version.py 2>/dev/null | head -1)"
[[ -n "$DEV_NAME" ]] || DEV_NAME="windinchen"
# 标签前缀：最终标签是 com.<开发者>.autolabel.<实例名>
BASE="com.$DEV_NAME"

typeset -a NAMES
typeset -A PORT_OF SET_OF DEV_OF HALF_OF

label_of() { print -- "$BASE.autolabel.$1"; }

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

job_loaded() { launchctl list "$1" >/dev/null 2>&1; }

job_pid() { launchctl list "$1" 2>/dev/null | sed -n 's/^[[:space:]]*"PID" = \([0-9]*\);/\1/p'; }

ascii_only() { [[ "$1" =~ '^[ -~]*$' ]]; }

xml_text() {  # 路径里出现 & < > 也要能写进 plist
  local s="$1"
  s=${s//&/&amp;}
  s=${s//</&lt;}
  s=${s//>/&gt;}
  print -r -- "$s"
}

load_conf() {
  [[ -f "$CONF" ]] || bail "没有实例清单：$CONF" \
    "照下面这个格式建一个（一行一个实例）：" \
    "  mps 8765 datasets/demo - no" \
    "  cuda 8766 datasets/demo cuda no"
  local name port dataset dev half lineno=0
  while read -r name port dataset dev half || [[ -n "$name" ]]; do
    lineno=$((lineno + 1))
    [[ -z "$name" || "$name" == \#* ]] && continue
    [[ "$name" =~ '^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$' ]] \
      || bail "$CONF 第 $lineno 行的实例名不合规：$name" \
              "只能用字母、数字、- 和 _——它要写进 launchd 标签和软链文件名，不能带中文"
    [[ -z "$PORT_OF[$name]" ]] || bail "$CONF 第 $lineno 行实例名重复：$name"
    [[ "$port" =~ '^[0-9]{2,5}$' ]] \
      || bail "$CONF 第 $lineno 行端口不是数字：$port（实例 $name）"
    [[ -d "$dataset" ]] || bail "$CONF 第 $lineno 行的数据集不存在：$dataset（实例 $name）" \
                          "先 python3 -m autolabel init $dataset，或把这一行改成已有目录"
    [[ -z "$half" ]] && half="no"
    NAMES+=("$name")
    PORT_OF[$name]="$port"
    SET_OF[$name]="$dataset"
    DEV_OF[$name]="$([[ "$dev" == "-" ]] && print -r -- "" || print -r -- "$dev")"
    HALF_OF[$name]="$half"
  done < "$CONF"
  (( ${#NAMES} )) || bail "$CONF 里一个实例都没有" "去掉注释至少留一行：mps 8765 datasets/demo - no"
}

render_plist() {
  local name="$1"
  local label="$(label_of "$name")" port="${PORT_OF[$name]}"
  local shim="$SHIM_DIR/$name.sh"
  local log="/tmp/autolabel-$port.log"
  print -r -- '<?xml version="1.0" encoding="UTF-8"?>'
  print -r -- '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
  print -r -- '<plist version="1.0">'
  print -r -- '<dict>'
  print -r -- "  <key>Label</key><string>$(xml_text "$label")</string>"
  print -r -- '  <key>ProgramArguments</key>'
  print -r -- '  <array>'
  print -r -- '    <string>/bin/zsh</string>'
  print -r -- "    <string>$(xml_text "$shim")</string>"
  print -r -- "    <string>$(xml_text "$name")</string>"
  print -r -- '  </array>'
  print -r -- '  <key>EnvironmentVariables</key>'
  print -r -- '  <dict>'
  print -r -- "    <key>HOME</key><string>$(xml_text "$HOME")</string>"
  print -r -- '  </dict>'
  print -r -- '  <key>RunAtLoad</key><true/>'
  print -r -- '  <key>KeepAlive</key>'
  print -r -- '  <dict><key>SuccessfulExit</key><false/></dict>'
  print -r -- '  <key>ThrottleInterval</key><integer>10</integer>'
  print -r -- "  <key>StandardOutPath</key><string>$log</string>"
  print -r -- "  <key>StandardErrorPath</key><string>$log</string>"
  print -r -- '</dict>'
  print -r -- '</plist>'
}

# 软链 + plist 一起落盘，返回 plist 路径
write_files() {
  local name="$1"
  local label="$(label_of "$name")"
  local shim="$SHIM_DIR/$name.sh"
  ascii_only "$shim" || bail "后台启动路径含非 ASCII 字符：$shim" \
    "launchd 会把这种路径传成乱码，任务每 10 秒崩一次。" \
    "要么把家目录换成纯英文名，要么先别用后台服务，改双击 运行工作台.command。"
  mkdir -p "$SHIM_DIR" || bail "建不了目录：$SHIM_DIR"
  ln -sfn "$RUNNER" "$shim" || bail "建软链失败：$shim → $RUNNER"
  local tmp="$AGENT_DIR/$label.plist.tmp"
  mkdir -p "$AGENT_DIR" || bail "建不了目录：$AGENT_DIR"
  render_plist "$name" >"$tmp" 2>&1 || { rm -f "$tmp"; bail "生成 plist 失败：$label"; }
  mv -f "$tmp" "$AGENT_DIR/$label.plist"
  chmod 644 "$AGENT_DIR/$label.plist"
  print -r -- "$AGENT_DIR/$label.plist"
}

explain_down() {  # 实例没起来时把最可能的原因翻成人话，别只甩几行日志
  local port="$1"
  local log="/tmp/autolabel-$port.log"
  tail -8 "$log" 2>/dev/null | sed 's/^/    /'
  if grep -q "can't open input file" "$log" 2>/dev/null; then
    say "这不是端口被占：launchd 起的后台任务读不到 ~/Desktop 里的项目文件。"
    say "macOS 默认不让后台任务访问「桌面文件夹」，而且这类任务不会弹窗问你要权限。"
    say "两个办法，任选一个："
    say "  ① 系统设置 → 隐私与安全性 → 文件与文件夹（或「完全磁盘访问权限」）里给"
    say "     「终端」勾上「桌面文件夹」，然后在终端里重跑一次 ./后台服务.command；"
    say "  ② 把整个项目挪出 ~/Desktop（比如 ~/autolabel），再跑一次 ./后台服务.command。"
    say "权限办好之前想先用界面：双击 运行工作台.command，终端里的前台进程不受这条限制。"
  fi
}

wait_ready() {  # 端口真的监听了才算起来：不让人对着「无法访问此网站」猜
  local port="$1" tries="${2:-40}"
  local _
  for _ in {1..$tries}; do
    port_busy "$port" && return 0
    sleep 1
  done
  return 1
}

prune_stale() {  # 清单里删掉的实例，别让 launchd 按旧 plist 反复重启一个不存在的实例
  local dry="$1" keep keep_name f nm stale=0
  for f in "$AGENT_DIR"/"$BASE".autolabel.*.plist(N); do
    nm="$(basename "$f" .plist)"
    nm="${nm#${BASE}.autolabel.}"
    keep=0
    for keep_name in "${NAMES[@]}"; do [[ "$keep_name" == "$nm" ]] && keep=1; done
    (( keep )) && continue
    stale=$((stale + 1))
    if [[ -z "$dry" ]]; then
      launchctl bootout "$DOMAIN/$(label_of "$nm")" 2>/dev/null
      rm -f "$f" "$SHIM_DIR/$nm.sh"
    fi
    warn "清单里已经没有 $nm，注销并清掉它留下的 plist 和软链"
  done
  return 0
}

cmd_install() {
  local dry="$1"
  step "读实例清单 $CONF（${#NAMES} 个）"
  local name label plist port skipped=0
  for name in "${NAMES[@]}"; do
    label="$(label_of "$name")"
    port="${PORT_OF[$name]}"
    plist="$(write_files "$name")" || return 1
    if [[ -n "$dry" ]]; then
      ok "$name · 端口 $port · plist 与软链已写出（没碰 launchd）"
      say "$plist"
      continue
    fi
    if job_loaded "$label"; then
      step "$name · 先摘掉在跑的旧任务"
      launchctl bootout "$DOMAIN/$label" 2>/dev/null
      local _
      for _ in {1..10}; do port_busy "$port" || break; sleep 1; done
    fi
    if port_busy "$port"; then
      skipped=$((skipped + 1))
      warn "$name · 端口 $port 还被别的程序占着，这个实例先不起"
      say "多半是你自己双击过 运行工作台.command。要么关掉那个窗口再跑一次，"
      say "要么把 $CONF 里这一行的端口换一个。"
      continue
    fi
    launchctl bootstrap "$DOMAIN" "$plist" \
      || bail "$name · 注册失败（$label）" "launchd 拒绝了 $plist" "看看属主是不是你：ls -l $plist"
    ok "$name · 端口 $port · 已注册（登录/开机自动加载）"
  done
  prune_stale "$dry"
  if [[ -n "$dry" ]]; then
    print
    say "这是 --dry-run：launchd 没被动过，在跑的还是原来的定义。"
    return 0
  fi
  print
  step "等服务起来"
  local failed=0
  for name in "${NAMES[@]}"; do
    port="${PORT_OF[$name]}"
    if wait_ready "$port" 40; then
      ok "$name → https://localhost:$port/（数据集 ${SET_OF[$name]}）"
    elif job_loaded "$(label_of "$name")"; then
      failed=$((failed + 1))
      warn "$name · 注册上了但 40 秒没监听："
      explain_down "$port"
    else
      failed=$((failed + 1))
      warn "$name · launchd 没接这个任务，见 $plist"
    fi
  done
  print
  if (( failed )); then
    bad "有 $failed 个实例没起来，上面是原因。"
    return 1
  fi
  (( skipped )) && say "另外 $skipped 个因为端口被占跳过了，见上面。"
  ok "后台服务就位。以后开机自己就起来，不用再双击任何东西。"
}

cmd_status() {
  print -P "%B后台服务实例%b（清单 $CONF）"
  local name label port pid extra
  for name in "${NAMES[@]}"; do
    label="$(label_of "$name")"
    port="${PORT_OF[$name]}"
    extra="算力跟系统设置"
    [[ -n "${DEV_OF[$name]}" ]] && extra="算力指定为 ${DEV_OF[$name]}"
    if ! job_loaded "$label"; then
      bad "$name 未注册 · https://localhost:$port/"
      say "跑一次 ./后台服务.command 就注册上了"
      continue
    fi
    pid="$(job_pid "$label")"
    if [[ -n "$pid" ]]; then
      ok "$name 在跑 · PID $pid · https://localhost:$port/ · 数据集 ${SET_OF[$name]}"
    else
      bad "$name 已注册但没在跑 · https://localhost:$port/"
    fi
    say "$extra · 日志 /tmp/autolabel-$port.log"
    if [[ -z "$pid" ]]; then explain_down "$port"; print; fi
  done
}

cmd_restart() {
  local target="$1" name label done=0
  for name in "${NAMES[@]}"; do
    [[ -n "$target" && "$name" != "$target" ]] && continue
    label="$(label_of "$name")"
    if ! job_loaded "$label"; then
      bad "$name 还没注册，先跑 ./后台服务.command"
      continue
    fi
    if launchctl kickstart -k "$DOMAIN/$label"; then
      ok "$name 已重启（端口 ${PORT_OF[$name]}）"
      done=$((done + 1))
    else
      bad "$name 重启失败（$label）"
    fi
  done
  (( done )) || exit 1
  local port failed=0
  for name in "${NAMES[@]}"; do
    [[ -n "$target" && "$name" != "$target" ]] && continue
    port="${PORT_OF[$name]}"
    if wait_ready "$port" 40; then
      ok "$name → https://localhost:$port/ 已就绪"
    else
      warn "$name · 40 秒没监听，日志最后几行："
      tail -8 "/tmp/autolabel-$port.log" 2>/dev/null | sed 's/^/    /'
      failed=$((failed + 1))
    fi
  done
  (( failed )) && exit 1
}

cmd_remove() {
  local target="$1" name label
  for name in "${NAMES[@]}"; do
    [[ -n "$target" && "$name" != "$target" ]] && continue
    label="$(label_of "$name")"
    if launchctl bootout "$DOMAIN/$label" 2>/dev/null; then
      ok "$name 已注销"
    else
      say "$name 本来就没注册"
    fi
    rm -f "$AGENT_DIR/$label.plist" "$SHIM_DIR/$name.sh"
  done
  ok "清单里的实例都处理完了；要恢复后台服务再跑一次 ./后台服务.command"
}

cmd_print_env() {
  local name="$1"
  [[ -n "${PORT_OF[$name]}" ]] \
    || bail "清单里没有实例：$name（现有 ${#NAMES} 个：${NAMES[*]}）" \
            "改过 configs/后台实例.conf 就重跑一次 ./后台服务.command"
  printf 'PORT=%q\n' "${PORT_OF[$name]}"
  printf 'DATASET=%q\n' "${SET_OF[$name]}"
  printf 'DEVICE=%q\n' "${DEV_OF[$name]}"
  printf 'HALF=%q\n' "${HALF_OF[$name]}"
}

MODE="install"
DRY=""
TARGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --status) MODE="status"; shift ;;
    --remove) MODE="remove"; [[ $# -ge 2 && "$2" != -* ]] && TARGET="$2" && shift; shift ;;
    --restart) MODE="restart"; [[ $# -ge 2 && "$2" != -* ]] && TARGET="$2" && shift; shift ;;
    --dry-run) DRY="--dry-run"; shift ;;
    plist|print-env) MODE="$1"; [[ $# -ge 2 ]] && TARGET="$2" && shift; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) bad "不认识的参数：$1（./后台服务.command --help 看用法）"; exit 2 ;;
  esac
done

[[ -f "$RUNNER" ]] || bail "缺启动器：$RUNNER" "它应该和这个脚本一起放在项目根目录"
load_conf

case "$MODE" in
  install) cmd_install "$DRY" ;;
  status) cmd_status ;;
  restart) cmd_restart "$TARGET" ;;
  remove) cmd_remove "$TARGET" ;;
  plist)
    [[ -n "$TARGET" ]] || bail "plist 后面要跟实例名：./后台服务.command plist mps"
    [[ -n "${PORT_OF[$TARGET]}" ]] || bail "清单里没有实例：$TARGET（现有 ${#NAMES} 个：${NAMES[*]}）"
    render_plist "$TARGET"
    ;;
  print-env)
    [[ -n "$TARGET" ]] || bail "print-env 后面要跟实例名：./后台服务.command print-env mps"
    cmd_print_env "$TARGET"
    ;;
esac
