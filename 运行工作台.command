#!/bin/zsh
# 双击启动「自动标注」复核工作台：起好了自动打开浏览器，关掉这个窗口 = 停掉工作台。
#
# 用法（都可以只双击，不带参数）：
#   ./运行工作台.command                              默认数据集 datasets/demo，端口 8765
#   ./运行工作台.command datasets/vidtest             换一份数据集
#   ./运行工作台.command datasets/demo --device cuda  指定只用 N 卡
#   PORT=8766 DEVICE=cuda ./运行工作台.command         老写法照样认（多卡一卡开一个窗口）
#
# 这个文件只干一件事：把自己交给 python -m autolabel launch。
# 环境自检、签本地证书、起服务、「端口真监听了才开浏览器」整套流程都在
# autolabel/launch.py 里，和 Windows 的 运行工作台.bat 共用同一份判断。
# 以前两条链路各抄一遍 zsh，日志位置、等端口、失败提示全都对不上，改一处忘另一处。
#
# 走 HTTPS：摄像头只在"安全网页"里开放，HTTP 会被浏览器关掉录像权限。
# 第一次自动签一张本地证书并加进当前用户的信任列表，不会再要管理员密码。
# 只往 PATH 前面补已知的 uv 安装位置，**保留原来的 PATH**：uv 装在别处（自定义目录、
# pyenv、conda）时才不会"明明装了却说没找到"
export PATH="$HOME/.local/bin:$HOME/Library/Python/3.9/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")" || exit 1

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
  exit 0
fi

# PORT / DEVICE / HALF 这三个环境变量是老写法，转成参数交给同一套逻辑
# ${(P)NAME} 是 zsh 的「间接取值」：NAME 里装的是变量名，要取的是它的内容。
# 少写这个 (P) 就把变量名本身当值传出去，变成 --port PORT --device DEVICE，
# 参数校验当场报错——症状是双击之后窗口一闪，什么都没起来。
ARGS=()
for PAIR in "PORT:--port" "DEVICE:--device"; do
  NAME="${PAIR%%:*}"; FLAG="${PAIR##*:}"
  VALUE="${(P)NAME}"
  [[ -n "$VALUE" ]] && ARGS+=("$FLAG" "$VALUE")
done
# HALF 是开关不是取值，写 no 就是不开——别让人写了 no 还被他吐槽开了半精度
case "${(L)HALF}" in
  yes|y|true|1|on) ARGS+=(--half) ;;
esac
if [[ -n "$1" && "${1:0:1}" != "-" ]]; then
  ARGS+=("$1")
  shift
fi
ARGS+=("$@")

# 找一个能跑 autolabel 的 Python：有 uv 就用 uv（依赖它自己会备齐），没有再退回系统 python3
BOOT=()
for CAND in "$HOME/.local/bin/uv" "$HOME/Library/Python/3.9/bin/uv" /opt/homebrew/bin/uv \
            /usr/local/bin/uv "$(command -v uv)"; do
  [[ -n "$CAND" && -x "$CAND" ]] || continue
  BOOT=("$CAND" run --offline --quiet --with numpy --with pillow --with pyyaml
        --with requests --python 3.11 python3)
  break
done

if (( ${#BOOT} )); then
  exec "${BOOT[@]}" -m autolabel launch "${ARGS[@]}"
fi
if command -v python3 >/dev/null 2>&1 &&
   python3 -c "import numpy, PIL, yaml, requests" >/dev/null 2>&1; then
  exec python3 -m autolabel launch "${ARGS[@]}"
fi
print -P "%F{red}✗%f 没找到 uv，系统 Python 也没装齐依赖，工作台起不来。"
print "  装一次 uv 以后就不用管了：curl -LsSf https://astral.sh/uv/install.sh | sh"
print "  或者手动装依赖：python3 -m pip install -r requirements.txt"
print "  Windows 上对应的是 运行工作台.bat / 后台服务.bat，逻辑同一份。"
read -r "?按回车关闭本窗口…"
exit 1
