#!/bin/zsh
# 把复核工作台注册成系统后台服务：登录自起、崩了自动拉起，不占终端窗口。
#
# 用法（双击本文件 = 跑第一种；带参数就在终端里跑）：
#   ./后台服务.command                       注册并启动清单里的全部实例
#   ./后台服务.command --status              谁在跑、端口、退出原因、日志尾几行
#   ./后台服务.command --restart [名字]       改完代码重启（不给名字就全部重启）
#   ./后台服务.command --remove [名字]        注销并删掉生成的任务定义与启动片
#   ./后台服务.command --show <名字>          只打印这个实例的任务定义（排查用）
#   ./后台服务.command --print-env <名字>     只打印这个实例的参数（启动片自己用）
#   ./后台服务.command --dry-run             只把任务定义写出来，不碰服务管理器
#
# 实例清单在 configs/后台实例.conf，一行一个：名字 端口 数据集 设备 半精度。
# macOS 用 launchd（~/Library/LaunchAgents 里一个 plist），Windows 用任务计划程序，
# 两边都指向 ~/.autolabel/services/<名字> 这个纯 ASCII 的小启动片——服务管理器把
# 中文路径传成乱码是老毛病，所以定义里绝不出现中文。注册 / 状态 / 重启 / 注销的
# 流程只写一份，在 autolabel/service.py，两个系统共用。
# 只往 PATH 前面补已知的 uv 安装位置，**保留原来的 PATH**：uv 装在别处（自定义目录、
# pyenv、conda）时才不会"明明装了却说没找到"
export PATH="$HOME/.local/bin:$HOME/Library/Python/3.9/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")" || exit 1

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
  exit 0
fi

# 老写法 plist <名字> / print-env <名字> 照样认，换成现在这两个参数
ARGS=("$@")
case "${ARGS[1]}" in
  plist)     ARGS[1]="--show" ;;
  print-env) ARGS[1]="--print-env" ;;
esac

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
  exec "${BOOT[@]}" -m autolabel service "${ARGS[@]}"
fi
if command -v python3 >/dev/null 2>&1 &&
   python3 -c "import numpy, PIL, yaml, requests" >/dev/null 2>&1; then
  exec python3 -m autolabel service "${ARGS[@]}"
fi
print -P "%F{red}✗%f 没找到 uv，系统 Python 也没装齐依赖，工作台起不来。"
print "  装一次 uv 以后就不用管了：curl -LsSf https://astral.sh/uv/install.sh | sh"
print "  或者手动装依赖：python3 -m pip install -r requirements.txt"
print "  Windows 上对应的是 运行工作台.bat / 后台服务.bat，逻辑同一份。"
read -r "?按回车关闭本窗口…"
exit 1
