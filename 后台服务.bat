@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
rem 把复核工作台注册成 Windows 后台服务：登录自起、崩了自动拉起，不占终端窗口。
rem
rem 用法（双击本文件 = 跑第一种；在命令行里带参数）：
rem   后台服务.bat                        注册并启动清单里的全部实例
rem   后台服务.bat --status               谁在跑、端口、退出原因、日志尾几行
rem   后台服务.bat --restart [名字]        改完代码重启（不给名字就全部重启）
rem   后台服务.bat --remove [名字]        注销并删掉生成的任务定义与启动片
rem   后台服务.bat --show 名字            只打印这个实例的任务定义（排查用）
rem   后台服务.bat --dry-run             只把任务定义写出来，不碰任务计划程序
rem
rem 实例清单在 configs\后台实例.conf，一行一个：名字 端口 数据集 设备 半精度。
rem macOS 用 launchd，Windows 用任务计划程序（登录触发 + 失败自动重启，语义和
rem launchd 的 KeepAlive 一致），两边都指向 .autolabel\services\ 下的纯 ASCII 启动片——
rem 计划任务把中文路径传成乱码是老毛病。注册 / 状态 / 重启 / 注销只写一份，
rem 在 autolabel\service.py，两个系统共用。

set "UV="
for %%P in ("%USERPROFILE%\.local\bin\uv.exe" "%LOCALAPPDATA%\Programs\uv\uv.exe" "%USERPROFILE%\.cargo\bin\uv.exe") do call :find_uv "%%~P"
if not defined UV for /f "delims=" %%I in ('where uv 2^>nul') do call :find_uv "%%~I"

if defined UV goto :with_uv
echo.
echo ! 没找到 uv，改用系统 Python。依赖要自己装好：python -m pip install -r requirements.txt
echo   想省事就装一次 uv：winget install astral-sh.uv
echo.
python -m autolabel service %*
goto :end

:with_uv
"%UV%" run --offline --quiet --with numpy --with pillow --with pyyaml --with requests --python 3.11 python3 -m autolabel service %*

:end
echo.
echo 上面就是结果。注册成功以后登录自己就起来，崩了自动拉起，不用再双击任何东西。
pause
exit /b 0

:find_uv
rem uv 是本项目用来备 Python 环境的工具；Windows 上装完常常不在 PATH 里，得自己找
if not defined UV if exist "%~1" set "UV=%~1"
exit /b 0
