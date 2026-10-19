@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
rem 双击启动「自动标注」复核工作台：起好了自动打开浏览器，关掉这个窗口 = 停掉工作台。
rem
rem 用法（也可以只双击，不带参数）：
rem   运行工作台.bat                                  默认数据集 datasets\demo，端口 8765
rem   运行工作台.bat datasets\vidtest                 换一份数据集
rem   运行工作台.bat datasets\demo --device cuda      指定只用 N 卡
rem   set PORT=8766 && 运行工作台.bat                老写法照样认（PORT / DEVICE / HALF）
rem
rem 这个文件只干一件事：把自己交给 python -m autolabel launch。环境自检、签本地证书、
rem 起服务、等端口真监听了再开浏览器，整套判断都在 autolabel\launch.py 里，和 macOS
rem 的 运行工作台.command 共用同一份逻辑，不会改一处忘另一处。
rem
rem 走 HTTPS：摄像头只在"安全网页"里开放，HTTP 会被浏览器关掉录像权限。
rem 第一次自动签一张本地证书，并引导你把它加进当前用户的受信任根证书（不用管理员）。

rem PORT / DEVICE / HALF 这三个环境变量是老写法（一卡开一个窗口用得上），转成参数
set "EXTRA="
if defined PORT set "EXTRA=%EXTRA% --port %PORT%"
if defined DEVICE set "EXTRA=%EXTRA% --device %DEVICE%"
if defined HALF if not "%HALF%"=="no" set "EXTRA=%EXTRA% --half"

set "UV="
for %%P in ("%USERPROFILE%\.local\bin\uv.exe" "%LOCALAPPDATA%\Programs\uv\uv.exe" "%USERPROFILE%\.cargo\bin\uv.exe") do call :find_uv "%%~P"
if not defined UV for /f "delims=" %%I in ('where uv 2^>nul') do call :find_uv "%%~I"

if defined UV goto :with_uv
echo.
echo ! 没找到 uv，改用系统 Python。依赖要自己装好：python -m pip install -r requirements.txt
echo   想省事就装一次 uv：winget install astral-sh.uv
echo.
python -m autolabel launch %EXTRA% %*
goto :end

:with_uv
"%UV%" run --offline --quiet --with numpy --with pillow --with pyyaml --with requests --python 3.11 python3 -m autolabel launch %EXTRA% %*

:end
echo.
echo 关掉这个窗口就是停掉服务。想让它登录自己起、不占窗口：双击 后台服务.bat
pause
exit /b 0

:find_uv
rem uv 是本项目用来备 Python 环境的工具；Windows 上装完常常不在 PATH 里，得自己找
if not defined UV if exist "%~1" set "UV=%~1"
exit /b 0
