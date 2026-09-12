@echo off
chcp 65001 >nul
title 个人财富管理配置器
cd /d "%~dp0"

rem 使用 Kimi Work 自带的 Python 环境
set "PY=D:\KimiData\daimon-share\daimon\runtime\python\.venv\Scripts\python.exe"

rem 如果 8501 端口被上次残留的进程占用，先释放
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F /T >nul 2>&1
)

echo 正在启动应用，请稍候……
echo 启动后会自动打开浏览器；关闭本窗口即可停止应用。
echo.

rem 8 秒后在默认浏览器打开「财富方案推荐」页面
start "" cmd /c "timeout /t 8 /nobreak >nul & start """" "http://localhost:8501/?page=plan""

"%PY%" -m streamlit run app.py --server.port 8501 --server.address localhost --server.headless true --browser.gatherUsageStats false

pause
