@echo off
chcp 65001 > nul
echo ==========================================
echo   KHOI DONG HE THONG (NEW)
echo ==========================================

echo [INFO] Dang cai dat cac thu vien can thiet...
python -m pip install -r requirements.txt

echo.
echo [INFO] Dang khoi dong Streamlit App...
python -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Co loi xay ra roi!
    pause
    exit /b %errorlevel%
)

pause
