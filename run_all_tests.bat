@echo off
echo =========================================
echo 🧪 OLFACTURA TEST ^& EVALUATION SUITE
echo =========================================

echo.
echo [1/2] RUNNING UNIT ^& INTEGRATION TESTS...
echo -----------------------------------------
python -m pytest tests/ -v
if %errorlevel% neq 0 (
    echo.
    echo ❌ Unit tests failed! Please check the output above.
    exit /b %errorlevel%
)
echo ✅ Unit tests passed successfully.

echo.
echo [2/2] RUNNING RAG AUTOMATED EVALUATION...
echo -----------------------------------------
python evaluation/run_evals.py

echo.
echo =========================================
echo 🎉 ALL CHECKS COMPLETED!
echo =========================================
pause
