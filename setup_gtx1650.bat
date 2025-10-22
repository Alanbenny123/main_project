@echo off
echo ===============================================
echo GTX 1650 GPU Setup for Training
echo ===============================================
echo.
echo Step 1: Uninstalling CPU-only PyTorch...
pip uninstall torch torchvision -y

echo.
echo Step 2: Installing CUDA PyTorch (for GTX 1650)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo Step 3: Verifying GPU detection...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"

echo.
echo ===============================================
echo Setup Complete!
echo ===============================================
pause

