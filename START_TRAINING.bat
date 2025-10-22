@echo off
echo ===============================================
echo STARTING BEHAVIOR MODEL TRAINING
echo ===============================================
echo.
echo This will take 2-4 hours on CPU
echo You can close this window and training will continue
echo.
echo To check progress later, run: python check_training_progress.py
echo.
echo ===============================================
echo.

python train_behavior_model.py

echo.
echo ===============================================
echo TRAINING COMPLETE!
echo ===============================================
pause

