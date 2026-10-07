@echo off
setlocal

REM Bu dosyayı Anaconda Prompt'ta, proje kökünden calistirin:
REM   scripts\01_install_gpu_pytorch.bat

call conda activate docentlik
if errorlevel 1 (
    echo [HATA] docentlik ortami aktif edilemedi.
    exit /b 1
)

echo.
echo ================= NVIDIA SURUCU / CUDA BILGISI ================
nvidia-smi
echo ================================================================
echo.
echo Yuklu surucu CUDA 12.8 destekliyorsa 1'i, emin degilseniz 2'yi secin.
echo [1] PyTorch CUDA 12.8 wheel (onerilen; yeni NVIDIA suruculeri)
echo [2] PyTorch CUDA 12.6 wheel (uyumluluk secenegi)
set /p CUDA_CHOICE="Secim [1]: "
if "%CUDA_CHOICE%"=="" set CUDA_CHOICE=1

python -m pip uninstall -y torch torchvision torchaudio

if "%CUDA_CHOICE%"=="1" (
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
) else (
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
)

if errorlevel 1 (
    echo [HATA] CUDA destekli PyTorch kurulamadi.
    echo PyTorch'un resmi yukleme secicisinden surucunuze uygun komutu alin:
    echo https://pytorch.org/get-started/locally/
    exit /b 1
)

echo.
echo ================= PYTORCH GPU DOGRULAMA =================
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK')"
echo ===========================================================
endlocal
