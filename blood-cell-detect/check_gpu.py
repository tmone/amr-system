import torch
import sys
import subprocess

def check_cuda_installation():
    print("=== Python Environment ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    print("\n=== CUDA Environment ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode('utf-8')
        print("\n=== NVIDIA-SMI ===")
        print(nvidia_smi)
    except:
        print("\nNVIDIA-SMI not found. Please ensure NVIDIA drivers are installed.")
    
    print("\n=== Troubleshooting Steps ===")
    if not torch.cuda.is_available():
        print("1. Check NVIDIA drivers are installed")
        print("2. Verify CUDA toolkit is installed")
        print("3. Ensure PyTorch CUDA version matches your CUDA installation")
        print("4. Try reinstalling PyTorch with CUDA support:")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    check_cuda_installation()
