import importlib.util
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys


def run(command: list[str]) -> str:
    if shutil.which(command[0]) is None:
        return "not found"

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "timed out"
    return result.stdout.strip() or result.stderr.strip()


def installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


print("OS:", platform.platform())
print("kernel:", platform.release())
print("Python:", sys.version.replace("\n", " "))
active_venv = os.getenv("VIRTUAL_ENV")
if active_venv is None and sys.prefix != sys.base_prefix:
    active_venv = sys.prefix
print("venv:", active_venv or "not active")
print("pip:", installed_version("pip"))

for name, command in {
    "uv": ["uv", "--version"],
    "git": ["git", "--version"],
    "gcc": ["gcc", "--version"],
    "docker": ["docker", "--version"],
    "docker daemon": ["docker", "info", "--format", "{{.ServerVersion}}"],
    "ollama": ["ollama", "--version"],
    "nvcc": ["nvcc", "--version"],
    "nvidia-smi": [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ],
}.items():
    print(f"{name}:", run(command))

packages = [
    "torch",
    "transformers",
    "FlagEmbedding",
    "scipy",
    "pandas",
    "pyarrow",
    "tqdm",
    "yaml",
    "ollama",
    "requests",
]
print("packages:")
for package in packages:
    print(f"  {package}:", "yes" if importlib.util.find_spec(package) else "no")

if importlib.util.find_spec("torch"):
    import torch

    print("torch version:", torch.__version__)
    print("torch CUDA runtime:", torch.version.cuda)
    print("torch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch device:", torch.cuda.get_device_name(0))
