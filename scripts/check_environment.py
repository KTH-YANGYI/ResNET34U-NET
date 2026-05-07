import argparse
import importlib
import os
import platform
import sys


REQUIRED_MODULES = [
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("numpy", "NumPy"),
    ("PIL", "Pillow"),
]

OPTIONAL_MODULES = [
    ("yaml", "PyYAML, optional because configs have a built-in fallback parser"),
    ("tqdm", "tqdm, optional progress bars"),
    ("scipy", "SciPy, faster connected components"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Check Python environment for UNET two-stage training")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when required packages or CUDA are missing")
    parser.add_argument("--no-cuda-required", action="store_true", help="Do not fail strict mode when CUDA is unavailable")
    return parser.parse_args()


def import_module(module_name):
    try:
        module = importlib.import_module(module_name)
        return module, None
    except Exception as exc:
        return None, exc


def module_version(module):
    for attr in ["__version__", "VERSION"]:
        if hasattr(module, attr):
            return str(getattr(module, attr))
    return "unknown"


def check_modules(items):
    results = []
    for module_name, label in items:
        module, exc = import_module(module_name)
        if module is None:
            results.append(
                {
                    "module": module_name,
                    "label": label,
                    "ok": False,
                    "version": "",
                    "error": str(exc),
                }
            )
        else:
            results.append(
                {
                    "module": module_name,
                    "label": label,
                    "ok": True,
                    "version": module_version(module),
                    "error": "",
                }
            )
    return results


def print_module_results(title, results):
    print(title)
    for item in results:
        status = "OK" if item["ok"] else "MISSING"
        version = f" {item['version']}" if item["version"] else ""
        print(f"  [{status}] {item['module']} ({item['label']}){version}")
        if item["error"]:
            print(f"         {item['error']}")


def check_cuda():
    torch, exc = import_module("torch")
    if torch is None:
        return {"ok": False, "error": str(exc), "device_count": 0, "devices": []}

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = []
    for index in range(device_count):
        try:
            devices.append(torch.cuda.get_device_name(index))
        except Exception:
            devices.append("unknown")

    return {
        "ok": cuda_available,
        "error": "",
        "device_count": device_count,
        "devices": devices,
    }


def main():
    args = parse_args()

    print("System")
    print(f"  python: {sys.version.replace(os.linesep, ' ')}")
    print(f"  executable: {sys.executable}")
    print(f"  platform: {platform.platform()}")
    print(f"  hostname: {platform.node()}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    print(f"  SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', '')}")

    required_results = check_modules(REQUIRED_MODULES)
    optional_results = check_modules(OPTIONAL_MODULES)
    print_module_results("Required Python modules", required_results)
    print_module_results("Optional Python modules", optional_results)

    cuda = check_cuda()
    print("CUDA")
    print(f"  available: {cuda['ok']}")
    print(f"  device_count: {cuda['device_count']}")
    for index, name in enumerate(cuda["devices"]):
        print(f"  device {index}: {name}")
    if cuda["error"]:
        print(f"  error: {cuda['error']}")

    missing_required = [item["module"] for item in required_results if not item["ok"]]
    should_fail_cuda = args.strict and not args.no_cuda_required and not cuda["ok"]

    if missing_required:
        print(f"Missing required modules: {', '.join(missing_required)}")

    if args.strict and (missing_required or should_fail_cuda):
        raise SystemExit(1)

    print("Environment check finished.")


if __name__ == "__main__":
    main()
