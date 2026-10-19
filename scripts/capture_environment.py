"""Record versions and compute backend without exporting paths or credentials."""
import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def capture():
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name and name.lower().replace("_", "-") != "eeg-siamese":
            packages[name.lower().replace("_", "-")] = dist.version
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Current environment snapshot; not a historical training lockfile",
        "python": platform.python_version(),
        "system": platform.system(),
        "system_release": platform.release(),
        "architecture": platform.machine(),
        "packages": dict(sorted(packages.items())),
    }
    try:
        import torch
        result["compute"] = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        result["compute"] = {"torch": None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = capture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("Environment snapshot saved.")


if __name__ == "__main__":
    main()
