"""Create a self-contained Hugging Face Gradio Space repository."""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="build/hf-space")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = Path(args.output).resolve()
    if target == root or target in root.parents or target == root / "src":
        raise ValueError("Choose a separate output directory")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        root / "src/csfsm_guardrails",
        target / "csfsm_guardrails",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ["README.md", "requirements.txt"]:
        shutil.copy2(root / "space" / name, target / name)
    for name in ["app.py", "LICENSE", "NOTICE"]:
        shutil.copy2(root / name, target / name)
    print(target)


if __name__ == "__main__":
    main()
