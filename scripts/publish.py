"""Explicit release helper; credentials come only from gh/hf login or environment."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--github", help="owner/repository; creates a new public repository")
    p.add_argument("--space", help="owner/space; uploads to a new or existing Space")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.github and not args.space:
        p.error("Specify --github and/or --space")
    if args.github:
        subprocess.run(["gh", "auth", "status"], check=True)
        if not (root / ".git").exists():
            subprocess.run(["git", "init", "-b", "main"], cwd=root, check=True)
        subprocess.run(["git", "add", "."], cwd=root, check=True)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root)
        if changed.returncode == 1:
            subprocess.run(
                ["git", "commit", "-m", "Release CSFSM Guardrails 0.1.0"], cwd=root, check=True
            )
        subprocess.run(
            [
                "gh",
                "repo",
                "create",
                args.github,
                "--public",
                "--source",
                str(root),
                "--remote",
                "origin",
                "--push",
                "--description",
                "Conversational stochastic finite-state guardrails: Python SDK, REST API and CPU Gradio Space",
            ],
            check=True,
        )
    if args.space:
        from huggingface_hub import HfApi

        target = root / "build/hf-space"
        subprocess.run(
            [sys.executable, str(root / "scripts/build_space.py"), "--output", str(target)],
            check=True,
        )
        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=str(target),
            commit_message="Release CSFSM Guardrails 0.1.0",
        )
        print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
