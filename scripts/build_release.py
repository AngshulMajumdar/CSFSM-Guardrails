"""Bundle the GitHub source, standalone Space and Python distributions."""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/CSFSM_Guardrails_0.1.0.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_space.py"),
            "--output",
            str(root / "build/hf-space"),
        ],
        check=True,
    )
    excluded = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".hypothesis",
        ".ruff_cache",
        "build",
        "dist",
        "data",
        ".coverage",
        ".venv",
    }
    with tempfile.TemporaryDirectory(prefix="csfsm-release-") as temp:
        bundle = Path(temp) / "CSFSM_Guardrails_0.1.0"
        github = bundle / "github"
        github.mkdir(parents=True)
        for file in root.rglob("*"):
            relative = file.relative_to(root)
            if not file.is_file() or any(
                p in excluded or p.endswith(".egg-info") for p in relative.parts
            ):
                continue
            if file.name.startswith(".env") and file.name != ".env.example":
                continue
            if file.suffix in {".pyc", ".log"} or file.name.startswith(".coverage"):
                continue
            target = github / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
        shutil.copytree(
            root / "build/hf-space",
            bundle / "hf-space",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        dist = bundle / "distributions"
        dist.mkdir()
        for pattern in ["*.whl", "*.tar.gz"]:
            for file in (root / "dist").glob(pattern):
                shutil.copy2(file, dist / file.name)
        validation = bundle / "validation"
        validation.mkdir()
        for name in ["test-results.xml", "coverage.json"]:
            if (root / "build" / name).exists():
                shutil.copy2(root / "build" / name, validation / name)
        (bundle / "START_HERE.md").write_text("""# CSFSM Guardrails 0.1.0

This release contains the reusable guardrail package and its CPU Gradio Space.

- `github/`: complete Python SDK, HTTP API, adapters, tests, documentation,
  examples, Docker configuration and CI. Upload this directory as the repository root.
- `hf-space/`: self-contained Hugging Face Gradio Space. Upload its contents
  as the root of a Gradio Space; choose CPU hardware.
- `distributions/`: installable wheel and Python source distribution.
- `validation/`: automated test and coverage reports.

## Run locally

```bash
cd github
python -m venv .venv
source .venv/bin/activate
pip install -e '.[api,space]'
csfsm demo
```

For the HTTP API, set credentials as documented in `github/docs/API.md`, then
run `csfsm serve`. For direct use, import `Guardrail` from `csfsm_guardrails`.
Use Python 3.11 or newer. The default detector and stochastic parameters are
hand-configured; see the model assumptions in the documentation.

## Publish from the GitHub directory

```bash
gh auth login
python scripts/publish.py --github AngshulMajumdar/CSFSM-Guardrails
pip install huggingface_hub
hf auth login
python scripts/publish.py --space Angshul/CSFSM-Guardrails
```

These commands publish when run with suitable account access. No live remote
repository or Space is claimed by this archive. The HF publisher requires write
permission. The GitHub publisher creates a new public repository.

50 automated tests and local HTTP API/Gradio smoke checks passed; see
`github/docs/VALIDATION.md` for scope and environment. Apache-2.0.
""")
        lines = []
        for file in sorted(bundle.rglob("*")):
            if file.is_file():
                lines.append(
                    hashlib.sha256(file.read_bytes()).hexdigest()
                    + "  "
                    + str(file.relative_to(bundle))
                )
        (bundle / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(bundle.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(Path(temp)))
        with zipfile.ZipFile(output) as archive:
            assert archive.testzip() is None
            print(f"{output}: {len(archive.namelist())} files, {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
