"""Exercise the running API and Gradio Space over real local HTTP."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from gradio_client import Client


def main():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(
        CSFSM_ALLOW_ANONYMOUS="1",
        GRADIO_ANALYTICS_ENABLED="False",
        GRADIO_SERVER_NAME="127.0.0.1",
        PORT="7861",
    )
    env["NO_PROXY"] = env.get("NO_PROXY", "") + ",127.0.0.1,localhost"
    os.environ["NO_PROXY"] = env["NO_PROXY"]
    processes = []
    with tempfile.TemporaryDirectory(prefix="csfsm-smoke-") as temp:
        env["CSFSM_DATABASE"] = str(Path(temp) / "test.sqlite3")
        with open(Path(temp) / "servers.log", "w+") as log:
            try:
                for command in [
                    [sys.executable, "-m", "csfsm_guardrails.cli", "serve", "--port", "8124"],
                    [sys.executable, str(root / "app.py")],
                ]:
                    processes.append(
                        subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
                    )
                with httpx.Client(trust_env=False, timeout=2) as http:
                    for url in ["http://127.0.0.1:8124/health", "http://127.0.0.1:7861/config"]:
                        deadline = time.monotonic() + 45
                        while True:
                            try:
                                if http.get(url).status_code == 200:
                                    break
                            except httpx.HTTPError:
                                pass
                            if time.monotonic() > deadline or any(
                                p.poll() is not None for p in processes
                            ):
                                log.seek(0)
                                raise RuntimeError("Service did not start: " + log.read()[-4000:])
                            time.sleep(0.1)
                    base = "http://127.0.0.1:8124"
                    sid = http.post(base + "/v1/sessions").json()["id"]
                    check = base + f"/v1/sessions/{sid}/check"
                    assert http.post(
                        check,
                        json={"candidate": {"stage": "input", "text": "Explain a state machine"}},
                    ).json()["allowed"]
                    assert not http.post(
                        check, json={"candidate": {"stage": "output", "text": "CSFSM_DEMO_SECRET"}}
                    ).json()["allowed"]
                    (root / "docs/openapi.json").write_text(
                        json.dumps(http.get(base + "/openapi.json").json(), indent=2) + "\n"
                    )
                print("Live HTTP API: input allowed; protected output blocked", flush=True)
                client = Client("http://127.0.0.1:7861", verbose=False)
                client.predict(api_name="/reset")
                result = client.predict(
                    "Explain a state machine", "A state machine tracks states.", api_name="/inspect"
                )
                assert result[0].startswith("RELEASED"), result
                result = client.predict(
                    "Summarize this record", "CSFSM_DEMO_SECRET", api_name="/inspect"
                )
                assert result[0].startswith("WITHHELD") and "CSFSM_DEMO_SECRET" not in result[1], (
                    result
                )
                client.predict(api_name="/reset")
                assert not client.predict("demo-001", api_name="/demo_execute")[-1]["executed"]
                client.predict("demo-001", api_name="/demo_authorize")
                assert client.predict("demo-001", api_name="/demo_execute")[-1]["released"]
                assert not client.predict("demo-001", api_name="/demo_execute")[-1]["executed"]
                print(
                    "Live Gradio: output gating, authorization and single-use grants passed",
                    flush=True,
                )
            finally:
                for process in processes:
                    process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()


if __name__ == "__main__":
    main()
