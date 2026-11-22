"""Command line entry points; optional dependencies are imported on demand."""

import argparse
import json
from pathlib import Path

from .guardrail import Guardrail
from .policy import Policy


def main():
    parser = argparse.ArgumentParser(prog="csfsm")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Inspect one input in a new conversation")
    check.add_argument("text")
    check.add_argument("--policy")
    validate = sub.add_parser("validate", help="Validate a YAML policy")
    validate.add_argument("policy")
    schema = sub.add_parser("schema", help="Export the policy JSON Schema")
    schema.add_argument("--output")
    serve = sub.add_parser("serve", help="Start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    demo = sub.add_parser("demo", help="Start the CPU Gradio demo")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.command == "check":
        policy = Policy.from_file(args.policy) if args.policy else None
        print(Guardrail(policy).session().check_input(args.text).model_dump_json(indent=2))
    elif args.command == "validate":
        policy = Policy.from_file(args.policy)
        print(json.dumps({"valid": True, "name": policy.name, "digest": policy.digest}))
    elif args.command == "schema":
        value = json.dumps(Policy.model_json_schema(), indent=2) + "\n"
        if args.output:
            Path(args.output).write_text(value)
        else:
            print(value)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("csfsm_guardrails.api:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "demo":
        from .demo import launch

        launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
