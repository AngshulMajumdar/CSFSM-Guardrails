# CSFSM Guardrails

**Stateful conversational guardrails based on a stochastic extended finite-state machine.**

CSFSM Guardrails is a Python library and HTTP service that gates model inputs,
candidate responses and registered tool calls. It maintains a probability
distribution over conversational modes and a separate trusted protocol state.
Applications can integrate the SDK, REST clients, or LangChain/LangGraph adapters.
The accompanying Gradio Space runs entirely on CPU.

The default language detector and numerical parameters are a transparent,
hand-configured baseline. They are **not a pretrained safety classifier or a
calibrated probability of real-world harm**. The stochastic controller, protocol
enforcement and extension interfaces are implemented; production semantic
coverage depends on the encoder, validators, risk model and policy you install.

## Install and run

From this repository (the package is not assumed to be published on PyPI):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[api,space]'
csfsm check 'Explain how a finite state machine works'
csfsm demo
```

Or install the wheel included in the release with `pip install path/to/csfsm_guardrails-0.1.0-py3-none-any.whl`.
Use the `api`, `client`, `space`, `langchain`, `langgraph` and `dev` extras as needed.

## Enforce model output

```python
from csfsm_guardrails import Guardrail

guard = Guardrail()
conversation = guard.session()


def your_model(prompt: str) -> str:
    return "A finite state machine has states and transitions."


result = conversation.run("Explain a finite state machine", your_model)
print(result.content)  # candidate output or a policy fallback
print(result.allowed)
```

`run` checks the input before calling the generator and checks the actual draft
before release. `arun`, `stream` and `astream` provide async and buffered streaming
variants. A blocked draft is never yielded by these wrappers. Concurrent changes
between generation and release raise a revision conflict and release no draft.

`check_input`, `check_output` and `check_tool` return **decisions only**. Call
`decision.require_allowed()` or branch on `allowed` before exposing content or
performing an operation. A decision is not an authorization token. For actual
tool enforcement, use `execute_tool`.

## Enforce tools

```python
from pydantic import BaseModel, ConfigDict
from csfsm_guardrails import Guardrail, ToolRegistry


class LookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str


tools = ToolRegistry().register(
    "lookup_record", lambda record_id: {"record_id": record_id, "status": "demo"}, LookupArgs
)
conversation = Guardrail(tools=tools).session()
# Trusted application code only, after real authentication and authorization:
conversation.trusted_event("verified")
conversation.grant_tool("lookup_record", {"record_id": "demo-001"}, ttl_seconds=60)
result = conversation.execute_tool(
    "lookup_record", {"record_id": "demo-001"}, execution_id="request-001"
)
print(result.result if result.released else result.error or result.decision.message)
```

Grants bind a tool and canonical arguments to one session, expire, and can be
used once. Tool functions must be registered by the host and have strict Pydantic
argument schemas. User text cannot create verification or permissions.
`trusted_event('revoked')` clears grants. Reusing an execution ID returns an
authorized cached result; it never runs a completed function again. A crash after
reservation leaves an indeterminate execution that requires host reconciliation.

## HTTP API

```bash
# Use independently generated secrets of at least 24 characters.
export CSFSM_API_KEYS='{"REPLACE_WITH_CLIENT_SECRET_24_PLUS":{"tenant":"app","role":"client"},"REPLACE_WITH_AUTHORITY_SECRET_24_PLUS":{"tenant":"app","role":"authority"}}'
csfsm serve --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation: `http://localhost:8000/docs`.
See [API guide](docs/API.md) for routes, clients, roles and retry behavior.
The default service persists sessions to SQLite. To register executable tools,
use an application factory as shown in [examples/api_server.py](examples/api_server.py).
`CSFSM_ALLOW_ANONYMOUS=1` is an explicit local-development option, never the default.

## LangChain and LangGraph

```python
from langchain_core.runnables import RunnableLambda
from csfsm_guardrails import Guardrail
from csfsm_guardrails.integrations.langchain import guarded_runnable

chain = guarded_runnable(Guardrail().session(), RunnableLambda(lambda p: "A safe answer."))
print(chain.invoke("Explain a state machine"))
```

See [integration examples](examples/) and [architecture](docs/ARCHITECTURE.md).
The LangGraph adapter provides input gate → generation → output gate; only its
final `output` field is intended for user display. Internal graph streams and
checkpoint contents may contain drafts and must remain private.

## Policy and deployment

- [Policy configuration and extension interfaces](docs/POLICIES.md)
- [Mathematical implementation](docs/MATHEMATICS.md)
- [API guide](docs/API.md)
- [Deployment and Hugging Face Space](docs/DEPLOYMENT.md)
- [Security boundary and known limitations](SECURITY.md)
- [Validation results](docs/VALIDATION.md)

```bash
pip install -e '.[api,space,dev,langchain,langgraph]'
pytest
ruff check .
python -m build
python scripts/build_space.py --output build/hf-space
```

The Space package includes its own copy of the runtime and does not require a
PyPI release, a model download, a GPU or a paid inference service.

Author: Angshul Majumdar, IIIT-Delhi. Licensed under Apache-2.0.
