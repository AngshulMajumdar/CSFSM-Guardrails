import pytest
from pydantic import BaseModel, ConfigDict

from csfsm_guardrails import Guardrail, MemoryStore, SQLiteStore, ToolRegistry


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    return (
        MemoryStore() if request.param == "memory" else SQLiteStore(tmp_path / "sessions.sqlite3")
    )


@pytest.fixture
def registered():
    calls = []

    def lookup(record_id):
        calls.append(record_id)
        return {"record_id": record_id, "status": "synthetic"}

    return ToolRegistry().register("lookup_record", lookup, Args), calls


@pytest.fixture
def session(store, registered):
    tools, calls = registered
    return Guardrail(store=store, tools=tools).session(), calls
