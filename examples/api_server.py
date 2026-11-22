"""Run: uvicorn examples.api_server:create_app --factory --port 8000"""

from pydantic import BaseModel, ConfigDict

from csfsm_guardrails import Guardrail, SQLiteStore, ToolRegistry
from csfsm_guardrails.api import APISettings
from csfsm_guardrails.api import create_app as guardrail_app


class LookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    record_id: str


def create_app():
    settings = APISettings.from_env()
    tools = ToolRegistry().register(
        "lookup_record",
        lambda record_id: {"record_id": record_id, "status": "synthetic example"},
        LookupArgs,
    )
    guard = Guardrail(store=SQLiteStore(settings.database), tools=tools)
    return guardrail_app(guard, settings)
