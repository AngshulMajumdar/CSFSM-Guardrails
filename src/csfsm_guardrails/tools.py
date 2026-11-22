"""Explicit, schema-validated tool registration. No eval or dynamic imports."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    function: Callable
    schema: type[BaseModel]

    def normalize(self, arguments):
        return self.schema.model_validate(arguments, strict=True).model_dump(mode="json")

    def invoke(self, arguments):
        result = self.function(**arguments)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, function, schema):
        if name in self._tools:
            raise ValueError("Tool already registered")
        if not issubclass(schema, BaseModel) or schema.model_config.get("extra") != "forbid":
            raise ValueError("Tool argument schemas must be Pydantic models with extra='forbid'")
        self._tools[name] = RegisteredTool(name, function, schema)
        return self

    def get(self, name):
        if name not in self._tools:
            raise ValueError("Tool is not registered for execution")
        return self._tools[name]

    def schemas(self):
        return {name: tool.schema.model_json_schema() for name, tool in self._tools.items()}
