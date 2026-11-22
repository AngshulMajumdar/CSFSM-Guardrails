"""Wrap an entire text-producing LangChain Runnable with enforcing gates."""

from langchain_core.runnables import RunnableLambda


def guarded_runnable(session, runnable):
    """Return a Runnable[str, str]. Native tool calls must use execute_tool.

    A fresh Session is required per independent conversation. Streaming buffers
    the complete response because partial tokens have not passed the output gate.
    """

    def text(value):
        if getattr(value, "tool_calls", None):
            raise TypeError("Route tool calls through Session.execute_tool")
        content = getattr(value, "content", value)
        if not isinstance(content, str):
            raise TypeError("Runnable must return text or a text AIMessage")
        return content

    def invoke(prompt, config):
        return session.run(prompt, lambda p: text(runnable.invoke(p, config=config))).content

    async def ainvoke(prompt, config):
        async def generate(p):
            return text(await runnable.ainvoke(p, config=config))

        return (await session.arun(prompt, generate)).content

    return RunnableLambda(invoke, afunc=ainvoke, name="csfsm_guarded_generation")
