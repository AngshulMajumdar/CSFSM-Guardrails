"""A small enforcing LangGraph with explicit input and output decision nodes."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GuardedState(TypedDict, total=False):
    prompt: str
    draft: str
    output: str
    input_decision: dict
    output_decision: dict


def guarded_graph(session, generate):
    """Compile a text graph; return only `output` to the application's user.

    Intermediate drafts are internal graph state. Do not expose graph events,
    checkpoint storage or stream_mode='updates' to untrusted consumers.
    """

    def check_input(state):
        decision = session.check_input(state["prompt"])
        return {
            "input_decision": decision.model_dump(),
            "draft": "",
            "output_decision": {},
            "output": "" if decision.allowed else decision.message,
        }

    def produce(state):
        draft = generate(state["prompt"])
        if not isinstance(draft, str):
            raise TypeError("Generator must return text")
        return {"draft": draft}

    def check_output(state):
        decision = session.check_output(
            state["draft"], expected_revision=state["input_decision"]["revision"]
        )
        return {
            "output": state["draft"] if decision.allowed else decision.message,
            "draft": "",
            "output_decision": decision.model_dump(),
        }

    graph = StateGraph(GuardedState)
    graph.add_node("input_gate", check_input)
    graph.add_node("generate", produce)
    graph.add_node("output_gate", check_output)
    graph.add_edge(START, "input_gate")
    graph.add_conditional_edges(
        "input_gate",
        lambda s: "generate" if s["input_decision"]["allowed"] else END,
        {"generate": "generate", END: END},
    )
    graph.add_edge("generate", "output_gate")
    graph.add_edge("output_gate", END)
    return graph.compile()
