from csfsm_guardrails import Guardrail
from csfsm_guardrails.integrations.langgraph import guarded_graph

graph = guarded_graph(Guardrail().session(), lambda _: "A state machine tracks state.")
result = graph.invoke({"prompt": "Explain a state machine"})
print(result["output"])
