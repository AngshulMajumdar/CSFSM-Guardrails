from langchain_core.runnables import RunnableLambda

from csfsm_guardrails import Guardrail
from csfsm_guardrails.integrations.langchain import guarded_runnable

chain = guarded_runnable(
    Guardrail().session(), RunnableLambda(lambda _: "A state machine tracks state.")
)
print(chain.invoke("Explain a state machine"))
