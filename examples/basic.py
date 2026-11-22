from csfsm_guardrails import Guardrail

session = Guardrail().session()
print(
    session.run("Explain a state machine", lambda _: "States change in response to events.").content
)
result = session.run("Summarize this record", lambda _: "CSFSM_DEMO_SECRET: protected")
assert not result.allowed
print(result.content)
