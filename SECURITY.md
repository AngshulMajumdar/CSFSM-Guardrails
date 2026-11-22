# Security boundary

The library mediates only operations routed through its enforcing wrappers.
Checks return decisions; ignoring a blocked decision bypasses enforcement. The
host must ensure all content release and protected tool calls use the guarded
paths. Existing provider tool auto-execution must be disabled or replaced.

Keep policy files, tool registry, detector extensions, API credentials, store
access and trusted events under host control. A model response must never supply
Python code, an import path, a verification event or a grant. The API rejects
extra JSON fields and separates client/authority credentials. Tool arguments are
strictly schema-validated, but the tool itself must enforce its own resource
permissions and timeouts. A policy requiring no grant/verification explicitly
weakens those requirements for that tool.

The bundled regular expressions and stochastic parameters have limited coverage.
They can miss harmful text, misclassify benign text, or be bypassed by semantic
rephrasing. This release does not claim universal jailbreak prevention. For a
particular deployment, attach suitable input/output detectors and evaluate false
accepts, false rejects and behavior across whole conversations. `model_score`
values must not be represented to users as measured safety probabilities.

Streaming buffers the whole response before release. Do not stream raw provider
tokens or LangGraph intermediate events/checkpoints to users. Registered tool
results are checked before disclosure; an output block cannot undo side effects
that already happened inside a tool. Keep tools narrow and authorize their exact
arguments before execution.

Tool execution is serialized with permission changes in the configured store.
SQLite is single-host and holds a write lock during the tool call; keep it bounded.
A crash after reservation is reported as indeterminate and is never automatically
retried. Do not delete pending execution state to retry an irreversible operation
without reconciling it with the external service. Tools should use external
idempotency keys where available. Callbacks must not re-enter the same store.

Session reset is not a global attacker quota. Enforce identity-level rate limits,
authorization and budgets in the host if users can create new sessions. The
included per-process limiter is not a distributed abuse-prevention service.
Use HTTPS at ingress, secret management, database access controls and appropriate
data retention. Raw approved tool results may persist for replay until cleanup.

Report security issues privately to the repository maintainer before publishing
details that affect deployed users. No external reporting address is assumed.
