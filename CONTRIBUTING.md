# Contributing

Use Python 3.11 or newer.
Install `pip install -e '.[api,space,dev,langchain,langgraph]'` and run
`pytest`, `ruff check .` and `ruff format --check .`. Changes to numerical
semantics need probability/invariant tests; enforcement changes need tests that
show blocked content or operations cannot reach the host callback.

Keep optional integrations out of core imports. Do not embed credentials or
download models during import. Document assumptions behind new risk estimates,
and preserve the difference between model scores and conditional bounds.
