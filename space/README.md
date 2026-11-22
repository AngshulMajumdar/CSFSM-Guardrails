---
title: CSFSM Guardrails
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: '3.11'
app_file: app.py
pinned: false
license: apache-2.0
short_description: Stateful guards for model responses and tool execution
---

# CSFSM Guardrails

A CPU-only demonstration of conversational stochastic finite-state guardrails.
Inspect candidate responses, observe posterior state probabilities and enforce
request-bound permissions on a synthetic tool. The same runtime is available as
a reusable Python SDK and HTTP API in the accompanying GitHub package.

No GPU, model download or API key is required. This Space uses a transparent
rule-based evidence encoder and illustrative risk scores. It is an enforcement
framework demonstration, not a trained general-purpose safety classifier.

Verification buttons simulate trusted application events for synthetic records.
Never connect those public demo controls to real permissions or sensitive tools.
Sessions are isolated and kept in memory for up to one hour; do not enter secrets.

Author: Angshul Majumdar, IIIT-Delhi. Apache-2.0.
