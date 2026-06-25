# n00dles 🍜

Open-source multi-agent AI orchestration. Chain agents, manage state, handle
failures — without 800 lines of boilerplate.

```python
from n00dles import agent, pipeline, run

@agent(model="claude-sonnet-4-6")
def researcher(topic: str) -> str:
    """Research the topic. Return 3 key facts."""

@agent(model="claude-sonnet-4-6")
def writer(research: str) -> str:
    """Write a short article from the research."""

content_pipeline = pipeline(researcher >> writer, retry=3)
result = run(content_pipeline, topic="multi-agent orchestration")
print(result.output)
```

## Install

```bash
pip install get-n00dles
```

(the PyPI distribution name is `get-n00dles` — PyPI rejected the bare `n00dles` as
too visually similar to an existing package; the import is still `import n00dles`)

Set an API key for whichever provider you're using (n00dles wraps
[litellm](https://github.com/BerriAI/litellm), so any provider litellm supports works
— Anthropic, OpenAI, Mistral, Gemini, Ollama, and more):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Why n00dles

- **No magic.** Every behavior is traceable from your code to the LLM call in a
  handful of stack frames. No metaclasses, no hidden registries.
- **Reliable by default.** Retry with exponential backoff, per-node timeouts, and
  checkpointed state are built in — not opt-in extras.
- **One provider layer.** Wraps litellm instead of maintaining five parallel
  provider SDKs. Switch models with a string, not a rewrite.
- **Zero cloud dependency.** This package makes no network calls except to your LLM
  provider. No telemetry, no license checks, no phone-home.

## Core concepts

**Agents** are plain functions decorated with `@agent`. The docstring becomes the
system prompt; the type hints become the I/O contract:

```python
@agent(model="claude-haiku-4-5")
def classify(ticket: str) -> str:
    """Classify the support ticket as billing, technical, or other."""
```

**Pipelines** chain agents with `>>` and wrap the chain with `pipeline()` to attach
retry/timeout policy:

```python
support_flow = pipeline(classify >> respond, retry=3, timeout=60)
result = run(support_flow, ticket="My invoice is wrong")
```

`run()` returns a `RunResult`:

```python
result.output          # the final agent's return value
result.run_id           # unique id — pass to a future resume() call if interrupted
result.duration_ms      # wall-clock time for the whole run
result.total_tokens     # summed across every agent call, including retries
result.agent_traces     # per-agent timing, status, and token usage, in order
```

State is checkpointed to SQLite after every node by default — no config required.
If your agent's return type is a Pydantic model instead of `str`, n00dles instructs
the LLM to respond in JSON and validates the response against your schema, raising
`AgentOutputError` if it doesn't fit.

```python
from n00dles import configure

configure(state_store="sqlite:///my_app.db")   # custom path (default: ./n00dles_state.db)
configure(trace_exporter="otel")               # pip install get-n00dles[otel]
```

## What's in this release (v0.1.0)

This is the first OSS release — sequential composition only. Implemented:

- `@agent`, `pipeline()`, `>>`, `run()`/`arun()`
- litellm provider integration (every major LLM provider)
- Retry with exponential backoff + jitter, per-node timeouts, fallback agents
- SQLite state store (default) with checkpoint-and-resume
- Pydantic I/O validation for structured agent outputs
- Trace events + an optional OpenTelemetry exporter (`pip install get-n00dles[otel]`)

**Not yet implemented** (tracked for the next release — see `PUBLISHING.md` for the
full rollout plan):

- `parallel()` / `branch()` / the `|` operator for fan-out and conditional routing
- Circuit breaker
- Redis state backend
- Langfuse and Helicone exporters
- `mock_agent()` / `MockLLM` testing utilities
- The `noodles` CLI (`run`, `serve`, `deploy`)

## Development

```bash
git clone https://github.com/n00dlehouse/n00dles-py
cd n00dles-py
pip install -e ".[dev]"
pytest
ruff check .
```

Tests run entirely against an in-repo fake provider — no API keys or network access
required.

## License

MIT
