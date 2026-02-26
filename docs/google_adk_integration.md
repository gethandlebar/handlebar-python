# Handlebar Google ADK Agents Integration Guide

## Prerequisits

- A [Handlebar account][platform]
- Handlebar API key (created on the platform)

## Google ADK

Install the adapter alongside the Google ADK SDK:

```bash
pip install handlebar-google-adk google-adk
```

### Minimal setup

Set your API key in the environment and add a single plugin to your runner:

```python
import os
os.environ["HANDLEBAR_API_KEY"] = "<your platform API key>"

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from handlebar.google_adk import HandlebarPlugin

agent = LlmAgent(
    model="gemini-2.5-flash",
    name="my_agent",
    tools=[...],
)

plugin = HandlebarPlugin(agent_slug="my_agent")
runner = InMemoryRunner(agent=agent, plugins=[plugin])

async for event in runner.run_async(
    user_id="user_123",
    session_id="session_456",
    new_message="Do the thing",
):
    if event.is_final_response():
        print(event.content.parts[0].text)
```

That's it. Every LLM call and tool invocation is now governed and audited.\
You can view your agent runs and configure rules on the [Handlebar platform][platform].

---

### Additional config

```python
from handlebar.core import AgentDescriptor, ConsoleSinkConfig
from handlebar.google_adk import HandlebarPlugin

plugin = HandlebarPlugin(
    agent=AgentDescriptor(
        slug="my-agent",
        name="My Agent",
        description="Does useful things",
    ),
    enforce_mode="enforce",   # "enforce" | "shadow" | "off"
    fail_closed=False,        # True = block all tool calls if API is unreachable
    sinks=[ConsoleSinkConfig(format="pretty")],  # override default HTTP sink to output Handlebar governance logs to console
)
```

| `enforce_mode` | Behaviour |
|---|---|
| `"enforce"` | Governance decisions are applied — blocked tools are stopped |
| `"shadow"` | Decisions are evaluated and logged but never enforced |
| `"off"` | No API calls; pass-through only |

---

### With a pre-initialised client

If you need multiple runners to share one client (and therefore one connection / audit stream):

```python
from handlebar.core import HandlebarClient, HandlebarClientConfig, AgentDescriptor
from handlebar.google_adk import HandlebarPlugin

config = HandlebarClientConfig(agent=AgentDescriptor(slug="my-agent"))
client = await HandlebarClient.init(config)

plugin = HandlebarPlugin(client=client)
runner = InMemoryRunner(agent=agent, plugins=[plugin])
```

---

### Tool tags

By defining metadata on your tools, you can enforce useful policies to prevent dangerous actions. For example:
- rate limiting "expensive" tool calls
- Blocking data exfil, e.g. a "pii read" operation flowing into "external"
- Redacting PII from external-facing tools

ADK tools don't have a native tags concept. Pass Handlebar tags via `custom_metadata` when you define the tool so that governance rules can match on them:

```python
from google.adk.tools import FunctionTool

def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email."""
    ...

tool = FunctionTool(func=send_email)
tool.custom_metadata = {"handlebar_tags": ["email", "external-comms"]}

agent = LlmAgent(model="gemini-2.5-flash", name="my_agent", tools=[tool])
```

---

### What happens on a block

When a governance rule blocks a tool call:

- The tool does **not** execute.
- The agent receives `{"error": "Blocked by Handlebar governance: <reason>"}` as the tool result.
- If the rule carries a `TERMINATE` control signal, the agent loop stops immediately after the block (no further LLM calls are made).
- The run is ended with status `"interrupted"` and all events are flushed to the audit log.

---

### Sync usage

If you are running outside an async context, use `run_sync` from Handlebar core to initialise the client before passing it to the plugin:

```python
from handlebar.core import HandlebarClient, HandlebarClientConfig, AgentDescriptor

config = HandlebarClientConfig(agent=AgentDescriptor(slug="my-agent"))
client = HandlebarClient.init_sync(config)

plugin = HandlebarPlugin(client=client)
```

Note: `InMemoryRunner.run_async()` still needs to be awaited — this only covers client initialisation.

[platform]: https://app.gethandlebar.com
