"""
Example: Google ADK agent governed by Handlebar.

Usage:
    uv sync --all-packages
    HANDLEBAR_API_KEY=... OPENAI_API_KEY=... uv run python examples/google_adk_agent.py

Environment variables:
    HANDLEBAR_API_KEY   Your Handlebar API key (required for governance / audit)
    OPENAI_API_KEY      Your OpenAI API key (required for the LLM) - yes, this demo uses OpenAI on a Google agent. sue me.
"""
import os
from typing import List

import asyncio
import datetime
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool

from handlebar.google_adk import HandlebarPlugin

load_dotenv()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_current_time(timezone: str = "UTC") -> dict:
    """Return the current date and time for a given timezone name.

    Args:
        timezone: A timezone name such as "UTC", "US/Eastern", or "Europe/London".

    Returns:
        A dict with keys 'timezone', 'datetime', and 'utc_offset'.
    """
    import zoneinfo

    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except zoneinfo.ZoneInfoNotFoundError:
        return {"error": f"Unknown timezone: {timezone!r}"}

    now = datetime.datetime.now(tz)
    return {
        "timezone": timezone,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "utc_offset": now.strftime("%z"),
    }


def add_numbers(a: float, b: float) -> dict:
    """Add two numbers together.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        A dict with key 'result' containing the sum.
    """
    return {"result": a + b}


# ---------------------------------------------------------------------------
# Build the agent
# ---------------------------------------------------------------------------

time_tool = FunctionTool(func=get_current_time)
math_tool = FunctionTool(func=add_numbers)

# Optionally tag tools so Handlebar rules can target them by tag.
# e.g. a rule could block all tools tagged "external" in production.
time_tool.custom_metadata = {"handlebar_tags": ["utility", "time"]}
math_tool.custom_metadata = {"handlebar_tags": ["utility", "math"]}

model_id = os.getenv("OPENAI_MODEL", "openai/gpt-5-nano")
model = LiteLlm(model=model_id)
agent = LlmAgent(
    model=model,
    name="example_agent",
    description="A simple agent that can tell the time and do arithmetic.",
    instruction=(
        "You are a helpful assistant. Use your tools to answer the user's question. "
        "Always use a tool rather than guessing when one is relevant."
    ),
    tools=[time_tool, math_tool],
)

# ---------------------------------------------------------------------------
# Attach Handlebar governance
# ---------------------------------------------------------------------------
#
# HandlebarPlugin reads HANDLEBAR_API_KEY from the environment automatically.
# Every LLM call and tool invocation is now evaluated, audited, and — if
# a governance rule matches — blocked before it executes.
#
# enforce_mode options:
#   "enforce" — decisions are applied (default)
#   "shadow"  — decisions are logged but not enforced (good for dry-runs)
#   "off"     — no API calls at all

plugin = HandlebarPlugin(
    agent_slug="example-agent",
    enforce_mode="enforce",
)

runner = InMemoryRunner(agent=agent, plugins=[plugin])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


async def main() -> None:
    user_id = "local-user"
    session_id = "local-session-001"

    questions: List[str] = [
        "What time is it in Tokyo right now?",
        "What is 1234 + 5678?",
    ]

    for question in questions:
        print(f"\nUser: {question}")
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=question,
        ):
            if event.is_final_response():
                text = event.content.parts[0].text if event.content and event.content.parts else "(no response)"
                print(f"Agent: {text}")


if __name__ == "__main__":
    asyncio.run(main())
