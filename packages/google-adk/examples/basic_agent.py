"""
Basic demo of the Handlebar plugin on google-adk.

To run:
    - set environment variables in root `.env`:
        - HANDLEBAR_API_KEY
        - OPENAI_API_KEY
    - from root, run `uv --package handlebar-google-adk packages/google-adk/examples/basic_agent.py`
"""

from dotenv import load_dotenv

import asyncio
import os
from datetime import datetime
from typing import Any, Dict

from google.adk.agents import Agent  # alias for LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types as genai_types

from handlebar_google_adk import HandlebarRunner


load_dotenv()

# ---------------------------
# Simple tool for the agent
# ---------------------------


def weather(city: str):
    """Get the current weather for a given city using a WeatherStation API"""
    return {
        "weather": "sunny",
    }


def timenow():
    """Get the current time in ISO format"""
    return datetime.utcnow().isoformat() + "Z"


async def call_agent_async(
    query: str,
    runner: Runner,
    user_id: str,
    session_id: str,
) -> str:
    print(f"\n>>> User: {query}")

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=query)],
    )

    final_text = "Agent did not produce a final response."

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_text = event.content.parts[0].text or final_text
            break  # If the break clause is in, after_run_callback is not invoked
        else:
            if (
                event.content
                and event.content.parts
                and isinstance(event.content.parts, list)
            ):
                event_text = ""
                for part in event.content.parts:
                    if part.text:
                        event_text += part.text + "; "
                print(event_text)

    print(f"\n<<< Agent: {final_text}")
    return final_text


# ---------------------------
# Main: wire ADK + OpenAI + Handlebar
# ---------------------------


async def main() -> None:
    # ---- Env / config ----
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in the environment.")

    handlebar_api_key = os.getenv("HANDLEBAR_API_KEY")
    if not handlebar_api_key:
        raise RuntimeError("HANDLEBAR_API_KEY must be set in the environment.")

    handlebar_app_name = os.getenv("HANDLEBAR_APP_NAME", "handlebar_openai_example")
    handlebar_base_url = os.getenv("HANDLEBAR_BASE_URL", "https://api.gethandlebar.com")
    print(f"Hitting {handlebar_base_url}")

    model_id = os.getenv("OPENAI_MODEL", "openai/gpt-5-nano")

    tools = [timenow, weather]
    tool_categories = {
        "get_current_time": ["time", "readonly", "low_risk"],
        "get_current_weather": ["readonly", "low_risk", "api"],
    }

    agent = Agent(
        name=handlebar_app_name,
        model=LiteLlm(model=model_id),
        instruction=(
            "You are a helpful, general assistant. "
            "Prefer calling tools instead of guessing. "
            "Use get_current_time when the user asks about the current time "
            "or anything that depends on 'now'."
        ),
        tools=tools,
    )

    # ---- Session service + HandlebarRunner ----
    session_service = InMemorySessionService()

    runner = HandlebarRunner(
        agent=agent,
        app_name=handlebar_app_name,
        session_service=session_service,
        handlebar_api_key=handlebar_api_key,
        handlebar_base_url=handlebar_base_url,
        # Optional extra metadata:
        tool_categories=tool_categories,
        default_uncategorised="allow",
    )

    user_id = "demo-user"
    session_id = "demo-session-1"

    await session_service.create_session(
        app_name=handlebar_app_name,
        user_id=user_id,
        session_id=session_id,
    )

    # ---- Simple REPL ----
    print("Handlebar + Google ADK + OpenAI demo")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        await call_agent_async(q, runner, user_id=user_id, session_id=session_id)


if __name__ == "__main__":
    asyncio.run(main())
