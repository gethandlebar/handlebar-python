# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
from handlebar.langchain import HandlebarMiddleware
from langchain.agents.middleware import AgentMiddleware

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

handler = HandlebarMiddleware(agent_slug="langchain-python")

agent = create_agent(
    model="openai:gpt-5.2-nano",
    tools=[get_weather],
    middleware=[handler],
    system_prompt="You are a helpful assistant",
)


# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]})
print(result)
