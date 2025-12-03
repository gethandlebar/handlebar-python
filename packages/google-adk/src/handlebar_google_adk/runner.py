from typing import Literal
from google.adk.runners import Runner
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.sessions import BaseSessionService

from .plugin import HandlebarPlugin

class HandlebarRunner(Runner):
    """
    Wrapper around google.adk.runners.Runner which adds HandlebarPlugin to the agent.

    This provides governance utilities (audit logs, rule enforcement) on tool callbacks.
    """
    def __init__(
        self,
        *,
        agent,
        app_name: str,
        session_service: BaseSessionService,
        plugins: list[BasePlugin] | None = None,
        handlebar_api_key: str,
        handlebar_base_url: str = "https://api.gethandlebar.com",
        handlebar_org_id: str | None = None,
        handlebar_user_category: str = "default",
        tool_categories: dict[str, list[str]] | None = None,
        default_uncategorised: Literal["allow", "block"] = "allow",
        **runner_kwargs,
    ):
        hb_plugin = HandlebarPlugin(
            app_name=app_name,
            handlebar_api_key=handlebar_api_key,
            handlebar_base_url=handlebar_base_url,
            handlebar_org_id=handlebar_org_id,
            handlebar_user_category=handlebar_user_category,
            tool_categories=tool_categories,
            default_uncategorised=default_uncategorised,
        )

        all_plugins = [hb_plugin, *(plugins or [])]

        super().__init__(
            agent=agent,
            app_name=app_name,
            session_service=session_service,
            plugins=all_plugins,
            **runner_kwargs,
        )
