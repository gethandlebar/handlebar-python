"""Budget manager — mirrors packages/core/src/budget-manager.ts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BudgetGrant:
    id: str
    decision: Literal["allow", "block", "defer"]
    grant: float | None  # remaining budget (None = unlimited)
    computed: dict | None  # {kind, value}
    expires_seconds: int | None = None


class BudgetManager:
    """Client-side cache of metric budget grants from the server.

    Tracks rolling metric windows and tells the caller when a server-side
    re-evaluation is needed.
    """

    def __init__(
        self,
        global_ttl_seconds: int = 60,
        budgets: list[BudgetGrant] | None = None,
    ) -> None:
        self.global_ttl_seconds = global_ttl_seconds
        self.budgets: list[BudgetGrant] = budgets or []
        self._last_evaluated_ms = time.time() * 1000

    def update_budgets(self, ttl_seconds: int, new_budgets: list[BudgetGrant]) -> None:
        """Replace the cached budget grants with fresh data from the server."""
        self.budgets = new_budgets
        self.global_ttl_seconds = ttl_seconds
        self._last_evaluated_ms = time.time() * 1000

    def usage(self, rule_metric_updates: dict[str, float]) -> None:
        """Subtract per-rule metric usage from cached grants."""
        new_budgets: list[BudgetGrant] = []
        for budget in self.budgets:
            grant_usage = rule_metric_updates.get(budget.id)
            if budget.grant is not None and grant_usage is not None:
                new_budgets.append(
                    BudgetGrant(
                        id=budget.id,
                        decision=budget.decision,
                        grant=budget.grant - grant_usage,
                        computed=budget.computed,
                        expires_seconds=budget.expires_seconds,
                    )
                )
            else:
                new_budgets.append(budget)
        self.budgets = new_budgets

    def reevaluate(self) -> bool:
        """Return True if the server should be asked for fresh budget grants."""
        now_ms = time.time() * 1000
        time_since_last_ms = now_ms - self._last_evaluated_ms
        time_until_next_ms = self.global_ttl_seconds * 1000 - time_since_last_ms

        if time_until_next_ms <= 0:
            return True

        for budget in self.budgets:
            if budget.grant is not None and budget.grant <= 0:
                return True

        return False
