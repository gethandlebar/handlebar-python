"""Metric aggregator — mirrors packages/core/src/metrics/aggregator.ts."""

from __future__ import annotations

from ..schema.metrics import InbuiltAgentMetricKind
from .types import MetricInfo
from .utils import validate_metric, validate_metric_key


class AgentMetricCollector:
    """Accumulates and aggregates per-step metrics."""

    def __init__(self) -> None:
        self._inbuilt: dict[str, MetricInfo] = {}
        self._custom: dict[str, MetricInfo] = {}
        self._aggregation_inbuilt: dict[str, MetricInfo] = {}
        self._aggregation_custom: dict[str, MetricInfo] = {}

    # ------------------------------------------------------------------
    # Inbuilt metrics
    # ------------------------------------------------------------------

    def set_inbuilt(
        self, kind: InbuiltAgentMetricKind, value: float, unit: str | None = None
    ) -> None:
        self._inbuilt[kind.value] = MetricInfo(value, unit)

    def add_inbuilt(
        self, kind: InbuiltAgentMetricKind, delta: float, unit: str | None = None
    ) -> None:
        prev = self._inbuilt.get(kind.value)
        prev_value = prev.value if prev else 0.0
        self._inbuilt[kind.value] = MetricInfo(
            prev_value + delta,
            unit if unit is not None else (prev.unit if prev else None),
        )

    # ------------------------------------------------------------------
    # Custom metrics
    # ------------------------------------------------------------------

    def set_custom(self, kind: str, value: float, unit: str | None = None) -> None:
        # Reject keys that collide with inbuilt metric names.
        try:
            InbuiltAgentMetricKind(kind)
            # If we get here the kind IS an inbuilt metric — reject it.
            raise ValueError(f'Custom metric kind "{kind}" collides with inbuilt metric name')
        except ValueError as exc:
            if "collides" in str(exc):
                raise  # re-raise our own error
            # Otherwise the enum lookup failed → kind is not inbuilt → safe to use.
        if not validate_metric_key(kind):
            raise ValueError(f'Invalid custom metric key "{kind}"')
        self._custom[kind] = MetricInfo(value, unit)

    def add_custom(self, kind: str, delta: float, unit: str | None = None) -> None:
        if not validate_metric_key(kind):
            raise ValueError(f'Invalid custom metric key "{kind}"')
        prev = self._custom.get(kind)
        prev_value = prev.value if prev else 0.0
        self._custom[kind] = MetricInfo(
            prev_value + delta,
            unit if unit is not None else (prev.unit if prev else None),
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> None:
        """Merge current step metrics into the running aggregation, then reset step metrics."""
        for key, info in self._inbuilt.items():
            existing = self._aggregation_inbuilt.get(key)
            self._aggregation_inbuilt[key] = MetricInfo(
                (existing.value if existing else 0.0) + info.value,
                existing.unit if existing else info.unit,
            )

        for key, info in self._custom.items():
            existing = self._aggregation_custom.get(key)
            self._aggregation_custom[key] = MetricInfo(
                (existing.value if existing else 0.0) + info.value,
                existing.unit if existing else info.unit,
            )

        self._inbuilt = {}
        self._custom = {}

    def to_event_payload(self, *, aggregate: bool = False) -> dict[str, dict[str, dict]] | None:
        """Return current metrics as a dict payload suitable for an audit event."""
        inbuilt_entries = {k: v for k, v in self._inbuilt.items() if validate_metric(v)}
        custom_entries = {k: v for k, v in self._custom.items() if validate_metric(v)}

        if aggregate:
            self.aggregate()

        if not inbuilt_entries and not custom_entries:
            return None

        def _info_dict(m: MetricInfo) -> dict:
            d: dict = {"value": m.value}
            if m.unit:
                d["unit"] = m.unit
            return d

        return {
            "inbuilt": {k: _info_dict(v) for k, v in inbuilt_entries.items()},
            "custom": {k: _info_dict(v) for k, v in custom_entries.items()},
        }
