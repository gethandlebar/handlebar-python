"""End-user / actor configuration types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _Base(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class EndUserGroupConfig(_Base):
    external_id: str
    metadata: dict[str, str] | None = None


class EndUserConfig(_Base):
    external_id: str
    metadata: dict[str, str] | None = None
