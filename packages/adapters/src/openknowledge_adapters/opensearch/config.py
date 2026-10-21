from __future__ import annotations

from pydantic import BaseModel, ConfigDict, HttpUrl, PositiveFloat, SecretStr, field_validator


class OpenSearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    url: HttpUrl
    user: str
    password: SecretStr
    timeout_seconds: PositiveFloat
    """Per request, where a bulk request carries a few megabytes."""

    @field_validator("url")
    @classmethod
    def _without_credentials(cls, url: HttpUrl) -> HttpUrl:
        """The user and password have their own fields, so the URL, which is no secret, has none."""
        if url.username is not None or url.password is not None:
            raise ValueError("the URL must not carry a user or password")
        return url
