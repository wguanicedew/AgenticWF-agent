"""
ModelCatalog: the models the runtime can use, from two sources (docs/agent-models.md).

    system configuration   [llm.model.<id>] sections of agent.cfg; credentials from the
                           environment (api_key_env) or [llm.provider.<name>]. Read-only at run time
    database               entries a manager adds, changes or removes while the system runs
                           (api/v1/admin/models); their API keys are encrypted in the database
                           (agenticwf.agent.runtime.secrets) and never returned

Database entries belong to the user who added them and are visible to that user, to their
organization, or (managers only) to everyone; only the owner may change or remove one.
Configured entries are visible to everyone and read-only.

Each user may choose their own default among the models they may use (api/v1/models/default): it
takes the place of [llm] default for them, and while they may no longer use it, [llm] default does.

A model id names one entry: provider + provider model name + options (effort, thinking, …).
Everything else in the runtime names models by id: the [llm] default, a thread's or a task
type's `model`, a chat message's model choice. Ids are unique across both sources; the
configuration's win.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.llm.base import LLMError
from agenticwf.agent.runtime.llm.routing import ROUTING_OPTION_KEYS, ModelProfile, check_options, coerce_option
from agenticwf.agent.runtime.types import Owner

LOG = get_logger(__name__)

PROVIDERS = ("anthropic", "openai", "vllm", "ollama", "echo")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
VISIBILITY = ("user", "organization", "all")
OPTION_KEYS = ("effort", "thinking", "thinking_display", "fallbacks", "max_tokens", "token_param",
               "stream_usage", "extra", "timeout", "max_retries", *ROUTING_OPTION_KEYS)


@dataclass(frozen=True)
class ModelEntry:
    id: str
    provider: str
    model: str
    label: str = ""
    description: str = ""
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)   # a database secret, decrypted; never shown
    api_key_env: str | None = None                         # configured models only: the environment variable holding it
    options: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    source: str = "config"                                 # config | database
    api_key_hint: str | None = None
    updated_at: float = field(default_factory=time.time)
    owner_subject: str | None = None     # database entries: who added it
    owner_org: str | None = None
    visibility: str = "all"              # user | organization | all
    api_key_unreadable: bool = False     # database entries: a stored key that cannot be decrypted here

    @property
    def profile(self) -> ModelProfile:
        """What routing knows of it: tier, size, capabilities, costs (docs/model-routing.md)."""
        return ModelProfile.of(self.options)

    def visible_to(self, owner: Owner | None) -> bool:
        """None: the runtime itself (no caller to filter for)."""
        if owner is None or self.visibility == "all":
            return True
        if self.visibility == "organization":
            return self.owner_org is not None and owner[1] == self.owner_org
        return (self.owner_subject, self.owner_org) == owner

    def owned_by(self, owner: Owner | None) -> bool:
        return self.source == "database" and owner is not None and (self.owner_subject, self.owner_org) == owner

    def public(self) -> dict[str, Any]:
        """What users see: no credentials, no endpoints."""
        profile = self.profile
        return {"id": self.id, "label": self.label or self.id, "description": self.description,
                "provider": self.provider, "model": self.model, "tier": profile.tier,
                "context_window": profile.context_window, "capabilities": sorted(profile.capabilities)}

    def admin_view(self) -> dict[str, Any]:
        """What managers see: everything but the secret itself."""
        return {**self.public(), "base_url": self.base_url, "api_key_env": self.api_key_env,
                "has_api_key": bool(self.api_key), "api_key_hint": self.api_key_hint,
                "options": dict(self.options), "enabled": self.enabled, "source": self.source,
                "visibility": self.visibility, "updated_at": int(self.updated_at)}

    def view_for(self, owner: Owner | None) -> dict[str, Any]:
        """The settings page's row: an owner sees their entry's settings (never the key itself);
        everyone else sees what any user sees, and which group it is in."""
        if self.owned_by(owner):
            return {**self.admin_view(), "editable": True, "group": "mine"}
        group = "system" if self.source == "config" else "shared"
        return {**self.public(), "source": self.source, "visibility": self.visibility, "enabled": self.enabled,
                "editable": False, "group": group}


def validate(entry: ModelEntry) -> None:
    if not ID_RE.match(entry.id):
        raise ValueError("model id: lowercase letters, digits, '.', '_' or '-' (at most 64)")
    if entry.id == "default":
        raise ValueError("'default' is reserved: it names the configured default model")
    if entry.provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}")
    if not entry.model.strip():
        raise ValueError("model: the provider's model name is required")
    if entry.base_url and not re.match(r"^https?://", entry.base_url):
        raise ValueError("base_url must be an http(s) URL")
    unknown = set(entry.options) - set(OPTION_KEYS)
    if unknown:
        raise ValueError(f"unknown options: {', '.join(sorted(unknown))}")
    check_options(entry.options)
    if entry.visibility not in VISIBILITY:
        raise ValueError(f"visibility must be one of {', '.join(VISIBILITY)}")
    if entry.visibility == "organization" and not entry.owner_org:
        raise ValueError("visibility 'organization' needs an organization-scoped sign-in")


class ModelStore(Protocol):
    """Database-held entries (agenticwf.agent.runtime.llm.db_catalog)."""

    def list_models(self) -> list[ModelEntry]: ...
    def save_model(self, entry: ModelEntry, keep_key: bool = False) -> None: ...
    def delete_model(self, model_id: str) -> bool: ...
    # optional: users' own defaults (a store without them: everyone has [llm] default)
    # def user_defaults(self) -> dict[Owner, str]: ...
    # def set_user_default(self, owner: Owner, model_id: str | None) -> None: ...


class ModelCatalog:
    def __init__(self, configured: list[ModelEntry] | None = None, provider_settings: dict[str, dict[str, Any]] | None = None,
                 default: str | None = None, fallbacks: list[str] | None = None, store: ModelStore | None = None,
                 refresh_seconds: float = 30.0, allowed_hosts: tuple[str, ...] = ()):
        for entry in configured or ():
            validate(entry)
        self._configured = {e.id: e for e in configured or ()}
        self.provider_settings = dict(provider_settings or {})
        self._default = default
        self.fallbacks = list(fallbacks or [])
        self.store = store
        self.refresh_seconds = refresh_seconds
        self.allowed_hosts = tuple(allowed_hosts)   # [llm] allowed_hosts: where users' own models may point
        self._database: dict[str, ModelEntry] = {}
        self._user_defaults: dict[Owner, str] = {}
        self._loaded_at = 0.0
        self._lock = threading.Lock()
        self.version = 0     # bumped on every change: the router drops adapters it built for older entries

    # -- reading --------------------------------------------------------------------------------
    def _refresh(self, force: bool = False) -> None:
        """Reload database entries every `refresh_seconds` (other processes may have changed them)."""
        if self.store is None or (not force and time.time() - self._loaded_at < self.refresh_seconds):
            return
        with self._lock:
            try:
                loaded = {e.id: e for e in self.store.list_models()}
                if hasattr(self.store, "user_defaults"):
                    self._user_defaults = dict(self.store.user_defaults())
            except Exception:   # keep serving the last good list rather than failing every request
                LOG.exception("could not load models from the database")
                self._loaded_at = time.time()
                return
            for clash in set(loaded) & set(self._configured):
                LOG.warning("database model %r ignored: the configuration defines that id", clash)
                loaded.pop(clash)
            if loaded != self._database:
                self._database = loaded
                self.version += 1
            self._loaded_at = time.time()

    def entries(self, include_disabled: bool = False, owner: Owner | None = None) -> list[ModelEntry]:
        """All entries, or those `owner` may use."""
        self._refresh()
        merged = [*self._configured.values(), *sorted(self._database.values(), key=lambda e: e.id)]
        return [e for e in merged if (include_disabled or e.enabled) and e.visible_to(owner)]

    def get(self, model_id: str) -> ModelEntry | None:
        self._refresh()
        return self._configured.get(model_id) or self._database.get(model_id)

    @property
    def default_id(self) -> str | None:
        if self._default:
            return self._default
        enabled = self.entries()
        return enabled[0].id if enabled else None

    def user_default(self, owner: Owner | None) -> str | None:
        """The default `owner` chose, while they may still use it; None: [llm] default applies."""
        if owner is None:
            return None
        self._refresh()
        chosen = self._user_defaults.get(tuple(owner))
        return chosen if chosen and self.is_available(chosen, owner) else None

    def default_for(self, owner: Owner | None) -> str | None:
        return self.user_default(owner) or self.default_id

    def resolve(self, ref: str | None, owner: Owner | None = None) -> ModelEntry:
        """A model id, or "default" / None for the default (`owner`'s own, else [llm] default). Unknown,
        disabled or (for `owner`) invisible ids fall back to it, so a task pinned to a removed model still runs."""
        wanted = None if ref in (None, "", "default") else ref
        entry = self.get(wanted) if wanted else None
        if entry is not None and entry.enabled and entry.visible_to(owner):
            return entry
        if wanted:
            LOG.warning("model %r is unknown or disabled; using the default", wanted)
        default_id = self.default_for(owner)
        default = self.get(default_id) if default_id else None
        if default is None or not default.enabled:
            raise LLMError("no model available: configure [llm] default or add a model", provider=None)
        return default

    def is_available(self, model_id: str, owner: Owner | None = None) -> bool:
        entry = self.get(model_id)
        return entry is not None and entry.enabled and entry.visible_to(owner)

    # -- changing (database entries only) -------------------------------------------------------
    def endpoint_refusal(self, entry: ModelEntry, resolve: bool = True) -> str | None:
        """Why a user's model may not be called where it points, or None. Configured models are the
        operators' and are not checked. A user's model reaches its provider's own API, the
        operators' [llm.provider.*] base_url, or a host on [llm] allowed_hosts; never anything else.
        `resolve=False` (at every call, on the event loop): the allow-list only, no DNS lookup."""
        if entry.source == "config":
            return None
        if entry.api_key_env:
            return "api_key_env is only for models operators configure in agent.cfg"
        url = entry.base_url
        if not url and entry.provider == "ollama" and not self.provider_settings.get("ollama", {}).get("base_url"):
            url = "http://localhost:11434/v1"   # the adapter's default: a server-side address
        if not url:
            return None
        from urllib.parse import urlsplit

        from agenticwf.agent.runtime.net import check_host, host_allowed

        host = urlsplit(url).hostname or ""
        if not resolve:
            return None if host_allowed(host, self.allowed_hosts) else f"host {host!r} is not on the allow-list for models"
        return check_host(host, self.allowed_hosts, "models ([llm] allowed_hosts)")

    def put(self, entry: ModelEntry, by: Owner | None = None, keep_key: bool = False) -> ModelEntry:
        """Add or replace a database entry. With `by`, an existing entry must belong to `by`.
        `keep_key`: leave the stored (encrypted) API key as it is, readable here or not."""
        if self.store is None:
            raise PermissionError("models can only be added to the database when it is configured")
        if entry.id in self._configured:
            raise PermissionError(f"model {entry.id!r} comes from the configuration and cannot be changed here")
        existing = self._database.get(entry.id)
        if by is not None and existing is not None and not existing.owned_by(by):
            raise PermissionError(f"model {entry.id!r} belongs to someone else")
        entry = replace(entry, source="database", updated_at=time.time())
        validate(entry)
        refusal = self.endpoint_refusal(entry)
        if refusal:
            raise ValueError(refusal)
        if keep_key:
            self.store.save_model(entry, keep_key=True)
        else:
            self.store.save_model(entry)
        self._refresh(force=True)
        return self.get(entry.id) or entry

    def set_user_default(self, owner: Owner, model_id: str | None) -> None:
        """Choose `owner`'s default: a model they may use, or None for [llm] default."""
        if self.store is None or not hasattr(self.store, "set_user_default"):
            raise PermissionError("a default of your own needs the database")
        if model_id is not None and not self.is_available(model_id, owner):
            raise ValueError(f"model {model_id!r} is unknown, disabled or not shared with you")
        self.store.set_user_default(tuple(owner), model_id)
        self._refresh(force=True)

    def delete(self, model_id: str, by: Owner | None = None) -> bool:
        if model_id in self._configured:
            raise PermissionError(f"model {model_id!r} comes from the configuration and cannot be removed here")
        if self.store is None:
            return False
        existing = self.get(model_id)
        if by is not None and existing is not None and not existing.owned_by(by):
            raise PermissionError(f"model {model_id!r} belongs to someone else")
        removed = self.store.delete_model(model_id)
        self._refresh(force=True)
        return removed


# -- configuration ------------------------------------------------------------------------------
def _coerce(key: str, value: str) -> Any:
    import json

    if key in ("max_tokens", "max_retries"):
        return int(value)
    if key == "timeout":
        return float(value)
    if key == "stream_usage":
        return value.strip().lower() in ("1", "true", "yes", "on")
    if key == "extra":
        return json.loads(value)
    if key in ROUTING_OPTION_KEYS:
        return coerce_option(key, value)
    return value


def entries_from_sections(sections: dict[str, dict[str, str]]) -> tuple[list[ModelEntry], dict[str, dict[str, Any]]]:
    """[llm.model.<id>] and [llm.provider.<name>] sections (name → options) → entries and provider settings."""
    models, providers = [], {}
    for name, values in sections.items():
        if name.startswith("llm.provider."):
            provider = name[len("llm.provider."):]
            providers[provider] = {k: (_coerce(k, v) if k in OPTION_KEYS else v) for k, v in values.items()}
        elif name.startswith("llm.model."):
            model_id = name[len("llm.model."):]
            options = {k: _coerce(k, v) for k, v in values.items() if k in OPTION_KEYS}
            models.append(ModelEntry(
                id=model_id, provider=values.get("provider", ""), model=values.get("model", ""),
                label=values.get("label", ""), description=values.get("description", ""),
                base_url=values.get("base_url") or None, api_key_env=values.get("api_key_env") or None,
                options=options, enabled=values.get("enabled", "true").lower() not in ("0", "false", "no", "off")))
    return models, providers
