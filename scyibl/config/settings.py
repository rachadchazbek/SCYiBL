"""Load ``config/agents.yaml`` and resolve the secrets it references.

Secrets come from the process environment first, then from an optional ``.env`` file.
Only the secrets actually needed by the active configuration are required, so switching
provider or model is a config-only change. All problems are reported together.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import dotenv_values
from pydantic import SecretStr, ValidationError

from scyibl.config.models import AppConfig

DEFAULT_CONFIG_PATH = Path("config/agents.yaml")
DEFAULT_ENV_FILE = Path(".env")
CONFIG_PATH_ENV = "SCYIBL_CONFIG"


class ConfigError(Exception):
    """Raised when the configuration is invalid or required secrets are missing."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        bullet_list = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"Invalid SCYiBL configuration:\n{bullet_list}")


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated configuration plus the secrets it needs. Secrets never appear in repr."""

    app: AppConfig
    source: Path
    secrets: Mapping[str, SecretStr] = field(repr=False)

    def provider_api_key(self, provider: str) -> SecretStr:
        env_name = self.app.providers[provider].api_key_env
        try:
            return self.secrets[env_name]
        except KeyError:
            raise ConfigError(
                [f"provider '{provider}' is not in use or its key was not loaded"]
            ) from None

    def secret(self, env_name: str) -> SecretStr | None:
        return self.secrets.get(env_name)


def load_app_config(path: Path) -> AppConfig:
    """Parse and validate the YAML file, turning validation errors into readable messages."""
    if not path.is_file():
        raise ConfigError([f"config file not found: {path}"])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError([f"{path}: invalid YAML ({exc})"]) from exc
    if not isinstance(raw, dict):
        raise ConfigError([f"{path}: top level must be a mapping"])
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ConfigError(problems) from exc


def read_environment(
    env_file: Path | None, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Merge ``.env`` values with the process environment (the process environment wins)."""
    merged: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        merged.update({k: v for k, v in dotenv_values(env_file).items() if v})
    merged.update({k: v for k, v in (os.environ if environ is None else environ).items() if v})
    return merged


def required_secrets(app: AppConfig) -> dict[str, str]:
    """Map each required env-var name to a human description of why it is needed."""
    needed: dict[str, str] = {}
    for provider in sorted(app.providers_in_use()):
        needed[app.providers[provider].api_key_env] = f"API key for provider '{provider}'"
    if app.dataset.required:
        needed[app.dataset.token_env] = f"read token for dataset '{app.dataset.repo_id}'"
    email = app.alerts.email
    if email.enabled:
        needed[email.to_env] = "recipient address for safety alert emails"
        needed[email.username_env] = "SMTP username for safety alert emails"
        needed[email.password_env] = "SMTP password for safety alert emails"
    sheet = app.alerts.google_sheet
    if sheet.enabled:
        needed[sheet.sheet_id_env] = "Google Sheet id for the safety alert log"
        needed[sheet.credentials_file_env] = "service-account credentials file for Google Sheets"
    return needed


def optional_secrets(app: AppConfig) -> list[str]:
    """Secrets that are loaded when present but not required by the current config."""
    names = [p.api_key_env for p in app.providers.values()]
    names.append(app.dataset.token_env)
    return names


def load_runtime_config(
    config_path: Path | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    """Load the config and resolve its secrets, failing with every problem listed at once."""
    env = read_environment(env_file, environ)
    path = config_path or Path(env.get(CONFIG_PATH_ENV, str(DEFAULT_CONFIG_PATH)))
    app = load_app_config(path)

    needed = required_secrets(app)
    missing = [
        f"environment variable {name} is not set ({why})"
        for name, why in needed.items()
        if name not in env
    ]
    if missing:
        hint = "Set them in your shell or in a .env file (see .env.example)."
        raise ConfigError([*missing, hint])

    names = set(needed) | set(optional_secrets(app))
    secrets = {name: SecretStr(env[name]) for name in names if name in env}
    return RuntimeConfig(app=app, source=path, secrets=secrets)
