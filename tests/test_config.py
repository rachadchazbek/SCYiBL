from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scyibl.cli import main
from scyibl.config import ConfigError, ModelRef, load_app_config, load_runtime_config
from scyibl.config.settings import required_secrets

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config" / "agents.yaml"
FAKE_KEY = "test-cohere-key-1234567890"


@pytest.fixture
def base_config() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))
    return data


def write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- schema -----------------------------------------------------------------


def test_repository_config_is_valid() -> None:
    app = load_app_config(REPO_CONFIG)
    assert app.providers_in_use() == {"cohere"}
    assert app.agents["classifier"].few_shot_k == 8


@pytest.mark.parametrize("value", ["command-a", "/model", "cohere/", ""])
def test_model_ref_requires_provider_and_name(value: str) -> None:
    with pytest.raises(ValueError, match="provider/model-name"):
        ModelRef.parse(value)


def test_model_ref_normalises_provider_case() -> None:
    assert str(ModelRef.parse("Cohere/command-a-03-2025")) == "cohere/command-a-03-2025"


def test_undeclared_provider_is_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["agents"]["adjudicator"]["model"] = "openai/gpt-x"
    with pytest.raises(ConfigError, match="undeclared provider"):
        load_app_config(write(tmp_path, base_config))


def test_missing_required_agent(tmp_path: Path, base_config: dict[str, Any]) -> None:
    del base_config["agents"]["safety"]
    with pytest.raises(ConfigError, match="missing required agents"):
        load_app_config(write(tmp_path, base_config))


def test_router_cannot_be_disabled(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["agents"]["router"]["enabled"] = False
    with pytest.raises(ConfigError, match="cannot be disabled"):
        load_app_config(write(tmp_path, base_config))


@pytest.mark.parametrize("band", [[0.6, 0.4], [0.5, 0.5], [-0.1, 0.5], [0.4, 1.2]])
def test_ambiguity_band_must_be_ordered_within_unit_interval(
    tmp_path: Path, base_config: dict[str, Any], band: list[float]
) -> None:
    base_config["escalation"]["ambiguity_band"] = band
    with pytest.raises(ConfigError, match="ambiguity_band"):
        load_app_config(write(tmp_path, base_config))


def test_confidence_out_of_range(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["escalation"]["min_confidence"] = 1.5
    with pytest.raises(ConfigError, match=r"escalation\.min_confidence"):
        load_app_config(write(tmp_path, base_config))


def test_unknown_keys_are_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["agents"]["classifier"]["temprature"] = 0.2  # typo
    with pytest.raises(ConfigError, match="temprature"):
        load_app_config(write(tmp_path, base_config))


def test_missing_file_and_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_app_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("agents: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_app_config(bad)


# --- secrets ----------------------------------------------------------------


def test_missing_provider_key_gives_clear_error() -> None:
    with pytest.raises(ConfigError) as info:
        load_runtime_config(REPO_CONFIG, env_file=None, environ={})
    message = str(info.value)
    assert "COHERE_API_KEY" in message
    assert ".env.example" in message


def test_key_from_environment(tmp_path: Path) -> None:
    cfg = load_runtime_config(REPO_CONFIG, env_file=None, environ={"COHERE_API_KEY": FAKE_KEY})
    assert cfg.provider_api_key("cohere").get_secret_value() == FAKE_KEY


def test_key_from_env_file_and_environment_wins(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("COHERE_API_KEY=from-file\nHF_TOKEN=hf-from-file\n", encoding="utf-8")
    from_file = load_runtime_config(REPO_CONFIG, env_file=env_file, environ={})
    assert from_file.provider_api_key("cohere").get_secret_value() == "from-file"
    assert from_file.secret("HF_TOKEN") is not None

    overridden = load_runtime_config(
        REPO_CONFIG, env_file=env_file, environ={"COHERE_API_KEY": "from-env"}
    )
    assert overridden.provider_api_key("cohere").get_secret_value() == "from-env"


def test_only_providers_in_use_are_required(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["providers"]["openai"] = {"api_key_env": "OPENAI_API_KEY"}
    cfg = load_runtime_config(
        write(tmp_path, base_config), env_file=None, environ={"COHERE_API_KEY": FAKE_KEY}
    )
    assert "OPENAI_API_KEY" not in required_secrets(cfg.app)


def test_switching_a_provider_is_config_only(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["providers"]["openai"] = {"api_key_env": "OPENAI_API_KEY"}
    base_config["agents"]["adjudicator"]["model"] = "openai/some-model"
    path = write(tmp_path, base_config)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_runtime_config(path, env_file=None, environ={"COHERE_API_KEY": FAKE_KEY})
    cfg = load_runtime_config(
        path, env_file=None, environ={"COHERE_API_KEY": FAKE_KEY, "OPENAI_API_KEY": "sk-test"}
    )
    assert str(cfg.app.agents["adjudicator"].model) == "openai/some-model"


def test_enabled_alerts_and_required_dataset_need_their_secrets(
    tmp_path: Path, base_config: dict[str, Any]
) -> None:
    base_config["alerts"]["email"]["enabled"] = True
    base_config["dataset"]["required"] = True
    with pytest.raises(ConfigError) as info:
        load_runtime_config(
            write(tmp_path, base_config), env_file=None, environ={"COHERE_API_KEY": FAKE_KEY}
        )
    for name in ("ALERT_EMAIL_TO", "ALERT_SMTP_PASSWORD", "HF_TOKEN"):
        assert name in str(info.value)


def test_secrets_never_leak_in_repr() -> None:
    cfg = load_runtime_config(REPO_CONFIG, env_file=None, environ={"COHERE_API_KEY": FAKE_KEY})
    assert FAKE_KEY not in repr(cfg)
    assert FAKE_KEY not in str(cfg.secrets)


def test_config_path_from_environment(tmp_path: Path, base_config: dict[str, Any]) -> None:
    path = write(tmp_path, base_config)
    cfg = load_runtime_config(
        None, env_file=None, environ={"COHERE_API_KEY": FAKE_KEY, "SCYIBL_CONFIG": str(path)}
    )
    assert cfg.source == path


# --- CLI --------------------------------------------------------------------


def test_cli_check_ok_masks_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", FAKE_KEY)
    assert (
        main(["config", "check", "--config", str(REPO_CONFIG), "--env-file", "/nonexistent"]) == 0
    )
    out = capsys.readouterr().out
    assert "Config OK" in out
    assert "cohere/command-a-reasoning-08-2025" in out
    assert FAKE_KEY not in out
    assert "****7890" in out


def test_cli_check_reports_missing_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    assert (
        main(["config", "check", "--config", str(REPO_CONFIG), "--env-file", "/nonexistent"]) == 1
    )
    assert "COHERE_API_KEY" in capsys.readouterr().err


def test_graph_and_pricing_sections(tmp_path: Path, base_config: dict[str, Any]) -> None:
    app = load_app_config(REPO_CONFIG)
    assert app.graph.checkpointer == "none"
    assert app.agents["adjudicator"].thinking_budget == 2048
    base_config["pricing"] = {
        "Cohere/command-a-03-2025": {"input_per_mtok": 2.5, "output_per_mtok": 10}
    }
    app = load_app_config(write(tmp_path, base_config))
    price = app.price_for(ModelRef.parse("cohere/command-a-03-2025"))
    assert price is not None
    assert price.output_per_mtok == 10


def test_invalid_checkpointer_rejected(tmp_path: Path, base_config: dict[str, Any]) -> None:
    base_config["graph"]["checkpointer"] = "postgres"
    with pytest.raises(ConfigError, match="checkpointer"):
        load_app_config(write(tmp_path, base_config))
