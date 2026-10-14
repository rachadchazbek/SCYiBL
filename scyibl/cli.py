"""Command-line entry point: ``scyibl config check`` and ``scyibl data prepare``."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from scyibl.config.settings import (
    DEFAULT_ENV_FILE,
    ConfigError,
    RuntimeConfig,
    load_runtime_config,
    required_secrets,
)


def _mask(value: str) -> str:
    return "set (****" + value[-4:] + ")" if len(value) >= 12 else "set"


def _render(cfg: RuntimeConfig) -> str:
    app = cfg.app
    lines = [f"Config OK: {cfg.source}", "", "Agents:"]
    for name, agent in app.agents.items():
        state = "" if agent.enabled else "  (disabled)"
        lines.append(f"  {name:<12} {agent.model}{state}")
    lines += [
        "",
        "Retrieval:",
        f"  embed        {app.retrieval.embed_model}",
        f"  rerank       {app.retrieval.rerank_model or '-'}",
        "",
        "Escalation:",
        f"  min_confidence={app.escalation.min_confidence} "
        f"ambiguity_band={list(app.escalation.ambiguity_band)} "
        f"max_depth={app.escalation.max_depth} "
        f"max_cost_usd={app.escalation.max_cost_usd_per_request}",
        "",
        "Secrets:",
    ]
    needed = required_secrets(app)
    for name in sorted(set(needed) | set(cfg.secrets)):
        secret = cfg.secret(name)
        status = _mask(secret.get_secret_value()) if secret else "missing"
        role = needed.get(name, "optional")
        lines.append(f"  {name:<32} {status:<16} {role}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scyibl")
    sub = parser.add_subparsers(dest="command", required=True)
    config_cmd = sub.add_parser("config", help="configuration utilities")
    config_sub = config_cmd.add_subparsers(dest="action", required=True)
    check = config_sub.add_parser("check", help="validate config and required secrets")
    check.add_argument("--config", type=Path, default=None, help="path to agents.yaml")
    check.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="path to .env")

    data_cmd = sub.add_parser("data", help="dataset utilities (issue #3)")
    data_sub = data_cmd.add_subparsers(dest="action", required=True)
    prep = data_sub.add_parser("prepare", help="anonymise, clean and split the raw CSV")
    prep.add_argument("--input", type=Path, required=True, help="raw CSV (private storage)")
    prep.add_argument("--out-dir", type=Path, required=True, help="directory outside the repo")
    prep.add_argument("--seed", type=int, default=42, help="split seed")

    args = parser.parse_args(argv)
    if args.command == "data":
        return _data_prepare(args)
    try:
        cfg = load_runtime_config(config_path=args.config, env_file=args.env_file)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(_render(cfg))
    return 0


DATA_EXTRA_MODULES = ("pandas", "pyarrow", "iterstrat")
DATA_EXTRA_HINT = "Install the data extra: uv sync --extra data"


def _data_prepare(args: argparse.Namespace) -> int:
    # The `data` extra is optional and imported lazily. Prints aggregate stats only.
    if any(importlib.util.find_spec(name) is None for name in DATA_EXTRA_MODULES):
        print(DATA_EXTRA_HINT, file=sys.stderr)
        return 1
    from scyibl.data.prepare import DataPrepError, prepare

    try:
        stats = prepare(args.input, args.out_dir, seed=args.seed)
    except (DataPrepError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=2))
    print(f"\nWrote train/validation/test.parquet and stats.json to {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
