# SCYiBL

SCYiBL classifies journaling sentences into eight therapeutic frameworks (multi-label):
CP, Constructivism, Coping, ED, ER/SR, Empowerment, Exposure/Habit and PERMA.

> **Status: rewrite in progress.** The 2022 RoBERTa service lives in [`legacy/`](legacy/README.md).
> The new version is a configurable multi-agent service (router → classifier → adjudicator,
> with a safety agent). See the [epic #15](https://github.com/rachadchazbek/scyibl/issues/15).

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # install runtime + dev dependencies
cp .env.example .env             # then fill in COHERE_API_KEY (never commit .env)
uv run scyibl config check       # validate config/agents.yaml and required secrets
uv run pytest                    # tests run offline, without API keys
uv run pre-commit install        # lint/format + blocks data and .env files on commit
```

### Spike (issue #1)

```bash
uv run python scripts/spike_cohere.py --dry-run   # offline, fake LLM
uv run python scripts/spike_cohere.py             # live Cohere calls on invented sentences
```
The report is written to `reports/spike/` (git-ignored). Paste the printed summary in issue #1.

## Architecture

LangGraph orchestrates plain async agents that call providers through our own `LLMClient`
interface. See [ADR-001](docs/adr/0001-agent-framework.md).

## Configuration

| What | Where |
|---|---|
| Models per agent, escalation thresholds, retrieval, alerts | [`config/agents.yaml`](config/agents.yaml) |
| Secrets (provider API keys, dataset token, alert credentials) | Environment variables or a local `.env` ([`.env.example`](.env.example)) |
| Alternative config file | `SCYIBL_CONFIG=path/to/agents.yaml` |

Models are written as `provider/model`, for example `cohere/command-a-03-2025`. Only the secrets
needed by the active configuration are required, so switching provider or model is a config-only change.

## Data

The dataset is **not** in this repository. It contains mental-health journaling sentences and
lives in private, access-controlled storage (issue #16). Data and model files are blocked by
pre-commit and CI.
