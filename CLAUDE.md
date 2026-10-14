# SCYiBL: guide for Claude Code

SCYiBL classifies journaling sentences into 8 therapeutic-framework categories (multi-label):
CP, Constructivism, Coping, ED, ER/SR, Empowerment, Exposure/Habit, PERMA.
It is being rebuilt from a 2022 RoBERTa classifier (`legacy/`) into a multi-agent service.
Roadmap: epic #15. Decisions: `docs/adr/`.

## Commands

```bash
uv sync                                          # install (Python 3.12+)
uv run pytest                                    # offline tests, no API keys needed
uv run ruff check . && uv run ruff format .      # lint + format
uv run mypy                                      # strict type-check
uv run scyibl config check                       # validate config/agents.yaml + secrets
uv run python scripts/spike_cohere.py --dry-run  # spike with the fake LLM
uv run python scripts/spike_cohere.py            # live Cohere calls (invented sentences only)
uv run scyibl data prepare --input RAW.csv --out-dir DIR  # owner only; DIR outside the repo
```
CI (`.github/workflows/ci.yml`) runs lint, mypy, pytest and a data-file guard on every PR.
After changing dependencies, run `uv lock` and commit `uv.lock`.

## Architecture (ADR-001)

- **LangGraph** orchestrates (`scyibl/graph/`): `router → classify → (adjudicate) → assemble`.
  Retrieval (#6) and a parallel safety branch (#8) come next (#18).
- **Agents are plain async functions** (`scyibl/agents/`) over our own `LLMClient`
  protocol (`scyibl/llm/base.py`). They must not import LangGraph or LangChain types.
- **Providers**: `CohereClient` (official SDK v2) behind `ProviderRouter`. Models are
  `provider/model` strings in `config/agents.yaml`. Do **not** add `langchain-cohere`.
- **Escalation** is the pure function `scyibl/escalation.py::should_escalate`, used by a
  conditional edge. Keep decision logic out of graph nodes so it stays unit-testable.
- **Schemas**: `scyibl/schema.py`. The schema sent to Cohere must not use `minimum`/`maximum`,
  `additionalProperties`, `anyOf`/`oneOf`/`allOf`, `$ref`; every object needs `required`.
  Range checks belong in Pydantic validators.
- **Dataset prep** (`scyibl/data/prepare.py`, #3): drops index columns, maps `Source` to
  `origin` (web/business/ptsd/participant), drops exact duplicates, then a seeded iterative
  stratified 70/15/15 split to Parquet + aggregate `stats.json`. It refuses to write inside a
  git repo and never prints rows. Tests use `tests/fixtures/synthetic_raw.csv` (invented).
- **Categories** are defined once in `scyibl/taxonomy.py`. Definitions are provisional; the
  meanings of CP and ED are unknown (#3). Do not present them as confirmed.
- **Behaviour lives in config**, secrets in env/`.env`. Switching model or provider should be a
  config change only.

## Rules (privacy first: the data is mental-health journaling)

1. **Never commit data or model files** (`csv, parquet, pkl, ckpt, jsonl`, …) or `.env`.
   The dataset lives in private storage (#16). Pre-commit and CI block these files.
2. **Never send dataset sentences to an LLM provider with the trial key.** Spike scripts, tests
   and examples use **invented** sentences only.
3. **Never paste real participant text** into code, tests, issues, PRs or commit messages.
4. Secrets are `SecretStr`; never print or log them (the CLI masks to the last 4 characters).
5. Graph state contains messages: no persistent checkpointer and no LangSmith tracing unless a
   retention/data-handling decision says so (#13).
6. Tests must run offline: use `FakeLLMClient` or an injected fake SDK, never the network.

## Conventions

- Strict typing (`mypy --strict`), ruff line length 100, `from __future__ import annotations`.
- Async tests use `asyncio.run(...)` inside sync tests (no pytest-asyncio).
- One issue per change; reference it in commits/PRs (`Refs #n` / `Closes #n`).
- Open PRs as draft until CI is green; describe a test plan, including owner-only live steps.

## Current state (keep updated)

- Phase 0 done (#2, #4). ADR-001 spike merged (PR #19); live runs 1 and 2 posted in #1.
- Dataset prep implemented (#3); owner has not yet run it on the real data. Splits are frozen
  once its `stats.json` is posted in #3. Label definitions in `docs/labels.md` (unconfirmed).
- Interim `min_confidence: 0.4`; calibrate on the validation split (#11).
- `pricing:` in config is empty, so cost is reported as unknown until it is filled.
