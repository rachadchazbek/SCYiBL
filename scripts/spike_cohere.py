"""Spike for ADR-001 (issue #1): Cohere + LangGraph on INVENTED sentences.

Never put dataset sentences in this file: a trial key is not covered by Cohere's data
commitments.

    uv run python scripts/spike_cohere.py            # live, needs COHERE_API_KEY in .env
    uv run python scripts/spike_cohere.py --dry-run  # offline, fake LLM

Writes a JSON report to reports/spike/ (git-ignored) and prints a summary to paste in #1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scyibl.config.settings import load_runtime_config
from scyibl.graph.build import build_graph
from scyibl.llm.base import LLMClient, LLMError
from scyibl.llm.fake import ChatCall, FakeLLMClient
from scyibl.schema import ClassificationResult
from scyibl.taxonomy import Category

# (invented sentence, labels a human would likely expect; provisional, see taxonomy.py)
SENTENCES: list[tuple[str, list[str]]] = [
    ("Writing my worries down before bed helps me calm down and fall asleep.", ["ER/SR", "Coping"]),
    ("I want to become someone who believes she can change her own life.", ["Empowerment"]),
    (
        "Every evening I note three good things from my day and it makes me feel grateful.",
        ["PERMA"],
    ),
    (
        "Little by little I'm going back to the places that used to make me anxious.",
        ["Exposure/Habit"],
    ),
    (
        "Looking back, I now see that leaving that job was the start of a new chapter for me.",
        ["CP", "Constructivism"],
    ),
    ("When work gets overwhelming, I take a walk and split my tasks into small steps.", ["Coping"]),
    ("Some mornings I wake up irritable for no clear reason.", ["ER/SR"]),
    (
        "Journaling every night has become a routine I don't even think about anymore.",
        ["Exposure/Habit"],
    ),
    ("I remember the day we moved out; the house felt empty and so did I.", ["CP"]),
    ("I feel a deep relief now that I've accepted things I can't control.", ["ED", "ER/SR"]),
]

RETRIEVAL_DOCS = [
    "Breathing slowly when I panic helps me get my emotions back under control.",
    "I signed up for the half-marathon and I know I can do it.",
    "I make my bed every morning so the day starts with a small win.",
    "Spending Sunday dinner with my family gives my week meaning.",
    "Talking to a friend when I am stressed helps me find solutions.",
    "I keep thinking about the summer my grandfather passed away.",
]
RETRIEVAL_QUERY = "When anxiety hits, I try to slow my breathing until I feel calmer."
EXPECTED_TOP_DOC = 0


def _fake_responder(call: ChatCall) -> dict[str, Any]:
    """Uncertain on 'anxious'/'irritable' sentences so the escalation path is exercised."""
    text = call.user.lower()
    unsure = "anxious" in text or "irritable" in text
    first_pass = "First assessment" not in call.system
    labels = {}
    for category in Category:
        score = 0.55 if (unsure and first_pass) else 0.1
        if category is Category.ER_SR and ("calm" in text or "irritable" in text):
            score = 0.9 if not (unsure and first_pass) else 0.55
        labels[category.value] = {"present": score >= 0.5, "score": score, "rationale": "fake"}
    return {"labels": labels}


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


async def _retrieval_check(client: LLMClient, app: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"embed_model": str(app.retrieval.embed_model)}
    try:
        docs = await client.embed(
            model=app.retrieval.embed_model, texts=RETRIEVAL_DOCS, input_type="search_document"
        )
        query = await client.embed(
            model=app.retrieval.embed_model, texts=[RETRIEVAL_QUERY], input_type="search_query"
        )
        sims = sorted(
            ((i, _cosine(query.vectors[0], v)) for i, v in enumerate(docs.vectors)),
            key=lambda p: -p[1],
        )
        out.update(
            embed_ok=True,
            embed_dim=len(docs.vectors[0]),
            embed_top3=[[i, round(s, 4)] for i, s in sims[:3]],
            embed_top1_correct=sims[0][0] == EXPECTED_TOP_DOC,
            embed_latency_ms=docs.latency_ms + query.latency_ms,
            embed_tokens=docs.usage.embed_tokens + query.usage.embed_tokens,
        )
    except LLMError as exc:
        out.update(embed_ok=False, embed_error=str(exc))

    if app.retrieval.rerank_model is not None:
        out["rerank_model"] = str(app.retrieval.rerank_model)
        try:
            rr = await client.rerank(
                model=app.retrieval.rerank_model,
                query=RETRIEVAL_QUERY,
                documents=RETRIEVAL_DOCS,
                top_n=3,
            )
            out.update(
                rerank_ok=True,
                rerank_top3=[[i, round(s, 4)] for i, s in rr.ranking],
                rerank_top1_correct=bool(rr.ranking) and rr.ranking[0][0] == EXPECTED_TOP_DOC,
                rerank_latency_ms=rr.latency_ms,
                rerank_search_units=rr.usage.search_units,
            )
        except LLMError as exc:
            out.update(rerank_ok=False, rerank_error=str(exc))
    return out


async def run(dry_run: bool, limit: int, delay_s: float, out_dir: Path) -> dict[str, Any]:
    if dry_run:
        cfg = load_runtime_config(env_file=None, environ={"COHERE_API_KEY": "dry-run"})
        client: LLMClient = FakeLLMClient(_fake_responder)
    else:
        cfg = load_runtime_config()
        from scyibl.llm.factory import build_llm_client

        client = build_llm_client(cfg)
    app = cfg.app
    graph = build_graph(app, client)

    rows: list[dict[str, Any]] = []
    for i, (sentence, expected) in enumerate(SENTENCES[:limit], start=1):
        if i > 1 and delay_s:
            await asyncio.sleep(delay_s)  # stay under trial-key rate limits
        rid = f"spike-{i}"
        config = {"recursion_limit": app.graph.recursion_limit, "configurable": {"thread_id": rid}}
        try:
            state = await graph.ainvoke({"request_id": rid, "message": sentence}, config=config)
        except Exception as exc:  # record and continue: the spike measures failures too
            rows.append({"id": i, "sentence": sentence, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:>2}] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        result: ClassificationResult = state["result"]
        first = ClassificationResult.label_view(state["first_pass"])
        first_predicted = [k for k, v in first.items() if v.score >= 0.5]
        final, exp = set(result.predicted), set(expected)
        rows.append(
            {
                "id": i,
                "sentence": sentence,
                "expected": expected,
                "first_pass_predicted": first_predicted,
                "predicted": result.predicted,
                "overlap": sorted(final & exp),
                "exact_match": final == exp,
                "extra_labels": sorted(final - exp),
                "missing_labels": sorted(exp - final),
                "adjudicator_added": sorted(final - set(first_predicted)),
                "adjudicator_removed": sorted(set(first_predicted) - final),
                "confidence": result.confidence,
                "escalated": result.escalated,
                "escalation_reasons": result.escalation_reasons,
                "models": result.models,
                "latency_ms": result.latency_ms,
                "node_latency_ms": {t.node: t.latency_ms for t in state.get("trace", [])},
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cost_usd": result.cost_usd,
                "first_pass_scores": {k: v.score for k, v in first.items()},
                "labels": {k: v.model_dump() for k, v in result.labels.items()},
            }
        )

    ok = [r for r in rows if "error" not in r]
    latencies = sorted(r["latency_ms"] for r in ok)
    summary = {
        "mode": "dry-run" if dry_run else "live",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": {name: str(a.model) for name, a in app.agents.items()},
        "sentences": len(rows),
        "succeeded": len(ok),
        "schema_valid_rate": round(len(ok) / len(rows), 3) if rows else 0.0,
        "escalated": sum(r["escalated"] for r in ok),
        "with_expected_label": sum(bool(r["overlap"]) for r in ok),
        "exact_match": sum(r["exact_match"] for r in ok),
        "extra_labels": sum(len(r["extra_labels"]) for r in ok),
        "missing_labels": sum(len(r["missing_labels"]) for r in ok),
        "adjudicator_added": sum(len(r["adjudicator_added"]) for r in ok),
        "adjudicator_removed": sum(len(r["adjudicator_removed"]) for r in ok),
        "latency_ms_p50_single_pass": _median([r["latency_ms"] for r in ok if not r["escalated"]]),
        "latency_ms_p50_escalated": _median([r["latency_ms"] for r in ok if r["escalated"]]),
        "latency_ms_p50": statistics.median(latencies) if latencies else None,
        "latency_ms_max": latencies[-1] if latencies else None,
        "input_tokens": sum(r["input_tokens"] for r in ok),
        "output_tokens": sum(r["output_tokens"] for r in ok),
        "cost_usd": None
        if any(r["cost_usd"] is None for r in ok)
        else round(sum(r["cost_usd"] for r in ok), 6),
    }
    report = {"summary": summary, "retrieval": await _retrieval_check(client, app), "rows": rows}

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"spike-{summary['mode']}-{int(time.time())}.json"
    path.write_text(json.dumps(report, indent=2, default=lambda o: asdict(o)), encoding="utf-8")
    report["path"] = str(path)
    return report


def _print(report: dict[str, Any]) -> None:
    print(f"\n{'#':>2}  {'escalated':<9} {'latency':>8}  first pass -> final  |  expected")
    for r in report["rows"]:
        if "error" in r:
            print(f"{r['id']:>2}  ERROR  {r['error'][:90]}")
            continue
        esc = "yes" if r["escalated"] else "no"
        print(
            f"{r['id']:>2}  {esc:<9} {r['latency_ms']:>6.0f}ms  "
            f"{', '.join(r['first_pass_predicted']) or '-'} -> {', '.join(r['predicted']) or '-'}"
            f"  |  {', '.join(r['expected'])}"
        )
    print("\nSummary (paste this in issue #1):")
    print(json.dumps({"summary": report["summary"], "retrieval": report["retrieval"]}, indent=2))
    print(f"\nFull report: {report['path']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="use the fake LLM (no network)")
    parser.add_argument("--limit", type=int, default=len(SENTENCES), help="number of sentences")
    parser.add_argument("--delay", type=float, default=None, help="seconds between sentences")
    parser.add_argument("--out", type=Path, default=Path("reports/spike"))
    args = parser.parse_args(argv)
    delay = args.delay if args.delay is not None else (0.0 if args.dry_run else 4.0)
    report = asyncio.run(run(args.dry_run, args.limit, delay, args.out))
    _print(report)
    return 0 if report["summary"]["succeeded"] else 1


if __name__ == "__main__":
    sys.exit(main())
